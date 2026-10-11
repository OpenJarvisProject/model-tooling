"""Low-rank factorization: replace each large nn.Linear's weight
matrix W (shape [out, in]) with two smaller ones, W ≈ A @ B, via
truncated SVD. Unlike whole-layer removal (a discrete, extreme
operation - delete a whole block), this is a continuous approximation
of each targeted weight matrix, so it's expected to degrade more
gently - but that's an expectation to verify, not assume, given the
batch-removal lesson from pruning (see prune_layers.py's history).

W = U @ diag(S) @ V^T (full SVD). Truncate to rank k: keep the top-k
singular values/vectors. Split the singular values between two
layers so neither one is poorly scaled:
  - first Linear:  in -> k,   weight = diag(sqrt(S_k)) @ V_k^T
  - second Linear: k -> out,  weight = U_k @ diag(sqrt(S_k))
Applying first then second reconstructs U_k @ diag(S_k) @ V_k^T, the
rank-k truncated SVD approximation of W.

Targets the large attention (q/k/v/o_proj) and MLP (gate/up/down_proj)
linear layers - the same layers that dominate parameter count, and the
standard targets for this technique in the literature.
"""

import argparse

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

VALIDATION_TEXTS = [
    "The history of the Roman Empire spans over a thousand years of conquest and decline.",
    "To bake a good loaf of bread, you need flour, water, yeast, and salt.",
    "The human brain contains roughly eighty-six billion neurons.",
    "Renewable energy sources include solar, wind, and hydroelectric power.",
]

TARGET_SUBMODULE_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


class FactorizedLinear(nn.Module):
    """Two sequential Linear layers approximating one larger one."""

    def __init__(self, first: nn.Linear, second: nn.Linear):
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, x):
        return self.second(self.first(x))


def factorize_linear(linear: nn.Linear, keep_fraction: float) -> tuple[FactorizedLinear, int, int]:
    """Returns (factorized_module, original_param_count, new_param_count)."""
    W = linear.weight.data.to(torch.float32)
    out_features, in_features = W.shape
    max_rank = min(out_features, in_features)
    k = max(1, round(max_rank * keep_fraction))

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_k = U[:, :k]
    S_k = S[:k]
    Vh_k = Vh[:k, :]
    sqrt_S_k = torch.sqrt(S_k)

    first = nn.Linear(in_features, k, bias=False)
    first.weight.data = (sqrt_S_k.unsqueeze(1) * Vh_k).to(linear.weight.dtype)

    second = nn.Linear(k, out_features, bias=linear.bias is not None)
    second.weight.data = (U_k * sqrt_S_k.unsqueeze(0)).to(linear.weight.dtype)
    if linear.bias is not None:
        second.bias.data = linear.bias.data.clone()

    original_params = out_features * in_features + (out_features if linear.bias is not None else 0)
    new_params = k * (in_features + out_features) + (out_features if linear.bias is not None else 0)
    return FactorizedLinear(first, second), original_params, new_params


def find_target_linears(model):
    """Returns [(layer_idx, parent_module, parent_name, submodule_name,
    linear), ...] for every targeted matrix in the model. layer_idx and
    parent_name are tracked specifically so the exact set of replaced
    matrices can be saved as metadata and replayed on load - see
    `load_factorized_model`."""
    targets = []
    for layer_idx, layer in enumerate(model.model.layers):
        for submodule_name in TARGET_SUBMODULE_NAMES:
            for parent_name in ("self_attn", "mlp"):
                parent = getattr(layer, parent_name, None)
                if parent is None or not hasattr(parent, submodule_name):
                    continue
                linear = getattr(parent, submodule_name)
                if isinstance(linear, nn.Linear):
                    targets.append((layer_idx, parent, parent_name, submodule_name, linear))
    return targets


def factorize_iteratively(model, tokenizer, device: str, keep_fraction: float, max_ppl_ratio: float, ppl_fn=None):
    """Unlike the first (batch) attempt, which factorized all 168
    target matrices at once and broke the model outright (9509x
    baseline perplexity, see README) despite each individual matrix
    only having ~1% reconstruction error in isolation - this factorizes
    one matrix at a time, safest (lowest reconstruction error) first,
    validating perplexity after each and reverting any that fails.
    Same lesson as prune_layers.py's iterative rewrite, applied to a
    different technique."""
    if ppl_fn is None:
        ppl_fn = lambda m: compute_perplexity(m, tokenizer, VALIDATION_TEXTS, device)
    baseline_ppl = ppl_fn(model)
    print(f"Baseline perplexity: {baseline_ppl:.3f}")

    targets = find_target_linears(model)
    scored = []
    for layer_idx, parent, parent_name, name, linear in targets:
        built = factorize_linear(linear, keep_fraction)
        W = linear.weight.data.to(torch.float32)
        W_approx = built[0].second.weight.data.to(torch.float32) @ built[0].first.weight.data.to(torch.float32)
        error = ((W_approx - W).norm() / W.norm()).item()
        scored.append((error, layer_idx, parent, parent_name, name, linear, built))
    scored.sort(key=lambda t: t[0])  # safest (lowest error) first

    accepted = 0
    current_ppl = baseline_ppl
    total_original_params = 0
    total_new_params = 0
    applied = []  # metadata for load_factorized_model: which matrices were actually replaced

    for error, layer_idx, parent, parent_name, name, linear, built in scored:
        factorized, orig_p, new_p = built
        setattr(parent, name, factorized)
        trial_ppl = ppl_fn(model)
        ratio = trial_ppl / baseline_ppl
        if ratio <= max_ppl_ratio:
            accepted += 1
            current_ppl = trial_ppl
            total_original_params += orig_p
            total_new_params += new_p
            applied.append(
                {"layer_idx": layer_idx, "parent_name": parent_name, "submodule_name": name, "rank": factorized.first.out_features}
            )
            print(
                f"  [{accepted}/{len(scored)} tried] factorized (recon_err={error:.4f}): "
                f"perplexity now {trial_ppl:.3f} ({ratio:.2f}x baseline) -> kept"
            )
        else:
            setattr(parent, name, linear)
            print(
                f"  factorizing matrix (recon_err={error:.4f}) would push perplexity to "
                f"{trial_ppl:.3f} ({ratio:.2f}x baseline) -> reverted"
            )

    print(
        f"Done: {accepted}/{len(scored)} matrices factorized at keep_fraction={keep_fraction}, "
        f"final perplexity {current_ppl:.3f} ({current_ppl / baseline_ppl:.2f}x baseline)"
    )
    return total_original_params, total_new_params, applied


def apply_factorization_structure(model, applied: list[dict]):
    """Replaces the listed matrices with correctly-shaped (but
    untrained/empty) FactorizedLinear modules, matching exactly what
    factorize_iteratively produced - must be called BEFORE loading a
    state_dict that was saved from a factorized model. Plain
    AutoModelForCausalLM.from_pretrained has no idea FactorizedLinear
    exists, reconstructs the standard architecture, finds the
    checkpoint's `*.first.weight`/`*.second.weight` keys don't match
    the expected plain `*.weight` key, and silently random-inits those
    matrices instead of loading the real trained weights - this is
    exactly what broke the first saved checkpoint (see README)."""
    for entry in applied:
        layer = model.model.layers[entry["layer_idx"]]
        parent = getattr(layer, entry["parent_name"])
        linear = getattr(parent, entry["submodule_name"])
        k = entry["rank"]
        first = nn.Linear(linear.in_features, k, bias=False).to(linear.weight.dtype)
        second = nn.Linear(k, linear.out_features, bias=linear.bias is not None).to(linear.weight.dtype)
        setattr(parent, entry["submodule_name"], FactorizedLinear(first, second))


def load_factorized_model(path):
    """Correct counterpart to the broken plain from_pretrained reload.
    The weights saved by plain save_pretrained are actually fine -
    they're stored under the real key names (e.g.
    `...q_proj.first.weight`) with the real trained values. The only
    thing broken was the *reload* skeleton: plain from_pretrained
    builds a standard architecture that doesn't have `.first`/`.second`
    submodules at all, so those keys never match anything and get
    silently dropped/randomized. Fix: build the correct skeleton via
    `apply_factorization_structure` first, then load the same
    safetensors file into it directly."""
    import json
    import os

    from safetensors.torch import load_file
    from transformers import AutoConfig

    with open(os.path.join(path, "factorization_metadata.json")) as f:
        applied = json.load(f)
    # Build from config only - AutoModelForCausalLM.from_pretrained(path)
    # would immediately try (and fail) to load the mismatched checkpoint
    # into the plain architecture first, producing the exact confusing
    # missing/unexpected-key warnings this function exists to avoid.
    config = AutoConfig.from_pretrained(path)
    # from_config, unlike from_pretrained(dtype="auto"), does NOT read
    # config.torch_dtype automatically - it defaults to float32, which
    # then mismatches the actual (e.g. bf16) checkpoint data during the
    # forward pass. Resolve and pass it explicitly.
    config_dtype = getattr(config, "torch_dtype", None)
    if config_dtype is None:
        dtype = torch.float32
    elif isinstance(config_dtype, torch.dtype):
        dtype = config_dtype
    else:
        dtype = getattr(torch, config_dtype)
    model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    apply_factorization_structure(model, applied)
    state_dict = load_file(os.path.join(path, "model.safetensors"))
    # strict=False: tied weights (e.g. lm_head tied to the input
    # embedding when config.tie_word_embeddings) aren't saved twice, so
    # they're legitimately absent from the safetensors file - re-tie
    # explicitly afterward rather than silently trusting from_config's
    # random init for that tensor.
    result = model.load_state_dict(state_dict, strict=False)
    expected_missing = {"lm_head.weight"} if getattr(config, "tie_word_embeddings", False) else set()
    unexpected_missing = set(result.missing_keys) - expected_missing
    if unexpected_missing or result.unexpected_keys:
        raise RuntimeError(
            f"load_factorized_model found real mismatches beyond the expected tied "
            f"lm_head: missing={unexpected_missing}, unexpected={result.unexpected_keys}"
        )
    if getattr(config, "tie_word_embeddings", False):
        model.lm_head.weight = model.model.embed_tokens.weight
    return model


def compute_perplexity(model, tokenizer, texts, device: str) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            out = model(**inputs, labels=inputs["input_ids"])
            n_tokens = inputs["input_ids"].numel()
            total_loss += out.loss.item() * n_tokens
            total_tokens += n_tokens
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--keep-fraction",
        type=float,
        default=0.5,
        help="fraction of the max possible rank to keep per matrix (default 0.5)",
    )
    parser.add_argument(
        "--max-ppl-ratio",
        type=float,
        default=1.5,
        help="reject a matrix's factorization if it pushes perplexity above this multiple of baseline",
    )
    args = parser.parse_args()

    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype="auto")
    model.to(device)

    total_original, total_new, applied = factorize_iteratively(
        model, tokenizer, device, args.keep_fraction, args.max_ppl_ratio
    )
    if total_original > 0:
        reduction = 1 - total_new / total_original
        print(
            f"Factorized target layers: {total_original:,} -> {total_new:,} params "
            f"({reduction * 100:.1f}% reduction in targeted layers)"
        )

    import json
    import os

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "factorization_metadata.json"), "w") as f:
        json.dump(applied, f)
    print(f"Saved to {args.output_dir} ({len(applied)} factorized matrices recorded in metadata)")

    print("Verifying the saved checkpoint actually reloads correctly...")
    reloaded = load_factorized_model(args.output_dir)
    reload_ppl = compute_perplexity(reloaded, tokenizer, VALIDATION_TEXTS, device)
    print(f"Reloaded model perplexity: {reload_ppl:.3f} (should match the in-script final perplexity above)")


if __name__ == "__main__":
    main()
