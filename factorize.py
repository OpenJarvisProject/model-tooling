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
    """Returns [(parent_module, submodule_name, linear), ...] for every
    targeted matrix in the model."""
    targets = []
    for layer in model.model.layers:
        for submodule_name in TARGET_SUBMODULE_NAMES:
            for parent_name in ("self_attn", "mlp"):
                parent = getattr(layer, parent_name, None)
                if parent is None or not hasattr(parent, submodule_name):
                    continue
                linear = getattr(parent, submodule_name)
                if isinstance(linear, nn.Linear):
                    targets.append((parent, submodule_name, linear))
    return targets


def reconstruction_error(linear: nn.Linear, keep_fraction: float) -> float:
    """Cheap, forward-pass-free safety proxy: how much does the
    rank-truncated matrix actually differ from the original? Used to
    try the safest (lowest-error) matrices first, same role cosine
    similarity played for layer-removal pruning."""
    factorized, _, _ = factorize_linear(linear, keep_fraction)
    W_original = linear.weight.data.to(torch.float32)
    W_approx = factorized.second.weight.data.to(torch.float32) @ factorized.first.weight.data.to(torch.float32)
    return ((W_approx - W_original).norm() / W_original.norm()).item()


def factorize_iteratively(model, tokenizer, device: str, keep_fraction: float, max_ppl_ratio: float):
    """Unlike the first (batch) attempt, which factorized all 168
    target matrices at once and broke the model outright (9509x
    baseline perplexity, see README) despite each individual matrix
    only having ~1% reconstruction error in isolation - this factorizes
    one matrix at a time, safest (lowest reconstruction error) first,
    validating perplexity after each and reverting any that fails.
    Same lesson as prune_layers.py's iterative rewrite, applied to a
    different technique."""
    baseline_ppl = compute_perplexity(model, tokenizer, VALIDATION_TEXTS, device)
    print(f"Baseline perplexity: {baseline_ppl:.3f}")

    targets = find_target_linears(model)
    scored = [(reconstruction_error(linear, keep_fraction), parent, name, linear) for parent, name, linear in targets]
    scored.sort(key=lambda t: t[0])  # safest (lowest error) first

    accepted = 0
    current_ppl = baseline_ppl
    total_original_params = 0
    total_new_params = 0

    for error, parent, name, linear in scored:
        factorized, orig_p, new_p = factorize_linear(linear, keep_fraction)
        setattr(parent, name, factorized)
        trial_ppl = compute_perplexity(model, tokenizer, VALIDATION_TEXTS, device)
        ratio = trial_ppl / baseline_ppl
        if ratio <= max_ppl_ratio:
            accepted += 1
            current_ppl = trial_ppl
            total_original_params += orig_p
            total_new_params += new_p
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
    return total_original_params, total_new_params


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

    total_original, total_new = factorize_iteratively(
        model, tokenizer, device, args.keep_fraction, args.max_ppl_ratio
    )
    if total_original > 0:
        reduction = 1 - total_new / total_original
        print(
            f"Factorized target layers: {total_original:,} -> {total_new:,} params "
            f"({reduction * 100:.1f}% reduction in targeted layers)"
        )

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
