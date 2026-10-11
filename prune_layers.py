"""Whole-layer removal pruning (structural, depth pruning).

Removes entire transformer decoder blocks rather than individual
weights (unlike Wanda-style unstructured sparsity) — this genuinely
shrinks the model's tensor shapes, so it actually produces a smaller
file and faster inference on commodity runtimes like llama.cpp, which
has no sparse-tensor-core support to exploit zeroed-but-same-shaped
weights.

Importance metric: for each decoder layer, measure how close that
layer's output is to its own input (cosine similarity) across
calibration text. A layer whose output is nearly identical to its
input is contributing little — it's acting close to an identity
function — and is a safe removal candidate. This is the "Block
Influence" idea from depth-pruning work like ShortGPT.

Removal is ITERATIVE, not a single batch: importance is re-measured
after every single removal, and each removal is validated against a
held-out perplexity check before being kept. A first attempt that
removed 6 layers in one shot, based on one importance snapshot of the
*intact* model, broke the model outright on a 24-layer model — removing
one layer changes what the next layer actually receives as input, which
a one-shot ranking never accounts for. This is the correctness fix, not
an optional refinement.

CPU-only: ROCm is confirmed dead on this machine's RX 580 (see
README). This is a one-time, infrequent job, so CPU is an accepted
trade-off, not an oversight.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CALIBRATION_TEXTS = [
    "The capital of France is Paris, a city known for its art and culture.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "In software engineering, a bug is an error in a program that causes it to behave unexpectedly.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Machine learning models are trained on large datasets to recognize patterns.",
    "The stock market experienced significant volatility during the economic downturn.",
    "Mount Everest is the tallest mountain above sea level on Earth.",
    "A balanced diet includes proteins, carbohydrates, fats, vitamins, and minerals.",
]

# Deliberately distinct from CALIBRATION_TEXTS, so the perplexity check
# isn't validated against the exact same text the importance ranking
# was computed from.
VALIDATION_TEXTS = [
    "The history of the Roman Empire spans over a thousand years of conquest and decline.",
    "To bake a good loaf of bread, you need flour, water, yeast, and salt.",
    "The human brain contains roughly eighty-six billion neurons.",
    "Renewable energy sources include solar, wind, and hydroelectric power.",
]


def measure_layer_importance(model, tokenizer, device: str) -> list[float]:
    """Returns one importance score per remaining layer (higher = safer to remove)."""
    layers = model.model.layers
    num_layers = len(layers)
    similarity_sums = [0.0] * num_layers
    sample_count = 0

    hooks = []
    layer_io: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def make_hook(idx):
        def hook(_module, inputs, output):
            hidden_in = inputs[0]
            hidden_out = output[0] if isinstance(output, tuple) else output
            layer_io[idx] = (hidden_in.detach(), hidden_out.detach())

        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    model.eval()
    with torch.no_grad():
        for text in CALIBRATION_TEXTS:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            model(**inputs)
            for i in range(num_layers):
                hidden_in, hidden_out = layer_io[i]
                cos_sim = torch.nn.functional.cosine_similarity(
                    hidden_in.flatten(1), hidden_out.flatten(1), dim=1
                )
                similarity_sums[i] += cos_sim.mean().item()
            sample_count += 1

    for hook in hooks:
        hook.remove()

    return [s / sample_count for s in similarity_sums]


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


def _renumber_layer_idx(layers):
    for new_idx, layer in enumerate(layers):
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = new_idx
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_idx


def remove_one_layer(model, index: int):
    """Mutates model in place, removing the layer at `index`. Returns
    (removed_module, removed_layer_type) so the caller can restore it
    with `restore_layer` if this removal gets rejected - this avoids
    `copy.deepcopy`-ing the whole model per trial, which is fine at
    0.5-1.5B params but risks OOM at 7B on a 23GB-RAM machine (two full
    bf16 copies of a 7B model is ~28GB alone, before any working
    memory)."""
    layers = list(model.model.layers)
    removed_module = layers.pop(index)

    layer_types = None
    removed_layer_type = None
    if getattr(model.config, "layer_types", None) is not None:
        layer_types = list(model.config.layer_types)
        removed_layer_type = layer_types.pop(index)
        model.config.layer_types = layer_types

    # Each decoder layer (and its self_attn submodule) is constructed
    # with a fixed layer_idx baked in at __init__ time, used for KV-cache
    # indexing. Removing a layer shifts every subsequent layer's
    # *position*, but its stored layer_idx doesn't automatically follow -
    # it still has its original value, which then indexes into a
    # KV-cache sized for the new (smaller) layer count and goes out of
    # range. Must renumber explicitly to match new positions.
    _renumber_layer_idx(layers)

    model.model.layers = torch.nn.ModuleList(layers)
    model.config.num_hidden_layers = len(layers)
    return removed_module, removed_layer_type


def restore_layer(model, index: int, removed_module, removed_layer_type):
    """Undoes `remove_one_layer` - reinserts the module at its original
    position and renumbers layer_idx again (insertion shifts positions
    just as removal does)."""
    layers = list(model.model.layers)
    layers.insert(index, removed_module)
    _renumber_layer_idx(layers)
    model.model.layers = torch.nn.ModuleList(layers)
    model.config.num_hidden_layers = len(layers)

    if getattr(model.config, "layer_types", None) is not None or removed_layer_type is not None:
        layer_types = list(model.config.layer_types) if model.config.layer_types else []
        layer_types.insert(index, removed_layer_type)
        model.config.layer_types = layer_types


def prune_iteratively(model, tokenizer, device: str, num_to_remove: int, max_ppl_ratio: float, ppl_fn=None):
    # ppl_fn: optional model -> perplexity override (see evalharness.py); the
    # built-in 4-sentence VALIDATION_TEXTS proved too small and overfit.
    if ppl_fn is None:
        ppl_fn = lambda m: compute_perplexity(m, tokenizer, VALIDATION_TEXTS, device)
    baseline_ppl = ppl_fn(model)
    print(f"Baseline perplexity: {baseline_ppl:.3f}")

    removed_count = 0
    while removed_count < num_to_remove:
        importance = measure_layer_importance(model, tokenizer, device)
        # Try least-important-first; fall back to the next candidate if
        # removing the top pick fails the perplexity check.
        ranked = sorted(range(len(importance)), key=lambda i: importance[i], reverse=True)

        accepted = False
        for candidate in ranked:
            removed_module, removed_layer_type = remove_one_layer(model, candidate)
            trial_ppl = ppl_fn(model)
            ratio = trial_ppl / baseline_ppl
            print(
                f"  trying to remove layer {candidate} "
                f"(similarity={importance[candidate]:.4f}): "
                f"perplexity {trial_ppl:.3f} ({ratio:.2f}x baseline)"
            )
            if ratio <= max_ppl_ratio:
                removed_count += 1
                accepted = True
                print(
                    f"  -> accepted. Now {len(model.model.layers)} layers, "
                    f"removed {removed_count}/{num_to_remove}."
                )
                break
            else:
                restore_layer(model, candidate, removed_module, removed_layer_type)
                print("  -> rejected, perplexity too high. Trying next candidate.")

        if not accepted:
            print(
                f"Stopping early: no remaining layer could be removed without "
                f"exceeding {max_ppl_ratio}x baseline perplexity. "
                f"Removed {removed_count}/{num_to_remove}."
            )
            break

    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", help="HF model id or local path (non-quantized)")
    parser.add_argument("output_dir", help="where to save the pruned model")
    parser.add_argument(
        "--remove-fraction",
        type=float,
        default=0.25,
        help="fraction of layers to remove (default 0.25)",
    )
    parser.add_argument(
        "--max-ppl-ratio",
        type=float,
        default=1.5,
        help="reject a removal if it pushes perplexity above this multiple of baseline (default 1.5x)",
    )
    args = parser.parse_args()

    device = "cpu"
    print(f"Loading {args.model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype="auto")
    model.to(device)

    num_layers = len(model.model.layers)
    num_to_remove = max(1, round(num_layers * args.remove_fraction))
    print(f"Model has {num_layers} layers, targeting removal of {num_to_remove}.")

    model = prune_iteratively(model, tokenizer, device, num_to_remove, args.max_ppl_ratio)

    print(f"Saving pruned model ({len(model.model.layers)} layers) to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
