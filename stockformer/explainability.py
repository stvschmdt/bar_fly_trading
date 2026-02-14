"""
Explainability and visualization tools for neural network models.

Architecture-agnostic attention extraction and plotting. Works with any
nn.Module containing nn.MultiheadAttention layers — cross-attention
transformers, encoder-only, encoder-decoder, or even feedforward networks
(saliency still works without attention).

Usage:
    from stockformer.explainability import AttentionExtractor, plot_attention_heatmap

    # Auto-discover all MultiheadAttention layers
    extractor = AttentionExtractor(model)
    with extractor:
        output = model(x, market_x)
    data = extractor.get_data()

    # Or specify explicit layers for architecture-specific features
    extractor = AttentionExtractor(
        model,
        gates=[layer.gate for layer in model.cross_attn_layers],
        pool=model.attn_pool,
    )
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union


# =============================================================================
# Attention Extraction (hook-based, architecture-agnostic)
# =============================================================================

class AttentionExtractor:
    """
    Architecture-agnostic context manager that hooks nn.MultiheadAttention
    layers to capture per-head attention weights.

    Auto-discovers all MHA layers by walking the module tree. Optionally
    accepts explicit references for gates and pooling layers.

    The returned data dict has:
        attention_weights: OrderedDict[module_path -> Tensor]
            Each tensor is [batch, heads, q_len, k_len]
        gate_values: List[Tensor]  (if gates provided)
            Each tensor is [batch, seq_len, d_model]
        pool_weights: Tensor or None  (if pool provided)
            Shape [batch, seq_len]

    Example (auto-discover):
        extractor = AttentionExtractor(model)

    Example (with gates and pooling):
        extractor = AttentionExtractor(
            model,
            gates=[model.cross_attn_layers[i].gate for i in range(4)],
            pool=model.attn_pool,
        )

    Example (explicit MHA selection):
        extractor = AttentionExtractor(
            model,
            attention_layers=[model.encoder.layers[0].self_attn],
        )
    """

    def __init__(
        self,
        model: nn.Module,
        attention_layers: Optional[List[nn.MultiheadAttention]] = None,
        gates: Optional[List[nn.Module]] = None,
        pool: Optional[nn.Module] = None,
    ):
        """
        Args:
            model: Any nn.Module
            attention_layers: Explicit list of MHA modules to hook.
                              If None, auto-discovers all nn.MultiheadAttention
                              in the model.
            gates: Optional list of modules whose outputs to capture
                   (e.g., sigmoid gate layers in gated cross-attention).
            pool: Optional module whose output to capture as pooling weights
                  (assumes output is pre-softmax logits of shape [*, seq, 1]).
        """
        self.model = model
        self._hooks: List = []
        self._patched_forwards: Dict = {}
        self._data: Dict = _empty_data()

        # Discover or accept attention layers
        if attention_layers is not None:
            self._mha_modules = [(f"mha_{i}", m) for i, m in enumerate(attention_layers)]
        else:
            self._mha_modules = _find_mha_layers(model)

        self._gate_modules = gates or []
        self._pool_module = pool

    def __enter__(self):
        self._data = _empty_data()
        self._register_hooks()
        return self

    def __exit__(self, *args):
        self._remove_hooks()

    def get_data(self) -> Dict[str, Union[Dict, List[torch.Tensor], torch.Tensor, None]]:
        """Return captured data. Call after forward pass."""
        return dict(self._data)

    def clear(self):
        """Clear captured data for next forward pass."""
        self._data = _empty_data()

    # --- Convenience accessors for backward compatibility ---

    def get_attention_weights(self, pattern: Optional[str] = None) -> List[torch.Tensor]:
        """
        Get attention weight tensors, optionally filtered by module path pattern.

        Args:
            pattern: Substring to filter module paths (e.g., "cross_attn", "self_attn").
                     None returns all.

        Returns:
            List of tensors, each [batch, heads, q_len, k_len]
        """
        weights = self._data["attention_weights"]
        if pattern is None:
            return list(weights.values())
        return [w for name, w in weights.items() if pattern in name]

    # ----- internals -----

    def _register_hooks(self):
        # --- MultiheadAttention layers ---
        for name, mha in self._mha_modules:
            # Monkey-patch forward to force per-head weight output
            orig_forward = mha.forward
            self._patched_forwards[id(mha)] = orig_forward

            def _make_patched(orig_fn):
                def patched(*args, **kwargs):
                    kwargs["need_weights"] = True
                    kwargs["average_attn_weights"] = False
                    return orig_fn(*args, **kwargs)
                return patched
            mha.forward = _make_patched(orig_forward)

            # Hook to capture weights
            def _make_attn_hook(layer_name):
                def hook(module, inp, output):
                    _, attn_weights = output
                    if attn_weights is not None:
                        self._data["attention_weights"][layer_name] = attn_weights.detach().cpu()
                return hook
            self._hooks.append(mha.register_forward_hook(_make_attn_hook(name)))

        # --- Gate layers ---
        for i, gate in enumerate(self._gate_modules):
            def _make_gate_hook(idx):
                def hook(module, inp, output):
                    self._data["gate_values"].append(output.detach().cpu())
                return hook
            self._hooks.append(gate.register_forward_hook(_make_gate_hook(i)))

        # --- Pooling layer ---
        if self._pool_module is not None:
            def _pool_hook(module, inp, output):
                weights = torch.softmax(output.squeeze(-1), dim=1)
                self._data["pool_weights"] = weights.detach().cpu()
            self._hooks.append(self._pool_module.register_forward_hook(_pool_hook))

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        # Restore all patched MHA forwards
        for name, mha in self._mha_modules:
            if id(mha) in self._patched_forwards:
                mha.forward = self._patched_forwards[id(mha)]
        self._patched_forwards.clear()


def _find_mha_layers(model: nn.Module) -> List[Tuple[str, nn.MultiheadAttention]]:
    """Walk module tree and return all nn.MultiheadAttention with their paths."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            layers.append((name, module))
    return layers


def _empty_data():
    return {
        "attention_weights": {},    # Dict[name -> Tensor[batch, heads, q_len, k_len]]
        "gate_values": [],          # List[Tensor[batch, seq_len, d_model]]
        "pool_weights": None,       # Tensor[batch, seq_len] or None
    }


# =============================================================================
# Factory — pre-configured extractors for known architectures
# =============================================================================

def make_extractor(model: nn.Module) -> AttentionExtractor:
    """
    Create an AttentionExtractor pre-configured for known architectures.

    Detects architecture by checking for characteristic attributes:
    - CrossAttentionStockTransformer: cross_attn_layers, stock_layers, attn_pool
    - StockTransformer: encoder, attn_pool
    - Generic nn.Module: auto-discovers all MHA layers

    Returns:
        Configured AttentionExtractor
    """
    # CrossAttentionStockTransformer
    if hasattr(model, "cross_attn_layers") and hasattr(model, "stock_layers"):
        gates = [layer.gate for layer in model.cross_attn_layers
                 if hasattr(layer, "gate")]
        pool = getattr(model, "attn_pool", None)
        return AttentionExtractor(model, gates=gates, pool=pool)

    # StockTransformer (encoder-only)
    if hasattr(model, "encoder") and hasattr(model, "attn_pool"):
        pool = model.attn_pool
        return AttentionExtractor(model, pool=pool)

    # Generic fallback — auto-discover everything
    return AttentionExtractor(model)


# =============================================================================
# Input Saliency (gradient-based — works with any model)
# =============================================================================

@torch.enable_grad()
def compute_input_saliency(
    model: nn.Module,
    *inputs: torch.Tensor,
    target_class: Optional[int] = None,
    method: str = "input_x_gradient",
) -> List[torch.Tensor]:
    """
    Compute input saliency maps via backpropagation. Works with any model.

    Args:
        model: Any nn.Module (set to eval mode)
        *inputs: One or more input tensors [1, ...]. Each will get a saliency map.
        target_class: For classification, which class to explain.
                      None = predicted class. Ignored for regression.
        method: "gradient" (vanilla) or "input_x_gradient" (Grad x Input)

    Returns:
        List of saliency tensors, one per input, with batch dim squeezed.
        E.g., for inputs (x[1,60,41], market_x[1,60,11]) returns
              [saliency[60,41], saliency[60,11]]
    """
    model.eval()
    grad_inputs = []
    for inp in inputs:
        t = inp.clone().detach().requires_grad_(True)
        grad_inputs.append(t)

    output = model(*grad_inputs)

    # Select scalar to backprop from
    if output.dim() == 0 or (output.dim() == 1 and output.shape[0] == 1):
        scalar = output.squeeze()
    elif output.dim() == 1:
        if target_class is None:
            target_class = output.argmax().item()
        scalar = output[target_class]
    else:
        if target_class is None:
            target_class = output[0].argmax().item()
        scalar = output[0, target_class]

    scalar.backward()

    saliencies = []
    for inp, grad_inp in zip(inputs, grad_inputs):
        if method == "input_x_gradient":
            sal = (grad_inp.grad * grad_inp).detach().cpu().squeeze(0).abs()
        else:
            sal = grad_inp.grad.detach().cpu().squeeze(0).abs()
        saliencies.append(sal)

    return saliencies


# =============================================================================
# Plotting — Attention Heatmaps (generic)
# =============================================================================

_CMAP = "YlOrRd"
_FIGSIZE = (10, 6)


def plot_attention_heatmap(
    weights: torch.Tensor,
    head: Optional[int] = None,
    sample: int = 0,
    query_labels: Optional[List[str]] = None,
    key_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot attention heatmap from a weight tensor.

    Args:
        weights: [batch, heads, q_len, k_len] attention weight tensor
        head: Which head. None = average across heads.
        sample: Batch index
        query_labels: Y-axis labels
        key_labels: X-axis labels
        title: Custom title
        save_path: If provided, saves figure
    """
    w = weights[sample]  # [heads, q_len, k_len]

    if head is not None:
        w = w[head]
        head_label = f"Head {head}"
    else:
        w = w.mean(dim=0)
        head_label = "Avg heads"

    w = w.numpy()

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    im = ax.imshow(w, aspect="auto", cmap=_CMAP, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Attention weight")

    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    if query_labels:
        ax.set_yticks(range(len(query_labels)))
        ax.set_yticklabels(query_labels, fontsize=7)
    if key_labels:
        ax.set_xticks(range(len(key_labels)))
        ax.set_xticklabels(key_labels, fontsize=7, rotation=45, ha="right")

    ax.set_title(title or f"Attention — {head_label}")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_attention_all_heads(
    weights: torch.Tensor,
    sample: int = 0,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot all attention heads in a grid from a weight tensor."""
    w = weights[sample]  # [heads, q_len, k_len]
    n_heads = w.shape[0]
    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3 * rows))
    if n_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for h in range(n_heads):
        ax = axes[h]
        ax.imshow(w[h].numpy(), aspect="auto", cmap=_CMAP, interpolation="nearest")
        ax.set_title(f"Head {h}", fontsize=10)
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")

    for h in range(n_heads, len(axes)):
        axes[h].axis("off")

    fig.suptitle(title or "Attention Heads", fontsize=13, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# Backward-compatible aliases for cross/self attention plotting
def plot_cross_attention(attn_data, layer=0, head=None, sample=0, **kwargs):
    """Plot cross-attention heatmap. Wraps plot_attention_heatmap."""
    weights = _get_weights_by_pattern(attn_data, "cross_attn", layer)
    kwargs.setdefault("title", f"Cross-Attention — Layer {layer}")
    return plot_attention_heatmap(weights, head=head, sample=sample, **kwargs)


def plot_cross_attention_all_heads(attn_data, layer=0, sample=0, **kwargs):
    """Plot all cross-attention heads. Wraps plot_attention_all_heads."""
    weights = _get_weights_by_pattern(attn_data, "cross_attn", layer)
    kwargs.setdefault("title", f"Cross-Attention Heads — Layer {layer}")
    return plot_attention_all_heads(weights, sample=sample, **kwargs)


def plot_self_attention(attn_data, layer=0, head=None, sample=0, **kwargs):
    """Plot self-attention pattern. Wraps plot_attention_heatmap."""
    weights = _get_weights_by_pattern(attn_data, "self_attn", layer)
    kwargs.setdefault("title", f"Self-Attention — Layer {layer}")
    return plot_attention_heatmap(weights, head=head, sample=sample, **kwargs)


def _get_weights_by_pattern(attn_data, pattern, layer_idx):
    """Get attention weights matching a name pattern, indexed by layer order."""
    weights = attn_data["attention_weights"]
    matching = [(name, w) for name, w in weights.items() if pattern in name]
    if not matching:
        # Fall back to all weights in order
        matching = list(weights.items())
    if layer_idx >= len(matching):
        raise IndexError(f"Layer {layer_idx} requested but only {len(matching)} "
                         f"'{pattern}' layers found: {[n for n,_ in matching]}")
    return matching[layer_idx][1]


# =============================================================================
# Plotting — Gate Activations
# =============================================================================

def plot_gate_activations(
    attn_data: dict,
    layer: int = 0,
    sample: int = 0,
    top_k: int = 32,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot gate activation heatmap.

    The gate controls how much context is injected per feature per timestep.
    Values near 0 = primary signal dominates, near 1 = context dominates.
    """
    gate = attn_data["gate_values"][layer][sample]  # [seq_len, d_model]
    g = gate.numpy()

    variance = g.var(axis=0)
    top_dims = np.argsort(variance)[-top_k:][::-1]
    g_top = g[:, top_dims]

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    im = ax.imshow(g_top.T, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Gate value")

    ax.set_xlabel("Timestep")
    ax.set_ylabel(f"Feature dim (top {top_k} by variance)")
    ax.set_title(title or f"Gate Activations — Layer {layer}")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_gate_timeseries(
    attn_data: dict,
    sample: int = 0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot mean gate activation over time for each layer."""
    n_layers = len(attn_data["gate_values"])

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for layer in range(n_layers):
        gate = attn_data["gate_values"][layer][sample]
        mean_gate = gate.mean(dim=-1).numpy()
        ax.plot(mean_gate, label=f"Layer {layer}", linewidth=2, alpha=0.8)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean gate activation")
    ax.set_title("Context Gate Usage Over Time (per layer)")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# =============================================================================
# Plotting — Attention Pooling
# =============================================================================

def plot_attention_pooling(
    attn_data: dict,
    sample: int = 0,
    timestep_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot attention pooling weights across the temporal dimension.

    Shows which timesteps the model considers most important for its
    final prediction.
    """
    weights = attn_data["pool_weights"][sample].numpy()
    seq_len = len(weights)

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    bars = ax.bar(range(seq_len), weights, color="steelblue", alpha=0.8,
                  edgecolor="navy", linewidth=0.5)

    top3 = np.argsort(weights)[-3:]
    for idx in top3:
        bars[idx].set_color("orangered")
        bars[idx].set_alpha(1.0)

    ax.set_xlabel("Timestep (0 = oldest)")
    ax.set_ylabel("Pooling weight")
    ax.set_title("Attention Pooling — Temporal Importance for Final Prediction")
    if timestep_labels:
        step = max(1, seq_len // 15)
        ax.set_xticks(range(0, seq_len, step))
        ax.set_xticklabels(timestep_labels[::step], fontsize=7, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# =============================================================================
# Plotting — Layer Evolution
# =============================================================================

def plot_layer_evolution(
    attn_data: dict,
    attention_type: str = "cross",
    sample: int = 0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot how attention patterns evolve across layers (entropy + sparsity).

    Args:
        attention_type: Substring to filter attention layer names.
                        E.g., "cross_attn", "self_attn", or "" for all.
    """
    weights = attn_data["attention_weights"]
    matching = [(n, w) for n, w in weights.items() if attention_type in n]
    if not matching:
        matching = list(weights.items())

    entropies = []
    sparsities = []
    layer_names = []

    for name, layer_w in matching:
        w = layer_w[sample]  # [heads, q_len, k_len]
        w_flat = w.mean(dim=0).mean(dim=0).clamp(min=1e-10)

        entropy = -(w_flat * w_flat.log()).sum().item()
        uniform_thresh = 1.0 / w_flat.shape[0]
        sparsity = (w_flat < uniform_thresh).float().mean().item()

        entropies.append(entropy)
        sparsities.append(sparsity)
        layer_names.append(name.split(".")[-1] if "." in name else name)

    n = len(matching)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(range(n), entropies, color="teal", alpha=0.8)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Entropy (nats)")
    ax1.set_title(f"Attention Entropy ({attention_type})")
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(layer_names, fontsize=8, rotation=30, ha="right")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(range(n), sparsities, color="coral", alpha=0.8)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Sparsity (frac below uniform)")
    ax2.set_title(f"Attention Sparsity ({attention_type})")
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(layer_names, fontsize=8, rotation=30, ha="right")
    ax2.set_ylim(0, 1)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Attention Evolution — {attention_type}", fontsize=13, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# =============================================================================
# Plotting — Input Saliency Maps
# =============================================================================

def plot_saliency_map(
    saliency: torch.Tensor,
    feature_names: Optional[List[str]] = None,
    title: str = "Input Saliency Map",
    top_k: int = 20,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot input saliency as a heatmap (time x features)."""
    sal = saliency.numpy()

    total_sal = sal.sum(axis=0)
    top_features = np.argsort(total_sal)[-top_k:][::-1]
    sal_top = sal[:, top_features]

    if feature_names:
        labels = [feature_names[i] for i in top_features]
    else:
        labels = [f"feat_{i}" for i in top_features]

    fig, ax = plt.subplots(figsize=(12, max(4, top_k * 0.3)))
    im = ax.imshow(sal_top.T, aspect="auto", cmap="inferno", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Saliency (|grad x input|)")

    ax.set_xlabel("Timestep")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_saliency_summary(
    *saliencies: torch.Tensor,
    group_names: Optional[List[str]] = None,
    feature_names: Optional[List[List[str]]] = None,
    top_k: int = 15,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side feature importance for multiple input groups.

    Args:
        *saliencies: One or more saliency tensors [seq_len, feat_dim]
        group_names: Names for each group (e.g., ["Stock", "Market"])
        feature_names: Per-group feature name lists
        top_k: Show top-k features per group
    """
    n = len(saliencies)
    if group_names is None:
        group_names = [f"Input {i}" for i in range(n)]
    if feature_names is None:
        feature_names = [None] * n

    colors = ["steelblue", "coral", "teal", "orchid", "goldenrod"]
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for i, (sal, gname, fnames) in enumerate(zip(saliencies, group_names, feature_names)):
        importance = sal.sum(dim=0).numpy()
        k = min(top_k, len(importance))
        top_idx = np.argsort(importance)[-k:]

        vals = importance[top_idx]
        if fnames:
            names = [fnames[j] for j in top_idx]
        else:
            names = [f"feat_{j}" for j in top_idx]

        axes[i].barh(range(len(vals)), vals, color=colors[i % len(colors)], alpha=0.8)
        axes[i].set_yticks(range(len(names)))
        axes[i].set_yticklabels(names, fontsize=8)
        axes[i].set_xlabel("Total saliency")
        axes[i].set_title(f"Top {k} {gname} Features")
        axes[i].grid(True, axis="x", alpha=0.3)

    fig.suptitle("Feature Importance — Input Saliency", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# =============================================================================
# Plotting — Head Specialization
# =============================================================================

def plot_head_specialization(
    attn_data: dict,
    layer: int = 0,
    attention_type: str = "cross",
    sample: int = 0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Analyze what different attention heads specialize in.

    Plots per-head entropy, peak position, and recency bias.
    Different patterns across heads = healthy model.
    """
    weights = _get_weights_by_pattern(attn_data, attention_type, layer)
    w = weights[sample]  # [heads, q_len, k_len]
    n_heads = w.shape[0]
    k_len = w.shape[-1]

    avg_w = w.mean(dim=1)  # [heads, k_len]

    entropies = []
    peak_positions = []
    recency_biases = []

    for h in range(n_heads):
        wh = avg_w[h].clamp(min=1e-10)
        entropy = -(wh * wh.log()).sum().item()
        peak = wh.argmax().item()
        recency = wh[-(k_len // 4):].sum().item()

        entropies.append(entropy)
        peak_positions.append(peak)
        recency_biases.append(recency)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    heads = range(n_heads)

    axes[0].bar(heads, entropies, color="teal", alpha=0.8)
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Entropy")
    axes[0].set_title("Attention Entropy")
    axes[0].set_xticks(list(heads))

    axes[1].bar(heads, peak_positions, color="steelblue", alpha=0.8)
    axes[1].set_xlabel("Head")
    axes[1].set_ylabel("Peak position")
    axes[1].set_title("Peak Attention Position")
    axes[1].set_xticks(list(heads))

    axes[2].bar(heads, recency_biases, color="coral", alpha=0.8)
    axes[2].set_xlabel("Head")
    axes[2].set_ylabel("Weight on last 25%")
    axes[2].set_title("Recency Bias")
    axes[2].set_xticks(list(heads))
    axes[2].set_ylim(0, 1)

    fig.suptitle(f"Head Specialization — Layer {layer} ({attention_type})", fontsize=13, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# =============================================================================
# Composite Report
# =============================================================================

def generate_report(
    model: nn.Module,
    *inputs: torch.Tensor,
    output_dir: str = "stockformer/output/explainability",
    input_names: Optional[List[str]] = None,
    feature_names: Optional[List[List[str]]] = None,
    sample: int = 0,
) -> Dict[str, Path]:
    """
    Generate a full explainability report for a single input sample.

    Works with any model. Uses make_extractor() to auto-configure hooks.

    Args:
        model: Any nn.Module in eval mode
        *inputs: Input tensors for the model's forward pass
        output_dir: Directory for output PNGs
        input_names: Names for each input (e.g., ["Stock", "Market"])
        feature_names: Per-input feature name lists
        sample: Which batch sample to visualize

    Returns:
        Dict mapping plot name to saved file Path
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = {}

    model.eval()

    # --- Hook-based extraction ---
    extractor = make_extractor(model)
    with torch.no_grad():
        with extractor:
            _ = model(*inputs)
    attn = extractor.get_data()

    attn_weights = attn["attention_weights"]
    gate_values = attn["gate_values"]
    pool_weights = attn["pool_weights"]

    # --- Attention heatmaps ---
    for i, (name, weights) in enumerate(attn_weights.items()):
        short_name = name.replace(".", "_")

        path = str(out / f"attn_{short_name}_avg.png")
        plot_attention_heatmap(weights, sample=sample, title=f"Attention — {name}",
                               save_path=path)
        saved[f"attn_{short_name}"] = Path(path)
        plt.close("all")

    # All-heads grid for first attention layer
    if attn_weights:
        first_name, first_w = next(iter(attn_weights.items()))
        path = str(out / "attn_all_heads_first.png")
        plot_attention_all_heads(first_w, sample=sample,
                                 title=f"All Heads — {first_name}", save_path=path)
        saved["attn_all_heads_first"] = Path(path)
        plt.close("all")

    # --- Gate activations ---
    for i in range(len(gate_values)):
        path = str(out / f"gate_layer{i}.png")
        plot_gate_activations(attn, layer=i, sample=sample, save_path=path)
        saved[f"gate_L{i}"] = Path(path)
        plt.close("all")

    if gate_values:
        path = str(out / "gate_timeseries.png")
        plot_gate_timeseries(attn, sample=sample, save_path=path)
        saved["gate_timeseries"] = Path(path)
        plt.close("all")

    # --- Pooling ---
    if pool_weights is not None:
        path = str(out / "attention_pooling.png")
        plot_attention_pooling(attn, sample=sample, save_path=path)
        saved["pool_weights"] = Path(path)
        plt.close("all")

    # --- Layer evolution ---
    if len(attn_weights) > 1:
        # Try cross-attn specific
        cross_layers = [n for n in attn_weights if "cross_attn" in n]
        self_layers = [n for n in attn_weights if "self_attn" in n]

        if cross_layers:
            path = str(out / "layer_evolution_cross.png")
            plot_layer_evolution(attn, attention_type="cross_attn", sample=sample, save_path=path)
            saved["layer_evolution_cross"] = Path(path)
            plt.close("all")

        if self_layers:
            path = str(out / "layer_evolution_self.png")
            plot_layer_evolution(attn, attention_type="self_attn", sample=sample, save_path=path)
            saved["layer_evolution_self"] = Path(path)
            plt.close("all")

        if not cross_layers and not self_layers:
            path = str(out / "layer_evolution.png")
            plot_layer_evolution(attn, attention_type="", sample=sample, save_path=path)
            saved["layer_evolution"] = Path(path)
            plt.close("all")

    # --- Head specialization (first layer) ---
    if attn_weights:
        first_name = next(iter(attn_weights))
        # Determine attention type from name
        if "cross_attn" in first_name:
            at = "cross_attn"
        elif "self_attn" in first_name:
            at = "self_attn"
        else:
            at = ""
        path = str(out / "head_specialization.png")
        plot_head_specialization(attn, layer=0, attention_type=at, sample=sample, save_path=path)
        saved["head_specialization"] = Path(path)
        plt.close("all")

    # --- Gradient-based saliency ---
    saliencies = compute_input_saliency(model, *inputs)

    names = input_names or [f"input_{i}" for i in range(len(inputs))]
    fnames = feature_names or [None] * len(inputs)

    for i, (sal, name) in enumerate(zip(saliencies, names)):
        path = str(out / f"saliency_{name.lower().replace(' ', '_')}.png")
        top_k = min(20, sal.shape[-1])
        plot_saliency_map(sal, feature_names=fnames[i],
                          title=f"{name} Input Saliency", top_k=top_k, save_path=path)
        saved[f"saliency_{name}"] = Path(path)
        plt.close("all")

    if len(saliencies) > 1:
        path = str(out / "saliency_summary.png")
        plot_saliency_summary(*saliencies, group_names=names,
                               feature_names=fnames, save_path=path)
        saved["saliency_summary"] = Path(path)
        plt.close("all")

    print(f"Explainability report: {len(saved)} figures saved to {out}/")
    return saved
