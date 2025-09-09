import os, re
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def _safe_mkdir(p):
    try: os.makedirs(p, exist_ok=True)
    except Exception: pass

def _norm_to_HQLK(weights, q, module_name=""):
    w = weights
    if not torch.is_tensor(w):
        return None
    w = w.detach().cpu()
    name_l = module_name.lower()
    if "singleheadattention" in name_l or "pointer" in name_l:
        if w.dtype.is_floating_point:
            w = w.exp()
    B = None
    Lq = None
    try:
        if torch.is_tensor(q):
            B = int(q.shape[0])
            Lq = int(q.shape[1])
    except Exception:
        pass
    if w.dim() == 4:
        if B is not None and w.shape[1] == B:
            w = w[:, 0, :, :]
        elif B is not None and w.shape[0] == B:
            w = w.permute(1,0,2,3)
            w = w[:, 0, :, :]
        else:
            if w.shape[1] == 1:
                w = w[:, 0, :, :]
            elif w.shape[0] == 1:
                w = w[0].unsqueeze(0)
                w = w.squeeze(0)
    elif w.dim() == 3:
        if B is not None and w.shape[0] == B:
            w = w[0].unsqueeze(0)
        else:
            pass
    elif w.dim() == 2:
        w = w.unsqueeze(0)
    else:
        return None
    return w

class AttnRecord(object):
    def __init__(self, layer_name, weights):
        self.layer = layer_name
        if weights.dim() == 2:
            weights = weights.unsqueeze(0)
        self.weights = weights

class AttnRecorder(object):
    def __init__(self):
        self._hooks = []
        self._buffer = []
        self.enabled = True
        self.current_step = None

    def clear(self): self._buffer = []

    def remove(self):
        for h in self._hooks:
            try: h.remove()
            except Exception: pass
        self._hooks = []

    def register(self, policy_net):
        self.remove()
        def _make_hook(layer_name):
            def _hook(module, inp, out):
                if not self.enabled: return
                q = inp[0] if (isinstance(inp, (list, tuple)) and len(inp) > 0) else None
                weights = None
                if isinstance(out, (list, tuple)):
                    for cand in out[::-1]:
                        if torch.is_tensor(cand) and cand.dim() >= 2:
                            weights = cand
                            break
                elif torch.is_tensor(out):
                    weights = out
                if weights is None:
                    for key in ["attn", "attn_weights", "attention", "alpha", "last_attention", "attn_last"]:
                        if hasattr(module, key):
                            w = getattr(module, key)
                            if torch.is_tensor(w):
                                weights = w
                                break
                if weights is None:
                    return
                w = _norm_to_HQLK(weights, q, module.__class__.__name__)
                if w is not None:
                    self._buffer.append(AttnRecord(layer_name, w))
            return _hook
        hooked_names = []
        for name, m in policy_net.named_modules():
            cls = m.__class__.__name__.lower()
            if any(k in cls for k in ["attention", "attn", "gat", "pointer"]):
                self._hooks.append(m.register_forward_hook(_make_hook(name)))
                hooked_names.append(name)
        print(f"[AttnViz] Hooked modules: {hooked_names}")

    def begin_step(self, step_id):
        self.current_step = step_id
        self.clear()

    def end_forward(self):
        out = self._buffer[:]
        self.clear()
        return out

def viz_heads_on_neighbors(save_path,
                           attn,
                           head,
                           node_coords,
                           current_index,
                           neighbor_indices,
                           map_info,
                           title_prefix=""):
    from utils import get_cell_position_from_coords
    plt.switch_backend("agg")
    fig = plt.figure(figsize=(4.2, 4.2), dpi=120)
    ax = fig.add_subplot(111)
    ax.set_title(f"{title_prefix}Head {head}", fontsize=10)
    ax.imshow(map_info.map, cmap="gray", origin="lower")
    ax.set_axis_off()
    H, Lq, Lk = list(attn.shape)
    q_idx = 0 if current_index is None else int(current_index)
    q_idx = max(0, min(q_idx, Lq - 1))
    w = attn[head, q_idx]
    if w.dim() > 1:
        w = w.view(-1)
    w = w.numpy()
    K = len(neighbor_indices)
    w = w[:K] if K <= len(w) else np.pad(w, (0, K - len(w)))
    if w.max() > 0:
        k_norm = w / (w.max() + 1e-9)
    else:
        k_norm = w
    if neighbor_indices is not None and len(neighbor_indices) > 0 and node_coords is not None:
        neigh_coords = node_coords[neighbor_indices]
        cells = get_cell_position_from_coords(neigh_coords, map_info).reshape(-1, 2)
        ax.scatter(cells[:, 0], cells[:, 1],
                   s=60 * (0.4 + 0.6 * k_norm),
                   c=k_norm, marker='o', edgecolors='w', linewidths=0.5, alpha=0.9, zorder=5)
    if node_coords is not None and 0 <= current_index < node_coords.shape[0]:
        curr_cell = get_cell_position_from_coords(node_coords[current_index], map_info)
        ax.plot(curr_cell[0], curr_cell[1], 'ms', ms=6, zorder=6)
    fig.tight_layout(pad=0.1)
    _safe_mkdir(os.path.dirname(save_path))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def dump_attn_debug_pngs(run_dir,
                         step,
                         agent_id,
                         records,
                         node_coords,
                         current_index,
                         neighbor_indices,
                         map_info):
    base = os.path.join(run_dir, "attn", f"agent{agent_id}", f"t{step:04d}")
    _safe_mkdir(base)
    if not records:
        with open(os.path.join(base, "_WHY_EMPTY.txt"), "w") as f:
            f.write("No attention captured for this forward.\n")
        return
    per_layer_means = []
    for rec in records:
        attn = rec.weights
        if not torch.is_tensor(attn) or attn.dim() != 3:
            continue
        per_layer_means.append(attn.mean(0, keepdim=True))
    if not per_layer_means:
        with open(os.path.join(base, "_WHY_EMPTY.txt"), "w") as f:
            f.write("Captured records had unexpected shapes.\n")
        return
    max_Lq = max(int(x.shape[1]) for x in per_layer_means)
    max_Lk = max(int(x.shape[2]) for x in per_layer_means)
    def _pad_to(x, Lq, Lk):
        dq = Lq - x.shape[1]
        dk = Lk - x.shape[2]
        if dq == 0 and dk == 0:
            return x
        return torch.nn.functional.pad(x, (0, dk, 0, dq), mode="constant", value=0.0)
    stacked = torch.cat([_pad_to(x, max_Lq, max_Lk) for x in per_layer_means], dim=0)
    mean_all = stacked.mean(0, keepdim=True)
    outp = os.path.join(base, "attn_mean_all.png")
    viz_heads_on_neighbors(outp, mean_all, head=0,
                           node_coords=node_coords,
                           current_index=current_index,
                           neighbor_indices=neighbor_indices,
                           map_info=map_info,
                           title_prefix=f"Mean(All layers & heads) ")
    with open(os.path.join(base, "_INDEX.txt"), "w") as f:
        f.write("Generated: attn_mean_all.png (mean over ALL layers & heads)\n")
