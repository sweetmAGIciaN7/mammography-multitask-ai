"""Phase 5: attention maps and Grad-CAM from a trained ``MultiTaskNet`` (torch).

Grad-CAM (Selvaraju et al., 2017) is taken on the **last backbone feature map, before CBAM**, the same layer for
every model (``mt_plain`` has no CBAM at all). For a target score s:  weights_c = mean over space of ds/dA_c,
map = ReLU(sum_c weights_c * A_c). Targets: the malignancy logit, and the logit of the *predicted* density class.
Everything runs in float32, eval mode, no autocast.

The CBAM map is the 7x7-conv spatial gate after its sigmoid, returned by the model itself.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def forward_parts(model, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    """The model's own forward pass, split so the pre-CBAM feature map is available. Same numbers as ``model(x)``."""
    f = model.features(x)
    g, attn = (model.attention(f) if model.attention is not None else (f, None))
    z = model.shared(g.mean((2, 3)))
    logits = {}
    if model.pathology_head is not None:
        logits["pathology"] = model.pathology_head(z).squeeze(1)
    if model.density_head is not None:
        logits["density"] = model.density_head(z)
    return f, attn, logits


def explain_batch(model, x: torch.Tensor) -> dict:
    """Maps for one normalised batch (N,3,H,W). Returns numpy arrays:
    cbam (N,h,w) or absent, gradcam_mal (N,h,w), gradcam_dens (N,h,w), p_malignant (N,), p_density (N,4)."""
    model.eval()
    x = x.float()
    out = {}
    with torch.enable_grad(), torch.autocast(device_type=x.device.type, enabled=False):
        f, attn, logits = forward_parts(model, x)
        targets = []
        if "pathology" in logits:
            targets.append(("gradcam_mal", logits["pathology"]))
            out["p_malignant"] = torch.sigmoid(logits["pathology"]).detach().cpu().numpy()
        if "density" in logits:
            d = logits["density"]
            targets.append(("gradcam_dens", d.gather(1, d.argmax(1, keepdim=True)).squeeze(1)))
            out["p_density"] = torch.softmax(d, 1).detach().cpu().numpy()
        for i, (name, score) in enumerate(targets):
            # samples are independent in eval mode (BatchNorm uses running stats), so d(sum)/dA_i = d(score_i)/dA_i
            (grad,) = torch.autograd.grad(score.sum(), f, retain_graph=i < len(targets) - 1)
            cam = F.relu((grad.mean((2, 3), keepdim=True) * f).sum(1))
            out[name] = cam.detach().cpu().numpy()
    if attn is not None:
        out["cbam"] = attn.detach()[:, 0].cpu().numpy()
    return out
