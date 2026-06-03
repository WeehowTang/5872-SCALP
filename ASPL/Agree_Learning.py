import torch
import torch.nn.functional as F


def vlm_agree_ce_disagree_kd_loss(
    vlm_logits: torch.Tensor,         # [B, K]
    src_logits: torch.Tensor,         # [B, K]
    neg_logits: torch.Tensor = None,  # [B, K] or None
    *,
    alpha: float = 0.5,
    neg_scale: float = 0.1,
    kd_temp: float = 1.0,
    ce_weight: float = 1.0,
    kd_weight: float = 1.0,
    detach_fused: bool = True,
    use_forward_kl: bool = False,     # False: standard KD; True: KL(p_v || p_fuse)
    eps: float = 1e-6,
):
    """
    Agreement:
        y_v == y_s  -> CE(vlm_logits, y_s)

    Disagreement:
        y_v != y_s  -> KD between p_v and p_fuse

    Fused teacher:
        z_fused = (1-alpha) * src_logits + alpha * vlm_logits - neg_scale * neg_logits

    Returns:
        loss: scalar tensor
        stats: dict
    """
    vlm_logits = vlm_logits.float()
    src_logits = src_logits.float()

    if neg_logits is not None:
        neg_logits = neg_logits.float()

    # --------------------------------------------------
    # fused teacher logits
    # --------------------------------------------------
    z_fused = (1.0 - alpha) * src_logits + alpha * vlm_logits
    if neg_logits is not None:
        z_fused = z_fused - neg_scale * neg_logits

    if detach_fused:
        z_fused = z_fused.detach()

    # --------------------------------------------------
    # hard predictions
    # --------------------------------------------------
    yv = vlm_logits.argmax(dim=1)   # [B]
    ys = src_logits.argmax(dim=1)   # [B]

    agree_mask = (yv == ys)
    disagree_mask = ~agree_mask

    zero = vlm_logits.sum() * 0.0
    loss_ce = zero
    loss_kd = zero

    n_agree = int(agree_mask.sum().item())
    n_disagree = int(disagree_mask.sum().item())

    # --------------------------------------------------
    # agreement -> CE
    # --------------------------------------------------
    if n_agree > 0:
        loss_ce = F.cross_entropy(
            vlm_logits[agree_mask],
            ys[agree_mask],
            reduction="mean",
        )

    # --------------------------------------------------
    # disagreement -> KD
    # --------------------------------------------------
    if n_disagree > 0:
        v_dis = vlm_logits[disagree_mask]
        f_dis = z_fused[disagree_mask]
        T = max(float(kd_temp), eps)

        if use_forward_kl:
            # KL(p_v || p_fuse)
            p_v = F.softmax(v_dis / T, dim=1).clamp_min(eps)
            p_f = F.softmax(f_dis / T, dim=1).clamp_min(eps)
            loss_kd = torch.mean(torch.sum(p_v * (torch.log(p_v) - torch.log(p_f)), dim=1)) * (T ** 2)
        else:
            # standard KD: teacher = fused, student = vlm
            p_v = F.softmax(v_dis / T, dim=1).clamp_min(eps)
            p_f = F.softmax(f_dis / T, dim=1).clamp_min(eps)
            loss_kd = F.kl_div(torch.log(p_v), p_f, reduction="batchmean") * (T ** 2)

    # --------------------------------------------------
    # total
    # --------------------------------------------------
    # loss = ce_weight * loss_ce + kd_weight * loss_kd

    return loss_ce, loss_kd, agree_mask