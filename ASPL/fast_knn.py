import torch
import torch.nn as nn
import torch.nn.functional as F

class FrozenFeatBankNeighborNoCertFP32(nn.Module):
    """
    FP32 only, no continuous certainty.
    Speed-up version:
    - build feature bank once
    - precompute fixed KNN once after epoch0
    - later batches only index nn_idx_bank / nn_sim_bank
    """

    def __init__(
        self,
        N: int,
        D: int,
        K: int,
        k: int = 20,
        tau_n: float = 0.1,
        chunk: int = 20000,
        device: str = "cuda",
        enable_promotion: bool = True,
        src_ratio_thr: float = 1.5,
        promote_m_min: int = 3,
        promote_r_min: float = 1.0,
        enable_neighbor_gate: bool = True,
        cls_level: bool = False,
        min_nb: int = 5,
        agree_thr: float = 0.4,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.N, self.D, self.K = int(N), int(D), int(K)
        self.k = int(k)
        self.tau_n = float(tau_n)
        self.chunk = int(chunk)
        self.device = torch.device(device)

        self.enable_promotion = bool(enable_promotion)
        self.src_ratio_thr = float(src_ratio_thr)
        self.promote_m_min = int(promote_m_min)
        self.promote_r_min = float(promote_r_min)

        self.enable_neighbor_gate = bool(enable_neighbor_gate)
        self.cls_stas = bool(cls_level)
        self.min_nb = int(min_nb)
        self.agree_thr = float(agree_thr)
        self.eps = float(eps)

        # -------------------------
        # feature bank / stats
        # -------------------------
        self.register_buffer(
            "Z_mem",
            torch.zeros(self.N, self.D, dtype=torch.float32, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "Z_filled",
            torch.zeros(self.N, dtype=torch.bool, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "pseudo_labels",
            torch.zeros(self.N, dtype=torch.long, device=self.device),
            persistent=False,
        )

        self.register_buffer(
            "mu_v",
            torch.zeros(self.D, dtype=torch.float32, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "var_v",
            torch.zeros(self.D, dtype=torch.float32, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "mu_V",
            torch.zeros(self.K, self.D, dtype=torch.float32, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "var_V",
            torch.zeros(self.K, self.D, dtype=torch.float32, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "count_V",
            torch.zeros(self.K, dtype=torch.long, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "sum_V",
            torch.zeros(self.K, self.D, dtype=torch.float32, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "sqsum_V",
            torch.zeros(self.K, self.D, dtype=torch.float32, device=self.device),
            persistent=False,
        )

        # -------------------------
        # probability / mask banks
        # -------------------------
        self.register_buffer(
            "partial_label_bank_vlm",
            torch.zeros(self.N, self.K, dtype=torch.float32, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "partial_label_bank_src",
            torch.zeros(self.N, self.K, dtype=torch.float32, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "uncertain_selection_mask_bank_vlm",
            torch.zeros(self.N, dtype=torch.long, device=self.device),
            persistent=False,
        )

        # -------------------------
        # fixed KNN bank
        # -------------------------
        self.register_buffer(
            "nn_idx_bank",
            torch.full((self.N, self.k), -1, dtype=torch.long, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "nn_sim_bank",
            torch.full((self.N, self.k), -1e6, dtype=torch.float32, device=self.device),
            persistent=False,
        )

        self._has_Z = False
        self._has_P_vlm = False
        self._has_P_src = False
        self._has_knn_bank = False

    @torch.no_grad()
    def build_feature_bank(self, idx: torch.Tensor, feats: torch.Tensor):
        idx = idx.to(self.device, dtype=torch.long)
        f = feats.detach()
        self.Z_mem[idx] = f
        self.Z_filled[idx] = True

    @torch.no_grad()
    def initialize_statistics(
        self,
        idx: torch.Tensor,
        logits: torch.Tensor,
        feats: torch.Tensor,
        temp: float = 1.0,
    ):
        device = self.device
        idx = idx.to(device=device, dtype=torch.long)
        logits = logits.to(device=device, dtype=torch.float32)
        feats = feats.to(device=device, dtype=torch.float32)

        self.Z_mem[idx] = feats
        self.Z_filled[idx] = True

        probs = F.softmax(logits / temp, dim=-1)
        pseudo = probs.argmax(dim=-1)
        return probs, pseudo

    @torch.no_grad()
    def reset_visual_statistics(self):
        if self.cls_stas:
            self.count_V.zero_()
            self.sum_V.zero_()
            self.sqsum_V.zero_()

    @torch.no_grad()
    def finalize_bank(self, require_full: bool = True):
        if require_full and not bool(self.Z_filled.all()):
            missing = (~self.Z_filled).nonzero(as_tuple=False).flatten()
            raise RuntimeError(f"Z_mem not fully filled: missing {missing.numel()} samples")
        self._has_Z = True

    @torch.no_grad()
    def finalize_statistics(self, eps: float = 1e-6):
        if not self._has_Z:
            raise RuntimeError("Z_mem not ready. Call finalize_bank() first.")
    
        device = self.Z_mem.device
    
        # -----------------------------------
        # global certain mask over full bank
        # -----------------------------------
        certain_mask = (self.uncertain_selection_mask_bank_vlm == 0)
    
        if hasattr(self, "Z_filled"):
            certain_mask = certain_mask & self.Z_filled
    
        if not certain_mask.any():
            raise RuntimeError("No certain filled samples available for statistics.")
    
        # -----------------------------------
        # global statistics on certain samples
        # -----------------------------------
        z_certain = self.Z_mem[certain_mask]   # [Nc, D]
    
        if not self.cls_stas:
            mu = z_certain.mean(dim=0)
            var = z_certain.var(dim=0, unbiased=False)
    
            self.mu_v.copy_(mu)
            self.var_v.copy_(var.clamp_min(eps))
    
            print(f"  mu_v shape: {self.mu_v.shape}")
            print(f"  var_v shape: {self.var_v.shape}")
            print(f"  mu_v mean: {self.mu_v.mean().item():.7f}")
            print(f"  var_v mean: {self.var_v.mean().item():.7f}")
    
        else:
            # initialize defaults
            self.mu_V.zero_()
            self.var_V.fill_(eps)
    
            for c in range(self.K):
                class_mask = (self.pseudo_labels == c)
                mask = certain_mask & class_mask
    
                if mask.any():
                    z_c = self.Z_mem[mask]
                    self.mu_V[c].copy_(z_c.mean(dim=0))
                    self.var_V[c].copy_(z_c.var(dim=0, unbiased=False).clamp_min(eps))
    
            # global stats from all certain samples, not from mu_V
            self.mu_v.copy_(self.mu_V.mean(dim=0)) 
            self.var_v.copy_(self.mu_V.var(dim=0, unbiased=False))
    
            print(f"  mu_V shape: {self.mu_V.shape}")
            print(f"  var_V shape: {self.var_V.shape}")
            print(f"  mu_v mean: {self.mu_v.mean().item():.7f}")
            print(f"  var_v mean: {self.var_v.sum().item():.7f}")

    @torch.no_grad()
    def _confident_from_probs(self, probs, tau_type="fixed", ratio_thr=1.5, margin_thr=0.05):
        top2 = probs.topk(2, dim=-1).values
        p1, p2 = top2[:, 0], top2[:, 1]
        eps = 0.1
        ratio = p1 / (p2 + eps)
        margin = p1 - p2

        if tau_type in ["fixed", "stat"]:
            return ratio >= ratio_thr
        elif tau_type == "cal":
            return margin >= margin_thr
        else:
            raise ValueError(tau_type)

    @torch.no_grad()
    def update_mask_from_logits(
        self,
        tar_idx: torch.Tensor,
        probs: torch.Tensor,
        pseudo_labels: torch.Tensor,
        ratio_thr: float = 1.5,
        tau_type: str = "fixed",
        margin_thr: float = 0.05,
        temp: float = 1.0,
        update_prob_bank: bool = True,
        clear_before_write: bool = True,
        which_bank: str = "vlm",
    ):
        dev = self.device
        tar_idx = tar_idx.to(dev, dtype=torch.long)
        probs = probs.to(dev, dtype=torch.float32)
        pseudo_labels = pseudo_labels.to(dev, dtype=torch.long)

        self.pseudo_labels[tar_idx] = pseudo_labels

        confident = self._confident_from_probs(
            probs, tau_type=tau_type, ratio_thr=ratio_thr, margin_thr=margin_thr
        )
        uncertain_mask = ~confident

        uncertain_idx_batch = torch.where(uncertain_mask)[0]
        certain_idx_batch = torch.where(~uncertain_mask)[0]
        certain_labels = pseudo_labels.index_select(0, certain_idx_batch)

        if which_bank == "vlm":
            if clear_before_write:
                self.uncertain_selection_mask_bank_vlm[tar_idx] = 0
            self.uncertain_selection_mask_bank_vlm[tar_idx[uncertain_idx_batch]] = 1

            if update_prob_bank:
                self.partial_label_bank_vlm[tar_idx] = probs.detach()
                self._has_P_vlm = True

        elif which_bank == "src":
            if update_prob_bank:
                self.partial_label_bank_src[tar_idx] = probs.detach()
                self._has_P_src = True
        else:
            raise ValueError(which_bank)

        return probs, uncertain_idx_batch, certain_idx_batch, certain_labels

    @torch.no_grad()
    def _knn_global_cosine(self, feats: torch.Tensor, idx_self: torch.Tensor):
        if not self._has_Z:
            raise RuntimeError("Z_mem not ready. Run epoch0 build_feature_bank then finalize_bank().")

        feats = F.normalize(feats.to(self.device, dtype=torch.float32), dim=1)
        idx_self = idx_self.to(self.device, dtype=torch.long)

        B = feats.size(0)
        k = min(self.k, self.N - 1)

        best_sim = torch.full((B, k), -1e9, device=self.device, dtype=torch.float32)
        best_idx = torch.full((B, k), -1, device=self.device, dtype=torch.long)

        for start in range(0, self.N, self.chunk):
            end = min(self.N, start + self.chunk)
            z = self.Z_mem[start:end]
            sim = feats @ z.t()

            mask = (idx_self >= start) & (idx_self < end)
            if mask.any():
                pos = (idx_self[mask] - start).view(-1, 1)
                sim[mask] = sim[mask].scatter(1, pos, -1e9)

            kk = min(k, sim.size(1))
            sim_topk, idx_topk = sim.topk(kk, dim=1)
            idx_topk = idx_topk + start

            sim_cat = torch.cat([best_sim, sim_topk], dim=1)
            idx_cat = torch.cat([best_idx, idx_topk], dim=1)

            best_sim, pos = sim_cat.topk(k, dim=1)
            best_idx = idx_cat.gather(1, pos)

        return best_idx, best_sim

    @torch.no_grad()
    def build_knn_bank(self):
        """
        Call once after epoch0 feature bank is fully built.
        """
        if not self._has_Z:
            raise RuntimeError("Z_mem not ready. Call finalize_bank() first.")

        print("[KNN] building fixed knn bank ...")
        for start in range(0, self.N, self.chunk):
            end = min(self.N, start + self.chunk)

            feats_chunk = self.Z_mem[start:end]  # already normalized if build_feature_bank(normalize=True)
            idx_chunk = torch.arange(start, end, device=self.device, dtype=torch.long)

            nn_idx, nn_sim = self._knn_global_cosine(feats_chunk, idx_chunk)
            self.nn_idx_bank[start:end] = nn_idx
            self.nn_sim_bank[start:end] = nn_sim

            print(f"[KNN] done {end}/{self.N}")

        self._has_knn_bank = True
        print("[KNN] fixed knn bank ready.")

    @torch.no_grad()
    def get_knn_from_bank(self, idx: torch.Tensor):
        if not self._has_knn_bank:
            raise RuntimeError("KNN bank not ready. Call build_knn_bank() first.")
        idx = idx.to(self.device, dtype=torch.long)
        return self.nn_idx_bank[idx], self.nn_sim_bank[idx]

#     @torch.no_grad()
#     def forward(
#     self,
#     logits_vlm: torch.Tensor,
#     logits_src: torch.Tensor,
#     idx: torch.Tensor,
#     *,
#     src_tau_type: str = "fixed",
#     src_margin_thr: float = 0.05,
# ):
#         idx = idx.to(self.device, dtype=torch.long)
#         logits_vlm = logits_vlm.to(self.device, dtype=torch.float32)
#         logits_src = logits_src.to(self.device, dtype=torch.float32)
    
#         # ----------------------------------
#         # 1) current predictions
#         # ----------------------------------
#         p_vlm = F.softmax(logits_vlm, dim=1)               # [B, K]
#         p_src = F.softmax(logits_src, dim=1)    # [B, K]
    
#         # ----------------------------------
#         # 2) VLM bank split: 1 = uncertain
#         # ----------------------------------
#         u_mask = (self.uncertain_selection_mask_bank_vlm[idx] == 1)   # [B]
#         c_mask = ~u_mask
    
#         u_idx_batch = torch.where(u_mask)[0]     # batch-local uncertain indices
#         u_idx_global = idx[u_idx_batch]          # global uncertain indices
    
#         # ----------------------------------
#         # 3) default empty outputs
#         # ----------------------------------
#         p_calib_u = torch.empty(0, self.K, device=self.device, dtype=torch.float32)
#         y_calib_u = torch.empty(0, device=self.device, dtype=torch.long)
#         valid_u_mask = torch.empty(0, device=self.device, dtype=torch.bool)
#         mode_ratio_u = None
    
#         # ----------------------------------
#         # 4) calibrate uncertain samples by certain neighbors
#         # ----------------------------------
#         if self._has_knn_bank and self._has_P_vlm and u_idx_batch.numel() > 0:
#             nn_idx, nn_sim = self.get_knn_from_bank(idx)      # [B, k], [B, k]
    
#             # select only uncertain rows
#             nn_idx_u = nn_idx[u_idx_batch]                    # [M, k]
#             nn_sim_u = nn_sim[u_idx_batch]                    # [M, k]
    
#             # only use VLM-certain neighbors
#             nb_certain = (self.uncertain_selection_mask_bank_vlm[nn_idx_u] == 0).float()  # [M, k]
#             n_c = nb_certain.sum(dim=1)                                                     # [M]
#             valid_u_mask = (n_c > 0)
    
#             # neighbor soft labels
#             p_nb = self.partial_label_bank_vlm[nn_idx_u]       # [M, k, K]
    
#             # similarity weights
#             w = F.softmax(nn_sim_u / self.tau_n, dim=1)        # [M, k]
#             w = w * nb_certain
#             w = w / (w.sum(dim=1, keepdim=True) + 1e-6)
    
#             # calibrated soft pseudo labels for uncertain samples
#             p_calib_u = torch.sum(w.unsqueeze(-1) * p_nb, dim=1)   # [M, K]
#             p_calib_u = p_calib_u / (p_calib_u.sum(dim=1, keepdim=True) + 1e-6)
#             y_calib_u = p_calib_u.argmax(dim=1)                    # [M]
    
#         return {
#             "is_uncertain": u_mask,
#             "is_certain": c_mask,
    
#             "u_idx_batch": u_idx_batch.detach(),
#             "u_idx_global": u_idx_global.detach(),
    
#             # calibrated pseudo labels for uncertain samples
#             "p_calib_u": p_calib_u.detach(),         # [M, K]
#             "y_calib_u": y_calib_u.detach(),         # [M]
#             "valid_u_mask": valid_u_mask.detach(),   # [M], whether each uncertain sample has certain neighbors
    
#         }
    @torch.no_grad()
    def forward(
        self,
        logits_vlm: torch.Tensor,
        logits_src: torch.Tensor,
        idx: torch.Tensor,
        *,
        src_tau_type: str = "fixed",
        src_margin_thr: float = 0.05,
        src_temp: float = 1.0,
    ):
        idx = idx.to(self.device, dtype=torch.long)
        logits_vlm = logits_vlm.to(self.device, dtype=torch.float32)
        logits_src = logits_src.to(self.device, dtype=torch.float32)

        # probs
        p_vlm = F.softmax(logits_vlm, dim=1)            # (B,K)
        p_src = F.softmax(logits_src / src_temp, dim=1) # (B,K)

        # base split from VLM mask bank: 1=uncertain
        u_base = (self.uncertain_selection_mask_bank_vlm[idx] == 1)
        c_base = ~u_base

        # indices for logging
        u_idx_batch = torch.where(u_base)[0]
        u_idx_global = idx[u_idx_batch]

        # # source confident only inside VLM-uncertain
        # src_conf = self._confident_from_probs(
        #     p_src,
        #     tau_type=src_tau_type,
        #     ratio_thr=self.src_ratio_thr,
        #     margin_thr=src_margin_thr,
        # )

        u_src_certain_mask = u_base
        u_src_certain_idx_batch = torch.where(u_src_certain_mask)[0]
        u_src_certain_idx_global = idx[u_src_certain_idx_batch]

        promoted_mask = torch.zeros_like(u_base, dtype=torch.bool)
        mode_ratio = None

        if self.enable_promotion and self._has_knn_bank and self._has_P_vlm and u_src_certain_idx_batch.numel() > 0:
            nn_idx, nn_sim = self.get_knn_from_bank(idx)

            sel = u_src_certain_idx_batch

            # only count VLM-certain neighbors
            nb_certain = (self.uncertain_selection_mask_bank_vlm[nn_idx[sel]] == 0).float()
            n_c = nb_certain.sum(dim=1)

            p_nb = self.partial_label_bank_vlm[nn_idx[sel]]   # [M,k,K]
            y_nb = p_nb.argmax(dim=2)                         # [M,k]

            y_onehot = F.one_hot(y_nb, num_classes=self.K).float()  # [M,k,K]
            y_count = (y_onehot * nb_certain.unsqueeze(-1)).sum(dim=1)
            mode_count, _ = y_count.max(dim=1)
            mode_ratio = mode_count / (n_c + 1e-6)

            neighbor_ok = (n_c >= self.promote_m_min) & (mode_ratio >= self.promote_r_min)
            promoted_mask[sel] = neighbor_ok

        if self.enable_promotion:
            u_mask = u_base & (~promoted_mask)
            c_mask = c_base | promoted_mask
        else:
            u_mask = u_base
            c_mask = c_base

        return {
            "p_vlm": p_vlm,
            "p_src": p_src,
            "is_uncertain": u_mask,
            "is_certain": c_mask,

            "u_idx_batch": u_idx_batch.detach(),
            "u_idx_global": u_idx_global.detach(),
            "u_src_certain_idx_batch": u_src_certain_idx_batch.detach(),
            "u_src_certain_idx_global": u_src_certain_idx_global.detach(),

            "promoted_mask": promoted_mask.detach(),
        }