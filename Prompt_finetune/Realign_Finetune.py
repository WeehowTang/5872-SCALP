import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from Simple_Tokenizier import SimpleTokenizer as _Tokenizer
from clip import load, tokenize

_tokenizer = _Tokenizer()

def orthogonal_text_loss(text_features: torch.Tensor) -> torch.Tensor:
    """
    Encourage class text prototypes to be orthogonal (low inter-class cosine sim).
    text_features: [C, D]
    """
    T = F.normalize(text_features, dim=1)          # [C, D]
    G = T @ T.t()                                  # [C, C]
    C = G.size(0)
    I = torch.eye(C, device=G.device, dtype=G.dtype)
    off = G - I
    return (off ** 2).sum() / (C * (C - 1) + 1e-6)

def gram_preserve_loss(text_cur: torch.Tensor, text_init: torch.Tensor) -> torch.Tensor:
    """
    Preserve pairwise class similarity structure (optional).
    Penalize change in off-diagonal entries of Gram matrix.
    """
    tc = F.normalize(text_cur, dim=1)
    ti = F.normalize(text_init, dim=1)
    Gc = tc @ tc.t()
    Gi = ti @ ti.t()
    C = Gc.size(0)
    mask = ~torch.eye(C, device=Gc.device, dtype=torch.bool)
    return ((Gc[mask] - Gi[mask]) ** 2).mean()

class PromptLearner(nn.Module):
    def __init__(
        self,
        clip_model,
        classnames,
        n_ctx=16,
        ctx_init=None,
        ctx_position="end",
        learned_cls=False,
        batch_size=None,
        use_neg_boost=False,
        device="cuda",
    ):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        self.dtype = clip_model.dtype
        self.device = clip_model.visual.conv1.weight.device
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.batch_size = batch_size
        self.use_neg_boost = use_neg_boost
        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            neg_ctx_init = 'not' + ' ' + ctx_init if self.use_neg_boost else ' '
            n_neg_ctx = len(neg_ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
            
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None:
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  # (N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        # self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        # print(f"ctx learned size is {self.ctx.shape}!!!")
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        self.ctx_init = ctx_init
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames
        self.clip_model = clip_model
        # --- 构建固定 prefix/suffix ---
        self._build_prompts(prompts)

    def _build_prompts(self, prompts):
        """构建固定 prefix/suffix"""
        tokenized = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        self.tokenized_prompts = tokenized
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized).type(self.dtype)
            
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("origin_embedds", embedding) 

class Transpose(nn.Module):
    """
    Class-level learnable diagonal transport:
        t'_c = A_c * (t_c - mu_T) + b_c

    where:
        A_c: [D] for each class c
        b_c: [D] for each class c
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        init_scale: float = 1.0,
        use_visual_anchor: bool = True,
        use_log_scale: bool = True,
        scale_clamp=(0.1, 10.0),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_visual_anchor = use_visual_anchor
        self.use_log_scale = use_log_scale
        self.scale_clamp = scale_clamp

        if use_log_scale:
            # [K, D], initialized later
            self.log_align_scale = nn.Parameter(
                torch.log(torch.ones(num_classes, embed_dim) * init_scale)
            )
        else:
            self.align_scale = nn.Parameter(
                torch.ones(num_classes, embed_dim) * init_scale
            )

        # class-level bias: [K, D]
        self.align_bias = nn.Parameter(torch.zeros(num_classes, embed_dim))

        self.initialized = False

    def get_scale(self):
        if self.use_log_scale:
            scale = torch.exp(self.log_align_scale)
        else:
            scale = self.align_scale
        return scale.clamp(self.scale_clamp[0], self.scale_clamp[1])

    @torch.no_grad()
    def init_alignment_from_stats(
        self,
        text_feat: torch.Tensor,   # [K, D]
        mu_V: torch.Tensor,        # [K, D]
        var_V: torch.Tensor,       # [K, D]
        eps: float = 1e-6,
        bias_mode: str = "muV",    # ["muV", "transport", "zero"]
    ):
        t = text_feat.float()                                # [K, D]
        mu_T = t.mean(dim=0, keepdim=True)                   # [1, D]
        var_T = t.var(dim=0, unbiased=False, keepdim=True).clamp_min(eps)

        mu_V = mu_V.to(t.device, dtype=t.dtype)
        var_V = var_V.to(t.device, dtype=t.dtype).clamp_min(eps)

        if mu_V.dim() == 1:
            mu_V = mu_V.unsqueeze(0).expand_as(t)            # [K, D]
        if var_V.dim() == 1:
            var_V = var_V.unsqueeze(0).expand_as(t)          # [K, D]

        # class-wise covariance ratio initialization
        init_scale = torch.sqrt(var_V / var_T).clamp(*self.scale_clamp)   # [K, D]

        if self.use_log_scale:
            self.log_align_scale.data.copy_(torch.log(init_scale))
        else:
            self.align_scale.data.copy_(init_scale)

        if bias_mode == "muV":
            init_bias = mu_V
        elif bias_mode == "transport":
            # make A*(t-mu_T)+b ≈ mu_V
            init_bias = mu_V - init_scale * (t - mu_T)
        elif bias_mode == "zero":
            init_bias = torch.zeros_like(mu_V)
        else:
            raise ValueError(f"Unknown bias_mode: {bias_mode}")

        self.align_bias.data.copy_(init_bias)
        self.initialized = True

        print("[Transpose] initialized from class-level stats")
        print(f"  init_scale mean: {init_scale.mean().item():.6f}")
        print(f"  init_bias mean : {init_bias.mean().item():.6f}")

    def forward(
        self,
        text_feat: torch.Tensor,   # [K, D]
        mu_V: torch.Tensor = None, # optional [K, D]
        blend: float = 1.0,
        eps: float = 1e-6,
        normalize: bool = True,
    ):
        t = text_feat.float()                              # [K, D]
        mu_T = t.mean(dim=0, keepdim=True)                 # [1, D]
        var_T = t.var(dim=0, unbiased=False, keepdim=True).clamp_min(eps)

        scale = self.get_scale()                           # [K, D]

        t_transport = scale * (t - mu_T) + self.align_bias # [K, D]

        if self.use_visual_anchor and (mu_V is not None):
            mu_V = mu_V.to(t.device, dtype=t.dtype)
            if mu_V.dim() == 1:
                mu_V = mu_V.unsqueeze(0).expand_as(t)
            t_transport = t_transport + mu_V

        # if blend < 1.0:
        #     t_transport = blend * t_transport + (1.0 - blend) * t

        if normalize:
            t_transport = F.normalize(t_transport, dim=-1)

        return t_transport

    def reg_loss(self, lambda_scale=1e-4, lambda_bias=1e-5):
        scale = self.get_scale()
        loss_scale = ((scale - 1.0) ** 2).mean()
        loss_bias = (self.align_bias ** 2).mean()
        
        return lambda_scale * loss_scale + lambda_bias * loss_bias



class ClipTuningModel(nn.Module):
    def __init__(self, clip_arch, classnames, n_ctx=16, batch_size=None,
                 ctx_init=None, ctx_position="end", learned_cls=False,
                 device="cuda", dtype=torch.float32, use_neg_boost=False,
                 use_text_orth=True, use_text_gram=False):
        super().__init__()
        self.device = device
        self.use_text_orth = use_text_orth
        self.use_text_gram = use_text_gram
        self.use_neg_boost = use_neg_boost
        self.reg_losses = {}
        self.ctx_init = ctx_init
        clip_model, clip_preprocess = load(clip_arch)
        clip_model = clip_model.to(device=device, dtype=dtype)
        self.preprocess = clip_preprocess
        self.realigned = False
        self.image_encoder = clip_model.visual
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

        # [FIX] logit_scale keep as log-scale parameter (freeze if you want)
        self.logit_scale = clip_model.logit_scale
        self.logit_scale.requires_grad_(False)

        self.bs = batch_size
        
        self.prompt_learner = PromptLearner(
            clip_model, classnames, n_ctx=n_ctx,
            ctx_init=ctx_init, ctx_position=ctx_position,
            learned_cls=learned_cls, batch_size=batch_size, device=device,
        )
        with torch.no_grad():
            init_embeds = self.prompt_learner.origin_embedds
            tok_pos = self.prompt_learner.tokenized_prompts
            text_feat_tmp, _ = self.encode_text(init_embeds, tok_pos)   # [K, D]
            text_feat_tmp = F.normalize(text_feat_tmp, dim=-1)
            embed_dim = text_feat_tmp.shape[1]
        self.num_classes = len(classnames)

        self.transpose = Transpose(
            num_classes=self.num_classes,
            embed_dim=embed_dim,
            init_scale=1.0,
            use_visual_anchor=False,
            use_log_scale=True,
            scale_clamp=(0.1, 10.0),
        ).to(device=device, dtype=dtype)
        
        if self.use_neg_boost:
            if ctx_init is not None:
                neg_ctx_init = "Not_" + ctx_init.strip().lower()
            else:
                neg_ctx_init = "Not_a_photo_of_a"
            self.neg_prompt_learner = PromptLearner(
            clip_model, classnames, n_ctx=n_ctx,
            ctx_init=neg_ctx_init, ctx_position=ctx_position,
            learned_cls=learned_cls, batch_size=batch_size, device=device,
        )
        else:
            self.neg_prompt_learner = None
        # [FIX] cache init text feats if you want gram loss
        
        # self.register_buffer("text_feats_init", None, persistent=False)
        self.register_buffer("text_feats_fixed", None, persistent=False)
        self.register_buffer("text_feats_neg_anchor", None, persistent=False)
        self._build_text_feats()

        # [FIX] freeze text encoder ONCE
        self._freeze_text_encoder()

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype
        
    def get_preprocess(self):
        return self.preprocess
        
    def _freeze_text_encoder(self):
        for p in self.transformer.parameters():
            p.requires_grad_(False)
        for p in self.token_embedding.parameters():
            p.requires_grad_(False)
        for p in self.ln_final.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def init_transpose_from_stats(
        self,
        mu_V: torch.Tensor,
        var_V: torch.Tensor,
        bias_mode: str = "muV",
    ):
        self._build_text_feats()
        text_pos = self.text_feats_fixed
    
        mu_v_expand = mu_V.unsqueeze(0).expand_as(text_pos) if mu_V.dim() == 1 else mu_V
        var_v_expand = var_V.unsqueeze(0).expand_as(text_pos) if var_V.dim() == 1 else var_V
    
        self.transpose.init_alignment_from_stats(
            text_feat=text_pos,
            mu_V=mu_v_expand,
            var_V=var_v_expand,
            bias_mode=bias_mode,
        )
        
    @torch.no_grad()
    def build_negative_text_anchor(self, gap: int = 5, realign: bool = False, mu_v=None, var_v=None, blend: float = 1.0):
        """
        Build fixed negative text features once.
        This anchor is NOT trainable and is reused in forward().
        """
        init_embeds = self.prompt_learner.origin_embedds
        tok_pos = self.prompt_learner.tokenized_prompts
    
        text_feat, _ = self.encode_text(init_embeds, tok_pos)   # [C, D]
        text_feat = F.normalize(text_feat, dim=-1)
    
        C, D = text_feat.shape
        neg_feat = []
    
        for c in range(C):
            left = text_feat[:max(c - gap, 0), :]
            right = text_feat[min(c + gap + 1, C):, :]
    
            if left.numel() == 0 and right.numel() == 0:
                others = torch.cat([text_feat[:c, :], text_feat[c+1:, :]], dim=0)
            elif left.numel() == 0:
                others = right
            elif right.numel() == 0:
                others = left
            else:
                others = torch.cat([left, right], dim=0)
    
            neg_c = others.mean(dim=0, keepdim=True)
            neg_c = F.normalize(neg_c, dim=-1)
            neg_feat.append(neg_c)
    
        neg_feat = torch.cat(neg_feat, dim=0)   # [C, D]
    
        # optional one-time realignment
        if realign and (mu_v is not None) and (var_v is not None):
            neg_feat = self._realignment(neg_feat, mu_v, var_v, blend=blend)
    
        self.text_feats_neg_anchor = neg_feat.detach()
        print("[ClipTuningModel] Fixed negative text anchor built.")

        
    @torch.no_grad()
    def _build_text_feats(self):
        init_embeds = self.prompt_learner.origin_embedds          # [C, L, D]
        tok_pos = self.prompt_learner.tokenized_prompts          # [C, T]
    
        text_feat, _ = self.encode_text(init_embeds, tok_pos)    # [C, D]
        text_feat = F.normalize(text_feat, dim=-1)
        self.register_buffer("text_feats_fixed", text_feat)
        # self.text_feats_init = text_feat.detach()
            
        
    def _realignment(
        self,
        text_feat: torch.Tensor,       # [K,D]
        mu_V: torch.Tensor,
        var_V: torch.Tensor,
        eps: float = 1e-6,
        blend: float = 0.5,
    ):
        t = text_feat.float()
        mu_T = t.mean(dim=0, keepdim=True)
        var_T = t.var(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
        print(f"previous avg var_T is {var_T.mean().item():.6f}")
        
        mu_V = mu_V.to(t.device, dtype=t.dtype)
        var_V = var_V.to(t.device, dtype=t.dtype).clamp_min(eps)
        
        if mu_V.dim() == 1:
            mu_V = mu_V.unsqueeze(0).expand_as(t)
        if var_V.dim() == 1:
            var_V = var_V.unsqueeze(0).expand_as(t)
        
        t_transport = (t - mu_T) / torch.sqrt(var_T)
        t_transport = t_transport * torch.sqrt(var_V) + mu_V
        
        print(f"target avg var_V is {var_V.mean(dim=1).item():.6f}")
        print(f"transport avg var is {t_transport.var(dim=0, unbiased=False, keepdim=True).mean().item():.6f}")
        
        if blend < 1.0:
            t_transport = blend * t_transport + (1.0 - blend) * t
        
        t_align = t_transport/t_transport.norm(dim=-1, keepdim=True)
        print(f"after normalize avg var is {t_align.var(dim=0, unbiased=False, keepdim=True).mean().item():.6f}")
        
        return t_align
        
    def encode_text(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        X = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        X = X / X.norm(dim=-1, keepdim=True)
        return X, x

    def forward(
            self,
            images,
            features=False,
            return_neg=False,
            alpha=0.5,
            realign_blend=0.5,
        ):
        # -------- IMAGE --------
        with torch.no_grad():
            image_features = self.image_encoder(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # -------- TEXT --------
        if self.text_feats_fixed is None:
            self._build_text_feats()
        text_pos = self.text_feats_fixed   # [K, D]

        # -------- learnable class-level realignment --------
        text_pos = self.transpose(
            text_feat=text_pos,
            mu_V=None,
            blend=realign_blend,
            normalize=True,)

        logits_pos = self.logit_scale.exp() * image_features @ text_pos.t()

        # -------- NEG --------
        logits_neg, text_neg = None, None
        if getattr(self, "use_neg_boost", False):
            if self.text_feats_neg_anchor is None:
                raise RuntimeError("Negative text anchor is not initialized.")
    
            text_neg = F.normalize(self.text_feats_neg_anchor, dim=-1)
            logits_neg = self.logit_scale.exp() * image_features @ text_neg.t()
    
        # -------- reg losses --------
        self.reg_losses = {}
        if getattr(self, "use_text_orth", False):
            self.reg_losses["loss_text_orth"] = orthogonal_text_loss(text_pos)
    
        if getattr(self, "use_text_gram", False) and hasattr(self, "text_feats_anchor") and self.text_feats_anchor is not None:
            self.reg_losses["loss_text_gram"] = gram_preserve_loss(text_pos, self.text_feats_anchor)
    
        self.reg_losses["loss_transpose"] = self.transpose.reg_loss(
            lambda_scale=1e-4,
            lambda_bias=1e-5,
        )
    
        out = (logits_pos, image_features) if features else logits_pos
    
        if return_neg:
            return out, {
                "logits_pos": logits_pos,
                "logits_neg": logits_neg,
                "text_pos": text_pos,
                "text_neg": text_neg,
            }

        return out, {"text_pos": text_pos}
        # if mu_V_cls is not None and var_V_cls is not None:
        #     if not self.transpose.initialized:
        #         self.transpose.init_alignment_from_stats(
        #             text_feat=text_pos,
        #             mu_V=mu_V_cls,
        #             var_V=var_V_cls,
        #             bias_mode="muV",
        #         )



        # elif mu_v is not None and var_v is not None:
        #     mu_v_expand = mu_v.unsqueeze(0).expand_as(text_pos) if mu_v.dim() == 1 else mu_v
        #     var_v_expand = var_v.unsqueeze(0).expand_as(text_pos) if var_v.dim() == 1 else var_v

        #     if not self.transpose.initialized:
        #         self.transpose.init_alignment_from_stats(
        #             text_feat=text_pos,
        #             mu_V=mu_v_expand,
        #             var_V=var_v_expand,
        #             bias_mode="muV",
        #         )

        #     text_pos = self.transpose(
        #         text_feat=text_pos,
        #         mu_V=None,
        #         blend=realign_blend,
        #         normalize=True,)
        # else:
        #     text_pos = F.normalize(text_pos, dim=-1)


# class Transpose(nn.Module):
#     """
#     Class-level diagonal transport with fixed initialization + learnable residuals:

#         t'_c = (A0_c * S_c) * (t_c - mu_T) + (b0_c + db_c)

#     where:
#         A0_c : fixed initialized scale       [K, D]
#         S_c  : learnable residual scale      [K, D], initialized as 1
#         b0_c : fixed initialized bias        [K, D]
#         db_c : learnable residual bias       [K, D], initialized as 0
#     """

#     def __init__(
#         self,
#         num_classes: int,
#         embed_dim: int,
#         init_scale: float = 1.0,
#         use_log_res_scale: bool = True,
#         scale_clamp=(0.1, 10.0),
#     ):
#         super().__init__()
#         self.num_classes = num_classes
#         self.embed_dim = embed_dim
#         self.use_log_res_scale = use_log_res_scale
#         self.scale_clamp = scale_clamp

#         # ----- fixed buffers -----
#         self.register_buffer(
#             "base_scale",
#             torch.ones(num_classes, embed_dim) * init_scale
#         )  # A0
#         self.register_buffer(
#             "base_bias",
#             torch.zeros(num_classes, embed_dim)
#         )  # b0

#         # ----- learnable residual scale -----
#         # effective scale = base_scale * res_scale
#         if use_log_res_scale:
#             self.log_res_scale = nn.Parameter(
#                 torch.zeros(num_classes, embed_dim)
#             )  # exp(0)=1
#         else:
#             self.res_scale = nn.Parameter(
#                 torch.ones(num_classes, embed_dim)
#             )

#         # ----- learnable residual bias -----
#         self.res_bias = nn.Parameter(
#             torch.zeros(num_classes, embed_dim)
#         )

#         self.initialized = False

#     def get_res_scale(self):
#         if self.use_log_res_scale:
#             s = torch.exp(self.log_res_scale)
#         else:
#             s = self.res_scale
#         return s

#     def get_scale(self):
#         scale = self.base_scale * self.get_res_scale()
#         return scale.clamp(self.scale_clamp[0], self.scale_clamp[1])

#     def get_bias(self):
#         return self.base_bias + self.res_bias

#     @torch.no_grad()
#     def init_alignment_from_stats(
#         self,
#         text_feat: torch.Tensor,   # [K, D]
#         mu_V: torch.Tensor,        # [K, D] or [D]
#         var_V: torch.Tensor,       # [K, D] or [D]
#         eps: float = 1e-6,
#         bias_mode: str = "transport",   # ["muV", "transport", "zero"]
#     ):
#         t = text_feat.float()                                  # [K, D]
#         mu_T = t.mean(dim=0, keepdim=True)                     # [1, D]
#         var_T = t.var(dim=0, unbiased=False, keepdim=True).clamp_min(eps)

#         mu_V = mu_V.to(t.device, dtype=t.dtype)
#         var_V = var_V.to(t.device, dtype=t.dtype).clamp_min(eps)

#         if mu_V.dim() == 1:
#             mu_V = mu_V.unsqueeze(0).expand_as(t)              # [K, D]
#         if var_V.dim() == 1:
#             var_V = var_V.unsqueeze(0).expand_as(t)            # [K, D]

#         # fixed base scale from statistics
#         init_scale = torch.sqrt(var_V / var_T).clamp(*self.scale_clamp)   # [K, D]
#         self.base_scale.copy_(init_scale)

#         # residual scale starts from identity
#         if self.use_log_res_scale:
#             self.log_res_scale.data.zero_()
#         else:
#             self.res_scale.data.fill_(1.0)

#         # fixed base bias
#         if bias_mode == "muV":
#             init_bias = mu_V
#         elif bias_mode == "transport":
#             # with residual scale = 1 and residual bias = 0,
#             # want: scale*(t-mu_T) + bias ≈ mu_V
#             init_bias = mu_V - init_scale * (t - mu_T)
#         elif bias_mode == "zero":
#             init_bias = torch.zeros_like(mu_V)
#         else:
#             raise ValueError(f"Unknown bias_mode: {bias_mode}")

#         self.base_bias.copy_(init_bias)

#         # residual bias starts from zero
#         self.res_bias.data.zero_()

#         self.initialized = True

#         print("[Transpose] initialized from class-level stats")
#         print(f"  base_scale mean: {self.base_scale.mean().item():.6f}")
#         print(f"  base_bias  mean: {self.base_bias.mean().item():.6f}")

#     def forward(
#         self,
#         text_feat: torch.Tensor,   # [K, D]
#         normalize: bool = True,
#     ):
#         t = text_feat.float()                              # [K, D]
#         mu_T = t.mean(dim=0, keepdim=True)                 # [1, D]

#         scale = self.get_scale()                           # [K, D]
#         bias = self.get_bias()                             # [K, D]

#         t_transport = scale * (t - mu_T) + bias

#         if normalize:
#             t_transport = F.normalize(t_transport, dim=-1)

#         return t_transport

#     def reg_loss(
#         self,
#         lambda_scale=1e-4,
#         lambda_bias=1e-5,
#     ):
#         # regularize only residual parts
#         res_scale = self.get_res_scale()
#         loss_scale = ((res_scale - 1.0) ** 2).mean()
#         loss_bias = (self.res_bias ** 2).mean()
#         return lambda_scale * loss_scale + lambda_bias * loss_bias


def get_load_clip_tuning_model(
    clip_arch: str,
    classnames,
    device,
    n_ctx: int = 8,
    batch_size: int = 16,
    ctx_position: str = 'end',
    ctx_init: str = None,
    load: bool = False,
    learned_cls: bool = False,
    use_text_orth: bool = True,
    use_text_gram: bool = False,
    use_neg_boost: bool = False,
    dtype: torch.dtype = torch.float32,
    **model_kwargs
):
    # 1) init model
    model = ClipTuningModel(
        clip_arch=clip_arch,
        classnames=classnames,
        n_ctx=n_ctx,
        ctx_init=ctx_init,
        ctx_position=ctx_position,
        batch_size=batch_size,
        learned_cls=learned_cls,
        use_text_orth=use_text_orth,
        use_text_gram=use_text_gram,
        use_neg_boost=use_neg_boost,
        device=device,
        dtype=dtype,
    )
    clip_preprocess = model.get_preprocess()

    if not load:
        return model, clip_preprocess

    # 2) load ckpt
    ckpt = model_kwargs.get("ckpt_path", None)
    if ckpt is None:
        raise ValueError("Must provide 'ckpt_path' when load=True")

    # ckpt = torch.load(ckpt_path, map_location=device)
    print(f"Loaded checkpoint from {ckpt_path}, epoch={ckpt.get('epoch', 'unknown')}")

    # 3) restore POS prompt learner (full state_dict)
    # support both new/old key names
    pos_state = None
    if "prompt_state" in ckpt:
        pos_state = ckpt["prompt_state"]
    elif "model_state" in ckpt:
        pos_state = ckpt["model_state"]

    if pos_state is not None:
        missing, unexpected = model.prompt_learner.load_state_dict(pos_state, strict=False)
        if len(missing) or len(unexpected):
            print(f"[WARN] POS prompt_learner load_state_dict strict=False "
                  f"missing={missing}, unexpected={unexpected}")
        else:
            print("[OK] Loaded prompt_learner.")
    else:
        print("[WARN] No pos prompt state found in ckpt (expected prompt_state/model_state).")

    # 4) restore NEG ctx only (optional)
    # you saved neg_ctx_check = neg_prompt_learner.ctx.detach().cpu()
    if use_neg_boost and getattr(model, "neg_prompt_learner", None) is not None:
        neg_ctx = ckpt.get("neg_ctx_check", None)
        if neg_ctx is None:
            print("[WARN] use_neg_boost=True but ckpt has no 'neg_ctx_check'. NEG prompt will start from init.")
        else:
            neg_ctx = neg_ctx.to(device=device, dtype=model.neg_prompt_learner.ctx.dtype)
            with torch.no_grad():
                if model.neg_prompt_learner.ctx.shape != neg_ctx.shape:
                    raise ValueError(
                        f"NEG ctx shape mismatch: model {tuple(model.neg_prompt_learner.ctx.shape)} "
                        f"vs ckpt {tuple(neg_ctx.shape)}. "
                        f"Check n_ctx/batch_size or whether you used batch-wise ctx."
                    )
                model.neg_prompt_learner.ctx.copy_(neg_ctx)
            print("[OK] Loaded NEG ctx only (neg_ctx_check).")

    return model, clip_preprocess

    
