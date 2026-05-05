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
        device="cuda",
    ):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        self.dtype = clip_model.dtype
        self.device = clip_model.visual.conv1.weight.device
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.batch_size = batch_size
        
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
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        print(f"ctx learned size is {self.ctx.shape}!!!")
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, self.ctx_dim, dtype=self.dtype)  # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors)  # to be optimized

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
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized).type(self.dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])  # CLS, EOS
        self.tokenized_prompts = tokenized

    def reset(self):
        """恢复 ctx 和 learned cls 到初始状态"""
        self.ctx.copy_(self.ctx_init_state)
        if self.learned_cls and self.cls is not None:
            self.cls.copy_(self.cls_init_state)

    def reset_prompt(self, new_ctx_str):
        """用新字符串初始化 context"""
        n_ctx = len(new_ctx_str.split(" "))
        prompt = tokenize(new_ctx_str).to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(prompt).type(self.dtype)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        self.ctx = nn.Parameter(ctx_vectors)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.n_ctx = n_ctx

    def reset_classnames(self, classnames, arch=None, DOWNLOAD_ROOT=None):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim,
                                      dtype=self.dtype)  # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx:, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None:
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)
            
        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls if self.batch_size is not None else self.cls.unsqueeze(1)  # (n_cls, 1, D)
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,  # (n_cls, n_ctx, dim)
                        cls,  # (n_cls, 1, dim)
                        suffix,  # (n_cls, 1 + n_ctx:, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,  # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx  # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class ClipTuningModel(nn.Module):
    def __init__(self, clip_arch, classnames, n_ctx=16, batch_size=None,
                 ctx_init=None, ctx_position="end", learned_cls=False,
                 device="cuda", dtype=torch.float32,
                 use_text_orth=True, use_text_gram=False):
        super().__init__()
        self.device = device
        self.use_text_orth = use_text_orth
        self.use_text_gram = use_text_gram
        self.reg_losses = {}

        clip_model, clip_preprocess = load(clip_arch)
        clip_model = clip_model.to(device=device, dtype=dtype)
        self.preprocess = clip_preprocess

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

        # [FIX] cache init text feats if you want gram loss
        self.register_buffer("text_feats_init", None, persistent=False)
        if self.use_text_gram:
            self._build_text_init(classnames)

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
        # text_projection usually not trained; freeze if it's a Parameter
        if isinstance(self.text_projection, torch.nn.Parameter):
            self.text_projection.requires_grad_(False)

    @torch.no_grad()
    def _build_text_init(self, classnames):
        import clip
        texts = [f"a photo of a {c}" for c in classnames]
        tokenized = clip.tokenize(texts).to(self.device)

        x = self.token_embedding(tokenized).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        text_feat = x[torch.arange(x.shape[0]), tokenized.argmax(dim=-1)] @ self.text_projection
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        self.text_feats_init = text_feat.detach()

    def encode_text(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        X = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        X = X / X.norm(dim=-1, keepdim=True)
        return X, x

    def forward(self, images, topk=None, features=False):
        with torch.no_grad():
            image_features = self.image_encoder(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features, embedded_prompt = self.encode_text(prompts, tokenized_prompts)

        # [FIX] reg losses
        self.reg_losses = {}
        if self.use_text_orth:
            self.reg_losses["loss_text_orth"] = orthogonal_text_loss(text_features)
        if self.use_text_gram and (self.text_feats_init is not None):
            self.reg_losses["loss_text_gram"] = gram_preserve_loss(text_features, self.text_feats_init)

        logits = self.logit_scale.exp() * image_features @ text_features.t()

        if topk is not None:
            topk_vals, topk_idx = logits.topk(topk, dim=-1)
            topk_logits = torch.gather(logits, 1, topk_idx)
            top_k_prompt = embedded_prompt[topk_idx, :]
            self.top_k_prompt = top_k_prompt
            if features:
                return logits, image_features, topk_logits, top_k_prompt
            else:
                return logits, topk_logits, top_k_prompt

        return (logits, image_features) if features else logits



# class ClipTuningModel(nn.Module):
#     def __init__(
#         self,
#         clip_arch: str,
#         classnames: list[str],
#         n_ctx: int = 16,
#         batch_size: int = None,
#         ctx_init: str = None,
#         ctx_position: str = "end",
#         learned_cls: bool = False,
#         device: str = "cuda",
#         dtype = torch.float32,
#     ):
#         super().__init__()
#         self.device = device

#         # --- Load CLIP ---
#         clip_model, clip_preprocess = load(clip_arch)
#         clip_model = clip_model.to(device=device, dtype=dtype)
#         self.preprocess = clip_preprocess
#         self.image_encoder = clip_model.visual
#         self.transformer = clip_model.transformer
#         self.token_embedding = clip_model.token_embedding
#         self.positional_embedding = clip_model.positional_embedding
#         self.ln_final = clip_model.ln_final
#         self.text_projection = clip_model.text_projection
#         self.logit_scale = clip_model.logit_scale.exp().detach()
#         self.bs = batch_size
#         # --- Prompt Learner ---
#         self.prompt_learner = PromptLearner(
#             clip_model,
#             classnames,
#             n_ctx=n_ctx,
#             ctx_init=ctx_init,
#             ctx_position=ctx_position,
#             learned_cls=learned_cls,
#             batch_size=batch_size,
#             device=device,
#         )
#         self.register_buffer("text_feats_init", None, persistent=False)
#         # self._build_text_init(classnames)  # compute once

#         # [NEW] switches
#         self.use_text_orth = use_text_orth
#         self.use_text_gram = use_text_gram

#         # [NEW] a dict to store regularization losses computed in forward
#         self.reg_losses = {}
#     @property
#     def dtype(self):
#         return self.image_encoder.conv1.weight.dtype

#     @property
#     def _freeze_params(self):
#         for p in self.transformer.parameters():
#             p.requires_grad_(False)
#         for p in self.token_embedding.parameters():
#             p.requires_grad_(False)
#         for p in self.ln_final.parameters():
#             p.requires_grad_(False)
        
#     # text_projection 通常是 parameter 或 buffer，按需要冻结
#     @torch.no_grad()
#     def _build_text_init(self, classnames: list[str]):
#         """
#         text_init[c] = normalized CLIP text feature for "a photo of a {classname}"
#         """
#         import clip  # <-- [MAY NEED ADJUSTMENT] depending on your load()

#         texts = [f"a photo of a {c}" for c in classnames]
#         tokenized = clip.tokenize(texts).to(self.device)  # [C, 77]

#         x = self.token_embedding(tokenized).type(self.dtype)            # [C,77,D]
#         x = x + self.positional_embedding.type(self.dtype)
#         x = x.permute(1, 0, 2)                                          # LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)                                          # NLD
#         x = self.ln_final(x).type(self.dtype)

#         # EOT position: argmax over token ids (OpenAI CLIP convention)
#         text_feat = x[torch.arange(x.shape[0]), tokenized.argmax(dim=-1)] @ self.text_projection
#         text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

#         self.text_feats_init = text_feat.detach()
    
#     def get_preprocess(self):
#         return self.preprocess
    
#     def encode_text(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor):
#         """
#         将 prompt embedding 送入 CLIP text encoder
#         """
#         _freeze_params()
#         x = prompts + self.positional_embedding.type(self.dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x).type(self.dtype)

#         X = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
#         X = X / X.norm(dim=-1, keepdim=True)
#         return X, x
        
#     def forward(self, images: torch.Tensor, topk: int = None, features: bool = False):
#         """
#         images -> logits
#         If features=True, also return image_features
#         """
    
#         # -------------------------
#         # Image features
#         # -------------------------
#         with torch.no_grad():
#             image_features = self.image_encoder(images)
#             image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
#         # -------------------------
#         # Text features
#         # -------------------------
#         prompts = self.prompt_learner()  # (n_cls, L, D)
#         tokenized_prompts = self.prompt_learner.tokenized_prompts
    
#         text_features, embedded_prompt = self.encode_text(
#             prompts, tokenized_prompts
#         )  # (n_cls, D)
#         # ============================================================
#         self.reg_losses = {}
#         if self.use_text_orth:
#             self.reg_losses["loss_text_orth"] = orthogonal_text_loss(text_features)

#         if self.use_text_gram and (self.text_init is not None):
#             self.reg_losses["loss_text_gram"] = gram_preserve_loss(text_features, self.text_init)

#         # -------------------------
#         # Logits
#         # -------------------------
#         logits = self.logit_scale * image_features @ text_features.t()
    
#         # -------------------------
#         # Top-k branch (optional)
#         # -------------------------
#         if topk is not None:
#             topk_vals, topk_idx = logits.topk(topk, dim=-1)
#             top_k_prompt = embedded_prompt[topk_idx, :]
#             self.top_k_prompt = top_k_prompt
#             topk_logits = logits[:, topk_idx]
    
#             if features:
#                 return logits, image_features, topk_logits, top_k_prompt
#             else:
#                 return logits, topk_logits, top_k_prompt

#         # -------------------------
#         # Normal return
#         # -------------------------
#         if features:
#             return logits, image_features
#         else:
#             return logits

    # def forward(self, images: torch.Tensor, topk: int = None, features: bool = False):
    #     """
    #     images -> logits
    #     If features=True, also return image_features
    #     """
    #     # -------------------------
    #     # Image features (frozen)
    #     # -------------------------
    #     with torch.no_grad():
    #         image_features = self.image_encoder(images)
    #         image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    #     # -------------------------
    #     # Text features (learned)
    #     # -------------------------
    #     prompts = self.prompt_learner()  # (n_cls, L, D)
    #     tokenized_prompts = self.prompt_learner.tokenized_prompts

    #     text_features, embedded_prompt = self.encode_text(prompts, tokenized_prompts)  # (C, D)

    #     # ============================================================
    #     # [NEW] compute text regularizers and store them in self.reg_losses
    #     # ============================================================
    #     self.reg_losses = {}
    #     if self.use_text_orth:
    #         self.reg_losses["loss_text_orth"] = orthogonal_text_loss(text_features)

    #     if self.use_text_anchor and (self.text_init is not None):
    #         self.reg_losses["loss_text_anchor"] = text_anchor_loss(text_features, self.text_init)

    #     if self.use_text_gram and (self.text_init is not None):
    #         self.reg_losses["loss_text_gram"] = gram_preserve_loss(text_features, self.text_init)

    #     # -------------------------
    #     # Logits
    #     # -------------------------
    #     # ============================================================
    #     # [CHANGED] use logit_scale.exp() (since it's log-scale)
    #     # ============================================================
    #     logits = self.logit_scale.exp() * image_features @ text_features.t()

    #     # -------------------------
    #     # Top-k branch (optional)
    #     # -------------------------
    #     if topk is not None:
    #         topk_vals, topk_idx = logits.topk(topk, dim=-1)

    #         # NOTE: embedded_prompt is [C, L, D], so topk_idx [B,K] indexing gives [B,K,L,D]
    #         top_k_prompt = embedded_prompt[topk_idx, :]
    #         self.top_k_prompt = top_k_prompt

    #         # NOTE: logits[:, topk_idx] would be wrong shape due to advanced indexing.
    #         # safer:
    #         topk_logits = torch.gather(logits, 1, topk_idx)

    #         if features:
    #             return logits, image_features, topk_logits, top_k_prompt
    #         else:
    #             return logits, topk_logits, top_k_prompt

    #     # -------------------------
    #     # Normal return
    #     # -------------------------
    #     if features:
    #         return logits, image_features
    #     else:
    #         return logits

    # --- 接口 ---
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames: list[str]):
        self.prompt_learner.reset_classnames(classnames)

import torch

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
    
    **model_kwargs
):
    # 初始化模型
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
        device=device,
    )
    clip_preprocess = model.get_preprocess()

    if not load:
        return model, clip_preprocess

    # ---- 加载 checkpoint ----
    ckpt_path = model_kwargs.get("ckpt_path", None)
    if ckpt_path is None:
        raise ValueError("Must provide 'ckpt_path' when load=True")
    
    else:
        ckpt = torch.load(ckpt_path, map_location=device)
        print(f"Loaded checkpoint from {ckpt_path}, epoch={ckpt.get('epoch', 'unknown')}")
    
    # 恢复 prompt learner
    if "model_state" in ckpt:
        model.prompt_learner.load_state_dict(ckpt["model_state"], strict=True)
        print("Loaded successfully!!")
    
    # ckpt_path = model_kwargs.get("ckpt_path", None)
    # if ckpt_path is None:
    #     raise ValueError("Must provide 'ckpt_path' when load=True")

    # ckpt = torch.load(ckpt_path, map_location=device)
    # print(f"Loaded checkpoint from {ckpt_path}, epoch={ckpt.get('epoch', 'unknown')}")
    
    # 恢复 ctx
    # if "ctx" in ckpt and hasattr(model.prompt_learner, "ctx"):
    #     print(f"--> ckpt ctx shape: {ckpt['ctx'].shape}")
    #     print(f"--> model ctx shape: {model.prompt_learner.ctx.shape}")
    #     with torch.no_grad():
    #         model.prompt_learner.ctx.data = ckpt["ctx"].to(device=device, dtype=torch.float32)    
    #     print("✓ Loaded ctx")

    # # 恢复 cls
    # if ckpt.get("cls") is not None and hasattr(model.prompt_learner, "cls"):
    #     with torch.no_grad():
    #         model.prompt_learner.cls.copy_(ckpt["cls"].to(device=device, dtype=torch.float32))
    #     print("✓ Loaded cls")

    return model, clip_preprocess


# def get_clip_tuning_model(
#     clip_arch: str,
#     classnames,
#     device,
#     n_ctx: int = 16,
#     batch_size: int = 16,
#     dtype = torch.float32,
#     ctx_position: str = 'end',
#     ctx_init: str = None,
#     learned_cls: bool = False,
# ):

#     # --- 构建模型 ---
#     model = ClipTuningModel(
#         clip_arch=clip_arch,
#         classnames=classnames,
#         n_ctx=n_ctx,
#         ctx_init=ctx_init,
#         ctx_position=ctx_position,
#         batch_size=batch_size,
#         learned_cls=learned_cls,
#         dtype = torch.float32,
#         device=device,
#     )

#     clip_preprocess = model.get_preprocess()
#     return model, clip_preprocess
