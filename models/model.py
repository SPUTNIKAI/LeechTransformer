from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LeechConfig:
    vocab_size: int
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 8
    block_size: int = 512
    dropout: float = 0.05
    bias: bool = False
    tie_weights: bool = True
    lambda_geo: float = 0.01
    resonance_threshold: float = 0.95

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0
        head_dim = self.d_model // self.n_heads
        assert head_dim % 24 == 0, "head_dim должен быть кратен 24"


def generate_leech_kernel(dim: int = 24) -> torch.Tensor:
    """Ортогональная матрица dim x dim (PoC через QR)."""
    base = torch.zeros(dim, dim, dtype=torch.float32)
    for i in range(dim - 1):
        base[i, i] = 2.0
        base[i, i + 1] = 2.0
    base[-1, -1] = 2.0
    base[-1, 0] = -2.0
    q, _ = torch.linalg.qr(base)
    return q


class LeechAttention(nn.Module):
    def __init__(self, cfg: LeechConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = self.head_dim**-0.5
        self.num_blocks = self.head_dim // 24

        kernel = generate_leech_kernel(24)
        total_blocks = self.n_heads * self.num_blocks
        W_list = [kernel] * total_blocks
        self.register_buffer("W_leech", torch.block_diag(*W_list))

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(1, 1, cfg.block_size, cfg.block_size)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q.view(B, self.n_heads, T, self.num_blocks, 24)
        k = k.view(B, self.n_heads, T, self.num_blocks, 24)
        kernel = self.W_leech[0:24, 0:24]
        q = torch.einsum("...i,ij->...j", q, kernel)
        k = torch.einsum("...i,ij->...j", k, kernel)
        q = q.reshape(B, self.n_heads, T, self.head_dim)
        k = k.reshape(B, self.n_heads, T, self.head_dim)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.out(out)


class LeechBlock(nn.Module):
    def __init__(self, cfg: LeechConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = LeechAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class LeechResonanceBiasing(nn.Module):
    """Смещение логитов по резонансу с базисом Лича (как в ноутбуке)."""

    def __init__(self, d_model: int, vocab_size: int, leech_basis: torch.Tensor, alpha_init: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_blocks = d_model // 24
        self.register_buffer("basis", leech_basis)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.register_buffer("token_proj", torch.zeros(vocab_size, 24))

    @torch.no_grad()
    def compute_token_proj(self, tok_emb: torch.Tensor) -> None:
        emb_blocks = tok_emb.view(-1, self.num_blocks, 24)
        proj = torch.einsum("vki,ij->vkj", emb_blocks, self.basis)
        proj = proj.sum(dim=1)
        self.token_proj.copy_(proj)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)
        B, T, D = hidden_states.shape
        h_blocks = hidden_states.view(B, T, self.num_blocks, 24)
        h_proj = torch.einsum("btki,ij->btkj", h_blocks, self.basis)
        h_proj = h_proj.sum(dim=2)
        bias = self.alpha * (h_proj @ self.token_proj.T)
        if T == 1:
            bias = bias.squeeze(1)
        return bias


class LeechResonanceLoss(nn.Module):
    def __init__(self, leech_basis: torch.Tensor):
        super().__init__()
        self.register_buffer("basis", leech_basis)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, D = hidden_states.shape
        h = hidden_states.view(B, T, D // 24, 24)
        h_norm = F.normalize(h, dim=-1)
        b_norm = F.normalize(self.basis, dim=-1)
        sim = torch.matmul(h_norm, b_norm.T)
        max_sim = torch.max(sim, dim=-1)[0]
        return 1.0 - max_sim.mean()


class DreamDecoder:
    def __init__(self, leech_basis: torch.Tensor, threshold: float = 0.95):
        self.basis = leech_basis
        self.threshold = threshold

    def check(self, hidden_state: torch.Tensor) -> tuple[float, str]:
        h = hidden_state[:24].unsqueeze(0)
        h_norm = F.normalize(h, dim=-1)
        b_norm = F.normalize(self.basis, dim=-1)
        sim = torch.matmul(h_norm, b_norm.T)
        max_res = torch.max(sim).item()
        if max_res > 0.999:
            status = "ABSOLUTE GENESIS"
        elif max_res > self.threshold:
            status = "AWAKE"
        else:
            status = "DREAMING"
        return max_res, status


class LeechGPT(nn.Module):
    def __init__(self, cfg: LeechConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.blocks = nn.ModuleList([LeechBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        leech_basis = generate_leech_kernel(24)
        self.register_buffer("leech_basis", leech_basis)
        self.resonance_loss_fn = LeechResonanceLoss(leech_basis)

        self.resonator: LeechResonanceBiasing | None = None
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def attach_resonator(self, alpha_init: float = 0.1) -> None:
        self.resonator = LeechResonanceBiasing(
            d_model=self.cfg.d_model,
            vocab_size=self.cfg.vocab_size,
            leech_basis=self.leech_basis,
            alpha_init=alpha_init,
        ).to(self.tok_emb.weight.device)
        self.resonator.compute_token_proj(self.tok_emb.weight.detach())

    @torch.no_grad()
    def refresh_resonator_token_proj(self) -> None:
        if self.resonator is not None:
            self.resonator.compute_token_proj(self.tok_emb.weight.detach())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        use_resonator: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        b, t = idx.size()
        assert t <= self.cfg.block_size
        x = self.tok_emb(idx) + self.pos_emb[:, :t, :]
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.head(x)
        hidden = x

        if use_resonator and self.resonator is not None:
            logits = logits + self.resonator(hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1))
            if self.cfg.lambda_geo > 0 and not use_resonator:
                loss_geo = self.resonance_loss_fn(hidden)
                loss = loss + self.cfg.lambda_geo * loss_geo

        return logits, hidden, loss

