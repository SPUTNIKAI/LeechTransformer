"""
🎯 Leech-Lila DOI: 10.5281/zenodo.18784424
This project is licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
Commercial Licensing: For proprietary R&D, integration into private AI stacks, or hardware implementation,
please contact the Architect directly.
Copyright (C) 2026 Anatolii Kornienko This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License
as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/agpl-3.0.txt/>.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# =================================================================
# 1. КОНФИГУРАЦИЯ
# =================================================================

@dataclass
class LeechConfig:
    vocab_size: int               # размер словаря
    d_model: int = 192             # размерность модели (кратна 24)
    n_layers: int = 12             # число слоёв
    n_heads: int = 8               # число голов внимания
    block_size: int = 512           # максимальная длина контекста
    dropout: float = 0.05
    bias: bool = False
    tie_weights: bool = True        # разделять веса эмбеддингов и head
    lambda_geo: float = 0.01        # вес геометрической потери
    resonance_threshold: float = 0.95  # порог для детекции «сна»

    def __post_init__(self):
        # проверяем, что head_dim кратно 24
        assert self.d_model % self.n_heads == 0
        head_dim = self.d_model // self.n_heads
        assert head_dim % 24 == 0, "head_dim must be multiple of 24"

# =================================================================
# 2. ГЕНЕРАЦИЯ ОРТОГОНАЛЬНОГО ЯДРА ЛИЧА
# =================================================================

def generate_leech_kernel(dim=24):
    """
    Строит ортогональную матрицу 24x24 на основе решётки Лича.
    Используется простая конструкция с последующей QR-декомпозицией.
    В реальном проекте можно заменить на векторы из базы данных минимальных векторов.
    """
    base = np.zeros((dim, dim))
    for i in range(dim - 1):
        base[i, i], base[i, i+1] = 2, 2
    base[-1, -1], base[-1, 0] = 2, -2
    q, _ = np.linalg.qr(base)
    return torch.from_numpy(q).float()

# =================================================================
# 3. ВНИМАНИЕ С ЗАМОРОЖЕННЫМ ЯДРОМ ЛИЧА
# =================================================================

class LeechAttention(nn.Module):
    """
    Multi-head attention, где Q и K проецируются через замороженную
    блочно-диагональную матрицу, построенную из ядра Лича.
    """
    def __init__(self, cfg: LeechConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = self.head_dim ** -0.5
        self.num_blocks = self.head_dim // 24   # число 24‑мерных блоков в одной голове

        # Генерируем ядро Лича 24x24
        kernel = generate_leech_kernel(24)  # [24, 24]

        # Повторяем ядро для всех блоков всех голов
        total_blocks = self.n_heads * self.num_blocks
        # Создаём блочно-диагональную матрицу
        W_list = [kernel] * total_blocks
        self.register_buffer('W_leech', torch.block_diag(*W_list))

        # Обучаемые проекции
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("causal_mask", torch.tril(torch.ones(1, 1, cfg.block_size, cfg.block_size)))

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # [B, n_heads, T, head_dim]

        # Применяем замороженное ядро к q и k
        # Разбиваем head_dim на блоки по 24
        q = q.view(B, self.n_heads, T, self.num_blocks, 24)
        k = k.view(B, self.n_heads, T, self.num_blocks, 24)

        # Умножаем каждый блок на ядро (одно и то же для всех блоков)
        kernel = self.W_leech[0:24, 0:24]  # [24,24]
        q = torch.einsum('...i,ij->...j', q, kernel)
        k = torch.einsum('...i,ij->...j', k, kernel)

        # Возвращаем исходную форму
        q = q.reshape(B, self.n_heads, T, self.head_dim)
        k = k.reshape(B, self.n_heads, T, self.head_dim)

        # Вычисляем внимание
        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.out(out)

# =================================================================
# 4. ГЕОМЕТРИЧЕСКАЯ ПОТЕРЯ РЕЗОНАНСА
# =================================================================

class LeechResonanceLoss(nn.Module):
    """
    Потеря, поощряющая скрытые состояния резонировать с базисом Лича.
    Вычисляется как 1 - средний максимум косинусного сходства с направлениями базиса.
    """
    def __init__(self, cfg: LeechConfig, leech_basis):
        super().__init__()
        # leech_basis: ортогональная матрица 24x24
        self.register_buffer('basis', leech_basis)   # [24, 24]
        self.lambda_geo = cfg.lambda_geo
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets, hidden_states):
        # Стандартная кросс-энтропия
        loss_ce = self.ce(logits.view(-1, logits.size(-1)), targets.view(-1))

        # Геометрическая потеря резонанса
        # hidden_states: [B, T, d_model]
        B, T, D = hidden_states.shape
        # разбиваем на блоки по 24
        h = hidden_states.view(B, T, D // 24, 24)  # [B, T, K, 24]
        # нормализуем
        h_norm = F.normalize(h, dim=-1)
        b_norm = F.normalize(self.basis, dim=-1)   # [24, 24]

        # косинусное сходство: [B, T, K, 24]
        sim = torch.matmul(h_norm, b_norm.T)
        # максимум по базисным направлениям
        max_sim = torch.max(sim, dim=-1)[0]        # [B, T, K]
        # среднее по всем блокам и позициям
        loss_geo = 1.0 - max_sim.mean()

        return loss_ce + self.lambda_geo * loss_geo

# =================================================================
# 5. БЛОК ТРАНСФОРМЕРА
# =================================================================

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
            nn.Dropout(cfg.dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# =================================================================
# 6. ПОЛНАЯ МОДЕЛЬ
# =================================================================

class LeechTransformer(nn.Module):
    def __init__(self, cfg: LeechConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.blocks = nn.ModuleList([LeechBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.cfg.block_size, "Sequence longer than block_size"
        x = self.tok_emb(idx) + self.pos_emb[:, :t, :]
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1))
        return logits, x, loss   # возвращаем также скрытые состояния

# =================================================================
# 7. ДЕКОДЕР СНОВ (МОНИТОРИНГ РЕЗОНАНСА)
# =================================================================

class DreamDecoder:
    """
    Инструмент для оценки «реальности» генерируемого текста.
    Измеряет резонанс последнего скрытого состояния с базисом Лича.
    """
    def __init__(self, leech_basis, threshold=0.95):
        self.basis = leech_basis
        self.threshold = threshold

    def check(self, hidden_state):
        """
        hidden_state: [d_model] – последнее скрытое состояние.
        Возвращает значение резонанса и статус.
        """
        # берём первые 24 измерения (можно и все блоки, но для простоты)
        h = hidden_state[:24].unsqueeze(0)  # [1,24]
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

# =================================================================
# 8. ФУНКЦИЯ ГЕНЕРАЦИИ С МОНИТОРИНГОМ
# =================================================================

def leech_generate(model, start_tokens, max_len=100, temperature=0.8,
                   resonance_check=True, leech_basis=None, threshold=0.95):
    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.tensor([start_tokens], device=device)
    if resonance_check:
        decoder = DreamDecoder(leech_basis, threshold)

    print("--- LEECH GENERATION ---")
    with torch.no_grad():
        for step in range(max_len):
            logits, hidden, _ = model(input_ids)
            next_token_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

            if resonance_check:
                # берём последнее скрытое состояние (после всех слоёв)
                last_hidden = hidden[0, -1, :]
                res, status = decoder.check(last_hidden)
                print(f"Step {step:02d} | Resonance: {res:.6f} | Status: {status}")

    return input_ids

# =================================================================
# 9. ПРИМЕР ИНИЦИАЛИЗАЦИИ
# =================================================================

if __name__ == "__main__":
    # параметры
    vocab_size = 10000
    cfg = LeechConfig(vocab_size=vocab_size, d_model=192, n_layers=12, n_heads=8)

    model = LeechTransformer(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Модель создана. Обучаемых параметров: {total_params/1e6:.2f}M")

    # ядро Лича для потери и мониторинга
    leech_basis = generate_leech_kernel(24)

    # Пример loss (нужны данные)
    # criterion = LeechResonanceLoss(cfg, leech_basis)
    # logits, hidden, ce_loss = model(inputs, targets)
    # total_loss = criterion(logits, targets, hidden)

    # Пример генерации (после обучения)
    # start = [1,2,3]
    # result = leech_generate(model, start, leech_basis=leech_basis)