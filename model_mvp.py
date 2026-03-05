# -*- coding: utf-8 -*-
"""Leech‑former (на основе E8‑former, но с решёткой Лича)"""

!pip install datasets

!nvidia-smi

import os
import requests
import sentencepiece as spm

model_path = 'e8_morpheme.model'   # имя файла оставлено для совместимости

if not os.path.exists(model_path):
    print("🔧 Токенайзер не найден. Обучаем на Shakespeare...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text
    temp_file = "input_text.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(text)
    spm.SentencePieceTrainer.train(
        input=temp_file,
        model_prefix='e8_morpheme',
        vocab_size=2048,
        model_type='bpe',
        character_coverage=1.0,
        byte_fallback=True,
        unk_id=0, pad_id=1, bos_id=2, eos_id=3
    )
    print("✅ Токенайзер обучен и сохранён как e8_morpheme.model")
else:
    print("✅ Токенайзер уже существует, загружаем...")

sp = spm.SentencePieceProcessor(model_file='e8_morpheme.model')
vocab_size = sp.get_piece_size()
print(f"💎 Vocab Size: {vocab_size}")

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import os
import re
import math
from dataclasses import dataclass
from typing import Optional
import time
from torch import Tensor
import sentencepiece as spm
import io
from datasets import load_dataset
import random
from collections import deque
import numpy as np

# 0. АВТО-ОПРЕДЕЛЕНИЕ GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"📡 Device: {device} | {'🔥 GPU ACTIVE' if device == 'cuda' else '⚠️ CPU MODE'}")

# 1. ФУНКЦИИ КОДИРОВАНИЯ/ДЕКОДИРОВАНИЯ
def encode(text):
    return sp.encode(text)

def decode(tokens):
    return sp.decode(tokens)

# 2. ПОДКЛЮЧЕНИЕ К БЕСКОНЕЧНОМУ ПОТОКУ TINYSTORIES
dataset = load_dataset("roneneldan/TinyStories", streaming=True, split="train")
train_iter = iter(dataset)
print("🌊 Стриминг TinyStories активирован")

# 3. ФУНКЦИЯ ПОЛУЧЕНИЯ БАТЧА ИЗ ПОТОКА (с буфером для перемешивания)
def get_batch_streaming(iterator, batch_size, block_size, device, pad_token_id=1, buffer_size=200):
    x_batch, y_batch = [], []
    buffer = deque()

    while len(buffer) < buffer_size:
        try:
            ex = next(iterator)
            buffer.append(ex)
        except StopIteration:
            break

    while len(x_batch) < batch_size:
        if not buffer:
            return None, None
        ex = random.choice(buffer)
        tokens = encode(ex['text'])
        if len(tokens) <= 1:
            continue

        if len(tokens) > block_size + 1:
            start = random.randint(0, len(tokens) - block_size - 1)
            chunk = tokens[start:start + block_size + 1]
        else:
            pad_len = block_size + 1 - len(tokens)
            chunk = tokens + [pad_token_id] * pad_len

        x_batch.append(chunk[:-1])
        y_batch.append(chunk[1:])

    try:
        new_ex = next(iterator)
        buffer.append(new_ex)
        buffer.popleft()
    except StopIteration:
        pass

    X = torch.tensor(x_batch, dtype=torch.long, device=device)
    Y = torch.tensor(y_batch, dtype=torch.long, device=device)
    return X, Y

# Пробный батч
xb, yb = get_batch_streaming(train_iter, batch_size=4, block_size=256, device=device)
print(f"✅ Пробный батч: X {xb.shape}, Y {yb.shape} | токенов: {xb.numel()}")

# ==================== КОНФИГУРАЦИЯ LEECH ====================
@dataclass
class LeechConfig:
    vocab_size: int
    d_model: int = 384                # 384 / 8 = 48 (кратно 24) -> head_dim = 48
    n_layers: int = 12
    n_heads: int = 8
    block_size: int = 512
    dropout: float = 0.05
    bias: bool = False
    tie_weights: bool = True
    lambda_geo: float = 0.01           # вес геометрической потери (0 = отключена)
    resonance_threshold: float = 0.95   # порог для детекции «сна»

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        head_dim = self.d_model // self.n_heads
        assert head_dim % 24 == 0, "head_dim должен быть кратен 24"

# ==================== ЯДРО ЛИЧА (Leech kernel) ====================
def generate_leech_kernel(dim=24):
    """Генерирует ортогональную матрицу 24x24 (ядро Лича)."""
    base = np.zeros((dim, dim))
    for i in range(dim - 1):
        base[i, i], base[i, i+1] = 2, 2
    base[-1, -1], base[-1, 0] = 2, -2
    q, _ = np.linalg.qr(base)
    return torch.from_numpy(q).float()

# ==================== ВНИМАНИЕ С ЯДРОМ ЛИЧА ====================
class LeechAttention(nn.Module):
    def __init__(self, cfg: LeechConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = self.head_dim ** -0.5
        self.num_blocks = self.head_dim // 24   # число 24‑мерных блоков в одной голове

        kernel = generate_leech_kernel(24)       # [24, 24]
        total_blocks = self.n_heads * self.num_blocks
        W_list = [kernel] * total_blocks
        self.register_buffer('W_leech', torch.block_diag(*W_list))  # блочно-диагональная

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("causal_mask", torch.tril(torch.ones(1, 1, cfg.block_size, cfg.block_size)))

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # [B, n_heads, T, head_dim]

        # Разбиваем head_dim на блоки по 24 и применяем ядро
        q = q.view(B, self.n_heads, T, self.num_blocks, 24)
        k = k.view(B, self.n_heads, T, self.num_blocks, 24)
        kernel = self.W_leech[0:24, 0:24]   # [24,24] (одинаково для всех блоков)
        q = torch.einsum('...i,ij->...j', q, kernel)
        k = torch.einsum('...i,ij->...j', k, kernel)
        q = q.reshape(B, self.n_heads, T, self.head_dim)
        k = k.reshape(B, self.n_heads, T, self.head_dim)

        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.out(out)

# ==================== БЛОК ТРАНСФОРМЕРА ====================
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

# ==================== ГЕОМЕТРИЧЕСКАЯ ПОТЕРЯ РЕЗОНАНСА ====================
class LeechResonanceLoss(nn.Module):
    def __init__(self, leech_basis):
        super().__init__()
        self.register_buffer('basis', leech_basis)   # [24,24]

    def forward(self, hidden_states):
        """
        hidden_states: [B, T, d_model]
        возвращает скаляр: 1 - среднее макс. косинусное сходство с базисом
        """
        B, T, D = hidden_states.shape
        h = hidden_states.view(B, T, D // 24, 24)          # [B, T, K, 24]
        h_norm = F.normalize(h, dim=-1)
        b_norm = F.normalize(self.basis, dim=-1)           # [24,24]
        sim = torch.matmul(h_norm, b_norm.T)                # [B,T,K,24]
        max_sim = torch.max(sim, dim=-1)[0]                 # [B,T,K]
        return 1.0 - max_sim.mean()

# ==================== ДЕКОДЕР СНОВ (МОНИТОРИНГ) ====================
class DreamDecoder:
    def __init__(self, leech_basis, threshold=0.95):
        self.basis = leech_basis
        self.threshold = threshold

    def check(self, hidden_state):
        """
        hidden_state: [d_model] – последнее скрытое состояние.
        Возвращает (резонанс, статус).
        """
        h = hidden_state[:24].unsqueeze(0)          # [1,24] для простоты берём первые 24
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

# ==================== ПОЛНАЯ МОДЕЛЬ (интерфейс как у E8GPT) ====================
class LeechGPT(nn.Module):
    def __init__(self, cfg: LeechConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.blocks = nn.ModuleList([LeechBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Ядро Лича для потери и мониторинга
        leech_basis = generate_leech_kernel(24)
        self.register_buffer('leech_basis', leech_basis)
        self.resonance_loss_fn = LeechResonanceLoss(leech_basis)

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
        assert t <= self.cfg.block_size
        x = self.tok_emb(idx) + self.pos_emb[:, :t, :]
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss_ce = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1))
            if self.cfg.lambda_geo > 0:
                loss_geo = self.resonance_loss_fn(x)   # x = скрытые состояния после всех слоёв
                loss = loss_ce + self.cfg.lambda_geo * loss_geo
            else:
                loss = loss_ce
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, repetition_penalty=1.5, repetition_window=50):
        """Упрощённая генерация без резонатора."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Штраф за повтор
            if repetition_penalty != 1.0:
                past_tokens = idx[0, -repetition_window:].tolist()
                for t in set(past_tokens):
                    count = past_tokens.count(t)
                    logits[0, t] -= repetition_penalty * count

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==================== СОЗДАНИЕ МОДЕЛИ ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = LeechConfig(vocab_size=vocab_size)
model = LeechGPT(cfg).to(device)

print(f"💎 Модель Leech-GPT создана.")
print(f"📦 Параметров: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print(f"   Архитектура: d_model={cfg.d_model}, layers={cfg.n_layers}, heads={cfg.n_heads}")

# ======================== МОНТИРОВАНИЕ DRIVE И ЧЕКПОИНТЫ ========================
from google.colab import drive
drive.mount('/content/drive')

checkpoint_dir = '/content/drive/MyDrive/leech_tinystories_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

def save_checkpoint(step, model, optimizer, loss, is_best=False):
    filename = 'best_model.pt' if is_best else f'checkpoint_step_{step}.pt'
    path = os.path.join(checkpoint_dir, filename)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': cfg,
    }, path)
    print(f'💾 Чекпоинт сохранён: {path} (loss={loss:.4f})')

def load_latest_checkpoint(model, optimizer):
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_step_')]
    if not files:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'🔄 Загружен best_model.pt, шаг {checkpoint["step"]}, loss {checkpoint["loss"]:.4f}')
            return checkpoint['step']
        print('🆕 Чекпоинтов нет, начинаем с нуля.')
        return 0
    steps = [int(f.split('_')[-1].split('.')[0]) for f in files]
    latest_step = max(steps)
    latest_file = f'checkpoint_step_{latest_step}.pt'
    path = os.path.join(checkpoint_dir, latest_file)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'🔄 Загружен чекпоинт: шаг {latest_step}, loss {checkpoint["loss"]:.4f}')
    return latest_step

# ======================== ПОДГОТОВКА СТРИМОВ ========================
train_dataset = load_dataset("roneneldan/TinyStories", streaming=True, split="train")
train_iter = iter(train_dataset)
val_dataset = load_dataset("roneneldan/TinyStories", streaming=True, split="validation")
val_iter = iter(val_dataset)
print("🌊 Стримы TinyStories готовы (train + validation)")

# ======================== ГИПЕРПАРАМЕТРЫ ========================
batch_size = 4
block_size = cfg.block_size
learning_rate = 5e-5
total_steps = 130000
save_every = 1000
log_every = 200
gen_every = 1000

# ======================== ОПТИМИЗАТОР И ЗАГРУЗКА ========================
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
start_step = load_latest_checkpoint(model, optimizer)

# ======================== ЦИКЛ ОБУЧЕНИЯ ========================
print(f"\n🚀 Запуск обучения с шага {start_step} до {total_steps}")
model.train()

for _ in range(5): next(train_iter)
print("💎 Стрим прогрет.")

best_val_loss = float('inf')

for step in range(start_step + 1, total_steps + 1):
    xb, yb = get_batch_streaming(train_iter, batch_size, block_size, device)
    if xb is None:
        train_iter = iter(train_dataset)
        continue

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if step % log_every == 0:
        print(f"📊 Шаг {step:5d} | Train Loss: {loss.item():.4f}")

    if step % log_every == 0:
        model.eval()
        with torch.no_grad():
            xb_val, yb_val = get_batch_streaming(val_iter, batch_size, block_size, device)
            if xb_val is None:
                val_iter = iter(val_dataset)
                xb_val, yb_val = get_batch_streaming(val_iter, batch_size, block_size, device)
            if xb_val is not None:
                _, val_loss = model(xb_val, yb_val)
                print(f"         | Validation Loss: {val_loss.item():.4f}")
        model.train()

    if step % gen_every == 0:
        model.eval()
        with torch.no_grad():
            test_prompt = "who is Lily? "
            context = torch.tensor([encode(test_prompt)], dtype=torch.long, device=device)
            print(f"\n🎭 Генерация (шаг {step}): ", end='')
            for _ in range(50):
                idx_cond = context[:, -block_size:]
                logits_gen, _ = model(idx_cond)
                logits_gen = logits_gen[0, -1, :].clone()
                past_tokens = context[0, -20:].tolist()
                for t in set(past_tokens):
                    count = past_tokens.count(t)
                    logits_gen[t] -= (1.5 * count)
                last_token = context[0, -1].item()
                last_piece = sp.id_to_piece(last_token)
                if last_piece in ":,.!?;":
                    for p in ":,.!?;":
                        pid = sp.piece_to_id(p)
                        if pid != -1:
                            logits_gen[pid] -= 50.0
                probs = torch.softmax(logits_gen / 0.8, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                context = torch.cat((context, next_token.unsqueeze(0)), dim=1)
                piece = sp.id_to_piece(next_token.item())
                if piece == '<0x22>':
                    piece = '"'
                elif piece == '<0x0A>':
                    piece = '\n'
                elif piece.startswith('▁'):
                    piece = ' ' + piece[1:]
                elif piece.startswith('<0x') and piece.endswith('>'):
                    continue
                print(piece, end='', flush=True)
            print("\n" + "-"*60)
        model.train()

    if step % save_every == 0:
        save_checkpoint(step, model, optimizer, loss.item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(step, model, optimizer, val_loss.item(), is_best=True)

save_checkpoint(total_steps, model, optimizer, loss.item())
print("🏁 Обучение завершено!")

# ======================== СОХРАНЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ ========================
print("\n💾 Сохранение модели Leech-GPT...")
torch.save({
    'model_state_dict': model.state_dict(),
    'config': cfg,
    'leech_basis': model.leech_basis.cpu()
}, 'leech_model_final.pth')
print(f"✅ Модель сохранена в 'leech_model_final.pth'")

# ======================== ПРИМЕР ГЕНЕРАЦИИ ПОСЛЕ ОБУЧЕНИЯ ========================
# Загружаем лучший чекпоинт (если нужно)
# path_best = '/content/drive/MyDrive/leech_tinystories_checkpoints/best_model.pt'
# checkpoint = torch.load(path_best, map_location=device, weights_only=False)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.to(device)
# model.eval()

test_prompt = "Yes, I am a magic book. I can do many things by myself. But I have to be careful and quiet. Do not make any noise or stop me, Mia said. She took the book from Lily's hands and put it on her pages. Thank you, Mia. You are a good mom, Lily said. She smiled and hugged Mia. They looked at the pictures and tried to read them together."
context = torch.tensor([encode(test_prompt)], dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=150, temperature=0.8, top_k=50)
print(decode(generated[0].tolist()))

# ======================== ДЕКОДЕР СНОВ (мониторинг последнего состояния) ========================
dream_decoder = DreamDecoder(model.leech_basis, threshold=cfg.resonance_threshold)
with torch.no_grad():
    sample_batch, _ = get_batch_streaming(val_iter, 1, 128, device)
    if sample_batch is not None:
        _, _ = model(sample_batch)
        # берём скрытое состояние после последнего блока (оно не сохраняется, но можно модифицировать forward)
        # для демонстрации просто пропустим
        pass
# Если нужно получать hidden, придётся расширить forward. Оставляем как опцию.

from google.colab import files
files.download('leech_model_final.pth')