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

"""Генерация текста для LeechGPT (опционально с резонатором)."""

import re
import time
import torch
import torch.nn.functional as F


def generate(
    model,
    sp,
    start_str="who is Lily?",
    max_tokens=112,
    temperature=0.5,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    repetition_window=50,
    device=None,
    use_resonator=False,
    use_kv_cache=False,
    return_stats=False,
    print_tokens=True,
):
    """
    Генерация текста.
    use_resonator: если True — модель добавит резонансное смещение к логитам (если резонатор подключён).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    block_size = model.cfg.block_size
    model.eval()
    # Кодируем промпт
    prompt_ids = sp.encode(start_str)
    if not prompt_ids:
        prompt_ids = [sp.bos_id() if sp.bos_id() != -1 else sp.unk_id()]

    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # ID специальных токенов
    punct_ids = [pid for p in ':,.!?;' if (pid := sp.piece_to_id(p)) != -1]
    space_id = sp.piece_to_id('▁')
    if space_id == -1:
        space_id = sp.piece_to_id(' ')

    generated_tokens = []

    eos_id = sp.eos_id() if sp.eos_id() != -1 else sp.piece_to_id('<|endoftext|>')
    if eos_id == -1:
        eos_id = None

    # KV-cache работает только пока общая длина <= block_size
    use_kv_cache = bool(use_kv_cache) and (context.size(1) <= block_size)
    past_key_values = None

    t0 = time.perf_counter()
    with torch.no_grad():
        if use_kv_cache:
            # "префилл" кэша на всём промпте
            logits, _, _, past_key_values = model(context, use_resonator=use_resonator, use_cache=True, past_key_values=None)
            _ = logits  # logits от промпта сейчас не используем напрямую

        for _ in range(max_tokens):
            if use_kv_cache:
                # генерируем по 1 токену (fast path)
                idx_in = context[:, -1:]
                try:
                    logits, _, _, past_key_values = model(
                        idx_in,
                        use_resonator=use_resonator,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                except ValueError:
                    # превысили block_size -> откат на обычный режим
                    use_kv_cache = False
                    past_key_values = None
                    idx_cond = context[:, -block_size:]
                    logits, _, _, _ = model(idx_cond, use_resonator=use_resonator, use_cache=False, past_key_values=None)
                logits = logits[0, -1, :].clone()
            else:
                idx_cond = context[:, -block_size:]
                logits, _, _, _ = model(idx_cond, use_resonator=use_resonator, use_cache=False, past_key_values=None)
                logits = logits[0, -1, :].clone()

            # Штрафы за повторения
            past_tokens = context[0, -repetition_window:].tolist()
            for t_idx in set(past_tokens):
                count = past_tokens.count(t_idx)
                logits[t_idx] -= repetition_penalty * count

            if len(context[0]) > 0:
                last_token = context[0, -1].item()
                if last_token in punct_ids:
                    logits[last_token] -= 50.0

            greedy_decode = temperature is not None and temperature <= 0
            if greedy_decode:
                probs = None
            else:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)

            # Top-K
            if not greedy_decode and top_k is not None and top_k > 0:
                v, _ = torch.topk(probs, min(top_k, probs.size(-1)))
                probs[probs < v[-1]] = 0.0

            # Top-P
            if not greedy_decode and top_p is not None and 0.0 < top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = 0
                indices_to_remove = sorted_indices[mask]
                probs[indices_to_remove] = 0.0

            if greedy_decode:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs_sum = probs.sum()
                if probs_sum > 0:
                    probs /= probs_sum
                else:
                    probs = torch.zeros_like(probs)
                    probs[space_id if space_id != -1 else sp.unk_id()] = 1.0
                next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token.unsqueeze(0)), dim=1)
            token_id = next_token.item()

            # Проверка EOS до печати
            if eos_id is not None and token_id == eos_id:
                break

            generated_tokens.append(token_id)

            # Онлайн-вывод токенов (как в Colab-версии)
            piece = sp.id_to_piece(token_id)
            # Фильтрация служебных токенов
            if piece == '<pad>' or (piece.startswith('<0x') and piece.endswith('>')):
                continue
            if piece == '<0x22>':
                piece = '"'
            elif piece == '<0x0A>':
                piece = '\n'
            elif piece.startswith('▁'):
                piece = ' ' + piece[1:]
            if print_tokens:
                print(piece, end='', flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    dt = time.perf_counter() - t0

    if print_tokens:
        print()  # перевод строки после генерации
    full_text = sp.decode(generated_tokens)
    clean_text = re.split(r'<pad>|<0x[0-9A-Fa-f]+>', full_text)[0]
    if return_stats:
        return clean_text, {"generated_tokens": len(generated_tokens), "seconds": dt}
    return clean_text
