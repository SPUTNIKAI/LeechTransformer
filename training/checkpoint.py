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

import os
import torch


def save_checkpoint(step, model, optimizer, loss, checkpoint_dir, scheduler=None, is_best=False):
    """Сохраняет чекпоинт в указанную директорию."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = 'best_model.pt' if is_best else f'checkpoint_step_{step}.pt'
    path = os.path.join(checkpoint_dir, filename)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'loss': loss,
        'config': model.cfg,
    }, path)
    print(f'💾 Чекпоинт сохранён: {path} (loss={loss:.4f})')


def load_latest_checkpoint(model, optimizer, checkpoint_dir, device, scheduler=None):
    """Загружает последний чекпоинт или best_model.pt. Возвращает шаг с которого продолжить."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_step_')]
    if not files:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
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
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f'🔄 Загружен чекпоинт: шаг {latest_step}, loss {checkpoint["loss"]:.4f}')
    return latest_step
