import torch
from data_utils.streaming_dataset import get_batch_streaming, create_train_val_iterators
from tokenizer.tokenizer_utils import load_tokenizer
from models.model import LeechGPT
from config.config import LeechConfig
from training.checkpoint import save_checkpoint, load_latest_checkpoint
from torch.optim.lr_scheduler import LinearLR
import os


def train(
    checkpoint_dir="checkpoints",
    resume=True,
    total_steps=100000,
    learning_rate=1e-5,
    weight_decay=0.1,
    warmup_steps=1000,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Загружаем токенайзер
    sp = load_tokenizer()  # или передать путь
    vocab_size = sp.get_piece_size()

    # Конфиг
    cfg = LeechConfig(vocab_size=vocab_size)
    model = LeechGPT(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)

    start_step = 0
    if resume:
        start_step = load_latest_checkpoint(model, optimizer, checkpoint_dir, device, scheduler=scheduler)

    train_iter, val_iter = create_train_val_iterators()

    # Гиперпараметры
    batch_size = 4
    block_size = cfg.block_size
    log_every = 200
    save_every = 1000
    gen_every = 1000

    model.train()
    best_val_loss = float("inf")
    for step in range(start_step + 1, total_steps + 1):
        xb, yb = get_batch_streaming(train_iter, batch_size, block_size, device, sp)
        if xb is None:
            train_iter, _ = create_train_val_iterators()  # пересоздаём
            continue

        _, _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % log_every == 0:
            print(f"Step {step}: loss {loss.item():.4f}")

            model.eval()
            with torch.no_grad():
                xb_val, yb_val = get_batch_streaming(val_iter, batch_size, block_size, device, sp)
                if xb_val is None:
                    _, val_iter = create_train_val_iterators()
                    xb_val, yb_val = get_batch_streaming(val_iter, batch_size, block_size, device, sp)
                if xb_val is not None:
                    _, _, val_loss = model(xb_val, yb_val)
                    print(f"         val_loss {val_loss.item():.4f}")
                    if val_loss.item() < best_val_loss:
                        best_val_loss = val_loss.item()
                        save_checkpoint(step, model, optimizer, best_val_loss, checkpoint_dir, scheduler=scheduler, is_best=True)
            model.train()

        if step % save_every == 0:
            save_checkpoint(step, model, optimizer, loss.item(), checkpoint_dir, scheduler=scheduler)

    save_checkpoint(total_steps, model, optimizer, loss.item(), checkpoint_dir, scheduler=scheduler)