import argparse
import os
import sys
import time
import torch

# Добавляем корень проекта в path для импортов
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import LeechGPT
from tokenizer.tokenizer_utils import load_tokenizer
from inference.generate import generate
from config.config import LeechConfig

def main():
    parser = argparse.ArgumentParser(description="LILA-E8 inference: generate text from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--prompt", type=str, default="who is Lily?", help="Start prompt")
    parser.add_argument("--max_tokens", type=int, default=112, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature (0.1-2.0)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K sampling (0=disabled)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P (nucleus) sampling (0=disabled)")
    parser.add_argument("--repetition_penalty", type=float, default=1.4, help="Penalty for repeated tokens")
    parser.add_argument("--repetition_window", type=int, default=50, help="Window for repetition penalty")
    parser.add_argument("--use_resonator", action="store_true", help="Enable LeechResonanceBiasing logits bias")
    parser.add_argument("--resonator_alpha", type=float, default=0.1, help="Initial alpha for resonance biasing")
    parser.add_argument("--bench", action="store_true", help="Print tokens/sec for this generation run")
    parser.add_argument("--bench_warmup", type=int, default=1, help="Warmup runs before timing (default: 1)")
    parser.add_argument("--quiet", action="store_true", help="Disable streaming token printing (useful for benchmarking)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    sp = load_tokenizer()
    cfg = LeechConfig(vocab_size=sp.get_piece_size())
    model = LeechGPT(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if args.use_resonator:
        model.attach_resonator(alpha_init=args.resonator_alpha)

    if args.bench and args.bench_warmup > 0:
        for _ in range(args.bench_warmup):
            _ = generate(
                model, sp,
                start_str=args.prompt,
                max_tokens=min(16, args.max_tokens),
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else None,
                top_p=args.top_p if 0 < args.top_p < 1 else None,
                repetition_penalty=args.repetition_penalty,
                repetition_window=args.repetition_window,
                device=device,
                use_resonator=args.use_resonator,
                print_tokens=not args.quiet,
            )

    result = generate(
        model, sp,
        start_str=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p if 0 < args.top_p < 1 else None,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        device=device,
        use_resonator=args.use_resonator,
        return_stats=args.bench,
        print_tokens=not args.quiet,
    )
    # Текст уже выводится потоково внутри generate()
    if args.bench:
        _, stats = result
        toks = stats["generated_tokens"]
        sec = stats["seconds"]
        tps = (toks / sec) if sec > 0 else float("inf")
        print(f"[bench] generated_tokens={toks} seconds={sec:.4f} tok/s={tps:.2f}")

if __name__ == "__main__":
    main()