"""Generate deterministic predictions for dialogue smoke/regression sets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from nanochat.chat_format import encode_chat_prompt
from nanochat.dialogue_eval import validate_examples


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("base", "sft", "rl"), required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-type", choices=("cuda",), default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.temperature != 0:
        parser.error("dialogue comparison requires deterministic temperature=0")
    return args


def validate_eval_set(examples: list[dict]) -> None:
    validate_examples(examples)


def main() -> None:
    args = parse_args()
    examples = read_jsonl(args.eval_set)
    validate_eval_set(examples)
    expected_ids = {row["id"] for row in examples}

    completed = {}
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"{args.output} exists; pass --resume to continue")
        completed_rows = read_jsonl(args.output)
        completed = {row["id"]: row for row in completed_rows}
        if len(completed) != len(completed_rows):
            raise ValueError("existing prediction IDs must be unique")
        if not completed.keys() <= expected_ids:
            raise ValueError("existing output contains IDs outside the evaluation set")
        expected_model = (args.source, args.model_tag, args.step)
        for row in completed_rows:
            actual_model = (row.get("source"), row.get("model_tag"), row.get("step"))
            if actual_model != expected_model:
                raise ValueError(
                    f"existing output model {actual_model} != requested {expected_model}"
                )

    # Keep heavyweight imports after all data-safety validation.
    from nanochat.checkpoint_manager import load_model
    from nanochat.common import compute_cleanup, compute_init
    from nanochat.engine import Engine

    _, _, _, _, device = compute_init(args.device_type)
    model, tokenizer, _ = load_model(
        args.source,
        device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )
    engine = Engine(model, tokenizer)
    if args.max_new_tokens >= model.config.sequence_len:
        raise ValueError("max new tokens must be smaller than model context length")
    prompt_budget = model.config.sequence_len - args.max_new_tokens

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if completed else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        for index, example in enumerate(examples, start=1):
            if example["id"] in completed:
                continue
            prompt = encode_chat_prompt(tokenizer, example["messages"], prompt_budget)
            started = time.perf_counter()
            results, _ = engine.generate_batch(
                prompt,
                num_samples=1,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=args.seed,
            )
            completion = results[0][len(prompt):]
            row = {
                "id": example["id"],
                "prediction": tokenizer.decode(completion).strip(),
                "source": args.source,
                "model_tag": args.model_tag,
                "step": args.step,
                "prompt_tokens": len(prompt),
                "completion_tokens": len(completion),
                "latency_seconds": round(time.perf_counter() - started, 4),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(examples)}] {example['id']}", flush=True)
    compute_cleanup()


if __name__ == "__main__":
    main()
