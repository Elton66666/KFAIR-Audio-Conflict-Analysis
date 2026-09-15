import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_qwen_behavioral_replication import (
    CHOICES,
    LABELS,
    MODEL_PATH,
    OUT,
    PROMPT,
    chat_text,
    load_manifest,
    load_model,
    make_inputs,
)


RAW = OUT / "qwen_adapter_predictions.jsonl"
RUNS = OUT / "qwen_adapter_run_summary.jsonl"
SUMMARY = OUT / "qwen_adapter_summary.json"
CHECKPOINTS = OUT / "qwen_adapter_checkpoints"


class ResidualAdapter(nn.Module):
    def __init__(self, hidden_size, rank):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden):
        return self.up(F.gelu(self.down(hidden)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260903, 20260904, 20260905])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--layer", type=int, default=25)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--limit-train", type=int, default=0)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_rows():
    frame = load_manifest()
    ravdess = frame[frame.dataset.eq("RAVDESS")].copy()
    train = ravdess[ravdess.actor.le(16)].to_dict("records")
    validation = ravdess[ravdess.actor.between(17, 20)].to_dict("records")
    test = ravdess[ravdess.actor.ge(21)].to_dict("records")
    emis = frame[frame.dataset.eq("EMIS")].to_dict("records")
    assert (len(train), len(validation), len(test), len(emis)) == (128, 32, 32, 192)
    return train, validation, test, emis


def metrics(rows):
    frame = pd.DataFrame(rows)
    frame["acoustic_correct"] = frame.prediction.eq(frame.acoustic_label)
    frame["semantic_follow"] = frame.prediction.eq(frame.semantic_label)
    result = {
        "n": int(len(frame)),
        "acoustic_accuracy": float(frame.acoustic_correct.mean()),
        "mean_acoustic_margin": float(frame.acoustic_margin.mean()),
    }
    conflicts = frame[frame.conflict]
    aligned = frame[~frame.conflict]
    result.update(
        {
            "aligned_accuracy": float(aligned.acoustic_correct.mean()) if len(aligned) else None,
            "conflict_acoustic_follow": float(conflicts.acoustic_correct.mean()) if len(conflicts) else None,
            "conflict_semantic_follow": float(conflicts.semantic_follow.mean()) if len(conflicts) else None,
        }
    )
    return result


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    CHECKPOINTS.mkdir(exist_ok=True)
    train, validation, test, emis = split_rows()
    if args.limit_train:
        train = train[: args.limit_train]
        validation = validation[: min(8, len(validation))]
        test = test[: min(8, len(test))]
        emis = emis[: min(8, len(emis))]
    completed = set()
    if RUNS.exists():
        completed = {int(row["seed"]) for row in map(json.loads, RUNS.read_text(encoding="utf-8").splitlines())}
    model, processor = load_model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    template = chat_text(processor)
    token_ids = torch.tensor(
        [processor.tokenizer.encode(CHOICES[label], add_special_tokens=False)[0] for label in LABELS],
        device="cuda",
    )
    layer = model.thinker.model.layers[args.layer]

    for seed in args.seeds:
        if seed in completed and not args.limit_train:
            print(f"SKIP completed seed={seed}", flush=True)
            continue
        set_seed(seed)
        adapter = ResidualAdapter(model.config.thinker_config.text_config.hidden_size, args.rank).cuda()

        def hook(module, hook_args, hook_kwargs):
            if hook_args:
                hidden = hook_args[0]
                patched = hidden.clone()
                delta = adapter(hidden[:, -1].float()).to(hidden.dtype)
                patched[:, -1] = patched[:, -1] + delta
                return (patched,) + hook_args[1:], hook_kwargs
            hidden = hook_kwargs["hidden_states"]
            patched = hidden.clone()
            delta = adapter(hidden[:, -1].float()).to(hidden.dtype)
            patched[:, -1] = patched[:, -1] + delta
            hook_kwargs["hidden_states"] = patched
            return hook_args, hook_kwargs

        handle = layer.register_forward_pre_hook(hook, with_kwargs=True)

        def logits_for(row, gradient=False):
            inputs = make_inputs(processor, template, row["audio_path"])
            context = torch.enable_grad() if gradient else torch.inference_mode()
            with context:
                output = model.thinker(**inputs, use_cache=False, return_dict=True)
                return output.logits[0, -1].float().index_select(0, token_ids)

        def evaluate(rows, split):
            adapter.eval()
            predictions = []
            for index, row in enumerate(rows, 1):
                started = time.time()
                scores_tensor = torch.log_softmax(logits_for(row), dim=-1)
                scores = {label: float(scores_tensor[i].cpu()) for i, label in enumerate(LABELS)}
                predictions.append(
                    {
                        **row,
                        "seed": seed,
                        "split": split,
                        "adapted": True,
                        "prediction": max(scores, key=scores.get),
                        "scores": scores,
                        "acoustic_margin": scores[row["acoustic_label"]] - scores[row["semantic_label"]],
                        "seconds": time.time() - started,
                    }
                )
                if index % 32 == 0:
                    print(f"EVAL seed={seed} split={split} {index}/{len(rows)}", flush=True)
            return predictions

        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
        best = None
        best_state = None
        history = []
        started = time.time()
        for epoch in range(1, args.epochs + 1):
            adapter.train()
            order = list(train)
            random.Random(seed + epoch).shuffle(order)
            total = 0.0
            for index, row in enumerate(order, 1):
                logits = logits_for(row, gradient=True)
                target = torch.tensor([LABELS.index(row["acoustic_label"])], device="cuda")
                loss = F.cross_entropy(logits.unsqueeze(0), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total += float(loss.detach().cpu())
                if index % 32 == 0:
                    print(f"TRAIN seed={seed} epoch={epoch} {index}/{len(order)} loss={total/index:.4f}", flush=True)
            validation_rows = evaluate(validation, "validation")
            validation_metrics = metrics(validation_rows)
            history.append({"epoch": epoch, "loss": total / len(order), **validation_metrics})
            score = (
                validation_metrics["acoustic_accuracy"],
                validation_metrics["mean_acoustic_margin"],
            )
            print(f"EPOCH seed={seed} epoch={epoch} {json.dumps(history[-1])}", flush=True)
            if best is None or score > best:
                best = score
                best_state = {key: value.detach().cpu().clone() for key, value in adapter.state_dict().items()}
        adapter.load_state_dict(best_state)
        test_rows = evaluate(test, "test")
        emis_rows = evaluate(emis, "emis_zero_shot")
        for row in test_rows + emis_rows:
            append_jsonl(RAW, row)
        checkpoint = CHECKPOINTS / f"qwen_decision_adapter_layer{args.layer}_rank{args.rank}_seed{seed}.pt"
        torch.save(
            {
                "state_dict": best_state,
                "seed": seed,
                "layer": args.layer,
                "rank": args.rank,
                "parameters": sum(parameter.numel() for parameter in adapter.parameters()),
                "model": str(MODEL_PATH),
                "prompt": PROMPT,
            },
            checkpoint,
        )
        run = {
            "seed": seed,
            "layer": args.layer,
            "rank": args.rank,
            "parameters": sum(parameter.numel() for parameter in adapter.parameters()),
            "best_epoch": int(max(range(len(history)), key=lambda i: (history[i]["acoustic_accuracy"], history[i]["mean_acoustic_margin"])) + 1),
            "history": history,
            "test": metrics(test_rows),
            "emis_zero_shot": metrics(emis_rows),
            "seconds": time.time() - started,
            "checkpoint": str(checkpoint),
        }
        append_jsonl(RUNS, run)
        print("RESULT " + json.dumps(run), flush=True)
        handle.remove()
        del adapter
        torch.cuda.empty_cache()

    if not args.limit_train and RUNS.exists():
        runs = list(map(json.loads, RUNS.read_text(encoding="utf-8").splitlines()))
        unique = {int(row["seed"]): row for row in runs}
        runs = [unique[seed] for seed in sorted(unique)]
        summary = {
            "model": "Qwen/Qwen2.5-Omni-7B",
            "revision": "ae9e1690543ffd5c0221dc27f79834d0294cba00",
            "architecture_note": "Cross-architecture behavioral and low-rank decision-layer replication; not a native discrete/continuous stream swap.",
            "train_split": "RAVDESS actors 1-16",
            "validation_split": "RAVDESS actors 17-20",
            "test_split": "RAVDESS actors 21-24",
            "external_test": "EMIS, zero-shot after RAVDESS-only training",
            "runs": runs,
        }
        SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
