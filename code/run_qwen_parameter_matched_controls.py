import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import binomtest

from run_qwen_behavioral_replication import CHOICES, LABELS, OUT, load_manifest, load_model, chat_text, make_inputs

SEEDS = [20260903, 20260904, 20260905]
METHODS = [
    {"method": "full_sequence_adapter_layer22", "kind": "adapter", "layer": 22},
    {"method": "ordinary_lora_layer22", "kind": "lora", "layer": 22},
]
RUNS = OUT / "qwen_parameter_matched_run_summary.jsonl"
RAW = OUT / "qwen_parameter_matched_predictions.jsonl"
SUMMARY = OUT / "qwen_parameter_matched_summary.csv"
EFFECTS = OUT / "qwen_parameter_matched_paired_effects.csv"
CHECKPOINTS = OUT / "qwen_parameter_matched_checkpoints"


class Adapter(nn.Module):
    def __init__(self, width, rank=8):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(rank, width, bias=False, dtype=torch.float32)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden):
        return self.up(F.gelu(self.down(hidden.float()))).to(hidden.dtype)


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=8):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32)
        self.B = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32)
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, hidden):
        return self.base(hidden) + self.B(self.A(hidden.float())).to(hidden.dtype) * self.scale


def append(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def metrics(rows):
    frame = pd.DataFrame(rows)
    correct = frame.prediction.eq(frame.acoustic_label)
    semantic = frame.prediction.eq(frame.semantic_label)
    conflict = frame[frame.conflict]
    aligned = frame[~frame.conflict]
    return {
        "n": int(len(frame)),
        "acoustic_accuracy": float(correct.mean()),
        "semantic_follow": float(semantic.mean()),
        "mean_acoustic_margin": float(frame.acoustic_margin.mean()),
        "aligned_accuracy": float(aligned.prediction.eq(aligned.acoustic_label).mean()) if len(aligned) else None,
        "conflict_acoustic_follow": float(conflict.prediction.eq(conflict.acoustic_label).mean()) if len(conflict) else None,
        "conflict_semantic_follow": float(conflict.prediction.eq(conflict.semantic_label).mean()) if len(conflict) else None,
    }


def analyze():
    runs = [json.loads(line) for line in RUNS.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for run in runs:
        for split in ("test", "emis_zero_shot"):
            rows.append({"method": run["method"], "seed": run["seed"], "split": split, **run[split]})
    frame = pd.DataFrame(rows)
    summary = []
    for (method, split), group in frame.groupby(["method", "split"]):
        summary.append({
            "method": method,
            "split": split,
            "seeds": len(group),
            "accuracy_mean": group.acoustic_accuracy.mean(),
            "accuracy_std": group.acoustic_accuracy.std(ddof=1),
            "margin_mean": group.mean_acoustic_margin.mean(),
            "margin_std": group.mean_acoustic_margin.std(ddof=1),
            "aligned_accuracy_mean": group.aligned_accuracy.mean(),
            "conflict_acoustic_follow_mean": group.conflict_acoustic_follow.mean(),
            "conflict_semantic_follow_mean": group.conflict_semantic_follow.mean(),
        })
    pd.DataFrame(summary).to_csv(SUMMARY, index=False)

    base = pd.read_json(OUT / "qwen_behavioral_predictions.jsonl", lines=True).drop_duplicates(["dataset", "filename"], keep="last")
    adapted = pd.read_json(RAW, lines=True).drop_duplicates(["method", "seed", "split", "filename"], keep="last")
    effects = []
    for (method, seed, split), group in adapted.groupby(["method", "seed", "split"]):
        if split == "test":
            reference = base[(base.dataset == "RAVDESS") & (pd.to_numeric(base.actor, errors="coerce") >= 21)]
        else:
            reference = base[base.dataset == "EMIS"]
        joined = reference[["filename", "prediction", "acoustic_label"]].merge(
            group[["filename", "prediction", "acoustic_label"]], on=["filename", "acoustic_label"], suffixes=("_base", "_adapted"), validate="one_to_one"
        )
        base_ok = joined.prediction_base.eq(joined.acoustic_label)
        adapted_ok = joined.prediction_adapted.eq(joined.acoustic_label)
        improved = int((~base_ok & adapted_ok).sum())
        harmed = int((base_ok & ~adapted_ok).sum())
        discordant = improved + harmed
        effects.append({
            "method": method,
            "seed": seed,
            "split": split,
            "delta_accuracy": float(adapted_ok.mean() - base_ok.mean()),
            "improved": improved,
            "harmed": harmed,
            "mcnemar_exact_p": float(binomtest(min(improved, harmed), discordant, 0.5).pvalue) if discordant else 1.0,
        })
    pd.DataFrame(effects).to_csv(EFFECTS, index=False)
    print(pd.read_csv(SUMMARY).to_string(index=False), flush=True)
    print(pd.read_csv(EFFECTS).to_string(index=False), flush=True)


def main():
    CHECKPOINTS.mkdir(exist_ok=True)
    manifest = load_manifest()
    ravdess = manifest[manifest.dataset.eq("RAVDESS")]
    train = ravdess[ravdess.actor.le(16)].to_dict("records")
    validation = ravdess[ravdess.actor.between(17, 20)].to_dict("records")
    test = ravdess[ravdess.actor.ge(21)].to_dict("records")
    emis = manifest[manifest.dataset.eq("EMIS")].to_dict("records")
    assert (len(train), len(validation), len(test), len(emis)) == (128, 32, 32, 192)
    completed = set()
    if RUNS.exists():
        completed = {(row["method"], int(row["seed"])) for row in map(json.loads, RUNS.read_text(encoding="utf-8").splitlines())}

    model, processor = load_model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    template = chat_text(processor)
    token_ids = torch.tensor([processor.tokenizer.encode(CHOICES[label], add_special_tokens=False)[0] for label in LABELS], device="cuda")

    def input_logits(row, gradient=False):
        inputs = make_inputs(processor, template, row["audio_path"])
        context = torch.enable_grad() if gradient else torch.inference_mode()
        with context:
            output = model.thinker(**inputs, use_cache=False, return_dict=True)
            return output.logits[0, -1].float().index_select(0, token_ids)

    for config in METHODS:
        for seed in SEEDS:
            key = (config["method"], seed)
            if key in completed:
                print(f"SKIP {key}", flush=True)
                continue
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            layer = model.thinker.model.layers[config["layer"]]
            handle = None
            original = None
            if config["kind"] == "adapter":
                module = Adapter(model.config.thinker_config.text_config.hidden_size).cuda()
                trainable = list(module.parameters())

                def hook(_module, args, kwargs):
                    if args:
                        hidden = args[0]
                        return (hidden + module(hidden),) + args[1:], kwargs
                    hidden = kwargs["hidden_states"]
                    kwargs["hidden_states"] = hidden + module(hidden)
                    return args, kwargs
                handle = layer.register_forward_pre_hook(hook, with_kwargs=True)
            else:
                original = layer.self_attn.q_proj
                module = LoRALinear(original).cuda()
                layer.self_attn.q_proj = module
                trainable = list(module.A.parameters()) + list(module.B.parameters())
            optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)

            def evaluate(items, split):
                module.eval(); output = []
                with torch.inference_mode():
                    for index, row in enumerate(items, 1):
                        started = time.time()
                        scores_tensor = torch.log_softmax(input_logits(row), dim=-1)
                        scores = {label: float(scores_tensor[i].cpu()) for i, label in enumerate(LABELS)}
                        output.append({**row, "method": config["method"], "seed": seed, "split": split, "prediction": max(scores, key=scores.get), "scores": scores, "acoustic_margin": scores[row["acoustic_label"]] - scores[row["semantic_label"]], "seconds": time.time() - started})
                        if index % 64 == 0: print(f"EVAL {config['method']} {seed} {split} {index}/{len(items)}", flush=True)
                return output

            history = []; best = None; best_state = None; started = time.time()
            for epoch in range(1, 5):
                module.train(); order = list(train); random.Random(seed + epoch).shuffle(order); total = 0.0
                for index, row in enumerate(order, 1):
                    logits = input_logits(row, gradient=True)
                    target = torch.tensor([LABELS.index(row["acoustic_label"])], device="cuda")
                    loss = F.cross_entropy(logits.unsqueeze(0), target)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    total += float(loss.detach().cpu())
                validation_rows = evaluate(validation, "validation")
                score_row = metrics(validation_rows); history.append({"epoch": epoch, "loss": total / len(order), **score_row})
                score = (score_row["acoustic_accuracy"], score_row["mean_acoustic_margin"])
                if best is None or score > best:
                    best = score; best_state = copy.deepcopy(module.state_dict() if config["kind"] == "adapter" else {"A": module.A.state_dict(), "B": module.B.state_dict()})
                print(f"EPOCH {config['method']} {seed} {epoch} {json.dumps(history[-1])}", flush=True)
            if config["kind"] == "adapter":
                module.load_state_dict(best_state)
            else:
                module.A.load_state_dict(best_state["A"]); module.B.load_state_dict(best_state["B"])
            test_rows = evaluate(test, "test"); emis_rows = evaluate(emis, "emis_zero_shot")
            for row in test_rows + emis_rows: append(RAW, row)
            checkpoint = CHECKPOINTS / f"{config['method']}_rank8_seed{seed}.pt"
            torch.save({"method": config, "seed": seed, "rank": 8, "parameters": sum(p.numel() for p in trainable), "state_dict": best_state}, checkpoint)
            record = {"method": config["method"], "kind": config["kind"], "layer": config["layer"], "seed": seed, "parameters": sum(p.numel() for p in trainable), "best_epoch": max(range(len(history)), key=lambda i: (history[i]["acoustic_accuracy"], history[i]["mean_acoustic_margin"])) + 1, "history": history, "test": metrics(test_rows), "emis_zero_shot": metrics(emis_rows), "seconds": time.time() - started, "checkpoint": str(checkpoint)}
            append(RUNS, record); print("RESULT " + json.dumps(record), flush=True)
            if handle is not None: handle.remove()
            if original is not None: layer.self_attn.q_proj = original
            del module, optimizer; torch.cuda.empty_cache()
    analyze()
    print("QWEN_PARAMETER_CONTROLS_DONE", flush=True)


if __name__ == "__main__":
    main()
