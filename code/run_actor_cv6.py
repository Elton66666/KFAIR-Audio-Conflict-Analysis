import copy
import gc
import json
import random
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from kimia_infer.api.kimia import KimiAudio


ROOT = Path("/root/autodl-tmp/kfair")
OUT = ROOT / "outputs"
RUN_SUMMARY = OUT / "actor_cv6_run_summary.jsonl"
RAW = OUT / "actor_cv6_test_predictions.jsonl"
BASELINE_RAW = OUT / "actor_cv6_baseline_predictions.jsonl"
CHECKPOINT_DIR = OUT / "actor_cv6_checkpoints"

LABELS = ["neutral", "happy", "sad", "angry"]
CHOICES = {"neutral": "A", "happy": "B", "sad": "C", "angry": "D"}
PAIRS = [(a, b) for i, a in enumerate(LABELS) for b in LABELS[i + 1 :]]
ACTOR_GROUPS = [list(range(start, start + 4)) for start in range(1, 25, 4)]
SEEDS = [20260903, 20260904, 20260905]
METHODS = [
    {"method": "decision_adapter_layer25", "kind": "adapter", "layer": 25, "scope": "decision_token"},
    {"method": "full_sequence_adapter_layer22", "kind": "adapter", "layer": 22, "scope": "full_sequence"},
    {"method": "ordinary_lora_layer22", "kind": "lora", "layer": 22, "scope": "q_proj"},
]


class Adapter(nn.Module):
    def __init__(self, width, rank=8):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(rank, width, bias=False, dtype=torch.float32)
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(F.gelu(self.down(x.float()))).to(x.dtype)


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=8):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32)
        self.B = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32)
        nn.init.normal_(self.A.weight, std=0.01)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.B(self.A(x.float())).to(x.dtype) * self.scale


def build_examples(manifest):
    rows = []
    for (actor, statement), group in manifest.groupby(["actor", "statement"]):
        files = {r.label: r.filename for r in group.itertuples()}
        for label_a, label_b in PAIRS:
            for cell, discrete_label, acoustic_label in (
                ("AA", label_a, label_a),
                ("AB", label_a, label_b),
                ("BA", label_b, label_a),
                ("BB", label_b, label_b),
            ):
                rows.append(
                    {
                        "actor": int(actor),
                        "statement": int(statement),
                        "pair": f"{label_a}_{label_b}",
                        "cell": cell,
                        "discrete_label": discrete_label,
                        "acoustic_label": acoustic_label,
                        "discrete_file": files[discrete_label],
                        "continuous_file": files[acoustic_label],
                        "conflict": discrete_label != acoustic_label,
                    }
                )
    assert len(rows) == 1152
    return rows


def example_key(example):
    return (example["actor"], example["statement"], example["pair"], example["cell"])


def metrics(rows):
    frame = pd.DataFrame(rows)
    frame["correct"] = frame.prediction.eq(frame.acoustic_label)
    frame["semantic"] = frame.prediction.eq(frame.discrete_label)
    conflict = frame[frame.conflict]
    aligned = frame[~frame.conflict]
    return {
        "n": int(len(frame)),
        "ser": float(frame.correct.mean()),
        "aligned_ser": float(aligned.correct.mean()),
        "conflict_acoustic_follow": float(conflict.correct.mean()),
        "conflict_semantic_follow": float(conflict.semantic.mean()),
        "conflict_margin": float(conflict.acoustic_margin.mean()),
    }


def completed_runs():
    if not RUN_SUMMARY.exists():
        return set()
    done = set()
    for line in RUN_SUMMARY.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        done.add((int(row["fold"]), row["method"], int(row["seed"])))
    return done


def append_jsonl(path, rows):
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    first = pd.read_csv(OUT / "ravdess_expanded_manifest.csv")
    second = pd.read_csv(OUT / "ravdess_statement2_manifest.csv")
    first["statement"], second["statement"] = 1, 2
    manifest = pd.concat([first, second], ignore_index=True)
    assert len(manifest) == 192
    examples = build_examples(manifest)

    model = KimiAudio(model_path=str(ROOT / "models/Kimi-Audio-7B-Instruct"), load_detokenizer=False)
    [module for module in model.alm.modules() if hasattr(module, "vq_adaptor")][0].kfair_branch_mode = "full"
    for parameter in model.alm.parameters():
        parameter.requires_grad_(False)

    prompt = (
        "Classify only the speaker's vocal emotion. Choose exactly one: "
        "A=neutral, B=happy, C=sad, D=angry. Reply with only A, B, C, or D."
    )
    token_ids = torch.tensor(
        [model.prompt_manager.text_tokenizer.encode(CHOICES[label], bos=False, eos=False)[0] for label in LABELS],
        device="cuda",
    )

    cache = {}
    for index, row in enumerate(manifest.itertuples(), 1):
        item = model.prompt_manager.get_prompt(
            [
                {"role": "user", "message_type": "text", "content": prompt},
                {"role": "user", "message_type": "audio", "content": row.audio_path},
            ],
            output_type="text",
        )
        audio, text, mask, _, _ = item.to_tensor()
        cache[row.filename] = (audio.cpu(), text.cpu(), mask.cpu(), item.continuous_feature[0].cpu())
        if index % 48 == 0:
            print(f"CACHE {index}/192", flush=True)

    def prepare(example):
        audio, text, mask, target_feature = cache[example["discrete_file"]]
        feature = cache[example["continuous_file"]][3]
        if feature.shape[1] != target_feature.shape[1]:
            feature = F.interpolate(
                feature.transpose(1, 2).float(),
                size=target_feature.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).to(feature.dtype)
        audio, text, mask, feature = audio.cuda(), text.cuda(), mask.cuda(), feature.cuda()
        position = torch.arange(audio.shape[1], device="cuda").unsqueeze(0).long()
        return audio, text, mask, feature, position

    def forward(example):
        audio, text, mask, feature, position = prepare(example)
        output = model.alm(
            input_ids=audio,
            text_input_ids=text,
            whisper_input_feature=[feature],
            is_continuous_mask=mask,
            position_ids=position,
            use_cache=False,
            return_dict=True,
        )
        return output.logits[1][0, -1].index_select(0, token_ids).float()

    baseline = {}
    with torch.inference_mode():
        for index, example in enumerate(examples, 1):
            baseline[example_key(example)] = forward(example).cpu()
            if index % 288 == 0:
                print(f"BASELINE {index}/1152", flush=True)

    def make_rows(items, adapted, fold, method, seed):
        rows = []
        with torch.inference_mode():
            for example in items:
                logits = forward(example) if adapted else baseline[example_key(example)].cuda()
                scores = {label: float(logits[i].cpu()) for i, label in enumerate(LABELS)}
                rows.append(
                    {
                        "fold": fold,
                        "method": method,
                        "seed": seed,
                        **example,
                        "prediction": max(scores, key=scores.get),
                        "acoustic_margin": scores[example["acoustic_label"]] - scores[example["discrete_label"]],
                    }
                )
        return rows

    if not BASELINE_RAW.exists():
        baseline_rows = []
        for fold_index, test_actors in enumerate(ACTOR_GROUPS, 1):
            test = [example for example in examples if example["actor"] in test_actors]
            baseline_rows.extend(make_rows(test, False, fold_index, "baseline", 0))
        append_jsonl(BASELINE_RAW, baseline_rows)
        print("BASELINE_CV_SAVED", flush=True)

    done = completed_runs()
    total_runs = len(ACTOR_GROUPS) * len(METHODS) * len(SEEDS)
    run_index = 0

    for fold_index, test_actors in enumerate(ACTOR_GROUPS, 1):
        validation_actors = ACTOR_GROUPS[fold_index % len(ACTOR_GROUPS)]
        train = [e for e in examples if e["actor"] not in test_actors and e["actor"] not in validation_actors]
        validation = [e for e in examples if e["actor"] in validation_actors]
        test = [e for e in examples if e["actor"] in test_actors]
        assert len(train) == 768 and len(validation) == 192 and len(test) == 192

        for method_config in METHODS:
            for seed in SEEDS:
                run_index += 1
                method = method_config["method"]
                run_key = (fold_index, method, seed)
                if run_key in done:
                    print(f"SKIP {run_index}/{total_runs} fold={fold_index} method={method} seed={seed}", flush=True)
                    continue

                torch.manual_seed(seed)
                random.seed(seed)
                random.shuffle(train)
                started = time.time()
                hook_handle = None
                original_q_proj = None

                if method_config["kind"] == "adapter":
                    module = Adapter(model.alm.config.hidden_size, rank=8).cuda()
                    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3, weight_decay=1e-4)

                    def adapter_hook(layer_module, args, active=module, scope=method_config["scope"]):
                        hidden = args[0].clone()
                        if scope == "full_sequence":
                            hidden = hidden + active(hidden)
                        else:
                            hidden[:, -1] = hidden[:, -1] + active(hidden[:, -1])
                        return (hidden,) + args[1:]

                    hook_handle = model.alm.model.layers[method_config["layer"]].register_forward_pre_hook(adapter_hook)
                    trainable = list(module.parameters())
                else:
                    attention = model.alm.model.layers[method_config["layer"]].self_attn
                    original_q_proj = attention.q_proj
                    module = LoRALinear(original_q_proj, rank=8, alpha=8).cuda()
                    attention.q_proj = module
                    trainable = [*module.A.parameters(), *module.B.parameters()]
                    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)

                best = None
                history = []
                try:
                    for epoch in range(1, 5):
                        random.shuffle(train)
                        total_loss = 0.0
                        for example in train:
                            logits = forward(example)
                            target = torch.tensor([LABELS.index(example["acoustic_label"])], device="cuda")
                            ser_loss = F.cross_entropy(logits.unsqueeze(0), target)
                            if method_config["kind"] == "adapter":
                                pair_loss = ser_loss if example["conflict"] else logits.new_zeros(())
                                preserve_loss = logits.new_zeros(())
                                if not example["conflict"]:
                                    preserve_loss = F.kl_div(
                                        F.log_softmax(logits, dim=-1),
                                        F.softmax(baseline[example_key(example)].cuda(), dim=-1),
                                        reduction="sum",
                                    )
                                loss = ser_loss + 0.5 * pair_loss + 0.2 * preserve_loss
                            else:
                                loss = ser_loss
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                            total_loss += float(loss.detach().cpu())

                        validation_rows = make_rows(validation, True, fold_index, method, seed)
                        validation_metrics = metrics(validation_rows)
                        history.append({"epoch": epoch, "loss": total_loss / len(train), **validation_metrics})
                        score = (
                            validation_metrics["ser"],
                            validation_metrics["conflict_acoustic_follow"],
                            validation_metrics["aligned_ser"],
                        )
                        if best is None or score > best["score"]:
                            if method_config["kind"] == "adapter":
                                state = copy.deepcopy(module.state_dict())
                            else:
                                state = {"A": copy.deepcopy(module.A.state_dict()), "B": copy.deepcopy(module.B.state_dict())}
                            best = {"score": score, "epoch": epoch, "state": state, "validation": validation_metrics}
                        print(
                            f"RUN {run_index}/{total_runs} fold={fold_index} method={method} seed={seed} "
                            f"epoch={epoch} loss={total_loss / len(train):.4f} val_ser={validation_metrics['ser']:.4f}",
                            flush=True,
                        )

                    if method_config["kind"] == "adapter":
                        module.load_state_dict(best["state"])
                    else:
                        module.A.load_state_dict(best["state"]["A"])
                        module.B.load_state_dict(best["state"]["B"])

                    test_rows = make_rows(test, True, fold_index, method, seed)
                    test_metrics = metrics(test_rows)
                    checkpoint_path = CHECKPOINT_DIR / f"fold{fold_index}_{method}_seed{seed}.pt"
                    torch.save(
                        {
                            "fold": fold_index,
                            "train_actors": sorted({e["actor"] for e in train}),
                            "validation_actors": validation_actors,
                            "test_actors": test_actors,
                            "method": method_config,
                            "seed": seed,
                            "parameters": sum(p.numel() for p in trainable),
                            "best_epoch": best["epoch"],
                            "state_dict": best["state"],
                            "validation": best["validation"],
                            "test": test_metrics,
                        },
                        checkpoint_path,
                    )
                    append_jsonl(RAW, test_rows)
                    record = {
                        "fold": fold_index,
                        "train_actors": sorted({e["actor"] for e in train}),
                        "validation_actors": validation_actors,
                        "test_actors": test_actors,
                        "method": method,
                        "kind": method_config["kind"],
                        "layer": method_config["layer"],
                        "scope": method_config["scope"],
                        "seed": seed,
                        "parameters": sum(p.numel() for p in trainable),
                        "best_epoch": best["epoch"],
                        "history": history,
                        "validation": best["validation"],
                        "test": test_metrics,
                        "seconds": time.time() - started,
                        "checkpoint": str(checkpoint_path),
                    }
                    append_jsonl(RUN_SUMMARY, [record])
                    print("RESULT " + json.dumps(record, ensure_ascii=False), flush=True)
                finally:
                    if hook_handle is not None:
                        hook_handle.remove()
                    if original_q_proj is not None:
                        model.alm.model.layers[method_config["layer"]].self_attn.q_proj = original_q_proj
                    del module, optimizer
                    gc.collect()
                    torch.cuda.empty_cache()

    print("CV6_DONE", flush=True)


if __name__ == "__main__":
    main()
