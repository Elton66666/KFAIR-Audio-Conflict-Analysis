import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from kimia_infer.api.kimia import KimiAudio

ROOT = Path("/root/autodl-tmp/kfair")
OUT = ROOT / "outputs"
RAW = OUT / "expanded_asr_preservation_predictions.jsonl"
RUNS = OUT / "expanded_asr_preservation_run_summary.jsonl"
SUMMARY = OUT / "expanded_asr_preservation_summary.csv"
CHECKPOINTS = OUT / "actor_cv6_checkpoints"
SEEDS = [20260903, 20260904, 20260905]
METHODS = ["baseline", "decision_adapter_layer25", "full_sequence_adapter_layer22", "ordinary_lora_layer22"]


class Adapter(nn.Module):
    def __init__(self, width, rank=8):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(rank, width, bias=False, dtype=torch.float32)

    def forward(self, hidden):
        return self.up(F.gelu(self.down(hidden.float()))).to(hidden.dtype)


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=8):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32)
        self.B = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32)

    def forward(self, hidden):
        return self.base(hidden) + self.B(self.A(hidden.float())).to(hidden.dtype) * self.scale


def normalize(text):
    return re.findall(r"[a-z]+", text.lower())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def append(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def build_manifests():
    first = pd.read_csv(OUT / "ravdess_expanded_manifest.csv"); first["statement"] = 1
    second = pd.read_csv(OUT / "ravdess_statement2_manifest.csv"); second["statement"] = 2
    ravdess = pd.concat([first, second], ignore_index=True).sort_values(["actor", "statement", "label"])
    ravdess["corpus"] = "RAVDESS"
    ravdess["reference"] = ravdess.statement.map({1: "kids are talking by the door", 2: "dogs are sitting by the door"})

    records = []
    for path in sorted((ROOT / "data/tess/audio").glob("*.wav")):
        parts = path.stem.split("_")
        speaker, label = parts[0], parts[-1]
        word = " ".join(parts[1:-1])
        if speaker in {"YAF", "OAF"} and label in {"neutral", "happy", "sad", "angry"}:
            records.append({"filename": path.name, "audio_path": str(path), "speaker": speaker, "word": word, "label": label})
    tess_all = pd.DataFrame(records)
    common = sorted(set(tess_all[tess_all.speaker == "YAF"].word) & set(tess_all[tess_all.speaker == "OAF"].word))[:96]
    labels = ["neutral", "happy", "sad", "angry"]
    chosen = []
    for index, word in enumerate(common):
        for speaker_index, speaker in enumerate(["YAF", "OAF"]):
            label = labels[(index + speaker_index) % len(labels)]
            row = tess_all[(tess_all.word == word) & (tess_all.speaker == speaker) & (tess_all.label == label)].iloc[0].to_dict()
            chosen.append(row)
    tess = pd.DataFrame(chosen)
    tess["corpus"] = "TESS"
    # TESS recordings contain the carrier sentence "Say the word <target>",
    # not an isolated target word.  Score the complete spoken content.
    tess["reference"] = "say the word " + tess.word
    assert len(ravdess) == 192 and len(tess) == 192 and tess.word.nunique() == 96
    return ravdess.to_dict("records"), tess.to_dict("records")


def summarize_rows(rows):
    edits = sum(row["word_edits"] for row in rows)
    words = sum(row["reference_words"] for row in rows)
    char_edits = sum(row["char_edits"] for row in rows)
    chars = sum(row["reference_chars"] for row in rows)
    return {
        "n": len(rows),
        "micro_wer": edits / words,
        "micro_cer": char_edits / chars,
        "exact_match": sum(row["exact_match"] for row in rows) / len(rows),
        "mean_seconds": sum(row["seconds"] for row in rows) / len(rows),
    }


def main():
    ravdess, tess = build_manifests()
    completed = set()
    if RUNS.exists():
        completed = {(row["method"], int(row["seed"]), row["corpus"]) for row in map(json.loads, RUNS.read_text(encoding="utf-8").splitlines())}
    model = KimiAudio(model_path=str(ROOT / "models/Kimi-Audio-7B-Instruct"), load_detokenizer=False)
    for parameter in model.alm.parameters(): parameter.requires_grad_(False)
    prompt = "Transcribe the spoken words exactly. Reply with only the transcript."
    adapter = Adapter(model.alm.config.hidden_size).cuda()
    attention = model.alm.model.layers[22].self_attn
    original_q_proj = attention.q_proj

    def checkpoint_path(fold, method, seed):
        return CHECKPOINTS / f"fold{fold}_{method}_seed{seed}.pt"

    def run_group(method, seed, corpus, rows, fold):
        key = (method, seed, corpus)
        if key in completed:
            print(f"SKIP {key}", flush=True); return
        handle = None; lora = None
        if method != "baseline":
            checkpoint = torch.load(checkpoint_path(fold, method, seed), map_location="cpu", weights_only=True)
            if method.startswith("decision"):
                adapter.load_state_dict(checkpoint["state_dict"]); adapter.eval()
                def hook(_module, args):
                    hidden = args[0].clone(); hidden[:, -1] = hidden[:, -1] + adapter(hidden[:, -1]); return (hidden,) + args[1:]
                handle = model.alm.model.layers[25].register_forward_pre_hook(hook)
            elif method.startswith("full"):
                adapter.load_state_dict(checkpoint["state_dict"]); adapter.eval()
                def hook(_module, args):
                    hidden = args[0]; return (hidden + adapter(hidden),) + args[1:]
                handle = model.alm.model.layers[22].register_forward_pre_hook(hook)
            else:
                lora = LoRALinear(original_q_proj).cuda()
                lora.A.load_state_dict(checkpoint["state_dict"]["A"]); lora.B.load_state_dict(checkpoint["state_dict"]["B"]); lora.eval()
                attention.q_proj = lora
        predictions = []; started_run = time.time()
        for index, row in enumerate(rows, 1):
            messages = [{"role": "user", "message_type": "text", "content": prompt}, {"role": "user", "message_type": "audio", "content": row["audio_path"]}]
            started = time.time()
            _, hypothesis = model.generate(messages, output_type="text", max_new_tokens=32, text_temperature=0.0, text_top_k=5)
            reference_tokens = normalize(row["reference"]); hypothesis_tokens = normalize(hypothesis)
            reference_chars = list("".join(reference_tokens)); hypothesis_chars = list("".join(hypothesis_tokens))
            result = {**row, "method": method, "seed": seed, "fold_checkpoint": fold if method != "baseline" else 0, "transcript": hypothesis, "word_edits": edit_distance(reference_tokens, hypothesis_tokens), "reference_words": len(reference_tokens), "char_edits": edit_distance(reference_chars, hypothesis_chars), "reference_chars": len(reference_chars), "exact_match": reference_tokens == hypothesis_tokens, "seconds": time.time() - started}
            append(RAW, result); predictions.append(result)
            if index % 32 == 0: print(f"ASR {method} {seed} {corpus} {index}/{len(rows)}", flush=True)
        summary = {"method": method, "seed": seed, "corpus": corpus, "fold_checkpoint": fold if method != "baseline" else 0, "seconds": time.time() - started_run, **summarize_rows(predictions)}
        append(RUNS, summary); print("RESULT " + json.dumps(summary), flush=True)
        if handle is not None: handle.remove()
        if lora is not None: attention.q_proj = original_q_proj; del lora
        torch.cuda.empty_cache()

    run_group("baseline", 0, "RAVDESS", ravdess, 0)
    run_group("baseline", 0, "TESS", tess, 0)
    for method in METHODS[1:]:
        for seed in SEEDS:
            for fold in range(1, 7):
                subset = [row for row in ravdess if (int(row["actor"]) - 1) // 4 + 1 == fold]
                run_group(method, seed, f"RAVDESS_FOLD{fold}", subset, fold)
            run_group(method, seed, "TESS", tess, 6)

    runs = pd.DataFrame(map(json.loads, RUNS.read_text(encoding="utf-8").splitlines()))
    rows = []
    baseline_ravdess = runs[(runs.method == "baseline") & (runs.corpus == "RAVDESS")].iloc[0]
    rows.append({"method": "baseline", "corpus": "RAVDESS", "seeds": 1, **{name: baseline_ravdess[name] for name in ["n", "micro_wer", "micro_cer", "exact_match"]}})
    baseline_tess = runs[(runs.method == "baseline") & (runs.corpus == "TESS")].iloc[0]
    rows.append({"method": "baseline", "corpus": "TESS", "seeds": 1, **{name: baseline_tess[name] for name in ["n", "micro_wer", "micro_cer", "exact_match"]}})
    for method in METHODS[1:]:
        method_runs = runs[runs.method == method]
        fold_rows = method_runs[method_runs.corpus.str.startswith("RAVDESS_FOLD")]
        for seed, group in fold_rows.groupby("seed"):
            weights = group.n.to_numpy()
            rows.append({"method": method, "corpus": "RAVDESS", "seeds": seed, "n": int(weights.sum()), "micro_wer": float(np.average(group.micro_wer, weights=weights)), "micro_cer": float(np.average(group.micro_cer, weights=weights)), "exact_match": float(np.average(group.exact_match, weights=weights))})
        for _, row in method_runs[method_runs.corpus == "TESS"].iterrows():
            rows.append({"method": method, "corpus": "TESS", "seeds": row.seed, "n": row.n, "micro_wer": row.micro_wer, "micro_cer": row.micro_cer, "exact_match": row.exact_match})
    detailed = pd.DataFrame(rows)
    aggregate = []
    for (method, corpus), group in detailed.groupby(["method", "corpus"]):
        aggregate.append({"method": method, "corpus": corpus, "runs": len(group), "n_per_run": int(group.n.iloc[0]), "micro_wer_mean": group.micro_wer.mean(), "micro_wer_std": group.micro_wer.std(ddof=1) if len(group) > 1 else 0.0, "micro_cer_mean": group.micro_cer.mean(), "micro_cer_std": group.micro_cer.std(ddof=1) if len(group) > 1 else 0.0, "exact_match_mean": group.exact_match.mean(), "exact_match_std": group.exact_match.std(ddof=1) if len(group) > 1 else 0.0})
    pd.DataFrame(aggregate).to_csv(SUMMARY, index=False)
    print(pd.read_csv(SUMMARY).to_string(index=False), flush=True)
    print("EXPANDED_ASR_DONE", flush=True)


if __name__ == "__main__":
    import numpy as np
    main()
