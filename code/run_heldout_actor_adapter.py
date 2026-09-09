import gc
import json
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from kimia_infer.api.kimia import KimiAudio

ROOT = Path("/root/autodl-tmp/kfair")
OUT = ROOT / "outputs"
RAW = OUT / "heldout_actor_adapter_predictions.jsonl"
SUMMARY = OUT / "heldout_actor_adapter_summary.json"
CHECKPOINT = OUT / "heldout_actor_adapter_layer22_rank8.pt"
LABELS = ["neutral", "happy", "sad", "angry"]
CHOICES = {"neutral": "A", "happy": "B", "sad": "C", "angry": "D"}
PAIRS = [(a, b) for index, a in enumerate(LABELS) for b in LABELS[index + 1:]]
LAYER = 22
RANK = 8


class ResidualAdapter(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(rank, width, bias=False, dtype=torch.float32)
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden):
        return self.up(F.gelu(self.down(hidden.float()))).to(hidden.dtype)


def build_examples(manifest):
    examples = []
    for (actor, statement), group in manifest.groupby(["actor", "statement"]):
        files = {row.label: row.filename for row in group.itertuples()}
        for a, b in PAIRS:
            for cell, discrete, acoustic in (("AA", a, a), ("AB", a, b), ("BA", b, a), ("BB", b, b)):
                examples.append({"actor": int(actor), "statement": int(statement), "pair": f"{a}_{b}", "cell": cell,
                                 "discrete_label": discrete, "acoustic_label": acoustic,
                                 "discrete_file": files[discrete], "continuous_file": files[acoustic],
                                 "conflict": discrete != acoustic})
    return examples


def metrics(rows):
    frame = pd.DataFrame(rows)
    frame["acoustic_correct"] = frame.prediction.eq(frame.acoustic_label)
    frame["semantic_follow"] = frame.prediction.eq(frame.discrete_label)
    result = {}
    for adapted, group in frame.groupby("adapted"):
        conflict = group[group.conflict]
        aligned = group[~group.conflict]
        result["adapter" if adapted else "baseline"] = {
            "n": len(group), "ser": float(group.acoustic_correct.mean()),
            "aligned_ser": float(aligned.acoustic_correct.mean()),
            "conflict_acoustic_follow": float(conflict.acoustic_correct.mean()),
            "conflict_semantic_follow": float(conflict.semantic_follow.mean()),
            "conflict_mean_margin": float(conflict.acoustic_margin.mean()),
        }
    return result


def main():
    torch.manual_seed(20260903)
    random.seed(20260903)
    first = pd.read_csv(OUT / "ravdess_expanded_manifest.csv"); first["statement"] = 1
    second = pd.read_csv(OUT / "ravdess_statement2_manifest.csv"); second["statement"] = 2
    manifest = pd.concat([first, second], ignore_index=True)
    examples = build_examples(manifest)
    model = KimiAudio(model_path=str(ROOT / "models/Kimi-Audio-7B-Instruct"), load_detokenizer=False)
    [module for module in model.alm.modules() if hasattr(module, "vq_adaptor")][0].kfair_branch_mode = "full"
    for parameter in model.alm.parameters():
        parameter.requires_grad_(False)
    prompt = "Classify only the speaker's vocal emotion. Choose exactly one: A=neutral, B=happy, C=sad, D=angry. Reply with only A, B, C, or D."
    token_ids = torch.tensor([model.prompt_manager.text_tokenizer.encode(CHOICES[label], bos=False, eos=False)[0] for label in LABELS], device="cuda")
    cache = {}
    for index, row in enumerate(manifest.itertuples(), 1):
        item = model.prompt_manager.get_prompt([{"role": "user", "message_type": "text", "content": prompt},
                                                {"role": "user", "message_type": "audio", "content": row.audio_path}], output_type="text")
        audio, text, mask, _, _ = item.to_tensor()
        cache[row.filename] = (audio.cpu(), text.cpu(), mask.cpu(), item.continuous_feature[0].cpu())
        if index % 48 == 0:
            print(f"CACHE {index}/192", flush=True)

    def prepare(example):
        audio, text, mask, discrete_feature = cache[example["discrete_file"]]
        feature = cache[example["continuous_file"]][3]
        if feature.shape[1] != discrete_feature.shape[1]:
            feature = F.interpolate(feature.transpose(1, 2).float(), size=discrete_feature.shape[1], mode="linear", align_corners=False).transpose(1, 2).to(feature.dtype)
        audio, text, mask, feature = audio.cuda(), text.cuda(), mask.cuda(), feature.cuda()
        positions = torch.arange(audio.shape[1], device="cuda").unsqueeze(0).long()
        return audio, text, mask, feature, positions

    def forward(example):
        audio, text, mask, feature, positions = prepare(example)
        output = model.alm(input_ids=audio, text_input_ids=text, whisper_input_feature=[feature],
                           is_continuous_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
        return output.logits[1][0, -1].index_select(0, token_ids).float()

    test = [example for example in examples if example["actor"] >= 21]
    baseline = {}
    with torch.inference_mode():
        for index, example in enumerate(examples, 1):
            baseline[(example["actor"], example["statement"], example["pair"], example["cell"])] = forward(example).cpu()
            if index % 192 == 0:
                print(f"BASELINE {index}/{len(examples)}", flush=True)

    width = model.alm.config.hidden_size
    adapter = ResidualAdapter(width, RANK).cuda()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    layer = model.alm.model.layers[LAYER]

    def adapter_hook(module, args):
        hidden = args[0].clone()
        hidden[:, -1] = hidden[:, -1] + adapter(hidden[:, -1])
        return (hidden,) + args[1:]

    handle = layer.register_forward_pre_hook(adapter_hook)
    train = [example for example in examples if example["actor"] <= 16]
    history = []
    for epoch in range(2):
        random.shuffle(train)
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for index, example in enumerate(train, 1):
            logits = forward(example)
            target = torch.tensor([LABELS.index(example["acoustic_label"])], device="cuda")
            ser_loss = F.cross_entropy(logits.unsqueeze(0), target)
            pair_loss = ser_loss if example["conflict"] else logits.new_zeros(())
            preserve = logits.new_zeros(())
            if not example["conflict"]:
                key = (example["actor"], example["statement"], example["pair"], example["cell"])
                preserve = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(baseline[key].cuda(), dim=-1), reduction="sum")
            loss = ser_loss + 0.5 * pair_loss + 0.2 * preserve
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach().cpu())
            if index % 96 == 0:
                print(f"TRAIN epoch={epoch + 1} {index}/{len(train)} loss={running/index:.4f}", flush=True)
        history.append(running / len(train))
    torch.save({"layer": LAYER, "rank": RANK, "state_dict": adapter.state_dict(), "history": history,
                "train_actors": list(range(1, 17)), "validation_actors": list(range(17, 21)), "test_actors": list(range(21, 25))}, CHECKPOINT)

    rows = []
    with torch.inference_mode():
        for adapted in (False, True):
            for example in test:
                key = (example["actor"], example["statement"], example["pair"], example["cell"])
                logits = forward(example) if adapted else baseline[key].cuda()
                scores = {label: float(logits[i].cpu()) for i, label in enumerate(LABELS)}
                rows.append({**example, "adapted": adapted, "prediction": max(scores, key=scores.get), "scores": scores,
                             "acoustic_margin": scores[example["acoustic_label"]] - scores[example["discrete_label"]]})
    handle.remove()
    RAW.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    result = {"layer": LAYER, "rank": RANK, "loss": "L_SER + 0.5 L_pair + 0.2 L_preserve", "history": history,
              "split": {"train": "actors 1-16", "validation_reserved": "actors 17-20", "test": "actors 21-24"},
              "test_metrics": metrics(rows)}
    SUMMARY.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
