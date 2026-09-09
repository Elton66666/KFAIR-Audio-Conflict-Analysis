import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kimia_infer.api.kimia import KimiAudio
from run_heldout_actor_adapter import ResidualAdapter

ROOT = Path("/root/autodl-tmp/kfair")
OUT = ROOT / "outputs"
DATA = ROOT / "data/emis_balanced192"
RAW = OUT / "adapter_emis_zeroshot_predictions.jsonl"
STATS = OUT / "adapter_emis_zeroshot_statistics.csv"
LABELS = ["neutral", "happy", "sad", "angry"]
CHOICES = {"neutral": "A", "happy": "B", "sad": "C", "angry": "D"}


def manifest():
    rows = []
    for filename, conflict in (("emis_balanced144_manifest.jsonl", True), ("emis_aligned48_manifest.jsonl", False)):
        for line in (ROOT / "data" / filename).read_text(encoding="utf-8").splitlines():
            row = json.loads(line); row["conflict"] = conflict; row["audio_path"] = str(DATA / row["filename"]); rows.append(row)
    assert len(rows) == 192
    return rows


def cluster_bootstrap(group, column, repetitions=10000):
    clustered = group.groupby(["generator", "text_id"])[column].agg(["sum", "count"])
    totals, counts = clustered["sum"].to_numpy(), clustered["count"].to_numpy()
    rng = np.random.default_rng(20260903)
    indices = rng.integers(0, len(totals), size=(repetitions, len(totals)))
    samples = totals[indices].sum(1) / counts[indices].sum(1)
    return float(group[column].mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def main():
    model = KimiAudio(model_path=str(ROOT / "models/Kimi-Audio-7B-Instruct"), load_detokenizer=False)
    [module for module in model.alm.modules() if hasattr(module, "vq_adaptor")][0].kfair_branch_mode = "full"
    checkpoint = torch.load(OUT / "heldout_actor_adapter_layer22_rank8.pt", map_location="cpu", weights_only=True)
    adapter = ResidualAdapter(model.alm.config.hidden_size, checkpoint["rank"]).cuda()
    adapter.load_state_dict(checkpoint["state_dict"]); adapter.eval()
    layer = model.alm.model.layers[checkpoint["layer"]]

    def hook(module, args):
        hidden = args[0].clone(); hidden[:, -1] = hidden[:, -1] + adapter(hidden[:, -1]); return (hidden,) + args[1:]

    prompt = "Classify only the speaker's vocal emotion. Choose exactly one: A=neutral, B=happy, C=sad, D=angry. Reply with only A, B, C, or D."
    token_ids = {label: model.prompt_manager.text_tokenizer.encode(choice, bos=False, eos=False)[0] for label, choice in CHOICES.items()}
    rows = []
    for adapted in (False, True):
        handle = layer.register_forward_pre_hook(hook) if adapted else None
        for index, row in enumerate(manifest(), 1):
            item = model.prompt_manager.get_prompt([{"role": "user", "message_type": "text", "content": prompt},
                                                    {"role": "user", "message_type": "audio", "content": row["audio_path"]}], output_type="text")
            audio, text, mask, _, _ = item.to_tensor(); audio, text, mask = audio.cuda(), text.cuda(), mask.cuda()
            positions = torch.arange(audio.shape[1], device="cuda").unsqueeze(0).long(); started = time.time()
            with torch.inference_mode():
                output = model.alm(input_ids=audio, text_input_ids=text, whisper_input_feature=[item.continuous_feature[0].cuda()],
                                   is_continuous_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
            log_probs = torch.log_softmax(output.logits[1][0, -1].float(), dim=-1)
            scores = {label: float(log_probs[token].cpu()) for label, token in token_ids.items()}
            rows.append({**row, "adapted": adapted, "prediction": max(scores, key=scores.get), "scores": scores,
                         "acoustic_margin": scores[row["acoustic_label"]] - scores[row["semantic_label"]],
                         "seconds": time.time() - started})
            if index % 48 == 0: print(f"adapted={adapted} {index}/192", flush=True)
        if handle: handle.remove()
    RAW.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    frame = pd.DataFrame(rows); frame["acoustic_correct"] = frame.prediction.eq(frame.acoustic_label); frame["semantic_follow"] = frame.prediction.eq(frame.semantic_label)
    frame["other"] = ~frame.acoustic_correct & ~frame.semantic_follow
    stats = []
    for keys, group in frame.groupby(["adapted", "conflict", "generator"]):
        for metric in ("acoustic_correct", "semantic_follow", "other", "acoustic_margin", "seconds"):
            mean, low, high = cluster_bootstrap(group, metric); stats.append({"adapted": keys[0], "conflict": keys[1], "generator": keys[2], "metric": metric, "mean": mean, "ci_low": low, "ci_high": high, "n": len(group)})
    for keys, group in frame.groupby(["adapted", "conflict"]):
        for metric in ("acoustic_correct", "semantic_follow", "other", "acoustic_margin", "seconds"):
            mean, low, high = cluster_bootstrap(group, metric); stats.append({"adapted": keys[0], "conflict": keys[1], "generator": "ALL", "metric": metric, "mean": mean, "ci_low": low, "ci_high": high, "n": len(group)})
    table = pd.DataFrame(stats); table.to_csv(STATS, index=False); print(table[table.generator.eq("ALL")].to_string(index=False), flush=True)


if __name__ == "__main__": main()
