import json
import re
from pathlib import Path

import pandas as pd
import torch

from kimia_infer.api.kimia import KimiAudio
from run_heldout_actor_adapter import ResidualAdapter

ROOT = Path("/root/autodl-tmp/kfair")
OUT = ROOT / "outputs"
PRED = OUT / "heldout_actor_adapter_asr.jsonl"
SUMMARY = OUT / "heldout_actor_adapter_asr_summary.csv"


def words(text):
    return re.findall(r"[a-z]+", text.lower())


def distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def main():
    first = pd.read_csv(OUT / "ravdess_expanded_manifest.csv"); first["statement"] = 1
    second = pd.read_csv(OUT / "ravdess_statement2_manifest.csv"); second["statement"] = 2
    manifest = pd.concat([first, second], ignore_index=True)
    manifest = manifest[(manifest.actor >= 21) & manifest.label.isin(["neutral", "angry"])]
    assert len(manifest) == 16
    model = KimiAudio(model_path=str(ROOT / "models/Kimi-Audio-7B-Instruct"), load_detokenizer=False)
    checkpoint = torch.load(OUT / "heldout_actor_adapter_layer22_rank8.pt", map_location="cpu", weights_only=True)
    adapter = ResidualAdapter(model.alm.config.hidden_size, checkpoint["rank"]).cuda()
    adapter.load_state_dict(checkpoint["state_dict"]); adapter.eval()
    layer = model.alm.model.layers[checkpoint["layer"]]

    def hook(module, args):
        hidden = args[0].clone()
        hidden[:, -1] = hidden[:, -1] + adapter(hidden[:, -1])
        return (hidden,) + args[1:]

    prompt = "Transcribe the spoken words exactly. Reply with only the transcript."
    rows = []
    for adapted in (False, True):
        handle = layer.register_forward_pre_hook(hook) if adapted else None
        for row in manifest.to_dict("records"):
            messages = [{"role": "user", "message_type": "text", "content": prompt},
                        {"role": "user", "message_type": "audio", "content": row["audio_path"]}]
            _, transcript = model.generate(messages, output_type="text", max_new_tokens=48, text_temperature=0.0, text_top_k=5)
            reference = "kids are talking by the door" if row["statement"] == 1 else "dogs are sitting by the door"
            edits = distance(words(reference), words(transcript))
            result = {**row, "adapted": adapted, "reference": reference, "transcript": transcript,
                      "edits": edits, "ref_words": len(words(reference)), "wer": edits / len(words(reference))}
            rows.append(result); print(json.dumps(result, ensure_ascii=False), flush=True)
        if handle:
            handle.remove()
    PRED.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    frame = pd.DataFrame(rows)
    summary = pd.DataFrame([{"adapted": adapted, "n": len(group), "wer_micro": group.edits.sum()/group.ref_words.sum(), "wer_mean": group.wer.mean()}
                            for adapted, group in frame.groupby("adapted")])
    summary.to_csv(SUMMARY, index=False); print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
