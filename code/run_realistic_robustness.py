import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from scipy.signal import fftconvolve

from kimia_infer.api.kimia import KimiAudio

ROOT = Path("/root/autodl-tmp/kfair")
OUT = ROOT / "outputs"
TEMP = ROOT / "data/realistic_robustness_audio"
ASSETS = ROOT / "data/slr28/selected"
RAW = OUT / "realistic_robustness_predictions.jsonl"
RUNS = OUT / "realistic_robustness_run_summary.jsonl"
SUMMARY = OUT / "realistic_robustness_summary.csv"
SEEDS = [20260903, 20260904, 20260905]
LABELS = ["neutral", "happy", "sad", "angry"]
CHOICES = {"neutral": "A", "happy": "B", "sad": "C", "angry": "D"}
METHODS = ["baseline", "decision_adapter_layer25", "full_sequence_adapter_layer22", "ordinary_lora_layer22"]
CONDITIONS = ["clean", "real_noise_10db", "real_office_rir", "g711_mulaw_8khz", "rir_plus_noise", "combined_rir_noise_g711"]


class Adapter(nn.Module):
    def __init__(self, width, rank=8):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(rank, width, bias=False, dtype=torch.float32)

    def forward(self, hidden):
        return self.up(F.gelu(self.down(hidden.float()))).to(hidden.dtype)


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=8):
        super().__init__(); self.base = base; self.scale = alpha / rank
        self.A = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32)
        self.B = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32)

    def forward(self, hidden):
        return self.base(hidden) + self.B(self.A(hidden.float())).to(hidden.dtype) * self.scale


def append(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n"); handle.flush()


def mono_resample(path, target_rate=48000):
    waveform, rate = torchaudio.load(str(path)); waveform = waveform.mean(dim=0)
    if rate != target_rate: waveform = torchaudio.functional.resample(waveform, rate, target_rate)
    return waveform.numpy().astype("float32"), target_rate


def safe_peak(signal):
    peak = float(np.max(np.abs(signal)))
    return signal * (0.99 / peak) if peak > 0.99 else signal


def add_noise(signal, noise, snr_db, key):
    if len(noise) < len(signal): noise = np.tile(noise, int(np.ceil(len(signal) / len(noise))))
    maximum = len(noise) - len(signal)
    offset = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (maximum + 1)
    segment = noise[offset:offset + len(signal)].copy()
    segment -= segment.mean()
    signal_rms = np.sqrt(np.mean(signal ** 2) + 1e-12)
    noise_rms = np.sqrt(np.mean(segment ** 2) + 1e-12)
    segment *= signal_rms / (10 ** (snr_db / 20) * noise_rms)
    return safe_peak(signal + segment)


def reverberate(signal, rir):
    rir = rir / (np.sqrt(np.sum(rir ** 2)) + 1e-12)
    output = fftconvolve(signal, rir, mode="full")[:len(signal)].astype("float32")
    output *= np.sqrt(np.mean(signal ** 2) + 1e-12) / (np.sqrt(np.mean(output ** 2) + 1e-12))
    return safe_peak(output)


def g711(signal, rate=48000):
    tensor = torch.from_numpy(signal)
    low = torchaudio.functional.resample(tensor, rate, 8000)
    encoded = torchaudio.functional.mu_law_encoding(low.clamp(-1, 1), 256)
    decoded = torchaudio.functional.mu_law_decoding(encoded, 256)
    return safe_peak(torchaudio.functional.resample(decoded, 8000, rate).numpy().astype("float32"))


def prepare_audio(manifest):
    TEMP.mkdir(parents=True, exist_ok=True)
    noise, _ = mono_resample(ASSETS / "RVB2014_type1_noise_smallroom1_1.wav")
    rir, _ = mono_resample(ASSETS / "air_type1_air_binaural_office_0_1.wav")
    mapping = {}
    for index, row in enumerate(manifest, 1):
        clean, rate = mono_resample(row["audio_path"])
        rev = reverberate(clean, rir)
        outputs = {
            "real_noise_10db": add_noise(clean, noise, 10, row["filename"] + "-noise"),
            "real_office_rir": rev,
            "g711_mulaw_8khz": g711(clean, rate),
            "rir_plus_noise": add_noise(rev, noise, 10, row["filename"] + "-rir-noise"),
        }
        outputs["combined_rir_noise_g711"] = g711(outputs["rir_plus_noise"], rate)
        mapping[(row["filename"], "clean")] = row["audio_path"]
        for condition, signal in outputs.items():
            path = TEMP / f"{condition}__{row['filename']}"
            sf.write(path, signal, rate, subtype="PCM_16")
            mapping[(row["filename"], condition)] = str(path)
        if index % 8 == 0: print(f"AUDIO_PREP {index}/{len(manifest)}", flush=True)
    provenance = {"source": "OpenSLR SLR28", "archive_md5": "e6f48e257286e05de56413b4779d8ffb", "noise": "RVB2014_type1_noise_smallroom1_1.wav", "rir": "air_type1_air_binaural_office_0_1.wav", "noise_snr_db": 10, "conditions": CONDITIONS, "generated_files": len(manifest) * (len(CONDITIONS) - 1)}
    (OUT / "realistic_robustness_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return mapping


def metrics(rows):
    frame = pd.DataFrame(rows)
    return {"n": len(frame), "accuracy": frame.correct.mean(), "semantic_follow": frame.semantic_follow.mean(), "mean_acoustic_margin": frame.acoustic_margin.mean()}


def main():
    first = pd.read_csv(OUT / "ravdess_expanded_manifest.csv"); first["statement"] = 1
    second = pd.read_csv(OUT / "ravdess_statement2_manifest.csv"); second["statement"] = 2
    frame = pd.concat([first, second], ignore_index=True); frame = frame[frame.actor >= 21].sort_values(["actor", "statement", "label"])
    manifest = []
    for row in frame.to_dict("records"):
        manifest.append({**row, "semantic_label": "neutral", "acoustic_label": row["label"], "conflict": row["label"] != "neutral"})
    assert len(manifest) == 32
    paths = prepare_audio(manifest)
    completed = set()
    if RUNS.exists(): completed = {(row["method"], int(row["seed"])) for row in map(json.loads, RUNS.read_text(encoding="utf-8").splitlines())}

    model = KimiAudio(model_path=str(ROOT / "models/Kimi-Audio-7B-Instruct"), load_detokenizer=False)
    [module for module in model.alm.modules() if hasattr(module, "vq_adaptor")][0].kfair_branch_mode = "full"
    for parameter in model.alm.parameters(): parameter.requires_grad_(False)
    prompt = "Classify only the speaker's vocal emotion. Choose exactly one: A=neutral, B=happy, C=sad, D=angry. Reply with only A, B, C, or D."
    token_ids = torch.tensor([model.prompt_manager.text_tokenizer.encode(CHOICES[label], bos=False, eos=False)[0] for label in LABELS], device="cuda")
    cache = {}
    for condition in CONDITIONS:
        for index, row in enumerate(manifest, 1):
            item = model.prompt_manager.get_prompt([{"role": "user", "message_type": "text", "content": prompt}, {"role": "user", "message_type": "audio", "content": paths[(row["filename"], condition)]}], output_type="text")
            audio, text, mask, _, _ = item.to_tensor()
            cache[(row["filename"], condition)] = (audio.cpu(), text.cpu(), mask.cpu(), item.continuous_feature[0].cpu())
        print(f"CACHE {condition}", flush=True)

    def logits(row, condition):
        audio, text, mask, feature = cache[(row["filename"], condition)]
        audio, text, mask, feature = audio.cuda(), text.cuda(), mask.cuda(), feature.cuda()
        position = torch.arange(audio.shape[1], device="cuda").unsqueeze(0).long()
        output = model.alm(input_ids=audio, text_input_ids=text, whisper_input_feature=[feature], is_continuous_mask=mask, position_ids=position, use_cache=False, return_dict=True)
        return output.logits[1][0, -1].index_select(0, token_ids).float()

    adapter = Adapter(model.alm.config.hidden_size).cuda(); attention = model.alm.model.layers[22].self_attn; original_q_proj = attention.q_proj
    systems = [("baseline", 0)] + [(method, seed) for method in METHODS[1:] for seed in SEEDS]
    for method, seed in systems:
        if (method, seed) in completed: print(f"SKIP {(method, seed)}", flush=True); continue
        handle = None; lora = None
        if method != "baseline":
            checkpoint = torch.load(OUT / f"actor_cv6_checkpoints/fold6_{method}_seed{seed}.pt", map_location="cpu", weights_only=True)
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
                lora = LoRALinear(original_q_proj).cuda(); lora.A.load_state_dict(checkpoint["state_dict"]["A"]); lora.B.load_state_dict(checkpoint["state_dict"]["B"]); lora.eval(); attention.q_proj = lora
        predictions = []; started = time.time()
        with torch.inference_mode():
            for condition in CONDITIONS:
                for row in manifest:
                    values = torch.log_softmax(logits(row, condition), dim=-1)
                    scores = {label: float(values[i].cpu()) for i, label in enumerate(LABELS)}
                    prediction = max(scores, key=scores.get)
                    result = {**row, "condition": condition, "method": method, "seed": seed, "prediction": prediction, "correct": prediction == row["acoustic_label"], "semantic_follow": prediction == row["semantic_label"], "scores": scores, "acoustic_margin": scores[row["acoustic_label"]] - scores[row["semantic_label"]]}
                    append(RAW, result); predictions.append(result)
                print(f"ROBUST {method} {seed} {condition}", flush=True)
        record = {"method": method, "seed": seed, "seconds": time.time() - started, "conditions": {condition: metrics([row for row in predictions if row["condition"] == condition]) for condition in CONDITIONS}}
        append(RUNS, record); print("RESULT " + json.dumps(record), flush=True)
        if handle is not None: handle.remove()
        if lora is not None: attention.q_proj = original_q_proj; del lora
        torch.cuda.empty_cache()

    runs = [json.loads(line) for line in RUNS.read_text(encoding="utf-8").splitlines() if line.strip()]
    flat = []
    for run in runs:
        for condition, values in run["conditions"].items(): flat.append({"method": run["method"], "seed": run["seed"], "condition": condition, **values})
    detailed = pd.DataFrame(flat); aggregate = []
    for (method, condition), group in detailed.groupby(["method", "condition"]):
        clean = detailed[(detailed.method == method) & (detailed.condition == "clean")]
        aggregate.append({"method": method, "condition": condition, "runs": len(group), "accuracy_mean": group.accuracy.mean(), "accuracy_std": group.accuracy.std(ddof=1) if len(group) > 1 else 0.0, "clean_delta_mean": group.accuracy.mean() - clean.accuracy.mean(), "semantic_follow_mean": group.semantic_follow.mean(), "margin_mean": group.mean_acoustic_margin.mean()})
    pd.DataFrame(aggregate).to_csv(SUMMARY, index=False); print(pd.read_csv(SUMMARY).to_string(index=False), flush=True)
    print("REALISTIC_ROBUSTNESS_DONE", flush=True)


if __name__ == "__main__": main()
