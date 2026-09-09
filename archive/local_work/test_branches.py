import json
import time

import torch

from kimia_infer.api.kimia import KimiAudio


MODEL = "/root/autodl-tmp/kfair/models/Kimi-Audio-7B-Instruct"
MODES = ["full", "no_continuous", "no_discrete", "rolled_continuous"]

model = KimiAudio(model_path=MODEL, load_detokenizer=False)
fusion_modules = [module for module in model.alm.modules() if hasattr(module, "vq_adaptor")]
if len(fusion_modules) != 1:
    raise RuntimeError(f"Expected one fusion module, found {len(fusion_modules)}")
fusion = fusion_modules[0]

messages = [
    {"role": "user", "message_type": "text", "content": "Please transcribe the following audio:"},
    {"role": "user", "message_type": "audio", "content": "test_audios/asr_example.wav"},
]
sampling = {
    "audio_temperature": 0.0,
    "audio_top_k": 5,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
    "max_new_tokens": 64,
}

results = []
for mode in MODES:
    fusion.kfair_branch_mode = mode
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    _, text = model.generate(messages, output_type="text", **sampling)
    results.append(
        {
            "mode": mode,
            "seconds": round(time.time() - started, 3),
            "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "output": text,
        }
    )
print("BRANCH_RESULTS=" + json.dumps(results, ensure_ascii=False))
