import json
import time

import torch

from kimia_infer.api.kimia import KimiAudio


MODEL = "/root/autodl-tmp/kfair/models/Kimi-Audio-7B-Instruct"

started = time.time()
model = KimiAudio(model_path=MODEL, load_detokenizer=False)
loaded = time.time()

sampling = {
    "audio_temperature": 0.0,
    "audio_top_k": 5,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}

messages = [
    {"role": "user", "message_type": "text", "content": "Please transcribe the following audio:"},
    {"role": "user", "message_type": "audio", "content": "test_audios/asr_example.wav"},
]
_, text = model.generate(messages, output_type="text", **sampling)

result = {
    "model_load_seconds": round(loaded - started, 3),
    "total_seconds": round(time.time() - started, 3),
    "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    "output": text,
}
print("SMOKE_RESULT=" + json.dumps(result, ensure_ascii=False))
