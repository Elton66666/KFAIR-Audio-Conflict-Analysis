import json
import types
import torch
import torch.nn.functional as F

from kimia_infer.api.kimia import KimiAudio

ROOT = "/root/autodl-tmp/kfair"
A = ROOT + "/data/ravdess_eval_audio/03-01-01-01-01-01-01.wav"
B = ROOT + "/data/ravdess_eval_audio/03-01-05-01-01-01-01.wav"
LABELS = ["neutral", "happy", "sad", "angry"]

model = KimiAudio(model_path=ROOT + "/models/Kimi-Audio-7B-Instruct", load_detokenizer=False)
pm = model.prompt_manager
original_extract = pm.extract_whisper_feat
fusion = [m for m in model.alm.modules() if hasattr(m, "vq_adaptor")][0]
fusion.kfair_branch_mode = "full"
prompt = "Classify the speaker's vocal emotion. Reply with exactly one lowercase label: neutral, happy, sad, or angry."


def feature(path, target_len=None):
    feat = original_extract(path)
    if target_len is not None and feat.shape[1] != target_len:
        feat = F.interpolate(feat.transpose(1, 2).float(), size=target_len, mode="linear", align_corners=False).transpose(1, 2).to(feat.dtype)
    return feat


def history(disc_path, cont_path):
    target_len = feature(disc_path).shape[1]
    override = feature(cont_path, target_len)
    pm.extract_whisper_feat = types.MethodType(lambda self, wav: override, pm)
    chats = [
        {"role": "user", "message_type": "text", "content": prompt},
        {"role": "user", "message_type": "audio", "content": disc_path},
    ]
    h = pm.get_prompt(chats, output_type="text")
    pm.extract_whisper_feat = original_extract
    return chats, h


def score(disc_path, cont_path):
    chats, h = history(disc_path, cont_path)
    audio_ids, text_ids, mask, _, _ = h.to_tensor()
    audio_ids, text_ids, mask = audio_ids.cuda(), text_ids.cuda(), mask.cuda()
    feats = [x.cuda() for x in h.continuous_feature]
    pos = torch.arange(audio_ids.shape[1], device="cuda").unsqueeze(0).long()
    with torch.inference_mode():
        _, logits, _ = model.alm.forward(input_ids=audio_ids, text_input_ids=text_ids, whisper_input_feature=feats, is_continuous_mask=mask, position_ids=pos, past_key_values=None, return_dict=False)
    last = torch.log_softmax(logits[0, -1].float(), dim=-1)
    label_ids = {label: pm.text_tokenizer.encode(label, bos=False, eos=False) for label in LABELS}
    result = {label: {"ids": ids, "first_logprob": float(last[ids[0]].cpu())} for label, ids in label_ids.items()}
    _, generated = model.generate(chats, output_type="text", max_new_tokens=12, text_temperature=0.0, text_top_k=5)
    return {"disc": disc_path, "cont": cont_path, "scores": result, "argmax": max(result, key=lambda x: result[x]["first_logprob"]), "generated_without_override": generated}


print(json.dumps(score(A, A), ensure_ascii=False, indent=2))
print(json.dumps(score(A, B), ensure_ascii=False, indent=2))
