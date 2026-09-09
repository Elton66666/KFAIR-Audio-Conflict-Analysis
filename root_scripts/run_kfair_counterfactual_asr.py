import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from kimia_infer.api.kimia import KimiAudio

ROOT = Path("/root/autodl-tmp/kfair")
OUT = ROOT / "outputs"
PRED = OUT / "kfair_counterfactual_asr.jsonl"
PAIR_TYPES = {("neutral", "angry"), ("happy", "sad")}
REFS = {1: "kids are talking by the door", 2: "dogs are sitting by the door"}


def words(x):
    import re
    return re.findall(r"[a-z]+", x.lower())


def distance(a, b):
    prev = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1]+1, prev[j]+1, prev[j-1]+(x != y)))
        prev = cur
    return prev[-1]


def main():
    pairs = pd.read_csv(OUT / "kfair_pair_manifest.csv")
    pairs = pairs[pairs.apply(lambda r: (r.label_a, r.label_b) in PAIR_TYPES, axis=1)]
    assert len(pairs) == 96
    model = KimiAudio(model_path=str(ROOT / "models/Kimi-Audio-7B-Instruct"), load_detokenizer=False)
    fusion = [m for m in model.alm.modules() if hasattr(m, "vq_adaptor")][0]
    fusion.kfair_branch_mode = "full"
    pm = model.prompt_manager
    prompt = "Transcribe the spoken words exactly. Reply with only the transcript."
    files = sorted(set(pairs.file_a) | set(pairs.file_b))
    path_map = {}
    for r in pairs.itertuples():
        path_map[r.file_a] = r.path_a; path_map[r.file_b] = r.path_b
    cache = {}
    for i, file in enumerate(files, 1):
        chats = [{"role":"user","message_type":"text","content":prompt},
                 {"role":"user","message_type":"audio","content":path_map[file]}]
        h = pm.get_prompt(chats, output_type="text")
        audio, text, mask, _, _ = h.to_tensor()
        cache[file] = (audio.cpu(), text.cpu(), mask.cpu(), h.continuous_feature[0].cpu())
        if i % 24 == 0: print(f"CACHE {i}/{len(files)}", flush=True)
    done = set()
    if PRED.exists():
        done = {(x["pair_id"],x["cell"]) for x in map(json.loads,PRED.read_text(encoding="utf-8").splitlines())}
    for i, pair in enumerate(pairs.itertuples(),1):
        for cell, dfile, cfile in [("AB",pair.file_a,pair.file_b),("BA",pair.file_b,pair.file_a)]:
            if (pair.pair_id,cell) in done: continue
            audio,text,mask,dfeat=cache[dfile]; cfeat=cache[cfile][3]
            if cfeat.shape[1] != dfeat.shape[1]:
                cfeat=F.interpolate(cfeat.transpose(1,2).float(),size=dfeat.shape[1],mode="linear",align_corners=False).transpose(1,2).to(cfeat.dtype)
            started=time.time()
            at,tt=model._generate_loop(audio_input_ids=audio.cuda(),text_input_ids=text.cuda(),max_new_tokens=48,
                audio_top_k=5,audio_temperature=0.0,audio_repetition_penalty=1.0,audio_repetition_window_size=64,
                text_top_k=5,text_temperature=0.0,text_repetition_penalty=1.0,text_repetition_window_size=16,
                is_continuous_mask=mask.cuda(),continous_feature=[cfeat.cuda()],output_type="text")
            transcript=model.detokenize_text(tt)
            ref=REFS[pair.statement]; edits=distance(words(ref),words(transcript))
            row={"pair_id":pair.pair_id,"actor":pair.actor,"statement":pair.statement,"label_a":pair.label_a,
                 "label_b":pair.label_b,"cell":cell,"discrete_file":dfile,"continuous_file":cfile,
                 "reference":ref,"transcript":transcript,"edits":edits,"wer":edits/len(words(ref)),
                 "exact_match":words(ref)==words(transcript),"seconds":round(time.time()-started,4)}
            with PRED.open("a",encoding="utf-8") as f: f.write(json.dumps(row,ensure_ascii=False)+"\n");f.flush()
        if i%24==0: print(f"PAIRS {i}/96",flush=True)
    df=pd.DataFrame(json.loads(x) for x in PRED.read_text(encoding="utf-8").splitlines())
    summary=df.groupby("cell").agg(n=("pair_id","size"),wer=("wer","mean"),exact_match=("exact_match","mean"),mean_seconds=("seconds","mean")).reset_index()
    summary.to_csv(OUT/"kfair_counterfactual_asr_summary.csv",index=False)
    print(summary.to_string(index=False),flush=True)


if __name__ == "__main__": main()
