import hashlib
import itertools
import json
import os
import gc
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from kimia_infer.api.kimia import KimiAudio

ROOT = Path("/root/autodl-tmp/kfair")
OUT = ROOT / "outputs"
PRED = OUT / "kfair_activation_patching.jsonl"
LABELS = ["neutral", "happy", "sad", "angry"]
CHOICES = {"neutral": "A", "happy": "B", "sad": "C", "angry": "D"}
CORE = [("neutral", "angry"), ("happy", "sad")]
MAIN_LAYERS = [24, 25, 26, 27]
WRONG_LAYERS = [12, 18, 22]
CAPTURE_LAYERS = sorted(set(MAIN_LAYERS + WRONG_LAYERS))


def load_data():
    a = pd.read_csv(OUT / "ravdess_expanded_manifest.csv")
    b = pd.read_csv(OUT / "ravdess_statement2_manifest.csv")
    a["statement"], b["statement"] = 1, 2
    manifest = pd.concat([a, b], ignore_index=True)
    pairs = []
    for (actor, statement), g in manifest.groupby(["actor", "statement"]):
        by = {r.label: r for r in g.itertuples()}
        for la, lb in CORE:
            x, y = by[la], by[lb]
            pairs.append(dict(pair_id=f"a{actor:02d}_s{statement}_{la}_{lb}", actor=int(actor),
                              statement=int(statement), label_a=la, label_b=lb,
                              file_a=x.filename, file_b=y.filename))
    return manifest, pairs


def resize_sequence(x, length):
    if x.shape[0] == length:
        return x
    return F.interpolate(x.T.unsqueeze(0).float(), size=length, mode="linear",
                         align_corners=False)[0].T.to(x.dtype)


def main():
    manifest, pairs = load_data()
    max_pairs = int(os.environ.get("KFAIR_MAX_PAIRS", "0"))
    if max_pairs:
        pairs = pairs[:max_pairs]
    model = KimiAudio(model_path=str(ROOT / "models/Kimi-Audio-7B-Instruct"), load_detokenizer=False)
    fusion = [m for m in model.alm.modules() if hasattr(m, "vq_adaptor")][0]
    fusion.kfair_branch_mode = "full"
    pm = model.prompt_manager
    prompt = "Classify only the speaker's vocal emotion. Choose exactly one: A=neutral, B=happy, C=sad, D=angry. Reply with only A, B, C, or D."
    choice_ids = {k: pm.text_tokenizer.encode(v, bos=False, eos=False)[0] for k, v in CHOICES.items()}
    cache = {}
    for i, r in enumerate(manifest.itertuples(), 1):
        h = pm.get_prompt([{"role":"user","message_type":"text","content":prompt},
                           {"role":"user","message_type":"audio","content":r.audio_path}], output_type="text")
        au, tx, ma, _, _ = h.to_tensor()
        cache[r.filename] = dict(audio=au.cpu(), text=tx.cpu(), mask=ma.cpu(),
                                 feature=h.continuous_feature[0].cpu(), label=r.label)
        if i % 48 == 0: print(f"CACHE {i}/192", flush=True)

    def prepare(dfile, cfile):
        d, c = cache[dfile], cache[cfile]
        feat = c["feature"]
        if feat.shape[1] != d["feature"].shape[1]:
            feat = F.interpolate(feat.transpose(1,2).float(), size=d["feature"].shape[1],
                                 mode="linear", align_corners=False).transpose(1,2).to(feat.dtype)
        au, tx, ma = d["audio"].cuda(), d["text"].cuda(), d["mask"].cuda()
        pos = torch.arange(au.shape[1], device="cuda").unsqueeze(0).long()
        return au, tx, ma, feat.cuda(), pos

    def forward(dfile, cfile, hidden=False):
        au, tx, ma, feat, pos = prepare(dfile, cfile)
        states = None
        handles = []
        captured = {}
        if hidden:
            def make_hook(layer):
                def hook(_module, args):
                    captured[layer] = args[0][0].detach().cpu()
                return hook
            for layer in CAPTURE_LAYERS:
                handles.append(model.alm.model.layers[layer].register_forward_pre_hook(make_hook(layer)))
        with torch.inference_mode():
            try:
                o = model.alm(input_ids=au, text_input_ids=tx, whisper_input_feature=[feat],
                              is_continuous_mask=ma, position_ids=pos, use_cache=False,
                              output_hidden_states=False, return_dict=True)
            finally:
                for handle in handles: handle.remove()
        lp = torch.log_softmax(o.logits[1][0,-1].float(), dim=-1)
        scores = {k: float(lp[v].cpu()) for k,v in choice_ids.items()}
        if hidden:
            states = captured
        return scores, states, ma.detach().cpu()

    def patched_forward(dfile, cfile, source_state, source_mask, layer, scope, seed):
        au, tx, ma, feat, pos = prepare(dfile, cfile)
        module = model.alm.model.layers[layer]
        def pre_hook(_module, args):
            h = args[0].clone()
            src = source_state.to(h.device)
            if scope == "full":
                h[0] = resize_sequence(src, h.shape[1])
            else:
                ti = ma[0].nonzero(as_tuple=False).squeeze(-1)
                si = source_mask[0].nonzero(as_tuple=False).squeeze(-1)
                vals = resize_sequence(src[si.cpu()].to(h.device), ti.numel())
                if scope == "random_positions":
                    gen = torch.Generator(device=h.device); gen.manual_seed(seed)
                    vals = vals[torch.randperm(vals.shape[0], generator=gen, device=h.device)]
                h[0, ti] = vals
            return (h,) + args[1:]
        handle = module.register_forward_pre_hook(pre_hook)
        try:
            with torch.inference_mode():
                o = model.alm(input_ids=au, text_input_ids=tx, whisper_input_feature=[feat],
                              is_continuous_mask=ma, position_ids=pos, use_cache=False, return_dict=True)
        finally:
            handle.remove()
        lp = torch.log_softmax(o.logits[1][0,-1].float(), dim=-1)
        return {k:float(lp[v].cpu()) for k,v in choice_ids.items()}

    completed = set()
    if PRED.exists():
        old = [json.loads(x) for x in PRED.read_text(encoding="utf-8").splitlines()]
        counts = pd.Series([x["pair_id"] for x in old]).value_counts()
        completed = set(counts[counts >= 50].index)
        kept = [x for x in old if x["pair_id"] in completed]
        PRED.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in kept),encoding="utf-8")
        print(f"RESUME completed_pairs={len(completed)} kept_rows={len(kept)}",flush=True)
    with PRED.open("a", encoding="utf-8") as out:
        for pi, p in enumerate(pairs, 1):
            if p["pair_id"] in completed:
                continue
            fa, fb = p["file_a"], p["file_b"]
            aa_scores, aa_states, aa_mask = forward(fa, fa, hidden=True)
            bb_scores, bb_states, bb_mask = forward(fb, fb, hidden=True)
            for target_cell, dfile, cfile in [("AB",fa,fb),("BA",fb,fa)]:
                base_scores, _, _ = forward(dfile,cfile,hidden=False)
                acoustic_label = p["label_b"] if target_cell == "AB" else p["label_a"]
                discrete_label = p["label_a"] if target_cell == "AB" else p["label_b"]
                base_margin = base_scores[acoustic_label] - base_scores[discrete_label]
                base = {**p,"target_cell":target_cell,"condition":"no_patch","layer":None,
                        "scope":"none","source_cell":None,"acoustic_label":acoustic_label,
                        "discrete_label":discrete_label,"prediction":max(base_scores,key=base_scores.get),
                        "scores":base_scores,"acoustic_margin":base_margin,"rescue":0.0}
                out.write(json.dumps(base,ensure_ascii=False)+"\n")
                # Acoustic-consistent source: AB<-BB, BA<-AA. Discrete-consistent is the reverse.
                sources = ({"acoustic":(bb_states,bb_mask,"BB"),"discrete":(aa_states,aa_mask,"AA")}
                           if target_cell == "AB" else
                           {"acoustic":(aa_states,aa_mask,"AA"),"discrete":(bb_states,bb_mask,"BB")})
                specs = []
                for layer in MAIN_LAYERS:
                    for direction in ["acoustic","discrete"]:
                        for scope in ["audio_tokens","full"]:
                            specs.append((layer,direction,scope))
                for layer in WRONG_LAYERS:
                    for direction in ["acoustic","discrete"]:
                        specs.append((layer,direction,"audio_tokens"))
                for direction in ["acoustic","discrete"]:
                    specs.append((25,direction,"random_positions"))
                for layer,direction,scope in specs:
                    states, smask, source_cell = sources[direction]
                    seed = int(hashlib.sha1(f"{p['pair_id']}-{target_cell}-{direction}".encode()).hexdigest()[:8],16)
                    started=time.time(); scores=patched_forward(dfile,cfile,states[layer],smask,layer,scope,seed)
                    margin=scores[acoustic_label]-scores[discrete_label]
                    rec={**p,"target_cell":target_cell,"condition":direction,"layer":layer,
                         "scope":scope,"source_cell":source_cell,"acoustic_label":acoustic_label,
                         "discrete_label":discrete_label,"prediction":max(scores,key=scores.get),
                         "scores":scores,"acoustic_margin":margin,"baseline_margin":base_margin,
                         "rescue":margin-base_margin,"seconds":round(time.time()-started,4)}
                    out.write(json.dumps(rec,ensure_ascii=False)+"\n")
                out.flush()
            del aa_states, bb_states
            gc.collect(); torch.cuda.empty_cache()
            if pi%8==0: print(f"PAIRS {pi}/{len(pairs)}",flush=True)
    print(f"DONE rows={sum(1 for _ in PRED.open(encoding='utf-8'))}",flush=True)

if __name__ == "__main__": main()
