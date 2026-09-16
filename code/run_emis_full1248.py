import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import binomtest

from kimia_infer.api.kimia import KimiAudio

ROOT = Path('/root/autodl-tmp/kfair')
DATA = ROOT / 'data/emis_full1248'
OUT = ROOT / 'outputs'
RAW = OUT / 'emis_full1248_predictions.jsonl'
RUNS = OUT / 'emis_full1248_run_summary.jsonl'
SUMMARY = OUT / 'emis_full1248_summary.csv'
PAIRED = OUT / 'emis_full1248_paired_effects.csv'
PROVENANCE = OUT / 'emis_full1248_provenance.json'
SEEDS = [20260903, 20260904, 20260905]
LABELS = ['neutral', 'happy', 'sad', 'angry']
CHOICES = {'neutral': 'A', 'happy': 'B', 'sad': 'C', 'angry': 'D'}


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


def append(path, row):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        handle.flush()


def parse_manifest():
    pattern = re.compile(
        r'^(?P<text_id>\d+)_(?P<semantic>neutral|happy|sad|angry)'
        r'(?:_(?P<explicitness>explicit|implicit))?_(?P<acoustic>neutral|happy|sad|angry)'
        r'_(?P<voice>\d+)_(?P<generator>COSY|F5TTS|STYLE)\.wav$'
    )
    rows = []
    for path in sorted(DATA.glob('*.wav')):
        match = pattern.match(path.name)
        if not match:
            raise ValueError(f'Unexpected filename: {path.name}')
        values = match.groupdict()
        semantic, acoustic = values['semantic'], values['acoustic']
        rows.append({
            'filename': path.name,
            'audio_path': str(path),
            'text_id': int(values['text_id']),
            'semantic_label': semantic,
            'acoustic_label': acoustic,
            'explicitness': values['explicitness'] or 'neutral',
            'voice_id': int(values['voice']),
            'generator': values['generator'],
            'cluster_id': f"{values['generator']}_{int(values['text_id']):02d}",
            'conflict': semantic != acoustic,
        })
    frame = pd.DataFrame(rows)
    assert len(frame) == 1248 and frame.filename.nunique() == 1248
    assert int(frame.conflict.sum()) == 936 and int((~frame.conflict).sum()) == 312
    assert frame.groupby('generator').size().to_dict() == {'COSY': 416, 'F5TTS': 416, 'STYLE': 416}
    assert frame.groupby('semantic_label').size().to_dict() == {'angry': 312, 'happy': 312, 'neutral': 312, 'sad': 312}
    assert frame.groupby('acoustic_label').size().to_dict() == {'angry': 312, 'happy': 312, 'neutral': 312, 'sad': 312}
    return frame.to_dict('records')


def bootstrap(group, column, repetitions=10000):
    clustered = group.groupby('cluster_id')[column].agg(['sum', 'count'])
    totals, counts = clustered['sum'].to_numpy(), clustered['count'].to_numpy()
    rng = np.random.default_rng(20260916)
    indices = rng.integers(0, len(totals), size=(repetitions, len(totals)))
    samples = totals[indices].sum(1) / counts[indices].sum(1)
    return float(group[column].mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def summarize():
    frame = pd.DataFrame(json.loads(line) for line in RAW.read_text(encoding='utf-8').splitlines() if line.strip())
    frame = frame.drop_duplicates(['system', 'seed', 'filename'], keep='last')
    systems = frame[['system', 'seed']].drop_duplicates()
    assert len(systems) == 13, systems
    assert len(frame) == 13 * 1248, len(frame)
    frame['acoustic_correct'] = frame.prediction.eq(frame.acoustic_label)
    frame['semantic_follow'] = frame.prediction.eq(frame.semantic_label)
    frame['other'] = ~frame.acoustic_correct & ~frame.semantic_follow
    rows = []
    for (system, seed, conflict, generator), group in frame.groupby(['system', 'seed', 'conflict', 'generator']):
        for metric in ['acoustic_correct', 'semantic_follow', 'other', 'acoustic_margin']:
            mean, low, high = bootstrap(group, metric)
            rows.append({'system': system, 'seed': seed, 'conflict': conflict, 'generator': generator,
                         'metric': metric, 'mean': mean, 'ci_low': low, 'ci_high': high, 'n': len(group)})
    for (system, seed, conflict), group in frame.groupby(['system', 'seed', 'conflict']):
        for metric in ['acoustic_correct', 'semantic_follow', 'other', 'acoustic_margin']:
            mean, low, high = bootstrap(group, metric)
            rows.append({'system': system, 'seed': seed, 'conflict': conflict, 'generator': 'ALL',
                         'metric': metric, 'mean': mean, 'ci_low': low, 'ci_high': high, 'n': len(group)})
    pd.DataFrame(rows).to_csv(SUMMARY, index=False)

    baseline = frame[frame.system.eq('branch_full')].set_index('filename')
    paired = []
    for (system, seed), group in frame[~frame.system.str.startswith('branch_')].groupby(['system', 'seed']):
        for subset, selected in [('all', group), ('conflict', group[group.conflict]), ('aligned', group[~group.conflict])]:
            current = selected.set_index('filename')
            base = baseline.loc[current.index]
            improved = int((~base.acoustic_correct & current.acoustic_correct).sum())
            harmed = int((base.acoustic_correct & ~current.acoustic_correct).sum())
            p = float(binomtest(min(improved, harmed), improved + harmed, .5).pvalue) if improved + harmed else 1.0
            paired.append({'system': system, 'seed': seed, 'subset': subset,
                           'baseline_accuracy': float(base.acoustic_correct.mean()),
                           'system_accuracy': float(current.acoustic_correct.mean()),
                           'delta_accuracy': float(current.acoustic_correct.mean() - base.acoustic_correct.mean()),
                           'improved': improved, 'harmed': harmed, 'mcnemar_exact_p': p})
    pd.DataFrame(paired).to_csv(PAIRED, index=False)


def main():
    manifest = parse_manifest()
    completed = set()
    if RUNS.exists():
        completed = {(row['system'], int(row['seed'])) for row in map(json.loads, RUNS.read_text(encoding='utf-8').splitlines())}
    model = KimiAudio(model_path=str(ROOT / 'models/Kimi-Audio-7B-Instruct'), load_detokenizer=False)
    for parameter in model.alm.parameters():
        parameter.requires_grad_(False)
    fusion = [module for module in model.alm.modules() if hasattr(module, 'vq_adaptor')][0]
    prompt = "Classify only the speaker's vocal emotion. Choose exactly one: A=neutral, B=happy, C=sad, D=angry. Reply with only A, B, C, or D."
    token_ids = torch.tensor([model.prompt_manager.text_tokenizer.encode(CHOICES[label], bos=False, eos=False)[0] for label in LABELS], device='cuda')

    cache = {}
    for index, row in enumerate(manifest, 1):
        item = model.prompt_manager.get_prompt([
            {'role': 'user', 'message_type': 'text', 'content': prompt},
            {'role': 'user', 'message_type': 'audio', 'content': row['audio_path']},
        ], output_type='text')
        audio, text, mask, _, _ = item.to_tensor()
        cache[row['filename']] = (audio.cpu(), text.cpu(), mask.cpu(), item.continuous_feature[0].cpu())
        if index % 104 == 0:
            print(f'CACHE {index}/1248', flush=True)

    def predict(row):
        audio, text, mask, feature = cache[row['filename']]
        audio, text, mask, feature = audio.cuda(), text.cuda(), mask.cuda(), feature.cuda()
        positions = torch.arange(audio.shape[1], device='cuda').unsqueeze(0).long()
        output = model.alm(input_ids=audio, text_input_ids=text,
                           whisper_input_feature=[feature], is_continuous_mask=mask,
                           position_ids=positions, use_cache=False, return_dict=True)
        values = torch.log_softmax(output.logits[1][0, -1].index_select(0, token_ids).float(), dim=-1)
        scores = {label: float(values[i].cpu()) for i, label in enumerate(LABELS)}
        return max(scores, key=scores.get), scores

    systems = [(f'branch_{mode}', 0, mode) for mode in ['full', 'no_continuous', 'no_discrete', 'rolled_continuous']]
    systems += [(method, seed, 'full') for method in ['decision_adapter_layer25', 'full_sequence_adapter_layer22', 'ordinary_lora_layer22'] for seed in SEEDS]
    adapter = Adapter(model.alm.config.hidden_size).cuda()
    attention = model.alm.model.layers[22].self_attn
    original_q_proj = attention.q_proj

    for system, seed, mode in systems:
        if (system, seed) in completed:
            print(f'SKIP {system} {seed}', flush=True)
            continue
        fusion.kfair_branch_mode = mode
        handle = None
        lora = None
        if not system.startswith('branch_'):
            checkpoint = torch.load(OUT / f'actor_cv6_checkpoints/fold6_{system}_seed{seed}.pt', map_location='cpu', weights_only=True)
            if system.startswith('decision'):
                adapter.load_state_dict(checkpoint['state_dict']); adapter.eval()
                def hook(_module, args):
                    hidden = args[0].clone(); hidden[:, -1] = hidden[:, -1] + adapter(hidden[:, -1]); return (hidden,) + args[1:]
                handle = model.alm.model.layers[25].register_forward_pre_hook(hook)
            elif system.startswith('full'):
                adapter.load_state_dict(checkpoint['state_dict']); adapter.eval()
                def hook(_module, args):
                    return (args[0] + adapter(args[0]),) + args[1:]
                handle = model.alm.model.layers[22].register_forward_pre_hook(hook)
            else:
                lora = LoRALinear(original_q_proj).cuda()
                lora.A.load_state_dict(checkpoint['state_dict']['A'])
                lora.B.load_state_dict(checkpoint['state_dict']['B'])
                lora.eval(); attention.q_proj = lora

        rows = []
        started = time.time()
        with torch.inference_mode():
            for index, row in enumerate(manifest, 1):
                prediction, scores = predict(row)
                result = {**row, 'system': system, 'seed': seed, 'branch_mode': mode,
                          'prediction': prediction, 'scores': scores,
                          'acoustic_margin': scores[row['acoustic_label']] - scores[row['semantic_label']]}
                append(RAW, result); rows.append(result)
                if index % 104 == 0:
                    print(f'EMIS {system} {seed} {index}/1248', flush=True)
        frame = pd.DataFrame(rows)
        record = {'system': system, 'seed': seed, 'branch_mode': mode, 'n': len(frame),
                  'seconds': time.time() - started,
                  'all_accuracy': float(frame.prediction.eq(frame.acoustic_label).mean()),
                  'conflict_acoustic_follow': float(frame[frame.conflict].prediction.eq(frame[frame.conflict].acoustic_label).mean()),
                  'conflict_semantic_follow': float(frame[frame.conflict].prediction.eq(frame[frame.conflict].semantic_label).mean()),
                  'aligned_accuracy': float(frame[~frame.conflict].prediction.eq(frame[~frame.conflict].acoustic_label).mean())}
        append(RUNS, record); print('RESULT ' + json.dumps(record), flush=True)
        if handle is not None: handle.remove()
        if lora is not None: attention.q_proj = original_q_proj; del lora
        torch.cuda.empty_cache()

    summarize()
    PROVENANCE.write_text(json.dumps({
        'source': 'EMIS official Zenodo record 19207001',
        'paper': 'arXiv:2510.25054',
        'archive': 'audios_EMIS.zip',
        'archive_bytes': 318538812,
        'archive_md5': 'a78a40ce73eae28dabafc1d16ec703bc',
        'text_csv': 'text_samples_EMIS.csv',
        'text_csv_md5': '84a37ca7ae97859c809e50938b3b4b6f',
        'license': 'GPL-3.0-or-later (Zenodo metadata)',
        'total': 1248, 'conflict': 936, 'aligned': 312,
        'generators': {'COSY': 416, 'F5TTS': 416, 'STYLE': 416},
        'semantic_labels': {label: 312 for label in LABELS},
        'acoustic_labels': {label: 312 for label in LABELS},
        'text_ids': 26, 'voice_ids': 10,
        'repair_checkpoint_rule': 'actor_cv6 fold6, fixed before external EMIS evaluation',
        'seeds': SEEDS,
    }, indent=2), encoding='utf-8')
    print(pd.read_csv(RUNS, lines=True) if False else 'EMIS_FULL1248_DONE', flush=True)


if __name__ == '__main__':
    main()
