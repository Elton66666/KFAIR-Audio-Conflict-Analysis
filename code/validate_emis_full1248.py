import json
from pathlib import Path

import numpy as np
import pandas as pd

root = Path('/root/autodl-tmp/kfair')
out = root / 'outputs'
rows = [json.loads(line) for line in (out / 'emis_full1248_predictions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
frame = pd.DataFrame(rows)
keys = frame[['system', 'seed', 'filename']].drop_duplicates()
assert len(frame) == len(keys) == 16224
assert frame.filename.nunique() == 1248
assert len(frame[['system', 'seed']].drop_duplicates()) == 13
assert not frame.prediction.isna().any()
assert set(frame.prediction.unique()) <= {'neutral', 'happy', 'sad', 'angry'}
assert int(frame.conflict.sum()) == 13 * 936
assert len(list((root / 'data/emis_full1248').glob('*.wav'))) == 1248

frame['acoustic_correct'] = frame.prediction.eq(frame.acoustic_label)
frame['semantic_follow'] = frame.prediction.eq(frame.semantic_label)
aggregate = []
for system, group in frame.groupby('system'):
    by_seed = []
    for seed, run in group.groupby('seed'):
        conflict = run[run.conflict]
        aligned = run[~run.conflict]
        by_seed.append({
            'seed': int(seed),
            'overall_accuracy': float(run.acoustic_correct.mean()),
            'conflict_acoustic_follow': float(conflict.acoustic_correct.mean()),
            'conflict_semantic_follow': float(conflict.semantic_follow.mean()),
            'aligned_accuracy': float(aligned.acoustic_correct.mean()),
            'conflict_margin': float(conflict.acoustic_margin.mean()),
        })
    item = {'system': system, 'runs': len(by_seed), 'by_seed': by_seed}
    for metric in ['overall_accuracy', 'conflict_acoustic_follow', 'conflict_semantic_follow', 'aligned_accuracy', 'conflict_margin']:
        values = np.array([row[metric] for row in by_seed])
        item[metric + '_mean'] = float(values.mean())
        item[metric + '_std'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    aggregate.append(item)

paired = pd.read_csv(out / 'emis_full1248_paired_effects.csv')
validation = {
    'rows': len(frame),
    'unique_keys': len(keys),
    'audio_files': frame.filename.nunique(),
    'systems': len(frame[['system', 'seed']].drop_duplicates()),
    'empty_predictions': int(frame.prediction.isna().sum()),
    'all_repair_conflict_mcnemar_p_below_0_05': bool((paired[paired.subset.eq('conflict')].mcnemar_exact_p < .05).all()),
    'aggregate': aggregate,
}
(out / 'emis_full1248_analysis.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(validation, ensure_ascii=False, indent=2))
