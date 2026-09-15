import json
from collections import Counter
from pathlib import Path

root = Path('/root/autodl-tmp/kfair/outputs')

def load(name):
    return [json.loads(x) for x in (root / name).read_text().splitlines() if x.strip()]

q = load('qwen_parameter_matched_predictions.jsonl')
r = load('realistic_robustness_predictions.jsonl')
a = load('expanded_asr_preservation_predictions.jsonl')

checks = {
    'qwen_rows': len(q),
    'qwen_unique': len({(x['method'], x['seed'], x['split'], x['filename']) for x in q}),
    'qwen_method_seed_split': dict(Counter(f"{x['method']}|{x['seed']}|{x['split']}" for x in q)),
    'robust_rows': len(r),
    'robust_unique': len({(x['method'], x['seed'], x['condition'], x['filename']) for x in r}),
    'robust_combinations': len({(x['method'], x['seed'], x['condition']) for x in r}),
    'asr_rows': len(a),
    'asr_unique': len({(x['method'], x['seed'], x['corpus'], x['filename']) for x in a}),
    'asr_combinations': len({(x['method'], x['seed'], x['corpus']) for x in a}),
    'asr_empty_transcripts': sum(not x['transcript'].strip() for x in a),
}
assert checks['qwen_rows'] == checks['qwen_unique'] == 1344
assert checks['robust_rows'] == checks['robust_unique'] == 1920
assert checks['robust_combinations'] == 60
assert checks['asr_rows'] == checks['asr_unique'] == 3840
assert checks['asr_combinations'] == 20
assert checks['asr_empty_transcripts'] == 0
(root / 'low_priority_validation.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(checks, ensure_ascii=False, indent=2))
