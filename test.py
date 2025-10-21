import yaml
from pathlib import Path

res = {'root_path': 'motion_dataset',
       'motions': []}
dataset_dir = 'retargeted_data'
for pkl_path in sorted(list(Path(dataset_dir).rglob('*.pkl'))):
    motion_info = {
        'file': pkl_path.relative_to(dataset_dir).as_posix(),
        'weight': 1.0,
        'description': 'general movement'
    }
    res['motions'].append(motion_info)
with open('motion_dataset.yaml', 'w') as f:
    yaml.dump(res, f)