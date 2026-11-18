"""Run the four-dataset desc20 suite: Base-to-New, Cross-Datasets and SSMT.

Seeds 1/2/3; PatternNet is the transfer source, PatternNetv2 the SSMT source.
18 training runs + 42 explicit evaluations, sequentially on the selected GPU.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('patternnet', 'rsicd', 'resisc45', 'mlrsnet')
FOLDERS = dict(zip(DATASETS + tuple(d + 'v2' for d in DATASETS),
                   ('PatternNet', 'RSICD', 'RESISC45', 'MLRSNet',
                    'PatternNetv2', 'RSICDv2', 'RESISC45v2', 'MLRSNetv2')))
SEEDS = (1, 2, 3)


def load_launcher():
    spec = importlib.util.spec_from_file_location('single_experiment_launcher', ROOT / 'scripts/run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_weights(weights):
    if weights is not None:
        weights = weights.expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(f'Missing CLIP weights: {weights}')
        return weights
    sys.path.insert(0, str(ROOT))
    from clip import clip
    caches = [ROOT / '.cache/clip', Path.home() / '.cache/clip']
    cache = next((path for path in caches if (path / 'ViT-B-16.pt').is_file()), caches[0])
    print(f'Automatic CLIP ViT-B/16 weights: {cache / "ViT-B-16.pt"}', flush=True)
    print('Using a verified cache if available; otherwise downloading official weights.', flush=True)
    return Path(clip._download(clip._MODELS['ViT-B/16'], root=str(cache))).resolve()


def dataset_roots(args):
    roots = {}
    overrides = {}
    for value in args.dataset_root:
        name, sep, path = value.partition('=')
        if not sep or name not in FOLDERS or name in overrides:
            raise ValueError('--dataset-root must be a unique DATASET=PARENT_PATH entry')
        overrides[name] = Path(path).expanduser().resolve()
    for name, folder in FOLDERS.items():
        candidates = [args.root]
        if name == 'patternnet':
            candidates.append(args.root / 'APPLeNet_PN')
        elif name == 'rsicd':
            candidates.append(args.root / 'APPLeNet_RSICD')
        elif name == 'resisc45':
            candidates.append(args.root / 'NWPU Data Set')
        matches = [parent for parent in candidates if (parent / folder / 'images').is_dir()]
        if name in overrides:
            roots[name] = overrides[name]
        elif len(matches) == 1:
            roots[name] = matches[0]
        elif len(matches) > 1:
            raise ValueError(f'Multiple copies of {folder}; choose --dataset-root {name}=PARENT_PATH')
        elif args.dry_run:
            roots[name] = args.root
        else:
            raise FileNotFoundError(f'Missing {folder}/images under {args.root}; use --dataset-root {name}=PARENT_PATH if needed')
    return roots


def make_plan(args, roots):
    launcher = load_launcher()
    jobs = []

    def add(experiment, protocol, phase, dataset, seed, source=None, subset=None):
        training_dataset = source or dataset
        config = f'vit_b16_c16_{training_dataset}_desc20'
        argv = [protocol, phase, dataset, str(seed), '--root', str(roots[dataset]),
                '--config', config,
                '--output-root', str(args.output_root)]
        if args.weights is not None:
            argv += ['--weights', str(args.weights)]
        if source:
            argv += ['--source', source]
        if subset:
            argv += ['--subset', subset]
        parsed = launcher.make_parser().parse_args(argv)
        # Preserve the SOURCE config tag for checkpoint lookup, but always
        # compute teacher features from the TARGET dataset's descriptions.
        overrides = ['TRAINER.HICROPL_NF.TEXT_PROMPTS', str(ROOT / 'gpt_files' / f'{dataset}_prompt.json')]
        command = launcher.build_command(parsed, overrides)
        command.insert(1, '-u')
        job_id = '_'.join(str(v) for v in (experiment, phase, source or dataset,
                                          dataset if source else None, subset, seed) if v is not None)
        jobs.append(dict(id=job_id, experiment=experiment, phase=phase, source=source or dataset,
                         target=dataset, seed=seed, subset=subset,
                         output=command[command.index('--output-dir') + 1], command=command))

    for dataset in DATASETS:
        for seed in SEEDS:
            add('base2new', 'base2new', 'train', dataset, seed)
            for subset in ('base', 'new'):
                add('base2new', 'base2new', 'test', dataset, seed, subset=subset)
    for experiment, protocol, suffix in [('crossdatasets', 'crossdata', ''), ('SSMT', 'domaingen', 'v2')]:
        source = 'patternnet' + suffix
        for seed in SEEDS:
            add(experiment, protocol, 'train', source, seed)
            for target in DATASETS[1:]:
                add(experiment, protocol, 'test', target + suffix, seed, source=source)
    return jobs


def preflight(args, roots):
    if not args.weights.is_file():
        raise FileNotFoundError(f'Missing CLIP weights: {args.weights}')
    files = [args.weights]
    v2_names = []
    for name, folder in FOLDERS.items():
        base = roots[name] / folder
        if not (base / 'images').is_dir():
            raise FileNotFoundError(base / 'images')
        split = base / f'{name}.json'
        if not split.is_file():
            split = ROOT / 'json' / f'{name}.json'
        data = json.loads(split.read_text(encoding='utf-8-sig'))
        if set(data) != {'train', 'val', 'test'} or not all(data.values()):
            raise ValueError(f'Invalid/empty split: {split}')
        classes = {row[2].replace('_', ' ') for group in data.values() for row in group}
        if name.endswith('v2'):
            if len(classes) != 16:
                raise ValueError(f'SSMT needs 16 shared classes: {name}')
            v2_names.append(classes)
        descriptions = ROOT / 'gpt_files' / f'{name}_prompt.json'
        texts = json.loads(descriptions.read_text(encoding='utf-8-sig'))
        missing = classes - set(texts)
        if missing or any(not isinstance(texts[c], list) or len(texts[c]) != 20
                          or not all(isinstance(t, str) and t.strip() for t in texts[c]) for c in classes):
            raise ValueError(f'Missing/invalid desc20 teacher text: {descriptions}; missing={missing}')
        config = ROOT / 'yaml/trainers/HiCroPLNoFusion' / f'vit_b16_c16_{name}_desc20.yaml'
        if not config.is_file():
            raise FileNotFoundError(config)
        files += [split, descriptions, config]
        print(f'Data: {name} -> {base / "images"}', flush=True)
    if any(names != v2_names[0] for names in v2_names[1:]):
        raise ValueError('SSMT v2 datasets do not share the same 16 class names')
    files += [p for dirname in ['models', 'clip', 'dassl', 'files'] for p in (ROOT / dirname).rglob('*.py')]
    files += [ROOT / 'train.py', ROOT / 'scripts/run.py', Path(__file__).resolve()]
    files += list((ROOT / 'yaml/datasets').glob('*.yaml'))
    hashes = {}
    for path in sorted(set(files)):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        hashes[str(path)] = digest.hexdigest()
    return hashes


def save_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_score(directory):
    logs = list(Path(directory).glob('log.txt*'))
    if not logs:
        raise FileNotFoundError(f'No evaluation log in {directory}')
    log = max(logs, key=lambda path: path.stat().st_mtime_ns)
    matches = re.findall(r'\* total: ([\d,]+)\s*\* correct: ([\d,]+)\s*\* accuracy: ([\d.]+)%',
                         log.read_text(encoding='utf-8-sig', errors='replace'))
    if not matches:
        raise ValueError(f'No completed evaluation result: {log}')
    total, correct, _ = matches[-1]
    total, correct = int(total.replace(',', '')), int(correct.replace(',', ''))
    if not 0 <= correct <= total or total == 0:
        raise ValueError(f'Invalid evaluation counts: {log}')
    return dict(accuracy=100 * correct / total, total=total, correct=correct, log=str(log))


def summarize(out, jobs, state):
    rows = []
    base = defaultdict(dict)
    for job in jobs:
        record = state.get(job['id'], {})
        if job['phase'] != 'test' or record.get('status') != 'completed':
            continue
        common = {k: job[k] for k in ['experiment', 'source', 'target', 'seed']}
        if job['experiment'] == 'base2new':
            key = (job['target'], job['seed'])
            base[key][job['subset']] = record['score']['accuracy']
        else:
            rows.append(dict(common, accuracy=record['score']['accuracy']))
    for (dataset, seed), values in base.items():
        if set(values) == {'base', 'new'}:
            b, n = values['base'], values['new']
            rows.append(dict(experiment='base2new', source=dataset, target=dataset, seed=seed,
                             base=b, new=n, H=2*b*n/(b+n) if b+n else 0.0))
    fields = ['experiment', 'source', 'target', 'seed', 'base', 'new', 'H', 'accuracy']
    with (out / 'results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    grouped = defaultdict(list)
    for row in rows:
        for metric in ['base', 'new', 'H', 'accuracy']:
            if metric in row:
                grouped[(row['experiment'], row['source'], row['target'], metric)].append(row[metric])
    with (out / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['experiment', 'source', 'target', 'metric', 'seeds_completed', 'mean', 'sample_std']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (experiment, source, target, metric), values in sorted(grouped.items()):
            writer.writerow(dict(experiment=experiment, source=source, target=target, metric=metric,
                                 seeds_completed=len(values), mean=statistics.mean(values),
                                 sample_std=statistics.stdev(values) if len(values) > 1 else ''))


def run_suite(args, jobs, fingerprints):
    out = args.output_root
    plan = dict(teacher='desc20', shots=16, seeds=list(SEEDS), jobs=jobs, sha256=fingerprints)
    if args.resume:
        previous = json.loads((out / 'run_plan.json').read_text(encoding='utf-8'))
        if previous != plan:
            raise ValueError('Run settings, data splits, descriptions, weights or code changed; choose a new --output-root')
        state = json.loads((out / 'status.json').read_text(encoding='utf-8'))
    else:
        out.mkdir(parents=True, exist_ok=False)
        state = {}
        save_json(out / 'run_plan.json', plan)
        save_json(out / 'status.json', state)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    for index, job in enumerate(jobs, 1):
        record = state.get(job['id'], {})
        if record.get('status') == 'completed':
            if not Path(record['artifact']).is_file():
                raise FileNotFoundError(f'Completed job artifact missing: {record["artifact"]}')
            print(f'[{index}/{len(jobs)}] SKIP completed: {job["id"]}', flush=True)
            continue
        folder = Path(job['output'])
        # Keep completed runs. For an interrupted training run, restart with
        # the same seed from scratch, rather than silently resuming with lost
        # RNG/best-val state. The old incomplete output is preserved separately.
        if folder.exists() and job['phase'] == 'train':
            archive = folder.with_name(folder.name + '.interrupted_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
            folder.rename(archive)
            print(f'Preserved incomplete training at {archive}; restarting this training run.', flush=True)
        print(f'[{index}/{len(jobs)}] {job["id"]}\n{shlex.join(job["command"])}', flush=True)
        state[job['id']] = {'status': 'running'}
        save_json(out / 'status.json', state)
        try:
            subprocess.run(job['command'], cwd=ROOT, env=env, check=True)
            if job['phase'] == 'train':
                artifact = folder / 'prompt_learner/model-best.pth.tar'
                if not artifact.is_file():
                    raise FileNotFoundError(artifact)
                result = dict(status='completed', artifact=str(artifact))
            else:
                score = read_score(folder)
                result = dict(status='completed', artifact=score['log'], score=score)
            state[job['id']] = result
        except BaseException as error:
            state[job['id']] = dict(status='failed', error=str(error))
            save_json(out / 'status.json', state)
            summarize(out, jobs, state)
            raise
        save_json(out / 'status.json', state)
        summarize(out, jobs, state)
    summarize(out, jobs, state)
    print(f'Completed all {len(jobs)} jobs. Results: {out / "results.csv"}; summary: {out / "summary.csv"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Server dataset parent')
    parser.add_argument('--weights', type=Path, help='Optional local weights; default: verified cache or official download')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/all_desc20_seed123')
    parser.add_argument('--dataset-root', action='append', default=[], metavar='DATASET=PARENT_PATH',
                        help='Optional per-dataset loader parent, e.g. rsicd=/data/APPLeNet_RSICD')
    parser.add_argument('--resume', action='store_true', help='Skip completed jobs; restart unfinished training from scratch')
    parser.add_argument('--dry-run', action='store_true', help='Print all 60 commands without training or writing files')
    args = parser.parse_args()
    args.root = args.root.expanduser().resolve()
    if args.weights is not None:
        args.weights = args.weights.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    roots = dataset_roots(args)
    print('desc20 | seeds 1,2,3 | 18 training runs + 42 evaluations | 16-shot, original hyperparameters', flush=True)
    if args.dry_run:
        jobs = make_plan(args, roots)
        for index, job in enumerate(jobs, 1):
            print(f'[{index}/{len(jobs)}] {job["id"]}\n{shlex.join(job["command"])}')
        return
    if args.output_root.exists() and not args.resume:
        parser.error('Output directory already exists; use a new --output-root or --resume for this same suite')
    args.weights = resolve_weights(args.weights)
    jobs = make_plan(args, roots)
    fingerprints = preflight(args, roots)
    run_suite(args, jobs, fingerprints)


if __name__ == '__main__':
    main()
