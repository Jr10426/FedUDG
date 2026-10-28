import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

from fedudg.config import DOMAINNET_DOMAINS, JOINT_SETTINGS, PACS_DOMAINS, VLCS_DOMAINS
from fedudg.io import atomic_json


ROOT = Path(__file__).resolve().parent
ACTIVE = set()
LOCK = threading.Lock()
STOP = threading.Event()


def entries(args):
    selected = ('pacs_domainnet', 'domainnet', 'vlcs') if args.datasets == 'all' else tuple(args.datasets.split(','))
    if len(set(selected)) != len(selected) or any(x not in ('pacs_domainnet', 'domainnet', 'vlcs') for x in selected):
        raise ValueError('Choose datasets from pacs_domainnet,domainnet,vlcs')
    seeds = [int(value) for value in args.seeds.split(',')]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be distinct')
    planned = []
    for dataset in selected:
        domains = PACS_DOMAINS if dataset == 'pacs_domainnet' else (
            DOMAINNET_DOMAINS if dataset == 'domainnet' else VLCS_DOMAINS)
        for seed in seeds:
            for target in domains:
                settings = JOINT_SETTINGS if dataset == 'pacs_domainnet' else ('within_domain',)
                for setting in settings:
                    alphas = (0.5,) if dataset == 'pacs_domainnet' else (10.0, 1.0, 0.1)
                    for alpha in alphas:
                        tag = '{}_{}_a{}_s{}'.format(target, setting, format(alpha, 'g'), seed)
                        table = 'table2' if dataset == 'pacs_domainnet' else ('table3' if dataset == 'domainnet' else 'table4')
                        planned.append({'dataset': dataset, 'target_domain': target, 'setting': setting,
                                        'alpha': alpha, 'seed': seed, 'name': tag,
                                        'output': str(Path(args.output_dir).resolve() / table / tag)})
    return planned


def child(command, log):
    if STOP.is_set():
        raise RuntimeError('Suite interrupted')
    process = subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=(os.name != 'nt'))
    with LOCK:
        ACTIVE.add(process)
    try:
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    finally:
        with LOCK:
            ACTIVE.discard(process)


def run_one(entry, args):
    if STOP.is_set():
        return
    output = Path(entry['output'])
    expected = {'dataset': entry['dataset'], 'target_domain': entry['target_domain'],
                'setting': entry['setting'], 'seed': entry['seed'], 'rounds': args.rounds,
                'batch_size': args.batch_size, 'dirichlet_alpha': entry['alpha']}
    config_path = output / 'config.json'
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding='utf-8'))
        if any(saved[key] != value for key, value in expected.items()):
            raise ValueError('Configuration changed for {}'.format(output))
    if (output / 'test_results.json').is_file() and (output / 'train_summary.json').is_file():
        print('[skipped] {}'.format(entry['name']), flush=True)
        return
    log_path = Path(args.output_dir).resolve() / '_logs' / (entry['name'] + '.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        if not (output / 'train_summary.json').is_file():
            command = [sys.executable, '-u', str(ROOT / 'train.py'),
                       '--dataset', entry['dataset'], '--target_domain', entry['target_domain'],
                       '--setting', entry['setting'], '--dirichlet_alpha', str(entry['alpha']),
                       '--seed', str(entry['seed']), '--rounds', str(args.rounds),
                       '--batch_size', str(args.batch_size), '--num_workers', str(args.num_workers),
                       '--device', args.device, '--data_dir', str(Path(args.data_dir).resolve()),
                       '--output_dir', str(output)]
            recovery = output / 'last.pth'
            if recovery.is_file():
                command.extend(['--resume', str(recovery)])
                print('[resume] {}'.format(entry['name']), flush=True)
            else:
                print('[train] {}'.format(entry['name']), flush=True)
            child(command, log)
        print('[test] {}'.format(entry['name']), flush=True)
        child([sys.executable, '-u', str(ROOT / 'test.py'), '--checkpoint', str(output / 'best.pth'),
               '--data_dir', str(Path(args.data_dir).resolve()), '--device', args.device,
               '--batch_size', str(args.batch_size), '--num_workers', str(args.num_workers)], log)
    log_path.unlink()
    print('[completed] {}'.format(entry['name']), flush=True)


def summarize(planned, root):
    rows = []
    for entry in planned:
        result = Path(entry['output']) / 'test_results.json'
        if result.is_file():
            values = json.loads(result.read_text(encoding='utf-8'))
            rows.append({name: values[name] for name in
                         ('dataset', 'target_domain', 'setting', 'dirichlet_alpha',
                          'seed', 'round', 'accuracy', 'accuracy_percent', 'source_validation')})
    path = root / 'results.csv'
    if rows:
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    atomic_json({'planned': len(planned), 'completed': len(rows)}, root / 'status.json')
    return len(rows)


def stop_children():
    STOP.set()
    with LOCK:
        processes = list(ACTIVE)
    for process in processes:
        if process.poll() is not None:
            continue
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def main():
    parser = argparse.ArgumentParser(description='Train and test the FedUDG entries in Tables II-IV')
    parser.add_argument('--datasets', default='all')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--output_dir', default='runs')
    parser.add_argument('--seeds', default='1337')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--rounds', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_parallel', type=int, default=1)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if args.max_parallel < 1:
        parser.error('--max_parallel must be positive')
    planned = entries(args)
    if args.plan:
        for entry in planned:
            print(entry)
        print('{} train/test pairs'.format(len(planned)))
        return
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(planned, root / 'manifest.json')
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel)
    failed = []
    try:
        futures = {pool.submit(run_one, entry, args): entry for entry in planned}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failed.append(futures[future]['name'])
                print('[failed] {}: {}'.format(futures[future]['name'], error), flush=True)
            summarize(planned, root)
    except KeyboardInterrupt:
        stop_children()
        raise
    finally:
        pool.shutdown(wait=True)
        completed = summarize(planned, root)
        print('Completed: {}/{}; results: {}'.format(completed, len(planned), root), flush=True)
    if failed:
        raise RuntimeError('{} entries failed; inspect _logs'.format(len(failed)))


if __name__ == '__main__':
    main()
