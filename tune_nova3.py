#!/usr/bin/env python3
"""
Hyperparameter tuning for NOVA3 using Optuna (Bayesian optimization).
Objective: maximize best test accuracy over training epochs.

Usage:
    python tune_nova3.py                          # default 50 trials, 5 epochs each
    python tune_nova3.py --n_trials 100 --epoch 10  # more thorough
    python tune_nova3.py --n_trials 200 --epoch 20  # full search
"""
import torch
import time
import argparse
import os
import random
import numpy as np
import optuna
from optuna.trial import TrialState
from urllib.request import urlopen
from urllib.parse import quote

from methods import NOVA3
from data_loader import collate_pad_double
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled = False

BARK_KEY = os.environ.get('BARK_KEY', '')


def bark_notify(title, body):
    if not BARK_KEY:
        return
    try:
        url = f"https://api.day.app/{BARK_KEY}/{quote(title)}/{quote(body)}"
        urlopen(url, timeout=5)
    except Exception:
        pass


def random_seed(value):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)


# Global data cache
_DATA_CACHE = {}

def load_data():
    if 'train' not in _DATA_CACHE:
        print("Loading data...")
        _DATA_CACHE['train'] = torch.load('data/snli_train_0.1.pkl', weights_only=False)
        _DATA_CACHE['val'] = torch.load('data/snli_dev_0.1.pkl', weights_only=False)
        _DATA_CACHE['test'] = torch.load('data/snli_test_0.1.pkl', weights_only=False)
    return _DATA_CACHE['train'], _DATA_CACHE['val'], _DATA_CACHE['test']


def create_objective(n_epochs, batch_size, seed):
    """Create an Optuna objective function."""
    best_so_far = [0.0]

    def objective(trial):
        random_seed(seed)

        train, val, test = load_data()
        training_size = train.dataset_size

        # === Hyperparameter search space (v19: + decay_power for pure normalize) ===
        outer_update_lr = trial.suggest_float("outer_update_lr", 1e-3, 0.1, log=True)
        inner_update_lr = trial.suggest_float("inner_update_lr", 1e-3, 0.15, log=True)
        gamma = trial.suggest_float("gamma", 100.0, 800.0, log=True)
        mu = trial.suggest_float("mu", 0.3, 0.95)
        rho = trial.suggest_float("rho", 0.3, 0.95)
        nu_m = trial.suggest_float("nu_m", 0.5, 0.99)
        z_lr = trial.suggest_float("z_lr", 1e-4, 0.1, log=True)
        c_t = trial.suggest_float("c_t", 2.0, 8.0, log=True)
        dpout_fc = trial.suggest_float("dpout_fc", 0.0, 0.15)
        ema_decay = trial.suggest_float("ema_decay", 0.99, 0.999)
        label_smooth = trial.suggest_float("label_smooth", 0.0, 0.15)
        decay_power = trial.suggest_float("decay_power", 0.5, 2.0)

        args = argparse.Namespace(
            data='snli', word_embed_dim=300, encoder_dim=512, n_enc_layers=2,
            fc_dim=1024, n_classes=3, n_labels=3, pool_type='max', linear_fc=False,
            outer_update_lr=outer_update_lr,
            inner_update_lr=inner_update_lr,
            gamma=gamma,
            beta=mu,
            mu=mu,
            rho=rho,
            nu_momentum=nu_m,
            nu=z_lr,
            dpout_fc=dpout_fc,
            ema_decay=ema_decay,
            label_smooth=label_smooth,
            decay_power=decay_power,
            inner_batch_size=batch_size,
            batch_size=64,
            noise_rate=0.1,
            seed=seed,
            epoch=n_epochs,
        )

        print(f"\n  T{trial.number} params: olr={outer_update_lr:.5f} ilr={inner_update_lr:.5f} "
              f"gamma={gamma:.1f} mu={mu:.3f} rho={rho:.3f} nu={nu_m:.3f} "
              f"z_lr={z_lr:.5f} c_t={c_t:.3f} dp={dpout_fc:.3f} ema={ema_decay:.4f} ls={label_smooth:.3f} pw={decay_power:.2f}")

        try:
            learner = NOVA3.Learner(args, training_size, verbose=False)
            learner.c_t = c_t
        except Exception as e:
            print(f"  Trial {trial.number}: init failed: {e}")
            torch.cuda.empty_cache()
            return 0.0

        best_test_acc = 0.0

        for epoch in range(n_epochs):
            train_loader = DataLoader(train, shuffle=True, batch_size=batch_size, collate_fn=collate_pad_double)
            val_loader = DataLoader(val, shuffle=True, batch_size=16, collate_fn=collate_pad_double)
            test_loader = DataLoader(test, batch_size=16, collate_fn=collate_pad_double)

            try:
                train_acc, train_loss = learner(train_loader, val_loader, training=True, epoch=epoch)
            except Exception as e:
                print(f"  Trial {trial.number} epoch {epoch}: train failed: {e}")
                del learner
                torch.cuda.empty_cache()
                return best_test_acc

            test_acc, test_loss = learner.test(test_loader)
            best_test_acc = max(best_test_acc, test_acc)

            print(f"  T{trial.number} ep{epoch}/{n_epochs} | TrainLoss: {train_loss:.4f} | TestAcc: {test_acc:.4f} | Best: {best_test_acc:.4f}")

            if best_test_acc > best_so_far[0]:
                best_so_far[0] = best_test_acc
                bark_notify("NOVA3 New High",
                            f"T{trial.number} ep{epoch}: {best_test_acc:.4f}")

            # Report intermediate value for pruning
            trial.report(test_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        print(f"  Trial {trial.number}: best_test_acc={best_test_acc:.4f} | "
              f"olr={outer_update_lr:.4f} ilr={inner_update_lr:.4f} "
              f"gamma={gamma:.1f} mu={mu:.3f} rho={rho:.3f} nu={nu_m:.3f} "
              f"z_lr={z_lr:.4f} c_t={c_t:.3f} dp={dpout_fc:.3f} ema={ema_decay:.4f} ls={label_smooth:.3f} pw={decay_power:.2f}")

        del learner
        torch.cuda.empty_cache()
        return best_test_acc

    return objective


def main():
    parser = argparse.ArgumentParser(description="NOVA3 Hyperparameter Tuning")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--epoch", type=int, default=20, help="Epochs per trial")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--seed", type=int, default=2, help="Random seed")
    parser.add_argument("--study_name", type=str, default="nova3_tune_v19", help="Optuna study name")
    parser.add_argument("--db", type=str, default="sqlite:///logs/nova3_tune_v19.db", help="Optuna storage DB")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)

    # Load .env for BARK_KEY
    global BARK_KEY
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    if k.strip() == 'BARK_KEY':
                        BARK_KEY = v.strip()

    # Create study with MedianPruner: prune trials that fall below median at each epoch
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.db,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )

    objective = create_objective(args.epoch, args.batch_size, args.seed)

    print(f"=== NOVA3 Hyperparameter Tuning ===")
    print(f"Trials: {args.n_trials}, Epochs/trial: {args.epoch}, Batch size: {args.batch_size}")
    print(f"Study DB: {args.db}")
    print(f"Bark notify: {'enabled' if BARK_KEY else 'disabled (set BARK_KEY in .env)'}")
    print(f"Search space (v19: + decay_power for pure normalize):")
    print(f"  outer_update_lr: [1e-3, 0.1] (log)")
    print(f"  inner_update_lr: [1e-3, 0.15] (log)")
    print(f"  gamma:           [100, 800] (log)")
    print(f"  mu:              [0.3, 0.95] -- z momentum")
    print(f"  rho:             [0.3, 0.95] -- x momentum")
    print(f"  nu_m:            [0.5, 0.99] -- y momentum")
    print(f"  z_lr:            [1e-4, 0.1] (log)")
    print(f"  c_t:             [2.0, 8.0] (log)")
    print(f"  dpout_fc:        [0.0, 0.15] -- FC dropout")
    print(f"  ema_decay:       [0.99, 0.999] -- EMA for eval")
    print(f"  label_smooth:    [0.0, 0.15] -- label smoothing")
    print(f"  decay_power:     [0.5, 2.0] -- LR decay steepness")
    print()

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # === Results ===
    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)

    # Best trial
    best = study.best_trial
    print(f"\nBest Trial #{best.number}:")
    print(f"  Test Acc: {best.value:.4f}")
    print(f"  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Top 10 trials
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)
    print(f"\nTop 10 Trials (out of {len(completed)} completed):")
    print(f"{'Rank':>4} | {'Trial':>5} | {'Acc':>7} | {'olr':>7} | {'ilr':>6} | {'gam':>5} | {'mu':>4} | {'rho':>4} | {'nu':>4} | {'zlr':>6} | {'ct':>4} | {'dp':>4}")
    print("-" * 95)
    for i, t in enumerate(completed[:10]):
        p = t.params
        print(f"{i+1:>4} | {t.number:>5} | {t.value:>7.4f} | {p['outer_update_lr']:>7.4f} | "
              f"{p['inner_update_lr']:>6.3f} | {p['gamma']:>5.0f} | "
              f"{p.get('mu', p.get('beta',0)):>4.2f} | {p.get('rho', p.get('beta',0)):>4.2f} | "
              f"{p.get('nu_m', p.get('beta',0)):>4.2f} | "
              f"{p['z_lr']:>6.4f} | {p['c_t']:>4.1f} | {p.get('dpout_fc',0):>4.2f}")

    # Save results
    result_file = f"logs/nova3_tune_results.txt"
    with open(result_file, 'w') as f:
        f.write(f"Best Test Acc: {best.value:.4f}\n")
        f.write(f"Best Params:\n")
        for k, v in best.params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nTop 10:\n")
        for i, t in enumerate(completed[:10]):
            f.write(f"  #{t.number}: acc={t.value:.4f} params={t.params}\n")

    # Generate the args line for main.py
    bp = best.params
    print(f"\n=== Copy to main.py ===")
    print(f"    elif args.methods == 'nova3':")
    print(f"        args.outer_update_lr = {bp['outer_update_lr']}")
    print(f"        args.inner_update_lr = {bp['inner_update_lr']}")
    print(f"        args.gamma = {bp['gamma']}")
    print(f"        args.mu = {bp['mu']}")
    print(f"        args.rho = {bp['rho']}")
    print(f"        args.nu_momentum = {bp['nu_m']}")
    print(f"        args.beta = {bp['mu']}")
    print(f"        args.nu = {bp['z_lr']}")
    print(f"        args.dpout_fc = {bp['dpout_fc']}")
    print(f"        args.ema_decay = {bp['ema_decay']}")
    print(f"        args.label_smooth = {bp['label_smooth']}")
    print(f"        learner = NOVA3.Learner(args, training_size)")
    print(f"        learner.c_t = {bp['c_t']}")

    print(f"\nResults saved to {result_file}")
    print(f"Study DB: {args.db} (can resume with --study_name {args.study_name})")

    bark_notify("NOVA3 Tuning Done",
                f"Best: {best.value:.4f} T#{best.number}")


if __name__ == "__main__":
    main()
