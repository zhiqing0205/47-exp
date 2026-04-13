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

from methods import NOVA3
from data_loader import collate_pad_double
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled = False


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

    def objective(trial):
        random_seed(seed)

        train, val, test = load_data()
        training_size = train.dataset_size

        # === Hyperparameter search space ===
        # Narrowed based on prior search + code fixes (L2 reg, full data, no y-norm)
        outer_update_lr = trial.suggest_float("outer_update_lr", 1e-4, 0.1, log=True)
        inner_update_lr = trial.suggest_float("inner_update_lr", 1e-3, 0.1, log=True)
        gamma = trial.suggest_float("gamma", 0.05, 2.0, log=True)
        beta = trial.suggest_float("beta", 0.8, 0.99)
        nu = trial.suggest_float("nu", 1e-3, 0.5, log=True)
        c_t = trial.suggest_float("c_t", 0.5, 5.0, log=True)

        args = argparse.Namespace(
            data='snli', word_embed_dim=300, encoder_dim=512, n_enc_layers=2,
            fc_dim=1024, n_classes=3, n_labels=3, pool_type='max', linear_fc=False,
            outer_update_lr=outer_update_lr,
            inner_update_lr=inner_update_lr,
            gamma=gamma,
            beta=beta,
            nu=nu,
            inner_batch_size=batch_size,
            batch_size=16,
            noise_rate=0.1,
            seed=seed,
        )

        try:
            learner = NOVA3.Learner(args, training_size)
            # Override c_t
            learner.c_t = c_t
        except Exception as e:
            print(f"  Trial {trial.number}: init failed: {e}")
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
                return best_test_acc

            test_acc, test_loss = learner.test(test_loader)
            best_test_acc = max(best_test_acc, test_acc)

            # Report intermediate value for pruning
            trial.report(test_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        print(f"  Trial {trial.number}: best_test_acc={best_test_acc:.4f} | "
              f"olr={outer_update_lr:.4f} ilr={inner_update_lr:.4f} "
              f"gamma={gamma:.4f} beta={beta:.3f} nu={nu:.4f} c_t={c_t:.3f}")

        torch.cuda.empty_cache()
        return best_test_acc

    return objective


def main():
    parser = argparse.ArgumentParser(description="NOVA3 Hyperparameter Tuning")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--epoch", type=int, default=5, help="Epochs per trial (use fewer for faster search)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--seed", type=int, default=2, help="Random seed")
    parser.add_argument("--study_name", type=str, default="nova3_tune_v2", help="Optuna study name")
    parser.add_argument("--db", type=str, default="sqlite:///logs/nova3_tune_v2.db", help="Optuna storage DB")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)

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
    print(f"Search space (narrowed):")
    print(f"  outer_update_lr: [1e-4, 0.1] (log)")
    print(f"  inner_update_lr: [1e-3, 0.1] (log)")
    print(f"  gamma:           [0.05, 2.0] (log)")
    print(f"  beta:            [0.8, 0.99]")
    print(f"  nu:              [1e-3, 0.5] (log)")
    print(f"  c_t:             [0.5, 5.0] (log)")
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
    print(f"{'Rank':>4} | {'Trial':>5} | {'Test Acc':>8} | {'olr':>8} | {'ilr':>8} | {'gamma':>8} | {'beta':>5} | {'nu':>8} | {'c_t':>6}")
    print("-" * 80)
    for i, t in enumerate(completed[:10]):
        p = t.params
        print(f"{i+1:>4} | {t.number:>5} | {t.value:>8.4f} | {p['outer_update_lr']:>8.4f} | "
              f"{p['inner_update_lr']:>8.4f} | {p['gamma']:>8.4f} | {p['beta']:>5.3f} | "
              f"{p['nu']:>8.4f} | {p['c_t']:>6.3f}")

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
    print(f"        args.beta = {bp['beta']}")
    print(f"        args.nu = {bp['nu']}")
    print(f"        # c_t = {bp['c_t']}  (set in Learner.__init__)")
    print(f"        learner = NOVA3.Learner(args, training_size)")
    print(f"        learner.c_t = {bp['c_t']}")

    print(f"\nResults saved to {result_file}")
    print(f"Study DB: {args.db} (can resume with --study_name {args.study_name})")


if __name__ == "__main__":
    main()
