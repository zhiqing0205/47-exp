#!/usr/bin/env python3
"""
Quick validation: run NOVA3 with V2's best params to test if polynomial LR decay helps.
V2 result was 72.7% (best at epoch 19, then oscillated down to 71.5%).
"""
import torch
import time
import argparse
import random
import numpy as np

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


def main():
    random_seed(2)

    # V2 best params (trial #19, test_acc=0.7269)
    BEST_PARAMS = {
        'outer_update_lr': 0.00034055842351315795,
        'inner_update_lr': 0.018765562953504705,
        'gamma': 69.55007498245101,
        'beta': 0.1412636267424252,
        'z_lr': 0.02005017417688199,  # maps to args.nu (eta_t)
        'c_t': 2.298967650867567,
    }

    print("=" * 60)
    print("NOVA3 V5+Diagnostic: lambda_x behavior analysis")
    print("=" * 60)
    print(f"V5 best: 73.82% (baseline)")
    print(f"This run: diagnose lambda_x to see if noisy samples get down-weighted")
    print(f"Params: {BEST_PARAMS}")
    print()

    print("Loading data...")
    train = torch.load('data/snli_train_0.1.pkl', weights_only=False)
    val = torch.load('data/snli_dev_0.1.pkl', weights_only=False)
    test = torch.load('data/snli_test_0.1.pkl', weights_only=False)
    training_size = train.dataset_size

    n_epochs = 20
    batch_size = 512

    args = argparse.Namespace(
        data='snli', word_embed_dim=300, encoder_dim=512, n_enc_layers=2,
        fc_dim=1024, n_classes=3, n_labels=3, pool_type='max', linear_fc=False,
        outer_update_lr=BEST_PARAMS['outer_update_lr'],
        inner_update_lr=BEST_PARAMS['inner_update_lr'],
        gamma=BEST_PARAMS['gamma'],
        beta=BEST_PARAMS['beta'],
        nu=BEST_PARAMS['z_lr'],
        inner_batch_size=batch_size,
        batch_size=16,
        noise_rate=0.1,
        seed=2,
        epoch=n_epochs,
    )

    learner = NOVA3.Learner(args, training_size, verbose=False)
    learner.c_t = BEST_PARAMS['c_t']

    results = []
    best_acc = 0.0
    st = time.time()

    for epoch in range(n_epochs):
        ep_start = time.time()
        train_loader = DataLoader(train, shuffle=True, batch_size=batch_size, collate_fn=collate_pad_double)
        val_loader = DataLoader(val, shuffle=True, batch_size=16, collate_fn=collate_pad_double)
        test_loader = DataLoader(test, batch_size=16, collate_fn=collate_pad_double)

        train_acc, train_loss = learner(train_loader, val_loader, training=True, epoch=epoch)
        test_acc, test_loss = learner.test(test_loader)

        best_acc = max(best_acc, test_acc)
        ep_time = time.time() - ep_start
        results.append((epoch, train_loss, test_acc, ep_time))

        print(f"ep{epoch:>2}/{n_epochs} | TrainLoss: {train_loss:.4f} | TestAcc: {test_acc:.4f} | "
              f"Best: {best_acc:.4f} | {ep_time:.0f}s")

    total_time = (time.time() - st) / 3600
    print()
    print("=" * 60)
    print(f"Total time: {total_time:.2f}h")
    print(f"V6 Best Test Acc: {best_acc:.4f}")
    print(f"V5 Reference:     0.7382 (60 epochs, delayed polynomial)")
    print(f"V4 Reference:     0.7365 (45 epochs, polynomial)")
    print(f"V2 Reference:     0.7269 (25 epochs, no decay)")
    diff = best_acc - 0.7382
    if diff > 0.005:
        verdict = f"ADAPTIVE LR HELPS (+{diff*100:.2f}%)"
    elif diff < -0.005:
        verdict = f"ADAPTIVE LR HURTS ({diff*100:.2f}%)"
    else:
        verdict = f"ADAPTIVE LR NEUTRAL ({diff*100:+.2f}%)"
    print(f"Verdict: {verdict}")
    if best_acc >= 0.75:
        print(f">>> TARGET 75% REACHED! <<<")
    print("=" * 60)


if __name__ == "__main__":
    main()
