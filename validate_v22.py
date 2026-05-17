#!/usr/bin/env python3
"""Multi-seed validation with v20 best params (T92, 75.96%)."""
import torch
import time
import argparse
import random
import numpy as np
import os
from urllib.request import urlopen
from urllib.parse import quote

from methods import NOVA3
from data_loader import collate_pad_double
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled = False

BARK_KEY = ''
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.isfile(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                if k.strip() == 'BARK_KEY':
                    BARK_KEY = v.strip()


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=20)
    cli_args = parser.parse_args()

    random_seed(cli_args.seed)

    # v20 T92 best params (rounded)
    args = argparse.Namespace(
        data='snli', word_embed_dim=300, encoder_dim=512, n_enc_layers=2,
        fc_dim=1024, n_classes=3, n_labels=3, pool_type='max', linear_fc=False,
        outer_update_lr=0.0018,
        inner_update_lr=0.127,
        gamma=760,
        beta=0.34,
        mu=0.34,
        rho=0.82,
        nu_momentum=0.89,
        nu=0.0852,
        dpout_fc=0.0,
        ema_decay=0.998,
        decay_power=1.0,
        reg_coeff=0.0001,
        inner_batch_size=512,
        batch_size=64,
        noise_rate=0.1,
        seed=cli_args.seed,
        epoch=cli_args.epoch,
        gamma_anneal=False,
    )

    print(f"=== v22 validate: seed={cli_args.seed}, epoch={cli_args.epoch} ===")
    print("Loading data...")
    train = torch.load('data/snli_train_0.1.pkl', weights_only=False)
    val = torch.load('data/snli_dev_0.1.pkl', weights_only=False)
    test = torch.load('data/snli_test_0.1.pkl', weights_only=False)

    learner = NOVA3.Learner(args, train.dataset_size, verbose=False)
    learner.c_t = 7.8

    best_acc = 0.0
    st = time.time()

    for epoch in range(cli_args.epoch):
        ep_start = time.time()
        train_loader = DataLoader(train, shuffle=True, batch_size=512, collate_fn=collate_pad_double)
        val_loader = DataLoader(val, shuffle=True, batch_size=64, collate_fn=collate_pad_double)
        test_loader = DataLoader(test, batch_size=64, collate_fn=collate_pad_double)

        train_acc, train_loss = learner(train_loader, val_loader, training=True, epoch=epoch)
        test_acc, test_loss = learner.test(test_loader)
        best_acc = max(best_acc, test_acc)
        ep_time = time.time() - ep_start

        print(f"  ep{epoch:>2}/{cli_args.epoch} | Loss: {train_loss:.4f} | TestAcc: {test_acc:.4f} | Best: {best_acc:.4f} | {ep_time:.0f}s")

        if test_acc >= 0.76:
            bark_notify("NOVA3 76%!", f"seed={cli_args.seed} ep{epoch}: {test_acc:.4f}")

    total_time = (time.time() - st) / 3600
    print(f"\n  seed={cli_args.seed} | Best: {best_acc:.4f} | Time: {total_time:.2f}h")

    if best_acc >= 0.76:
        bark_notify("NOVA3 76% REACHED", f"seed={cli_args.seed}: {best_acc:.4f}")

    return best_acc


if __name__ == "__main__":
    main()
