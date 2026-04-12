#!/usr/bin/env python3
"""
Visualization with wall-clock time as x-axis.
Reads epoch_times_sec from result txt if available, otherwise estimates from total time.
"""
import re, os, ast, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG_DIR = "logs"
OUTPUT_DIR = "figures_time"
os.makedirs(OUTPUT_DIR, exist_ok=True)

METHOD_CONFIG = {
    "stocbio":  ("stocbio_e20.log",  "StocBio",      "#1f77b4", "--", 1.5, "o"),
    "ttsa":     ("ttsa_e20.log",     "TTSA",         "#ff7f0e", "--", 1.5, "s"),
    "saba":     ("saba_e20.log",     "SABA",         "#2ca02c", "--", 1.5, "^"),
    "ma-soba":  ("ma-soba_e20.log",  "MA-SOBA",      "#9467bd", "--", 1.5, "D"),
    "bo-rep":   ("bo-rep_e20.log",   "BO-REP",       "#8c564b", "--", 1.5, "v"),
    "sustain":  ("sustain_e20.log",  "SUSTAIN",      "#7f7f7f", "--", 1.5, "p"),
    "NOVA2":    (None,               "NOVA2 (Ours)", "#d62728", "-",  3.0, "*"),
}


def parse_log(filepath):
    """Parse e20 log file."""
    with open(filepath) as f:
        lines = f.readlines()
    epoch = -1
    train = {}
    test = {}
    for line in lines:
        m = re.match(r'\[epoch/epochs\]:(\d+)/(\d+)', line)
        if m:
            epoch = int(m.group(1))
        m = re.match(r'Step (\d+) \| Task Loss: ([\d.]+) \| Acc: ([\d.]+)', line)
        if m and epoch >= 0:
            train[epoch] = (float(m.group(2)), float(m.group(3)))
        m = re.match(r'Task loss: ([\d.]+), Task acc: ([\d.]+)', line)
        if m and epoch >= 0:
            test[epoch] = (float(m.group(1)), float(m.group(2)))
        m = re.match(r'\s+Test\s+Loss:\s+([\d.]+)\s+\|\s+Test\s+Acc:\s+([\d.]+)', line)
        if m and epoch >= 0:
            test[epoch] = (float(m.group(1)), float(m.group(2)))
    return train, test


def parse_nova2_txt():
    files = glob.glob(os.path.join(LOG_DIR, "NOVA2_*.txt"))
    if not files:
        return None, None, None
    with open(files[0]) as f:
        raw = f.read()
    data = ast.literal_eval(_clean_np_str(raw))
    acc_str = data['AVG Train ACC']
    train_acc = ast.literal_eval(_clean_np_str(acc_str)) if isinstance(acc_str, str) else acc_str
    acc_str = data['AVG Test ACC']
    test_acc = ast.literal_eval(_clean_np_str(acc_str)) if isinstance(acc_str, str) else acc_str
    loss_str = data['AVG Train LOSS']
    train_loss = ast.literal_eval(_clean_np_str(loss_str)) if isinstance(loss_str, str) else loss_str
    loss_str = data['AVG Test LOSS']
    test_loss = ast.literal_eval(_clean_np_str(loss_str)) if isinstance(loss_str, str) else loss_str
    test_loss = [float(x) for x in test_loss]
    train = {i: (train_loss[i], train_acc[i]) for i in range(len(train_acc))}
    test = {i: (test_loss[i], test_acc[i]) for i in range(len(test_acc))}
    total_hours = float(data.get('time', 0))
    epoch_times = data.get('epoch_times_sec', None)
    if epoch_times:
        epoch_times = ast.literal_eval(epoch_times)
    else:
        n = len(train_acc)
        per_epoch = total_hours * 3600 / n
        epoch_times = [per_epoch] * n
    return train, test, epoch_times


def _clean_np_str(s):
    """Remove np.float32/64 wrappers for ast.literal_eval compatibility."""
    s = re.sub(r'np\.float\d+\((.*?)\)', r'\1', s)
    return s


def get_epoch_times(method):
    """Get per-epoch times (seconds) from result txt."""
    patterns = [
        os.path.join(LOG_DIR, f"{method}_*.txt"),
        os.path.join(LOG_DIR, f"{method.replace('-','_')}_*.txt"),
        os.path.join(LOG_DIR, f"{method.upper()}_*.txt"),
    ]
    files = []
    for p in patterns:
        files = glob.glob(p)
        if files:
            break
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    try:
        with open(files[0]) as f:
            raw = f.read()
        data = ast.literal_eval(_clean_np_str(raw))
        epoch_times = data.get('epoch_times_sec', None)
        if epoch_times:
            if isinstance(epoch_times, str):
                epoch_times = ast.literal_eval(epoch_times)
            return epoch_times
        total_hours = float(data.get('time', 0))
        test_acc = data['AVG Test ACC']
        if isinstance(test_acc, str):
            test_acc = ast.literal_eval(_clean_np_str(test_acc))
        n = len(test_acc)
        if total_hours > 0 and n > 0:
            per_epoch = total_hours * 3600 / n
            return [per_epoch] * n
    except Exception as e:
        print(f"  Warning: failed to parse {files[0]}: {e}")
    return None


def epoch_times_to_cumulative_minutes(epoch_times):
    """Convert per-epoch seconds to cumulative minutes."""
    cumul = []
    total = 0
    for t in epoch_times:
        total += t
        cumul.append(total / 60.0)
    return cumul


def load_all_data():
    all_data = {}
    for key, (logfile, name, *_) in METHOD_CONFIG.items():
        if key == "NOVA2":
            result = parse_nova2_txt()
            if result[0] is None:
                continue
            train, test, epoch_times = result
        else:
            path = os.path.join(LOG_DIR, logfile)
            if not os.path.exists(path):
                continue
            train, test = parse_log(path)
            epoch_times = get_epoch_times(key)
        if train and epoch_times:
            cumul_min = epoch_times_to_cumulative_minutes(epoch_times)
            all_data[key] = (train, test, cumul_min)
    return all_data


def plot_test_acc_vs_time(all_data):
    """Test Accuracy vs Wall-clock Time (minutes)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in METHOD_CONFIG:
        if key not in all_data:
            continue
        _, test, cumul_min = all_data[key]
        _, name, color, ls, lw, marker = METHOD_CONFIG[key]
        epochs = sorted(test.keys())
        accs = [test[e][1] * 100 for e in epochs]
        times = [cumul_min[e] for e in epochs if e < len(cumul_min)]
        accs = accs[:len(times)]
        ms = 8 if key == "NOVA2" else 5
        zorder = 10 if key == "NOVA2" else 2
        ax.plot(times, accs, color=color, linestyle=ls, linewidth=lw,
                marker=marker, markersize=ms, markevery=2, label=name, zorder=zorder)
    ax.set_xlabel("Training Time (minutes)", fontsize=14)
    ax.set_ylabel("Test Accuracy (%)", fontsize=14)
    ax.set_title("Test Accuracy vs Training Time", fontsize=16)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig1_test_acc_vs_time.png"), dpi=200)
    plt.close()
    print("  fig1_test_acc_vs_time.png")


def plot_test_loss_vs_time(all_data):
    """Test Loss vs Wall-clock Time (minutes)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in METHOD_CONFIG:
        if key not in all_data:
            continue
        _, test, cumul_min = all_data[key]
        _, name, color, ls, lw, marker = METHOD_CONFIG[key]
        epochs = sorted(test.keys())
        losses = [test[e][0] for e in epochs]
        times = [cumul_min[e] for e in epochs if e < len(cumul_min)]
        losses = losses[:len(times)]
        ms = 8 if key == "NOVA2" else 5
        zorder = 10 if key == "NOVA2" else 2
        ax.plot(times, losses, color=color, linestyle=ls, linewidth=lw,
                marker=marker, markersize=ms, markevery=2, label=name, zorder=zorder)
    ax.set_xlabel("Training Time (minutes)", fontsize=14)
    ax.set_ylabel("Test Loss", fontsize=14)
    ax.set_title("Test Loss vs Training Time", fontsize=16)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig2_test_loss_vs_time.png"), dpi=200)
    plt.close()
    print("  fig2_test_loss_vs_time.png")


def plot_train_loss_vs_time(all_data):
    """Training Loss vs Wall-clock Time (minutes)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in METHOD_CONFIG:
        if key not in all_data:
            continue
        train, _, cumul_min = all_data[key]
        _, name, color, ls, lw, marker = METHOD_CONFIG[key]
        epochs = sorted(train.keys())
        losses = [train[e][0] for e in epochs]
        times = [cumul_min[e] for e in epochs if e < len(cumul_min)]
        losses = losses[:len(times)]
        ms = 8 if key == "NOVA2" else 5
        zorder = 10 if key == "NOVA2" else 2
        ax.plot(times, losses, color=color, linestyle=ls, linewidth=lw,
                marker=marker, markersize=ms, markevery=2, label=name, zorder=zorder)
    ax.set_xlabel("Training Time (minutes)", fontsize=14)
    ax.set_ylabel("Training Loss", fontsize=14)
    ax.set_title("Training Loss vs Training Time", fontsize=16)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig3_train_loss_vs_time.png"), dpi=200)
    plt.close()
    print("  fig3_train_loss_vs_time.png")


def plot_combined_2x1(all_data):
    """Combined: Test Acc + Test Loss vs Time side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for key in METHOD_CONFIG:
        if key not in all_data:
            continue
        _, test, cumul_min = all_data[key]
        _, name, color, ls, lw, marker = METHOD_CONFIG[key]
        epochs = sorted(test.keys())
        accs = [test[e][1] * 100 for e in epochs]
        losses = [test[e][0] for e in epochs]
        times = [cumul_min[e] for e in epochs if e < len(cumul_min)]
        accs = accs[:len(times)]
        losses = losses[:len(times)]
        ms = 7 if key == "NOVA2" else 4
        zorder = 10 if key == "NOVA2" else 2

        ax1.plot(times, accs, color=color, linestyle=ls, linewidth=lw,
                 marker=marker, markersize=ms, markevery=2, label=name, zorder=zorder)
        ax2.plot(times, losses, color=color, linestyle=ls, linewidth=lw,
                 marker=marker, markersize=ms, markevery=2, label=name, zorder=zorder)

    ax1.set_xlabel("Training Time (minutes)", fontsize=13)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax1.set_title("(a) Test Accuracy vs Time", fontsize=14)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Training Time (minutes)", fontsize=13)
    ax2.set_ylabel("Test Loss", fontsize=13)
    ax2.set_title("(b) Test Loss vs Time", fontsize=14)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Bilevel Optimization: Performance vs Training Time", fontsize=15, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig4_combined_vs_time.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig4_combined_vs_time.png")


if __name__ == "__main__":
    print("Loading data with time info...")
    all_data = load_all_data()
    print(f"Loaded {len(all_data)} methods:")
    for key in all_data:
        _, _, cumul = all_data[key]
        print(f"  {key}: {len(cumul)} epochs, total={cumul[-1]:.1f} min ({cumul[-1]/60:.2f}h)")

    print("\nGenerating figures...")
    plot_test_acc_vs_time(all_data)
    plot_test_loss_vs_time(all_data)
    plot_train_loss_vs_time(all_data)
    plot_combined_2x1(all_data)
    print(f"\nAll figures saved to {OUTPUT_DIR}/")
