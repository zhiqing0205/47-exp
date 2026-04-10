#!/usr/bin/env python3
"""
Visualization of bilevel optimization experiment results.
Generates multiple publication-quality figures.
NOVA2 is highlighted as "Ours" method.
"""
import re, os, ast
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ============ Config ============
LOG_DIR = "logs"
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Method display config: (log_key, display_name, color, linestyle, linewidth, marker)
# NOVA2 is "Ours" - bold, prominent color
METHOD_CONFIG = {
    "stocbio":  ("stocbio_e20.log",  "StocBio",  "#1f77b4", "--", 1.5, "o"),
    "ttsa":     ("ttsa_e20.log",     "TTSA",     "#ff7f0e", "--", 1.5, "s"),
    "saba":     ("saba_e20.log",     "SABA",     "#2ca02c", "--", 1.5, "^"),
    "ma-soba":  ("ma-soba_e20.log",  "MA-SOBA",  "#9467bd", "--", 1.5, "D"),
    "bo-rep":   ("bo-rep_e20.log",   "BO-REP",   "#8c564b", "--", 1.5, "v"),
    "sustain":  ("sustain_e20.log",  "SUSTAIN",  "#7f7f7f", "--", 1.5, "p"),
    "NOVA2":    (None,               "NOVA2 (Ours)", "#d62728", "-", 3.0, "*"),
}

# Time data (hours)
TIME_DATA = {
    "StocBio": 2.5,
    "TTSA": 2.23,
    "SABA": 1.38,
    "MA-SOBA": 1.35,
    "BO-REP": 2.22,
    "SUSTAIN": 3.8,
    "NOVA2 (Ours)": 0.55,
}


def parse_log(filepath):
    """Parse e20 log file, return per-epoch train/test loss & acc."""
    with open(filepath) as f:
        lines = f.readlines()
    epoch = -1
    train = {}  # epoch -> (loss, acc)
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
    return train, test


def parse_nova2_txt():
    """Parse NOVA2's result txt file."""
    import glob
    files = glob.glob(os.path.join(LOG_DIR, "NOVA2_*.txt"))
    if not files:
        return None, None
    with open(files[0]) as f:
        data = ast.literal_eval(f.read())
    train_acc = ast.literal_eval(data['AVG Train ACC'])
    test_acc = ast.literal_eval(data['AVG Test ACC'])
    train_loss = ast.literal_eval(data['AVG Train LOSS'])
    test_loss = [float(x) for x in ast.literal_eval(data['AVG Test LOSS'])]
    train = {i: (train_loss[i], train_acc[i]) for i in range(len(train_acc))}
    test = {i: (test_loss[i], test_acc[i]) for i in range(len(test_acc))}
    return train, test


def load_all_data():
    """Load all method data."""
    all_data = {}
    for key, (logfile, name, *_) in METHOD_CONFIG.items():
        if key == "NOVA2":
            train, test = parse_nova2_txt()
        else:
            path = os.path.join(LOG_DIR, logfile)
            if not os.path.exists(path):
                continue
            train, test = parse_log(path)
        if train:
            all_data[key] = (train, test)
    return all_data


def plot_test_acc(all_data):
    """Fig 1: Test Accuracy curves - the main result figure."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for key in METHOD_CONFIG:
        if key not in all_data:
            continue
        _, test = all_data[key]
        _, name, color, ls, lw, marker = METHOD_CONFIG[key]
        epochs = sorted(test.keys())
        accs = [test[e][1] * 100 for e in epochs]
        ms = 8 if key == "NOVA2" else 5
        mevery = 2
        zorder = 10 if key == "NOVA2" else 2
        ax.plot(epochs, accs, color=color, linestyle=ls, linewidth=lw,
                marker=marker, markersize=ms, markevery=mevery,
                label=name, zorder=zorder)

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Test Accuracy (%)", fontsize=14)
    ax.set_title("Test Accuracy on SNLI Data Cleaning", fontsize=16)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(-0.5, 19.5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig1_test_acc.png"), dpi=200)
    plt.close()
    print("  fig1_test_acc.png")


def plot_test_loss(all_data):
    """Fig 2: Test Loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for key in METHOD_CONFIG:
        if key not in all_data:
            continue
        _, test = all_data[key]
        _, name, color, ls, lw, marker = METHOD_CONFIG[key]
        epochs = sorted(test.keys())
        losses = [test[e][0] for e in epochs]
        ms = 8 if key == "NOVA2" else 5
        zorder = 10 if key == "NOVA2" else 2
        ax.plot(epochs, losses, color=color, linestyle=ls, linewidth=lw,
                marker=marker, markersize=ms, markevery=2,
                label=name, zorder=zorder)

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Test Loss", fontsize=14)
    ax.set_title("Test Loss on SNLI Data Cleaning", fontsize=16)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(-0.5, 19.5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig2_test_loss.png"), dpi=200)
    plt.close()
    print("  fig2_test_loss.png")


def plot_train_loss(all_data):
    """Fig 3: Training Loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for key in METHOD_CONFIG:
        if key not in all_data:
            continue
        train, _ = all_data[key]
        _, name, color, ls, lw, marker = METHOD_CONFIG[key]
        epochs = sorted(train.keys())
        losses = [train[e][0] for e in epochs]
        ms = 8 if key == "NOVA2" else 5
        zorder = 10 if key == "NOVA2" else 2
        ax.plot(epochs, losses, color=color, linestyle=ls, linewidth=lw,
                marker=marker, markersize=ms, markevery=2,
                label=name, zorder=zorder)

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Training Loss", fontsize=14)
    ax.set_title("Training Loss on SNLI Data Cleaning", fontsize=16)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(-0.5, 19.5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig3_train_loss.png"), dpi=200)
    plt.close()
    print("  fig3_train_loss.png")


def plot_best_acc_bar(all_data):
    """Fig 4: Bar chart comparing best test accuracy."""
    methods = []
    accs = []
    colors = []

    for key in METHOD_CONFIG:
        if key not in all_data:
            continue
        _, test = all_data[key]
        _, name, color, *_ = METHOD_CONFIG[key]
        best_acc = max(test[e][1] for e in test) * 100
        methods.append(name)
        accs.append(best_acc)
        colors.append(color)

    # Sort by accuracy
    order = np.argsort(accs)
    methods = [methods[i] for i in order]
    accs = [accs[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(methods)), accs, color=colors, edgecolor='white', height=0.6)

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        is_ours = "Ours" in methods[i]
        weight = 'bold' if is_ours else 'normal'
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', va='center', fontsize=12, fontweight=weight)
        if is_ours:
            bar.set_edgecolor('#d62728')
            bar.set_linewidth(2.5)

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=12)
    ax.set_xlabel("Best Test Accuracy (%)", fontsize=14)
    ax.set_title("Best Test Accuracy Comparison", fontsize=16)
    ax.set_xlim(60, max(accs) + 3)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig4_best_acc_bar.png"), dpi=200)
    plt.close()
    print("  fig4_best_acc_bar.png")


def plot_time_comparison(all_data):
    """Fig 5: Training time vs Best accuracy scatter + bar."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Time bar chart
    names = []
    times = []
    colors = []
    for key in METHOD_CONFIG:
        if key not in all_data:
            continue
        _, name, color, *_ = METHOD_CONFIG[key]
        if name in TIME_DATA:
            names.append(name)
            times.append(TIME_DATA[name])
            colors.append(color)

    order = np.argsort(times)[::-1]
    names_s = [names[i] for i in order]
    times_s = [times[i] for i in order]
    colors_s = [colors[i] for i in order]

    bars = ax1.barh(range(len(names_s)), times_s, color=colors_s, edgecolor='white', height=0.6)
    for i, (bar, t) in enumerate(zip(bars, times_s)):
        is_ours = "Ours" in names_s[i]
        weight = 'bold' if is_ours else 'normal'
        ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{t:.1f}h', va='center', fontsize=11, fontweight=weight)
        if is_ours:
            bar.set_edgecolor('#d62728')
            bar.set_linewidth(2.5)

    ax1.set_yticks(range(len(names_s)))
    ax1.set_yticklabels(names_s, fontsize=12)
    ax1.set_xlabel("Training Time (hours)", fontsize=13)
    ax1.set_title("Training Time", fontsize=14)
    ax1.grid(True, axis='x', alpha=0.3)

    # Right: Scatter - time vs accuracy (efficiency plot)
    for key in METHOD_CONFIG:
        if key not in all_data:
            continue
        _, test = all_data[key]
        _, name, color, _, _, marker = METHOD_CONFIG[key]
        if name not in TIME_DATA:
            continue
        best_acc = max(test[e][1] for e in test) * 100
        t = TIME_DATA[name]
        is_ours = key == "NOVA2"
        ms = 200 if is_ours else 100
        zorder = 10 if is_ours else 2
        edgecolor = '#d62728' if is_ours else 'white'
        lw = 2.5 if is_ours else 1
        ax2.scatter(t, best_acc, c=color, s=ms, marker=marker, edgecolors=edgecolor,
                   linewidths=lw, zorder=zorder, label=name)

    # Add arrow annotation for NOVA2
    for key in METHOD_CONFIG:
        if key == "NOVA2" and key in all_data:
            _, test = all_data[key]
            best_acc = max(test[e][1] for e in test) * 100
            t = TIME_DATA["NOVA2 (Ours)"]
            ax2.annotate('Fastest & Competitive',
                        xy=(t, best_acc), xytext=(t + 0.8, best_acc - 2),
                        fontsize=10, fontweight='bold', color='#d62728',
                        arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5))

    ax2.set_xlabel("Training Time (hours)", fontsize=13)
    ax2.set_ylabel("Best Test Accuracy (%)", fontsize=13)
    ax2.set_title("Efficiency: Accuracy vs Time", fontsize=14)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig5_time_comparison.png"), dpi=200)
    plt.close()
    print("  fig5_time_comparison.png")


def plot_combined_2x2(all_data):
    """Fig 6: Combined 2x2 subplot for paper."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Test Accuracy
    ax = axes[0, 0]
    for key in METHOD_CONFIG:
        if key not in all_data:
            continue
        _, test = all_data[key]
        _, name, color, ls, lw, marker = METHOD_CONFIG[key]
        epochs = sorted(test.keys())
        accs = [test[e][1] * 100 for e in epochs]
        ms = 7 if key == "NOVA2" else 4
        zorder = 10 if key == "NOVA2" else 2
        ax.plot(epochs, accs, color=color, linestyle=ls, linewidth=lw,
                marker=marker, markersize=ms, markevery=2, label=name, zorder=zorder)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("(a) Test Accuracy", fontsize=13)
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # (0,1) Test Loss
    ax = axes[0, 1]
    for key in METHOD_CONFIG:
        if key not in all_data:
            continue
        _, test = all_data[key]
        _, name, color, ls, lw, marker = METHOD_CONFIG[key]
        epochs = sorted(test.keys())
        losses = [test[e][0] for e in epochs]
        ms = 7 if key == "NOVA2" else 4
        zorder = 10 if key == "NOVA2" else 2
        ax.plot(epochs, losses, color=color, linestyle=ls, linewidth=lw,
                marker=marker, markersize=ms, markevery=2, label=name, zorder=zorder)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Loss", fontsize=12)
    ax.set_title("(b) Test Loss", fontsize=13)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # (1,0) Best Accuracy Bar
    ax = axes[1, 0]
    methods_list = []
    accs_list = []
    colors_list = []
    for key in METHOD_CONFIG:
        if key not in all_data:
            continue
        _, test = all_data[key]
        _, name, color, *_ = METHOD_CONFIG[key]
        best_acc = max(test[e][1] for e in test) * 100
        methods_list.append(name)
        accs_list.append(best_acc)
        colors_list.append(color)
    order = np.argsort(accs_list)
    bars = ax.barh(range(len(methods_list)),
                   [accs_list[i] for i in order],
                   color=[colors_list[i] for i in order],
                   edgecolor='white', height=0.6)
    sorted_names = [methods_list[i] for i in order]
    for i, bar in enumerate(bars):
        is_ours = "Ours" in sorted_names[i]
        if is_ours:
            bar.set_edgecolor('#d62728')
            bar.set_linewidth(2.5)
        acc_val = [accs_list[j] for j in order][i]
        weight = 'bold' if is_ours else 'normal'
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{acc_val:.1f}%', va='center', fontsize=10, fontweight=weight)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel("Best Test Accuracy (%)", fontsize=12)
    ax.set_title("(c) Best Accuracy Comparison", fontsize=13)
    ax.set_xlim(60, max(accs_list) + 3)
    ax.grid(True, axis='x', alpha=0.3)

    # (1,1) Time vs Accuracy
    ax = axes[1, 1]
    for key in METHOD_CONFIG:
        if key not in all_data:
            continue
        _, test = all_data[key]
        _, name, color, _, _, marker = METHOD_CONFIG[key]
        if name not in TIME_DATA:
            continue
        best_acc = max(test[e][1] for e in test) * 100
        t = TIME_DATA[name]
        is_ours = key == "NOVA2"
        ms = 180 if is_ours else 80
        zorder = 10 if is_ours else 2
        ec = '#d62728' if is_ours else 'white'
        lw = 2.5 if is_ours else 1
        ax.scatter(t, best_acc, c=color, s=ms, marker=marker,
                  edgecolors=ec, linewidths=lw, zorder=zorder, label=name)
    ax.set_xlabel("Training Time (hours)", fontsize=12)
    ax.set_ylabel("Best Test Accuracy (%)", fontsize=12)
    ax.set_title("(d) Efficiency: Accuracy vs Time", fontsize=13)
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Bilevel Optimization for Data Hyper-Cleaning on SNLI", fontsize=15, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig6_combined_2x2.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig6_combined_2x2.png")


if __name__ == "__main__":
    print("Loading data...")
    all_data = load_all_data()
    print(f"Loaded {len(all_data)} methods: {list(all_data.keys())}")
    print("\nGenerating figures...")
    plot_test_acc(all_data)
    plot_test_loss(all_data)
    plot_train_loss(all_data)
    plot_best_acc_bar(all_data)
    plot_time_comparison(all_data)
    plot_combined_2x2(all_data)
    print(f"\nAll figures saved to {OUTPUT_DIR}/")
