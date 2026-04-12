#!/usr/bin/env python3
"""
Auto-generate training statistics report from log files.
Usage: python stats.py [--output results.md]
"""
import re, os, argparse
from datetime import datetime

LOG_DIR = "logs"
METHODS_ORDER = ["stocbio", "ttsa", "saba", "ma-soba", "bo-rep", "sustain", "vrbo", "accbo",
                 "meha", "nova2", "nova3", "s-pngbio"]


def parse_log(filepath):
    """Parse a single method's log file."""
    with open(filepath) as f:
        lines = f.readlines()

    epoch = -1
    train_last = {}  # epoch -> (step, loss, acc)
    test_results = {}  # epoch -> (loss, acc)
    max_epoch = 0

    for line in lines:
        m = re.match(r'\[epoch/epochs\]:(\d+)/(\d+)', line)
        if m:
            epoch = int(m.group(1))
            max_epoch = int(m.group(2))

        m = re.match(r'Step (\d+) \| Task Loss: ([\d.]+) \| Acc: ([\d.]+)', line)
        if m and epoch >= 0:
            train_last[epoch] = (int(m.group(1)), float(m.group(2)), float(m.group(3)))

        m = re.match(r'Task loss: ([\d.]+), Task acc: ([\d.]+)', line)
        if m and epoch >= 0:
            test_results[epoch] = (float(m.group(1)), float(m.group(2)))

    return train_last, test_results, max_epoch


def get_duration_from_txt(method):
    """Read training time from result txt files in logs/."""
    import glob, ast
    pattern = os.path.join(LOG_DIR, f"{method}_*.txt")
    # Also try with underscores for methods like ma-soba -> ma_soba
    files = glob.glob(pattern)
    if not files:
        alt = method.replace("-", "_")
        files = glob.glob(os.path.join(LOG_DIR, f"{alt}_*.txt"))
    if not files:
        # Try uppercase for NOVA2 etc
        files = glob.glob(os.path.join(LOG_DIR, f"{method.upper()}_*.txt"))
    if not files:
        return "—"
    # Use the most recent file
    files.sort(key=os.path.getmtime, reverse=True)
    try:
        with open(files[0]) as f:
            data = ast.literal_eval(f.read())
        hours = data.get('time', data.get('total_time_hours', None))
        if hours is not None:
            hours = float(hours)
            h = int(hours)
            m = int((hours - h) * 60)
            return f"{h}h{m:02d}m"
    except:
        pass
    return "—"


def generate_report(output_path):
    lines = []
    lines.append("# Training Results Report")
    lines.append("")
    lines.append(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Method | Status | Duration | Best Test Acc | Best Test Loss | Best Epoch |")
    lines.append("|--------|--------|----------|---------------|----------------|------------|")

    all_results = {}
    for method in METHODS_ORDER:
        logfile = os.path.join(LOG_DIR, f"{method}_e20.log")
        if not os.path.exists(logfile) or os.path.getsize(logfile) < 100:
            continue

        train_last, test_results, max_epoch = parse_log(logfile)
        if not train_last:
            continue

        all_results[method] = (train_last, test_results, max_epoch)

        # Status and duration
        last_ep = max(train_last.keys())
        duration = get_duration_from_txt(method)
        if last_ep >= max_epoch - 1:
            status = "✅"
        else:
            status = f"⏳ ep {last_ep}/{max_epoch}"

        # Best test
        if test_results:
            best_ep = max(test_results, key=lambda e: test_results[e][1])
            best_loss, best_acc = test_results[best_ep]
            lines.append(f"| {method} | {status} | {duration} | {best_acc:.1%} | {best_loss:.4f} | {best_ep} |")
        else:
            lines.append(f"| {method} | {status} | {duration} | — | — | — |")

    # Detailed trends for each method
    lines.append("")
    lines.append("## Detailed Training Curves")

    for method, (train_last, test_results, max_epoch) in all_results.items():
        lines.append("")
        lines.append(f"### {method}")
        lines.append("")
        lines.append(f"| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |")
        lines.append("|-------|------------|-----------|-----------|----------|")

        for ep in sorted(train_last.keys()):
            _, tl, ta = train_last[ep]
            if ep in test_results:
                vl, va = test_results[ep]
                lines.append(f"| {ep} | {tl:.4f} | {ta:.1%} | {vl:.4f} | {va:.1%} |")
            else:
                lines.append(f"| {ep} | {tl:.4f} | {ta:.1%} | — | — |")

    # Write
    content = "\n".join(lines) + "\n"
    with open(output_path, "w") as f:
        f.write(content)
    print(f"Report saved to {output_path}")
    print(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results.md", help="Output markdown file")
    args = parser.parse_args()
    generate_report(args.output)
