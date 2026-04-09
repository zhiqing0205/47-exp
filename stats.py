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


def parse_master_log(filepath):
    """Parse master log for timing info."""
    times = {}
    if not os.path.exists(filepath):
        return times
    with open(filepath) as f:
        for line in f:
            m = re.match(r'>>> (\S+) start .* (\d{2}:\d{2}:\d{2})', line)
            if m:
                times.setdefault(m.group(1), {})['start'] = m.group(2)
            m = re.match(r'<<< (\S+) exit=(\d+) .* (\d{2}:\d{2}:\d{2})', line)
            if m:
                times.setdefault(m.group(1), {})['end'] = m.group(3)
                times[m.group(1)]['exit'] = int(m.group(2))
    return times


def calc_duration(start_str, end_str):
    """Calculate duration between HH:MM:SS strings."""
    fmt = "%H:%M:%S"
    try:
        s = datetime.strptime(start_str, fmt)
        e = datetime.strptime(end_str, fmt)
        diff = (e - s).total_seconds()
        if diff < 0:
            diff += 86400
        h = int(diff // 3600)
        m = int((diff % 3600) // 60)
        return f"{h}h{m:02d}m"
    except:
        return "—"


def generate_report(output_path):
    master_log = os.path.join(LOG_DIR, "e20_master.log")
    # Try alternative master log locations
    for candidate in ["e20_master.log", "serial_e2.log"]:
        p = os.path.join(LOG_DIR, candidate)
        if os.path.exists(p):
            master_log = p
            break

    # Also check the background task output
    task_output = None
    import glob
    for f in glob.glob("/tmp/claude-0/-root-data-47-data-cleaning/*/tasks/*.output"):
        if os.path.exists(f):
            with open(f) as fh:
                content = fh.read()
            if ">>>" in content and "<<<" in content:
                task_output = f
                break

    times = {}
    if task_output:
        times = parse_master_log(task_output)
    if not times:
        times = parse_master_log(master_log)

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

        # Status
        t = times.get(method, {})
        last_ep = max(train_last.keys())
        if 'end' in t and t.get('exit', 1) == 0:
            status = "✅"
            duration = calc_duration(t['start'], t['end'])
        elif last_ep >= max_epoch - 1:
            status = "✅"
            duration = "—"
        else:
            status = f"⏳ ep {last_ep}/{max_epoch}"
            duration = "—"

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
