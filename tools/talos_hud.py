import time
import re
import subprocess
import collections
from datetime import datetime
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.console import Console
from rich import box

# CONFIGURATION
SERVICE_NAME = "talos-omni.service"
HISTORY_LEN = 50

# METRIC BUFFERS
loss_history = collections.deque(maxlen=HISTORY_LEN)
sat_history = collections.deque(maxlen=HISTORY_LEN)
rob_history = collections.deque(maxlen=HISTORY_LEN)

# UPDATED REGEX FOR PHASE 16 DAEMON
# Matches: [Step 123] [RETINA] Loss: 0.123 | Sat: 0.999 | Rob: 0.001
LOG_PATTERN = re.compile(r"\[Step (\d+)\] \[(\w+)\] Loss: ([\d\.]+) \| Sat: ([\d\.]+) \| Rob: ([\d\.]+)")

def get_journal_stream():
    cmd = ["journalctl", "-u", SERVICE_NAME, "-f", "-n", "0", "--output", "cat"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    return process

def generate_layout():
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3)
    )
    layout["main"].split_row(
        Layout(name="metrics"),
        Layout(name="log_feed")
    )
    return layout

def make_header():
    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="right")
    grid.add_row(
        "[b cyan]TALOS-O (OMNI) // COGNITIVE MONITOR[/]",
        f"[dim]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]"
    )
    return Panel(grid, style="white on blue")

def make_metrics_table(step, source, loss, sat, rob):
    table = Table(box=box.SIMPLE)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_column("Status", justify="right")

    # STATUS LOGIC
    sat_color = "green" if sat > 0.9 else "yellow" if sat > 0.7 else "red"
    rob_color = "green" if rob < 0.05 else "yellow" if rob < 0.1 else "red"
    
    # Source Logic
    src_style = "bold magenta" if source == "INJECT" else "dim white"

    table.add_row("Cognitive Step", str(step), "[dim]INCREMENTING[/]")
    table.add_row("Focus Source", f"[{src_style}]{source}[/]", "")
    table.add_row("Entropy (Loss)", f"{loss:.4f}", "")
    table.add_row("Satisfaction", f"[{sat_color}]{sat:.4f}[/]", "")
    table.add_row("Robustness", f"[{rob_color}]{rob:.4f}[/]", "")
    
    return Panel(table, title="System 2 State", border_style="cyan")

def make_ascii_chart(data):
    if not data: return ""
    min_val, max_val = min(data), max(data)
    range_val = max_val - min_val if max_val != min_val else 1.0
    chars = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    return "".join([chars[int((v - min_val) / range_val * 7)] for v in data])

def run_hud():
    process = get_journal_stream()
    layout = generate_layout()
    
    # State
    cur_step, cur_loss, cur_sat, cur_rob = 0, 0.0, 0.0, 0.0
    cur_source = "WAITING"
    last_line = ""

    with Live(layout, refresh_per_second=8, screen=True):
        while True:
            line = process.stdout.readline()
            if line:
                line = line.strip()
                last_line = line
                match = LOG_PATTERN.search(line)
                if match:
                    cur_step = int(match.group(1))
                    cur_source = match.group(2)
                    cur_loss = float(match.group(3))
                    cur_sat = float(match.group(4))
                    cur_rob = float(match.group(5))
                    
                    loss_history.append(cur_loss)
                    sat_history.append(cur_sat)
                    rob_history.append(cur_rob)

            layout["header"].update(make_header())
            layout["main"]["metrics"].split_column(
                Layout(make_metrics_table(cur_step, cur_source, cur_loss, cur_sat, cur_rob), size=8),
                Layout(Panel(make_ascii_chart(sat_history), title="Satisfaction", border_style="green"), size=5),
                Layout(Panel(make_ascii_chart(loss_history), title="Entropy", border_style="red"), size=5)
            )
            layout["main"]["log_feed"].update(Panel(Text(last_line, style="dim white"), title="Live Cortex Stream", border_style="blue"))
            layout["footer"].update(Panel("Active. Ctrl+C to Exit.", style="white on black"))

if __name__ == "__main__":
    try:
        run_hud()
    except KeyboardInterrupt:
        pass
