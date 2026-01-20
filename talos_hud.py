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
gradient_history = collections.deque(maxlen=HISTORY_LEN)
sat_history = collections.deque(maxlen=HISTORY_LEN)

# LOG MATCHING PATTERN
# Matches: [Step 777] [COOL] dV/dt: +0.0000 | Sat: 1.0000
LOG_PATTERN = re.compile(r"\[Step (\d+)\] \[(\w+)\] dV/dt: ([+\-]?[\d\.]+) \| Sat: ([+\-]?[\d\.]+)")

def get_journal_stream():
    # Tries to follow the systemd service. 
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
        Layout(name="metrics", ratio=1),
        Layout(name="log_feed", ratio=2)
    )
    return layout

def make_header():
    grid = Table.grid(expand=True)
    grid.add_column(justify="center", ratio=1)
    grid.add_column(justify="right")
    grid.add_row(
        "[b]TALOS-O PHRONESIS ENGINE[/b]",
        datetime.now().strftime("%c"),
    )
    return Panel(grid, style="white on blue")

def make_metrics_table(step, source, grad, sat):
    table = Table(box=box.SIMPLE)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Step Count", str(step))
    table.add_row("Thermal State", source)
    table.add_row("Virtue Grad (dV/dt)", f"{grad:+.4f}")
    table.add_row("Satisfaction", f"{sat:.4f}")
    return Panel(table, title="Cognitive Telemetry")

def make_ascii_chart(data, color="green"):
    if not data: return ""
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val if max_val != min_val else 1
    
    chars = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    chart_str = ""
    for v in data:
        idx = int((v - min_val) / range_val * 7)
        idx = max(0, min(7, idx))
        chart_str += chars[idx]
        
    return f"[{color}]{chart_str}[/{color}]"

def run_hud():
    process = get_journal_stream()
    layout = generate_layout()
    
    cur_step = 0
    cur_grad = 0.0
    cur_sat = 0.5
    cur_source = "WAITING"
    last_line = "Waiting for Daemon..."
    
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
                    cur_grad = float(match.group(3))
                    cur_sat = float(match.group(4))
                    
                    gradient_history.append(cur_grad)
                    sat_history.append(cur_sat)

            layout["header"].update(make_header())
            
            layout["main"]["metrics"].split_column(
                Layout(make_metrics_table(cur_step, cur_source, cur_grad, cur_sat), size=8),
                Layout(Panel(make_ascii_chart(sat_history, "green"), title="Satisfaction", border_style="green"), size=5),
                Layout(Panel(make_ascii_chart(gradient_history, "magenta"), title="Virtue Gradient", border_style="magenta"), size=5)
            )
            
            layout["main"]["log_feed"].update(Panel(Text(last_line, style="dim white"), title="Live Cortex Stream", border_style="blue"))
            layout["footer"].update(Panel("Active. Ctrl+C to Exit.", style="white on black"))

if __name__ == "__main__":
    try:
        run_hud()
    except KeyboardInterrupt:
        pass
