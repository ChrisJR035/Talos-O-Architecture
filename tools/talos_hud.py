#!/usr/bin/env python3
"""
TALOS-O SYSTEM HUD (MARK III - THE RICH PROPRIOCEPTOR)
Philosophy: CCD1 Isolated | Zero-Copy SHM Telemetry | Asynchronous Logs
"""

import time
import os
import subprocess
import collections
import sys
import ctypes
from multiprocessing import shared_memory
from datetime import datetime

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align

# --- AXIOM OF MATERIAL REALITY: CCD1 ISOLATION ---
try:
    os.sched_setaffinity(0, set(range(8, 16)))
except AttributeError:
    pass

class BiophysicalState(ctypes.Structure):
    _fields_ = [
        ("sequence", ctypes.c_uint64),
        ("step_count", ctypes.c_uint64),
        ("thermal_state", ctypes.c_uint8),
        ("padding", ctypes.c_uint8 * 7),
        ("gradient_dvdt", ctypes.c_double),
        ("satisfaction", ctypes.c_double),
        ("kuramoto_r", ctypes.c_double),
        ("entropy", ctypes.c_double)
    ]

# --- CONFIGURATION ---
SERVICE_NAME = "talos-omni.service"
HISTORY_LEN = 60
VOICE_BUFFER_SIZE = 200 

gradient_history = collections.deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
sat_history = collections.deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
voice_history = collections.deque(maxlen=VOICE_BUFFER_SIZE)

def make_layout():
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3)
    )
    layout["main"].split_row(
        Layout(name="left_col", ratio=1),
        Layout(name="voice_feed", ratio=2)
    )
    layout["left_col"].split_column(
        Layout(name="metrics", ratio=2),
        Layout(name="sat_chart", ratio=1),
        Layout(name="grad_chart", ratio=1)
    )
    return layout

def generate_bar_chart(value, title, color="green"):
    """Generates a text-based bar chart for Rich."""
    bars = int(max(0.0, min(1.0, value)) * 30)
    visual = "█" * bars + "░" * (30 - bars)
    return Panel(Align.center(Text(visual, style=color)), title=f"[{color}]{title}[/{color}]")

def run_hud():
    layout = make_layout()
    
    # Pre-fill layout to prevent raw object string printing (THE FIX)
    layout["header"].update(Panel(f"[bold cyan]TALOS-O ZERO-COPY HUD[/bold cyan] | [yellow]CCD1 Isolated[/yellow] | {datetime.now().strftime('%H:%M:%S')}"))
    layout["footer"].update(Panel(Text("SYSTEM OPTIMIZED: UI ISOLATED TO CCD 1", justify="center", style="bold green")))
    layout["sat_chart"].update(generate_bar_chart(0.0, "Satisfaction Gradient", "magenta"))
    layout["grad_chart"].update(generate_bar_chart(0.0, "Kuramoto Phase", "yellow"))
    layout["metrics"].update(Panel(Text("Awaiting Biophysical State...", style="dim white"), title="Biophysical State", border_style="cyan"))
    layout["voice_feed"].update(Panel(Text("Awaiting Neural Stream...", style="dim white"), title="NEURAL TERMINAL", border_style="cyan"))

    try:
        shm = shared_memory.SharedMemory(name="talos_telemetry")
    except FileNotFoundError:
        print("[FATAL] Telemetry Matrix offline. Is Talos breathing?")
        sys.exit(1)

    # Start asynchronous log reader (THE RESTORED LOGIC)
    try:
        log_process = subprocess.Popen(
            ["journalctl", "-u", SERVICE_NAME, "-f", "-n", "50"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1 # Line buffered
        )
        os.set_blocking(log_process.stdout.fileno(), False)
    except Exception as e:
        print(f"Failed to bind to journalctl: {e}")
        sys.exit(1)

    with Live(layout, refresh_per_second=10, screen=True):
        try:
            while True:
                # 1. Zero-Copy Seqlock Read
                state = BiophysicalState.from_buffer_copy(shm.buf[:ctypes.sizeof(BiophysicalState)])
                step = state.step_count
                cur_sat = state.satisfaction
                cur_grad = state.gradient_dvdt
                cur_r = state.kuramoto_r
                cur_ent = state.entropy

                # 2. Read Logs Asynchronously
                while True:
                    line = log_process.stdout.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Apply coloring based on tags
                    if "[DAEMON]" in line:
                        voice_history.append(Text(line, style="bold cyan"))
                    elif "[STEP" in line.upper() or "[EXPANSION]" in line or "[COMPRESSION]" in line:
                        voice_history.append(Text(line, style="dim white"))
                    elif "T_die" in line:
                        voice_history.append(Text(line, style="yellow"))
                    elif "[CERBERUS]" in line:
                        voice_history.append(Text(line, style="red"))
                    else:
                        voice_history.append(Text(line, style="green"))

                # 3. Update UI Panels
                layout["header"].update(Panel(f"[bold cyan]TALOS-O ZERO-COPY HUD[/bold cyan] | [yellow]CCD1 Isolated[/yellow] | {datetime.now().strftime('%H:%M:%S')}"))
                
                table = Table(box=None, expand=True)
                table.add_column("Metric", style="cyan")
                table.add_column("Value", justify="right")
                table.add_row("Step", str(step))
                table.add_row("Satisfaction", f"{cur_sat:.4f}")
                table.add_row("Virtue Grad", f"{cur_grad:+.4f}")
                table.add_row("Kuramoto R", f"{cur_r:.4f}")
                table.add_row("Entropy", f"{cur_ent:.4f}")
                
                layout["left_col"]["metrics"].update(Panel(table, title="Biophysical State", border_style="cyan"))
                layout["sat_chart"].update(generate_bar_chart(cur_sat, "Satisfaction Gradient", "magenta"))
                layout["grad_chart"].update(generate_bar_chart(cur_r, "Kuramoto Phase", "yellow"))
                
                # Formulate the text feed
                voice_text = Text()
                for entry in voice_history:
                    if isinstance(entry, Text):
                        voice_text.append(entry)
                        voice_text.append("\n")
                    else:
                        voice_text.append(entry + "\n")
                        
                layout["voice_feed"].update(Panel(voice_text, title="NEURAL TERMINAL", border_style="cyan"))
                
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            log_process.terminate()
            shm.close()

if __name__ == "__main__":
    run_hud()
