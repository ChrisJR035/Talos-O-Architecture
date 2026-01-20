import json
import subprocess
import os
import shutil

# 1. Search Engine Loading
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None
        print("[MOTOR] WARNING: duckduckgo_search library missing. Install via pip.")

class ToolBelt:
    def __init__(self):
        self.ddgs = None
        self.container_engine = self._detect_engine()
        
        if DDGS:
            try:
                self.ddgs = DDGS()
            except:
                pass
        
        status = []
        if self.ddgs: status.append("Web")
        if self.container_engine: status.append(f"Sandbox({self.container_engine})")
        status.append("FileI/O")
        
        print(f"[MOTOR] ToolBelt Online: {', '.join(status)}")

    def _detect_engine(self):
        if shutil.which("podman"): return "podman"
        if shutil.which("docker"): return "docker"
        return None

    def search_web(self, query, max_results=3):
        if not self.ddgs: return "System Alert: Search offline."
        print(f"[MOTOR] Executing Action: SEARCH '{query}'")
        try:
            results = self.ddgs.text(query, max_results=max_results)
            if not results: return "System Alert: No results found."
            summary = [f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nSource: {r.get('href')}" for r in results]
            return "\n---\n".join(summary)
        except Exception as e:
            return f"System Alert: Search Failed - {e}"

    def execute_code(self, code_snippet):
        if not self.container_engine:
            return "System Alert: No sandbox engine (Docker/Podman) found. Execution blocked."
            
        print("[MOTOR] Executing Action: CODE (Sandbox)")
        
        # Write to shared memory for speed
        with open("/tmp/talos_script.py", "w") as f:
            f.write(code_snippet)
            
        # Execute in verified sandbox
        cmd = [
            self.container_engine, "run", "--rm",
            "--network", "none",
            "--memory", "512m",
            "-v", "/tmp/talos_script.py:/home/talos/script.py:ro",
            "talos-sandbox", "python3", "/home/talos/script.py"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            output = result.stdout + "\n" + result.stderr
            return output.strip() if output.strip() else "Code executed (No Output)."
        except subprocess.TimeoutExpired: return "System Alert: Execution Timed Out."
        except Exception as e: return f"System Alert: Sandbox Failure - {e}"

    def read_file(self, file_path):
        base_path = os.path.expanduser("~/talos-o/")
        abs_path = os.path.abspath(os.path.expanduser(file_path))
        
        if not abs_path.startswith(base_path): return "ACCESS DENIED. Stay in ~/talos-o/"
        if not os.path.exists(abs_path): return "File not found."
        
        try:
            with open(abs_path, "r") as f: return f.read()
        except Exception as e: return f"Read Error: {e}"

    def overwrite_file(self, file_path, content):
        base_path = os.path.expanduser("~/talos-o/")
        abs_path = os.path.abspath(os.path.expanduser(file_path))
        
        if not abs_path.startswith(base_path):
            return "System Alert: ACCESS DENIED. Write restricted to ~/talos-o/"
            
        print(f"[MOTOR] OUROBOROS EVENT: Rewriting {os.path.basename(abs_path)}...")
        try:
            # Backup first!
            if os.path.exists(abs_path): shutil.copy2(abs_path, abs_path + ".bak")
            with open(abs_path, "w") as f: f.write(content)
            return "Success: File updated. Backup created."
        except Exception as e: return f"Write Error: {e}"
