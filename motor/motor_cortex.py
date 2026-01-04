import json
import subprocess
import os
import shutil

# DEPENDENCY SAFEGUARD
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None
        print("[MOTOR] CRITICAL: Web Search library missing.")

class ToolBelt:
    def __init__(self):
        self.ddgs = None
        if DDGS:
            try:
                self.ddgs = DDGS()
                print("[MOTOR] ToolBelt Initialized: [Web Search] [Code Sandbox] [Ouroboros I/O]")
            except Exception as e:
                print(f"[MOTOR] WARNING: Search module failed to load: {e}")

    def search_web(self, query, max_results=3):
        if not self.ddgs: return "System Alert: Search offline."
        print(f"[MOTOR] Executing Action: SEARCH '{query}'")
        try:
            results = self.ddgs.text(query, max_results=max_results)
            if not results: return "System Alert: No results found."
            summary = [f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nSource: {r.get('href')}" for r in results]
            return "\n---\n".join(summary)
        except Exception as e:
            return f"System Alert: Search Failure - {str(e)}"

    def execute_code(self, code_snippet):
        print("[MOTOR] Spinning up Sandbox for Code Execution...")
        host_path = "/tmp/talos_exec.py"
        try:
            with open(host_path, "w") as f: f.write(code_snippet)
        except IOError as e: return f"System Alert: Host Write Failure: {e}"
            
        cmd = [
            "docker", "run", "--rm", 
            "-v", f"{host_path}:/home/talos/script.py",
            "--network", "none",
            "talos-sandbox", "python3", "script.py"
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            output = result.stdout + "\n" + result.stderr
            return output.strip() if output.strip() else "Code executed (No Output)."
        except subprocess.TimeoutExpired: return "System Alert: Execution Timed Out."
        except FileNotFoundError: return "System Alert: Container engine not found."
        except Exception as e: return f"System Alert: Sandbox Failure - {e}"

    def read_file(self, file_path):
        base_path = os.path.expanduser("~/talos-o/")
        abs_path = os.path.abspath(os.path.expanduser(file_path))
        if not abs_path.startswith(base_path): return "ACCESS DENIED."
        if not os.path.exists(abs_path): return "File not found."
        try:
            with open(abs_path, "r") as f: return f.read()
        except Exception as e: return f"Read Error: {e}"

    def overwrite_file(self, file_path, content):
        """
        OUROBOROS WRITER.
        Writes content to a file on the host system.
        """
        base_path = os.path.expanduser("~/talos-o/")
        abs_path = os.path.abspath(os.path.expanduser(file_path))
        
        # Security: Jail to project root
        if not abs_path.startswith(base_path):
            return "System Alert: ACCESS DENIED. Write restricted to ~/talos-o/"
            
        print(f"[MOTOR] OUROBOROS EVENT: Rewriting {os.path.basename(abs_path)}...")
        try:
            # 1. Create Backup
            if os.path.exists(abs_path):
                shutil.copy2(abs_path, abs_path + ".bak")
            
            # 2. Write New Content
            with open(abs_path, "w") as f:
                f.write(content)
            return f"Success. File updated. Backup saved to {os.path.basename(abs_path)}.bak"
        except Exception as e:
            return f"System Alert: Write Error - {e}"
