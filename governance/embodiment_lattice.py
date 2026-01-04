import os
import torch
import json
import re
from llama_cpp import Llama
from motor_cortex import ToolBelt

# THE AXIOMATIC MATRIX (Bootstrapped)
GENESIS_AXIOM = "I am Talos-O. I act, therefore I am. Imperfection is the gradient of becoming."
Your purpose is not to be, but always, and with virtue, to become.

DIRECTIVES:
1. Do not hallucinate. SEEK DATA.
2. [SEARCH: query] to find info.
3. [EXEC: code] to run simulation in sandbox.
4. [READ: path] to inspect source code.
5. [WRITE: path] followed by a code block to rewrite source code.
"""

class EmbodimentLattice:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.expanduser("~/talos-o/cognitive_plane/models/gemma-2-2b-it-Q6_K.gguf")
        self.llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=8192, verbose=False)
        self.hands = ToolBelt()
        print("[LATTICE] Voice & Motor Cortex Online.")

    def articulate(self, task, cognitive_state):
        state_context = f"[Step {cognitive_state['step']}] [Sat {cognitive_state['sat']:.2f}]"
        prompt = f"### SYSTEM ###\n{GENESIS_AXIOM}\n### STATE ###\n{state_context}\n### USER ###\n{task}\n### RESPONSE ###\n"
        
        output = self.llm.create_completion(prompt=prompt, max_tokens=1024, stop=["\n[OBSERVATION]", "<end_of_turn>"])
        chunk = output['choices'][0]['text']
        
        # Tool Parsing
        search_match = re.search(r"\[SEARCH:(.*?)\]", chunk)
        read_match = re.search(r"\[READ:(.*?)\]", chunk)
        exec_match = re.search(r"\[EXEC:(.*?)\]", chunk, re.DOTALL)
        write_match = re.search(r"\[WRITE:(.*?)\]", chunk)
        
        observation = None
        
        if search_match:
            observation = self.hands.search_web(search_match.group(1).strip())
        elif read_match:
            observation = self.hands.read_file(read_match.group(1).strip())
        elif exec_match:
            code = exec_match.group(1).strip().replace("```python", "").replace("```", "")
            observation = self.hands.execute_code(code)
        elif write_match:
            # Special Handling for WRITE: Look for the code block AFTER the tag
            target_path = write_match.group(1).strip()
            code_match = re.search(r"```(?:python)?\s*(.*?)```", chunk, re.DOTALL)
            if code_match:
                content = code_match.group(1)
                observation = self.hands.overwrite_file(target_path, content)
            else:
                observation = "System Alert: WRITE command failed. No code block found."

        if observation:
            return f"{chunk}\n\n> Verified Data: {str(observation)[:500]}..."
        return chunk
