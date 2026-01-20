import os
import torch

class SafeLoader:
    """
    Secure File I/O handler to prevent Directory Traversal attacks.
    Uses realpath resolution and commonpath checks.
    """
    def __init__(self, base_dir):
        self.base_dir = os.path.realpath(os.path.expanduser(base_dir))
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def validate_path(self, user_input):
        # 1. Resolve target path
        target_abs = os.path.realpath(os.path.join(self.base_dir, user_input))
        
        # 2. Containment Check
        # os.path.commonpath returns the longest common sub-path
        try:
            common = os.path.commonpath([self.base_dir, target_abs])
        except ValueError:
            # Can happen on Windows if drives are different
            raise PermissionError("Access Denied: Path on different drive.")

        # 3. Verify strict prefix
        if common!= self.base_dir:
            raise PermissionError(f"Access Denied: Path traversal detected. {user_input} escapes jail.")
            
        return target_abs

    def load_weights(self, filename):
        safe_path = self.validate_path(filename)
        return torch.load(safe_path, map_location='cpu')

    def save_weights(self, state_dict, filename):
        safe_path = self.validate_path(filename)
        torch.save(state_dict, safe_path)
        print(f" Saved to {safe_path}")
