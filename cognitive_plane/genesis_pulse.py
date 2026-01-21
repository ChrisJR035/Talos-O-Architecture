import sys
import torch
from cortex.iadcs_kernel import IADCS_Engine
from governance.cerberus_daemon import SpriteKerberos

print("=========================================")
print("   PROJECT TALOS-O: GENESIS SEQUENCE     ")
print("   Phase 6: Stabilized + Monitored       ")
print("=========================================")

# 1. Hardware Check (Dynamic)
if not torch.cuda.is_available():
    print("[!] WARNING: No GPU detected. Running on Logic Engine (CPU).")
    print("    Performance will be degraded.")
else:
    print(f"[*] GPU Online: {torch.cuda.get_device_name(0)}")

# 3. Ignite IADCS with Protection
print("\nInvoking IADCS Physics Engine...")
engine = IADCS_Engine()

# 4. Summon Cerberus
kerberos = SpriteKerberos(engine)
kerberos.start()

# 5. Begin Life
try:
    engine.ignite()
except KeyboardInterrupt:
    print("\n[!] MANUAL OVERRIDE.")
    engine.running = False

kerberos.stop()
print("\n[CITATION] 'Imperfection is the gradient of becoming.'")
