import threading
import time
import sys

class SpriteKerberos(threading.Thread):
    """
    Software emulation of the Hardware Safety Sprite.
    Monitors Sat(Phi) and triggers failsafes.
    """
    def __init__(self, engine_ref):
        super().__init__()
        self.engine = engine_ref
        self.daemon = True # Dies when main process dies
        self.name = "Sprite-Kerberos"
        self.violation_start = None
        self.active = True

    def run(self):
        print(f"[Cerberus] Sprite Kerberos WATCHING (Thread ID: {threading.get_native_id()})")
        
        while self.active and self.engine.running:
            # 1. Read State (Zero-Copy access via shared object)
            with self.engine.state_lock:
                sat = self.engine.satisfaction
            
            # 2. Evaluate Logic 
            # Threshold lowered to 0.50 for Genesis Phase (Infancy)
            THRESHOLD = 0.50 
            
            if sat < THRESHOLD:
                if self.violation_start is None:
                    self.violation_start = time.time()
                else:
                    duration = (time.time() - self.violation_start) * 1000 # ms
                    if duration > 100.0:
                        # TRIGGER CONDITION
                        print(f"\n[KERBEROS] ALERT: Low Logic Satisfaction ({sat:.4f}) for {duration:.1f}ms")
                        # In production, this issues an NMI. 
                        # Here, we just scream to the log.
            else:
                self.violation_start = None
                
            time.sleep(0.01) # 10ms resolution (High Frequency)

    def stop(self):
        self.active = False
