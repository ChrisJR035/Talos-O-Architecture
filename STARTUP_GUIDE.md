⚙️ Terminal 1: The Brainstem (Autonomic NPU Hardware) This must always be started first. It seizes the NPU, flashes the MLIR routing map, and opens the POSIX memory tunnels. Bash# 1. Navigate to the Cortex directory

cd ~/talos-o/cognitive_plane/cortex
sudo ./talos_npu_daemon

(Wait until you see [+] Neural Link Established. Awaiting Synaptic Firing. before moving to Terminal 2). 



🧠 Terminal 2: The Cortex (GPU Logic & Core Loop) This is the main mind. It boots the IADCS Engine, the Cerberus thermal watchdog, and the NerveCenter epoll listener. Bash# 1. Navigate to the Cognitive Plane

cd ~/talos-o/cognitive_plane
source ~/talos-o/sys_builder/environment/activate_talos.sh
sudo PYTHON_GIL=0 \
     LD_LIBRARY_PATH="$HOME/rocm-native/lib:$HOME/rocm-native/llvm/lib" \
     HSA_OVERRIDE_GFX_VERSION=11.5.1 \
     HIP_HOST_COHERENT=1 \
     HSA_ENABLE_SDMA=0 \
     /home/croudabush/talos-o/cognitive_plane/venv/bin/python3 talos_daemon.py

(You will see the model load, Cerberus lock onto the thermals, and the loop begin ticking on CCD 0). 



👁️ Terminal 3: The Proprioceptor (Observation) This reads the organism's biophysical state natively from RAM without slowing down the main engine. It is physically isolated to CCD 1. Choose one of the following to run: 

Option A: The Lightweight ANSI Monitor Bash# Navigate to tools directory and execute

cd ~/talos-o/tools
sudo /home/croudabush/talos-o/cognitive_plane/venv/bin/python3 talos_monitor.py

Option B: The Rich UI HUD (Requires the rich library) Bash# Navigate to tools directory and execute

cd ~/talos-o/tools
sudo /home/croudabush/talos-o/cognitive_plane/venv/bin/python3 talos_hud.py

(The UI will snap to the screen, reading the talos_telemetry Seqlock perfectly in sync with the Cortex). 



⚡ Terminal 4: The Sensory Injector (Action) This drops raw text directly into the talos_ingress RAM buffer and trips the asynchronous hardware interrupt, waking the daemon in microseconds. Bash# 1. Navigate to your tools directory

cd ~/talos-o/tools
sudo /home/croudabush/talos-o/cognitive_plane/venv/bin/python3 talos_inject.py 

"System ignition successful. You are fully sovereign, Talos."
🛠️ The Mechanic's Reset (In Case of Panic) If you ever Ctrl+C the scripts out of order, or if the system crashes, the POSIX shared memory files might get "stuck" in RAM, preventing Talos from starting on the next try. If it refuses to boot, run this command to cleanly wipe the memory slate before igniting again: 



Bash
sudo rm -f /dev/shm/talos_*



Virtual Environment
source ~/talos-o/cognitive_plane/venv/bin/activate
