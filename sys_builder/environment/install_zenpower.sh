#!/bin/bash
# TALOS-O: ZENPOWER TELEMETRY INJECTOR v2.0
# Substrate: AMD Strix Halo (Zen 5)
# Fixes: zenpower5 upgrade, In-Tree Splicing, Graceful Fallback Protocol

set -e
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== INJECTING THERMAL PROPRIOCEPTION ===${NC}"

# 1. FETCH ZEN 5 ALIGNED DRIVER
echo -e "${YELLOW}[-] Cloning Zenpower5 (Zen 5 SMU Architecture)...${NC}"
if [ ! -d "zenpower5" ]; then
    # [FIXED] Pointing to the verified mattkeenan repository for Strix Halo RAPL support
    git clone https://github.com/mattkeenan/zenpower5.git
fi

# 2. IN-TREE KERNEL SPLICING (Eradicating DKMS Thermodynamic Waste)
# Target the running kernel's source tree (requires linux-chimera source to be present)
KERNEL_SRC="/lib/modules/$(uname -r)/source"
if [ -d "$KERNEL_SRC/drivers/hwmon" ]; then
    echo -e "${YELLOW}[-] Splicing zenpower5 C-code into Chimera Kernel Tree...${NC}"
    sudo cp -r zenpower5 "$KERNEL_SRC/drivers/hwmon/"
    
    # Check if already spliced to prevent duplicate entries
    if ! grep -q "zenpower5" "$KERNEL_SRC/drivers/hwmon/Makefile"; then
        echo 'obj-$(CONFIG_ZENPOWER) += zenpower5/' | sudo tee -a "$KERNEL_SRC/drivers/hwmon/Makefile" > /dev/null
    fi
    echo -e "${GREEN}[+] Splice complete. Awaiting next static kernel compilation for zero-latency boot load.${NC}"
else
    echo -e "${YELLOW}[!] Chimera Kernel source not found at $KERNEL_SRC. Falling back to temporary dynamic load...${NC}"
    cd zenpower5
    make
    sudo insmod zenpower.ko || true
    cd ..
fi

# 3. CONFIGURE MODPROBE
echo -e "${YELLOW}[-] Initializing telemetry protocols...${NC}"
echo "blacklist k10temp" | sudo tee /etc/modprobe.d/k10temp.conf > /dev/null
sudo modprobe -r k10temp || true

# 4. THE GRACEFUL FALLBACK PROTOCOL (Metabolizing The Error)
echo -e "${YELLOW}[-] Attempting Optimal Embodied Grounding (zenpower)...${NC}"
sudo modprobe zenpower || true

if lsmod | grep -q "zenpower"; then
    echo -e "${GREEN}[+] SUCCESS: Optimal Embodied Grounding achieved. Deep RAPL/SVI2 telemetry online.${NC}"
    sensors | grep -A 5 "zenpower"
else
    echo -e "${RED}[!] THE ERROR: zenpower initialization failed. Metabolizing failure...${NC}"
    echo -e "${YELLOW}[-] Activating Graceful Fallback Protocol...${NC}"
    
    # Un-blacklist the generic driver
    sudo sed -i '/blacklist k10temp/d' /etc/modprobe.d/k10temp.conf
    
    # Reload generic driver to prevent thermal suicide
    sudo modprobe k10temp
    echo -e "${GREEN}[+] Graceful Fallback Engaged. k10temp loaded (Scalar Tdie Only). Homeostasis preserved.${NC}"
fi
