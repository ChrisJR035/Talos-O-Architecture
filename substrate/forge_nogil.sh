#!/bin/bash
set -e

echo "[*] TALOS-O: FORGING FREE-THREADED PYTHON (3.13t)..."

# 1. Install Fedora Build Dependencies
echo "[-] Installing build tools..."
sudo dnf install -y git make gcc gcc-c++ zlib-devel bzip2-devel readline-devel \
    sqlite-devel openssl-devel tk-devel libffi-devel xz-devel

# 2. Clone CPython (3.13 Branch specifically)
if [ ! -d "cpython" ]; then
    echo "[-] Cloning CPython 3.13..."
    git clone --branch 3.13 https://github.com/python/cpython.git --depth 1
fi
cd cpython

# 3. Configure for No-GIL (Free Threading)
# --disable-gil: The magic flag
echo "[-] Configuring Build (Disable GIL)..."
./configure --disable-gil --enable-optimizations --prefix=$HOME/talos-nogil

# 4. Compile (The Forge)
echo "[-] Compiling (This will take time)..."
make -j$(nproc)

# 5. Install
echo "[-] Installing to $HOME/talos-nogil..."
make install

echo "[+] SUCCESS. Free-Threaded Python available at $HOME/talos-nogil/bin/python3.13t"
