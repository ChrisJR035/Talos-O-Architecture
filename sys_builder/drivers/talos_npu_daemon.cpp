// =============================================================================
// TALOS-O: NPU BRAINSTEM (v18.2 - The Context Matrix)
// Architecture: AMD Strix Halo (AIE2P / XDNA 2) / Linux amdxdna 6.19
// Philosophy: "Zero Abstraction. Absolute Thermodynamic Purity."
//
// [COMMIT: v18.2 - NPU Context Shift + Thermodynamic Yield]
// - Replaced legacy FPGA load_axlf with NPU-native xrt::hw_context.
// - Binds kernels and zero-copy BOs to the isolated hardware context.
// - Injected Thermodynamic Yield to prevent 100% spinlock CPU starvation.
// =============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>

__attribute__((constructor)) static void bind_xrt_root() { 
    setenv("XILINX_XRT", "/opt/xilinx", 0); 
}

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_hw_context.h>

#define TENSOR_BYTES 4096 
#define SHM_INGRESS "/talos_latent_in"
#define SHM_EGRESS "/talos_latent_out"

std::vector<uint32_t> load_instruction_sequence(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "\033[1;31m[FATAL] Missing NPU Instruction Map: " << filename << "\033[0m\n";
        exit(1);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint32_t> buffer(size / sizeof(uint32_t));
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        return buffer;
    }
    exit(1);
}

// [FIX M-7]: Dynamic path resolution ensuring the daemon can be launched anywhere
std::string get_xclbin_path() {
    const char* override_path = getenv("TALOS_XCLBIN");
    if (override_path) return std::string(override_path);
    const char* home = getenv("HOME");
    if (!home) { 
        std::cerr << "\033[1;31m[FATAL] $HOME not set. Cannot resolve cortex.xclbin\033[0m\n"; 
        exit(1); 
    }
    return std::string(home) + "/talos-o/cognitive_plane/cortex/cortex.xclbin";
}

int main() {
    std::cout << "\033[96m[NPU DAEMON] Igniting XDNA 2 Brainstem...\033[0m\n";

    std::string xclbin_path = get_xclbin_path();
    xrt::device device(0);
    xrt::xclbin xclbin_obj(xclbin_path);
    auto uuid = device.register_xclbin(xclbin_obj);
    
    xrt::hw_context hw_ctx(device, uuid);
    xrt::kernel execution_matrix(hw_ctx, "MLIR_AIE");

    auto instr_v = load_instruction_sequence(std::string(getenv("HOME")) + "/talos-o/cognitive_plane/cortex/insts.bin");

    int fd_in = shm_open(SHM_INGRESS, O_CREAT | O_RDWR, 0666);
    int fd_out = shm_open(SHM_EGRESS, O_CREAT | O_RDWR, 0666);
    
    ftruncate(fd_in, TENSOR_BYTES);
    ftruncate(fd_out, TENSOR_BYTES);
    
    void* shm_in_ptr = mmap(0, TENSOR_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd_in, 0);
    void* shm_out_ptr = mmap(0, TENSOR_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd_out, 0);

    std::cout << "  \033[33m[-] Allocating NPU Memory Banks...\033[0m\n";

    xrt::bo bo_instr(hw_ctx, instr_v.size() * sizeof(uint32_t), xrt::bo::flags::cacheable, execution_matrix.group_id(1));
    xrt::bo bo_in(hw_ctx, shm_in_ptr, TENSOR_BYTES, xrt::bo::flags::host_only, execution_matrix.group_id(3));
    xrt::bo bo_out(hw_ctx, shm_out_ptr, TENSOR_BYTES, xrt::bo::flags::host_only, execution_matrix.group_id(4));

    bo_instr.write(instr_v.data());
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "  \033[32m[+] Neural Link Established. Awaiting Synaptic Firing.\033[0m\n";

    unsigned long long cycle_count = 0;
    float* in_tensor = static_cast<float*>(shm_in_ptr);

    // =========================================================================
    // PHASE 4: THE DETERMINISTIC EXECUTION LOOP
    // =========================================================================
    while (true) {
        
        // [FIX: THERMODYNAMIC YIELD]
        // If there is no thought pending (0.0f), yield the CPU scheduler.
        // Prevents a 100% spinlock thermal meltdown on the host Zen 5 core.
        if (in_tensor[0] == 0.0f) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            continue;
        }

        float current_pulse = in_tensor[0];
        
        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Engage the XDNA 2 Silicon
        auto run = execution_matrix(3, bo_instr, instr_v.size(), bo_in, bo_out);
        run.wait();

        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        // Consume the thought
        in_tensor[0] = 0.0f;
        cycle_count++;
        std::cout << "\033[1;36m[SYNAPSE] Pulse Processed! Cycle count: \033[0m" << cycle_count << "\n";
    }

    return 0;
}
