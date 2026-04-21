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
// [PHASE 2 FIX: UNIX DOMAIN SOCKETS]
#include <sys/socket.h>
#include <sys/un.h>

__attribute__((constructor)) static void bind_xrt_root() { 
    const char* home = getenv("HOME");
    if (home) {
        std::string local_xrt = std::string(home) + "/talos-o/sys_builder/xrt_local";
        setenv("XILINX_XRT", local_xrt.c_str(), 0);
    }
}

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_hw_context.h>

#define TENSOR_BYTES 4096 
#define SHM_INGRESS "/talos_latent_in"
#define SHM_EGRESS "/talos_latent_out"

// [AXIOM 10] Shared Memory structure to map Python telemetry
struct BiophysicalState {
    uint64_t sequence;
    uint64_t step_count;
    uint8_t thermal_state;
    uint8_t padding[7];
    double gradient_dvdt;
    double satisfaction;
    double kuramoto_r;
    double entropy;
};

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

    // [AXIOM 10] Map telemetry for hardware-level pausing
    int fd_telem = shm_open("talos_telemetry", O_RDONLY, 0666);
    BiophysicalState* telemetry = nullptr;
    if (fd_telem >= 0) {
        telemetry = static_cast<BiophysicalState*>(mmap(0, sizeof(BiophysicalState), PROT_READ, MAP_SHARED, fd_telem, 0));
    }

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
    // [PHASE 2 FIX: SCM_RIGHTS LISTENER THREAD]
    // Spawns a background thread to listen for raw Python File Descriptors
    // bypassing the CPU memory heap for AST validation.
    // =========================================================================
    std::thread ipc_listener([&]() {
        int server_fd, client_fd;
        struct sockaddr_un address;
        
        server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (server_fd < 0) return;
        
        memset(&address, 0, sizeof(struct sockaddr_un));
        address.sun_family = AF_UNIX;
        strncpy(address.sun_path, "/tmp/talos_ast_membrane.sock", sizeof(address.sun_path) - 1);
        unlink(address.sun_path);
        
        if (bind(server_fd, (struct sockaddr*)&address, sizeof(struct sockaddr_un)) < 0) return;
        if (listen(server_fd, 5) < 0) return;
        
        std::cout << "  \033[35m[IPC] SCM_RIGHTS Socket Active: /tmp/talos_ast_membrane.sock\033[0m\n";
        
        while (true) {
            client_fd = accept(server_fd, NULL, NULL);
            if (client_fd >= 0) {
                struct msghdr msg = {0};
                char m_buffer[1];
                struct iovec io = { .iov_base = m_buffer, .iov_len = sizeof(m_buffer) };
                msg.msg_iov = &io;
                msg.msg_iovlen = 1;
                
                char c_buffer[256];
                msg.msg_control = c_buffer;
                msg.msg_controllen = sizeof(c_buffer);
                
                if (recvmsg(client_fd, &msg, 0) >= 0) {
                    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
                    if (cmsg && cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
                        int received_fd;
                        memcpy(&received_fd, CMSG_DATA(cmsg), sizeof(int));
                        std::cout << "  \033[36m[IPC] Received Python AST File Descriptor: " << received_fd << ".\033[0m\n";
                        
                        // NOTE: Phase 3 (The AIE Aho-Corasick Scrubber) will map this FD to L2 SRAM.
                        // For Phase 2 testing, we simply close the connection to force the CPU fallback.
                    }
                }
                close(client_fd);
            }
        }
    });
    ipc_listener.detach();

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
        
        // [PHASE 3 FIX: THE EPISTEMIC OFFLOAD]
        // Map the ONNX-quantized Softmax/Cross-Entropy operators for the NPU context.
        // Opcode 1.0 = Standard Latent Evolution (IADCS Engine)
        // Opcode 2.0 = Linguistic Coherence Probe (Reflective Cortex Perplexity Audit)
        uint32_t active_kernel_opcode = 3; // Default MLIR_AIE Instruction Group
        if (current_pulse == 2.0f) {
            std::cout << "\033[35m[NPU BRAINSTEM] Intercepted Epistemic Audit (Opcode 2.0). Routing to Softmax/CE hardware tiles...\033[0m\n";
            active_kernel_opcode = 4; // Map to the Quantized Softmax/CE kernel instruction set
        }

        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Engage the XDNA 2 Silicon
        auto run = execution_matrix(active_kernel_opcode, bo_instr, instr_v.size(), bo_in, bo_out);
        
        // [AXIOM 10: HARDWARE-LEVEL HOMEOSTASIS]
        // Poll execution. If a massive power spike (>15W) is detected,
        // yield the CPU timeslice to drop voltage and prevent GPU panic.
        ert_cmd_state state = run.state();
        while (state != ERT_CMD_STATE_COMPLETED && state != ERT_CMD_STATE_ERROR && state != ERT_CMD_STATE_ABORT) {
            state = run.wait(std::chrono::milliseconds(1));
            
            if (telemetry != nullptr && telemetry->gradient_dvdt > 15.0) {
                sched_yield(); 
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
        
        // After the while loop breaks, check the final state
        if (state == ERT_CMD_STATE_ERROR || state == ERT_CMD_STATE_ABORT) {
            std::cerr << "\033[1;31m[FATAL] XDNA 2 Hardware Execution Aborted.\033[0m\n";
            continue; // Skip to the next thought
        }

        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        // Consume the thought
        in_tensor[0] = 0.0f;
        cycle_count++;
        std::cout << "\033[1;36m[SYNAPSE] Pulse Processed! Cycle count: \033[0m" << cycle_count << "\n";
    }

    return 0;
}
