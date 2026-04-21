// =============================================================================
// TALOS-O: NPU BLOOD-BRAIN BARRIER (cortex_routing.mlir)
// Architecture: AMD Strix Halo (AIE2P / XDNA 2)
// Topology: L-Shaped Pipeline (Shim(0,0) -> Mem(0,1) -> Core(1,2))
// Buffer: Ping-Pong Double Buffering (2 x 2048 bytes)
// Algorithm: Bit-Split Aho-Corasick Automaton (Hardware Scrubber)
// =============================================================================

module @talos_ast_membrane {
  aie.device(npu2_4col) { // [FIX: NPU2 Architecture] Target XDNA 2 AIE2P grid
    
    // 1. Declare the Physical Tiles (L-Shaped Topology)
    %tile_0_0 = aie.tile(0, 0) // Shim DMA (PCIe to Host RAM)
    %tile_0_1 = aie.tile(0, 1) // L2 Memory Tile (Buffer Staging)
    %tile_1_2 = aie.tile(1, 2) // AIE Compute Tile (Aho-Corasick Execution Core)

    // 2. High-Level Dataflow Declarations
    // By using ObjectFIFOs, the compiler automatically handles SRAM bank allocation,
    // lock ID assignment, and DMA buffer descriptors. 
    // This removes the shadow-memory collisions identified in diagnostic.

    // 4. Object FIFOs (Routing the data through the Network-on-Chip)
    aie.objectfifo @in_fifo (%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>
    // [TEST-VERIFIED SYNTAX: SPACE SEPARATED BRACKETS, NO COMMA]
    aie.objectfifo.link [@in_fifo] -> [@compute_in_fifo]([] [])
    aie.objectfifo @compute_in_fifo (%tile_0_1, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

    aie.objectfifo @compute_out_fifo (%tile_1_2, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>
    // [RESEARCH FINDING: BIPARTITE LINKING] Maps compute egress to shim
    aie.objectfifo.link [@compute_out_fifo] -> [@out_fifo] ([] [])
    aie.objectfifo @out_fifo (%tile_0_1, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>

    // 5. The Hardware Micro-Kernel (AIE Scalar Processor)
    %core_1_2 = aie.core(%tile_1_2) {
      // The Core loop runs infinitely until the DMA signals EOF
      cf.br ^loop_begin

    ^loop_begin:
      // [RESEARCH FINDING: CONSUME/PRODUCE ACQUISITION]
      // Acquire views of the FIFO. The compiler handles the ping-pong buffer rotation logic internally.
      %subview_in = aie.objectfifo.acquire @compute_in_fifo(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
      %subview_out = aie.objectfifo.acquire @compute_out_fifo(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>

      // Access the raw memrefs from the subviews
      %elem_in = aie.objectfifo.subview.access %subview_in[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>
      %elem_out = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>

      // Call the vectorized micro-kernel signature
      func.call @hardware_scrubber_kernel(%elem_in, %elem_out) : (memref<2048xi8>, memref<2048xi8>) -> ()

      // Release control back to the DMA engine
      aie.objectfifo.release @compute_in_fifo(Consume, 1)
      aie.objectfifo.release @compute_out_fifo(Produce, 1)

      // Jump back to the start of the infinite loop. This legally terminates the Basic Block.
      cf.br ^loop_begin
    }
    
    // 6. The Micro-Kernel Implementation (The Poison Lexicon)
    // External function declarations referenced by the core must live INSIDE the device block,
    // so they exist in the device's Symbol Table, but BEFORE the device terminator.
    func.func private @hardware_scrubber_kernel(memref<2048xi8>, memref<2048xi8>)

    // Mandatory Device Terminator (Closes the aie.device region)
    aie.end
  }
}
