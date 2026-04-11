module @cortex_core {
  aie.device(npu2) {
    // 1. Explicit Tile Definitions (Row 0: Shim, Row 1: Mem Tile, Row 2: Compute)
    %tile_0_0 = aie.tile(0, 0) 
    %tile_0_1 = aie.tile(0, 1) 
    %tile_0_2 = aie.tile(0, 2) 

    // 2. Zero-Copy Pipeline Orchestration (ObjectFifo)
    // Shim -> Mem Tile (Ping-Pong buffering in L2 Cache)
    aie.objectfifo @in_stage0 (%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    
    // Mem Tile -> Compute Tile (High-Speed Local Delivery)
    aie.objectfifo @in_stage1 (%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>

    // 3. Memory Controller Stream Binding (Explicit AIE2P Syntax)
    aie.objectfifo.link [@in_stage0] -> [@in_stage1] ([] [])
  }
}