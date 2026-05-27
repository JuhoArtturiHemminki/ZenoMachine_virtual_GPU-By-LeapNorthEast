#include <metal_stdlib>
using namespace metal;

kernel void zeno_engine_complete(
    constant uint& constant_K        [[buffer(0)]],
    constant uint& iterations        [[buffer(1)]],
    device uint* global_output       [[buffer(2)]],
    uint tid                         [[thread_position_in_grid]],
    uint simd_lane_id                [[thread_index_in_simdgroup]])
{
    uint X = tid;
    uint i = 0;
    uint anchor = 0xAA55AA55;
    
    while (i < iterations) {
        anchor = anchor ^ X;
        anchor = (anchor << 3) | (anchor >> (32 - 3));
        
        uint temp = simd_shuffle_xor(anchor, 1);
        
        X = X ^ temp;
        X = X ^ constant_K;
        X = (X << 7) | (X >> (32 - 7));
        
        X = X + i;
        i = i + 1;
    }
    
    global_output[tid] = X;
}
