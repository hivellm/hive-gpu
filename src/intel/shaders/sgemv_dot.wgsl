// SGEMV-like compute shader: scores[i] = sum_d matrix[i, d] * query[d].
// matrix is row-major (n_vectors, dimension). One thread per output row.
//
// Matches src/shaders/metal_hnsw.metal::sgemv_dot and the behaviour of
// cuBLAS/rocBLAS SGEMV with trans=T used by the CUDA/ROCm backends.

struct Params {
    dimension: u32,
    n_vectors: u32,
}

@group(0) @binding(0) var<storage, read>        matrix: array<f32>;
@group(0) @binding(1) var<storage, read>        query: array<f32>;
@group(0) @binding(2) var<storage, read_write>  scores: array<f32>;

var<push_constant> pc: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= pc.n_vectors) {
        return;
    }
    let base = tid * pc.dimension;
    var sum: f32 = 0.0;
    for (var d: u32 = 0u; d < pc.dimension; d = d + 1u) {
        sum = sum + matrix[base + d] * query[d];
    }
    scores[tid] = sum;
}
