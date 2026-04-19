// SGEMM-lite compute shader: out[i, j] = sum_d samples[i, d] * centroids[j, d].
// samples row-major (n_samples, dimension); centroids row-major
// (n_list, dimension); output row-major (n_samples, n_list).
// 2D grid — one thread per output cell.

struct Params {
    dimension: u32,
    n_list: u32,
    n_samples: u32,
}

@group(0) @binding(0) var<storage, read>        samples: array<f32>;
@group(0) @binding(1) var<storage, read>        centroids: array<f32>;
@group(0) @binding(2) var<storage, read_write>  out_: array<f32>;

var<push_constant> pc: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= pc.n_samples || j >= pc.n_list) {
        return;
    }
    let sbase = i * pc.dimension;
    let cbase = j * pc.dimension;
    var sum: f32 = 0.0;
    for (var d: u32 = 0u; d < pc.dimension; d = d + 1u) {
        sum = sum + samples[sbase + d] * centroids[cbase + d];
    }
    out_[i * pc.n_list + j] = sum;
}
