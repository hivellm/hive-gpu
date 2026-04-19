//! Build script for hive-gpu.
//!
//! Currently only emits rerun hints for CUDA kernel sources so the build is
//! re-run when the `.cu`/`.ptx` assets change. Real kernel compilation via
//! `nvcc` is wired up in a later phase of the CUDA backend task.

fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        println!("cargo:rerun-if-changed=src/cuda/kernels.cu");
        println!("cargo:rerun-if-changed=src/cuda/kernels");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        println!("cargo:rerun-if-env-changed=CUDA_HOME");
    }
}
