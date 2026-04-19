//! Build script for hive-gpu.
//!
//! - Emits rerun hints for CUDA kernel sources so the build re-runs when
//!   the CUDA assets change.
//! - When the `intel` feature is active, compiles the GLSL compute
//!   shaders under `src/intel/shaders/` into SPIR-V via the `shaderc`
//!   crate and writes the binaries to `$OUT_DIR` so Rust can
//!   `include_bytes!` them at compile time.

fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        println!("cargo:rerun-if-changed=src/cuda/kernels.cu");
        println!("cargo:rerun-if-changed=src/cuda/kernels");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        println!("cargo:rerun-if-env-changed=CUDA_HOME");
    }

    #[cfg(feature = "intel")]
    compile_intel_shaders();
}

#[cfg(feature = "intel")]
fn compile_intel_shaders() {
    use std::path::Path;

    let shader_dir = Path::new("src/intel/shaders");
    let out_dir =
        std::env::var("OUT_DIR").expect("OUT_DIR is always set by cargo during build scripts");

    for entry in ["sgemv_dot.wgsl", "sgemm_dot.wgsl"] {
        let src_path = shader_dir.join(entry);
        println!("cargo:rerun-if-changed={}", src_path.display());
        let source = std::fs::read_to_string(&src_path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", src_path.display()));

        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("wgsl parse of {}: {e:?}", src_path.display()));
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::PUSH_CONSTANT,
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("wgsl validate of {}: {e:?}", src_path.display()));

        // Target Vulkan 1.2 / SPIR-V 1.5 to match the IntelContext
        // creation call.
        let spv_options = naga::back::spv::Options {
            lang_version: (1, 5),
            flags: naga::back::spv::WriterFlags::empty(),
            ..naga::back::spv::Options::default()
        };

        let spirv = naga::back::spv::write_vec(&module, &info, &spv_options, None)
            .unwrap_or_else(|e| panic!("spirv emit of {}: {e:?}", src_path.display()));
        // Naga returns Vec<u32>; flatten to bytes.
        let mut bytes = Vec::with_capacity(spirv.len() * 4);
        for word in &spirv {
            bytes.extend_from_slice(&word.to_le_bytes());
        }

        let out_name = entry.replace(".wgsl", ".spv");
        let out_path = Path::new(&out_dir).join(out_name);
        std::fs::write(&out_path, &bytes)
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", out_path.display()));
    }
}
