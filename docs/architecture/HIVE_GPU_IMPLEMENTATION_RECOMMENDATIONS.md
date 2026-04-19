# Recomendações de Implementação para hive-gpu

**Date:** 2025-01-07  
**Version:** 1.0  
**Author:** Vectorizer Team  
**Target:** hive-gpu Library (v0.2.0+)

---

## 📋 Executive Summary

Este documento detalha as **recomendações críticas** para expansão da biblioteca `hive-gpu`, necessárias para suportar aceleração GPU em **todas as plataformas principais** do Vectorizer.

### Current Status

| Backend | Status | Plataformas | Cobertura de Mercado |
|---------|--------|-------------|---------------------|
| **Metal** | ✅ Implementado | macOS | ~5% servidores |
| **CUDA** | ❌ Faltando | Linux/Windows (NVIDIA) | ~70% servidores |
| **ROCm** | ❌ Faltando | Linux (AMD) | ~15% servidores |
| **WebGPU** | ❌ Faltando | Cross-platform | ~10% uso geral |

### Impacto da Implementação

**Se CUDA for implementado:**
- ✅ **95% dos servidores de produção** poderão usar GPU
- ✅ **10-50x speedup** em operações vetoriais
- ✅ **Latência reduzida** de 10-30ms para 0.5-3ms
- ✅ **ROI altíssimo** para apenas 1-2 semanas de desenvolvimento

---

## 🎯 Objetivos Principais

### 1. **CUDA Backend** (Prioridade: 🔥 CRÍTICA)
- Suportar NVIDIA GPUs em Linux e Windows
- Atingir 70% do mercado de servidores ML/AI
- Performance competitiva com implementações nativas CUDA

### 2. **Device Info API** (Prioridade: 🔥 ALTA)
- Fornecer informações detalhadas do GPU
- Suportar consulta de VRAM disponível/total
- Identificar modelo de GPU e driver version

### 3. **ROCm Backend** (Prioridade: ⚡ MÉDIA)
- Suportar AMD GPUs em Linux
- Atingir 15% adicional do mercado

### 4. **WebGPU Backend** (Prioridade: 📦 BAIXA)
- Fallback universal cross-platform
- Suporte para deployment em browsers

### 5. **Memory Pooling** (Prioridade: ⚡ MÉDIA)
- Otimizar alocação/deallocação de memória
- Reduzir overhead de operações batch

### 6. **Async Operations** (Prioridade: 📦 BAIXA)
- Operações GPU assíncronas
- Melhor integração com Tokio runtime

---

## 🏗️ Arquitetura Recomendada

### Estrutura de Diretórios Proposta

```
hive-gpu/
├── src/
│   ├── lib.rs                 # Core traits and types
│   ├── error.rs               # Error types
│   ├── types.rs               # Common types (GpuVector, etc.)
│   │
│   ├── metal/                 # ✅ Existente
│   │   ├── mod.rs
│   │   ├── context.rs
│   │   └── storage.rs
│   │
│   ├── cuda/                  # ❌ IMPLEMENTAR (Phase 1)
│   │   ├── mod.rs
│   │   ├── context.rs
│   │   ├── storage.rs
│   │   ├── kernels.cu         # CUDA kernels
│   │   └── utils.rs
│   │
│   ├── rocm/                  # ❌ IMPLEMENTAR (Phase 2)
│   │   ├── mod.rs
│   │   ├── context.rs
│   │   ├── storage.rs
│   │   └── kernels.hip        # HIP kernels
│   │
│   ├── wgpu/                  # ❌ IMPLEMENTAR (Phase 3)
│   │   ├── mod.rs
│   │   ├── context.rs
│   │   ├── storage.rs
│   │   └── shaders.wgsl       # WGSL shaders
│   │
│   └── pool/                  # ❌ IMPLEMENTAR (Phase 4)
│       ├── mod.rs
│       └── memory_pool.rs
│
├── benches/                   # Benchmarks por backend
│   ├── metal_bench.rs
│   ├── cuda_bench.rs
│   └── comparison.rs
│
├── tests/                     # Integration tests
│   ├── metal_tests.rs
│   ├── cuda_tests.rs
│   └── cross_backend.rs
│
├── build.rs                   # Build script (CUDA/ROCm compilation)
├── Cargo.toml
└── README.md
```

---

## 📦 Phase 1: Device Info API (Prioridade: 🔥 ALTA)

**Tempo Estimado:** 1-2 dias  
**Complexidade:** Baixa  
**Impacto:** Alto (usado por todos backends)

### Especificação da API

#### 1.1 Nova Struct: `GpuDeviceInfo`

```rust
// src/types.rs

/// Detailed GPU device information
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// GPU device name (e.g., "Apple M1 Pro", "NVIDIA RTX 4090")
    pub name: String,
    
    /// Backend type
    pub backend: String,
    
    /// Total VRAM in bytes
    pub vram_total: usize,
    
    /// Available VRAM in bytes
    pub vram_available: usize,
    
    /// Currently used VRAM in bytes
    pub vram_used: usize,
    
    /// Driver version string
    pub driver_version: String,
    
    /// Compute capability (CUDA) or equivalent
    pub compute_capability: Option<String>,
    
    /// Maximum threads per block (for parallel operations)
    pub max_threads_per_block: Option<usize>,
    
    /// Maximum shared memory per block
    pub max_shared_memory: Option<usize>,
    
    /// Device ID (for multi-GPU systems)
    pub device_id: usize,
    
    /// PCI Bus ID (for identification)
    pub pci_bus_id: Option<String>,
}

impl GpuDeviceInfo {
    /// Calculate VRAM usage percentage
    pub fn vram_usage_percent(&self) -> f32 {
        (self.vram_used as f32 / self.vram_total as f32) * 100.0
    }
    
    /// Check if device has enough available VRAM
    pub fn has_available_vram(&self, required_bytes: usize) -> bool {
        self.vram_available >= required_bytes
    }
}
```

#### 1.2 Adicionar ao Trait `GpuContext`

```rust
// src/lib.rs

pub trait GpuContext: Send + Sync {
    /// Get detailed device information
    fn device_info(&self) -> Result<GpuDeviceInfo, HiveGpuError>;
    
    /// Get current VRAM usage in bytes
    fn vram_usage(&self) -> Result<usize, HiveGpuError> {
        Ok(self.device_info()?.vram_used)
    }
    
    /// Check if device has enough available VRAM
    fn has_available_vram(&self, required_bytes: usize) -> Result<bool, HiveGpuError> {
        Ok(self.device_info()?.has_available_vram(required_bytes))
    }
    
    // ... métodos existentes
}
```

### Implementação para Metal

```rust
// src/metal/context.rs

impl GpuContext for MetalNativeContext {
    fn device_info(&self) -> Result<GpuDeviceInfo, HiveGpuError> {
        let device = self.device();
        
        // Query Metal device properties
        let name = device.name().to_string();
        let vram_total = device.recommended_max_working_set_size() as usize;
        let vram_used = device.current_allocated_size() as usize;
        let vram_available = vram_total.saturating_sub(vram_used);
        
        // Get macOS version as "driver version"
        let driver_version = Self::get_macos_version();
        
        Ok(GpuDeviceInfo {
            name,
            backend: "Metal".to_string(),
            vram_total,
            vram_available,
            vram_used,
            driver_version,
            compute_capability: None, // Metal doesn't have this
            max_threads_per_block: Some(1024), // Metal typical
            max_shared_memory: Some(32 * 1024), // 32KB typical
            device_id: 0,
            pci_bus_id: None,
        })
    }
}

impl MetalNativeContext {
    fn get_macos_version() -> String {
        use std::process::Command;
        
        if let Ok(output) = Command::new("sw_vers").arg("-productVersion").output() {
            String::from_utf8_lossy(&output.stdout).trim().to_string()
        } else {
            "Unknown".to_string()
        }
    }
}
```

### Testes

```rust
// tests/device_info_tests.rs

#[test]
fn test_metal_device_info() {
    #[cfg(all(feature = "metal-native", target_os = "macos"))]
    {
        use hive_gpu::metal::MetalNativeContext;
        use hive_gpu::GpuContext;
        
        let ctx = MetalNativeContext::new().expect("Metal should be available");
        let info = ctx.device_info().expect("Should get device info");
        
        assert!(!info.name.is_empty());
        assert_eq!(info.backend, "Metal");
        assert!(info.vram_total > 0);
        assert!(info.vram_available <= info.vram_total);
        assert!(!info.driver_version.is_empty());
        
        println!("Device: {}", info.name);
        println!("VRAM: {:.2} GB total, {:.2} GB available", 
            info.vram_total as f64 / 1024.0 / 1024.0 / 1024.0,
            info.vram_available as f64 / 1024.0 / 1024.0 / 1024.0
        );
    }
}
```

---

## 🚀 Phase 2: CUDA Backend (Prioridade: 🔥 CRÍTICA)

**Tempo Estimado:** 1-2 semanas  
**Complexidade:** Alta  
**Impacto:** Crítico (70% do mercado)

### 2.1 Dependências Necessárias

```toml
# Cargo.toml

[dependencies]
# Existing...

# CUDA support (optional)
cuda-runtime-sys = { version = "0.3", optional = true }
cuda-driver-sys = { version = "0.3", optional = true }
cublas-sys = { version = "0.3", optional = true }

[build-dependencies]
cc = "1.0"         # Para compilar kernels CUDA

[features]
default = ["metal-native"]
metal-native = []
cuda = ["cuda-runtime-sys", "cuda-driver-sys", "cublas-sys"]

[lib]
name = "hive_gpu"
crate-type = ["rlib", "cdylib"]
```

### 2.2 Build Script

```rust
// build.rs

fn main() {
    #[cfg(feature = "cuda")]
    {
        build_cuda_kernels();
    }
}

#[cfg(feature = "cuda")]
fn build_cuda_kernels() {
    use std::env;
    use std::path::PathBuf;
    
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    
    println!("cargo:rerun-if-changed=src/cuda/kernels.cu");
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    
    // Compile CUDA kernels
    cc::Build::new()
        .cuda(true)
        .flag("-gencode")
        .flag("arch=compute_70,code=sm_70")  // Volta
        .flag("-gencode")
        .flag("arch=compute_75,code=sm_75")  // Turing
        .flag("-gencode")
        .flag("arch=compute_80,code=sm_80")  // Ampere
        .flag("-gencode")
        .flag("arch=compute_86,code=sm_86")  // Ampere
        .flag("-gencode")
        .flag("arch=compute_89,code=sm_89")  // Ada Lovelace
        .flag("-gencode")
        .flag("arch=compute_90,code=sm_90")  // Hopper
        .file("src/cuda/kernels.cu")
        .compile("hive_gpu_cuda_kernels");
}
```

### 2.3 CUDA Context Implementation

```rust
// src/cuda/context.rs

use cuda_runtime_sys::*;
use std::ffi::c_void;
use std::ptr;

use crate::{GpuContext, GpuDeviceInfo, GpuVectorStorage, HiveGpuError};

/// CUDA GPU Context
pub struct CudaContext {
    device_id: i32,
    stream: cudaStream_t,
    cublas_handle: cublasHandle_t,
}

impl CudaContext {
    /// Create a new CUDA context
    pub fn new() -> Result<Self, HiveGpuError> {
        Self::new_with_device(0)
    }
    
    /// Create a new CUDA context with specific device
    pub fn new_with_device(device_id: i32) -> Result<Self, HiveGpuError> {
        unsafe {
            // Check device count
            let mut device_count = 0;
            cuda_check(cudaGetDeviceCount(&mut device_count))?;
            
            if device_count == 0 {
                return Err(HiveGpuError::NoDevice);
            }
            
            if device_id >= device_count {
                return Err(HiveGpuError::InvalidDeviceId(device_id));
            }
            
            // Set device
            cuda_check(cudaSetDevice(device_id))?;
            
            // Create stream
            let mut stream = ptr::null_mut();
            cuda_check(cudaStreamCreate(&mut stream))?;
            
            // Create cuBLAS handle
            let mut cublas_handle = ptr::null_mut();
            cublas_check(cublasCreate_v2(&mut cublas_handle))?;
            cublas_check(cublasSetStream_v2(cublas_handle, stream))?;
            
            Ok(Self {
                device_id,
                stream,
                cublas_handle,
            })
        }
    }
    
    /// Get device count
    pub fn device_count() -> Result<i32, HiveGpuError> {
        unsafe {
            let mut count = 0;
            cuda_check(cudaGetDeviceCount(&mut count))?;
            Ok(count)
        }
    }
    
    /// Check if CUDA is available
    pub fn is_available() -> bool {
        Self::device_count().map(|c| c > 0).unwrap_or(false)
    }
}

impl GpuContext for CudaContext {
    fn device_info(&self) -> Result<GpuDeviceInfo, HiveGpuError> {
        unsafe {
            let mut props: cudaDeviceProp = std::mem::zeroed();
            cuda_check(cudaGetDeviceProperties(&mut props, self.device_id))?;
            
            // Get memory info
            let mut free_mem: usize = 0;
            let mut total_mem: usize = 0;
            cuda_check(cudaMemGetInfo(&mut free_mem, &mut total_mem))?;
            
            let name = std::ffi::CStr::from_ptr(props.name.as_ptr())
                .to_string_lossy()
                .into_owned();
            
            // Get driver version
            let mut driver_version = 0;
            cuda_check(cudaDriverGetVersion(&mut driver_version))?;
            
            Ok(GpuDeviceInfo {
                name,
                backend: "CUDA".to_string(),
                vram_total: total_mem,
                vram_available: free_mem,
                vram_used: total_mem - free_mem,
                driver_version: format!("{}.{}", driver_version / 1000, (driver_version % 1000) / 10),
                compute_capability: Some(format!("{}.{}", props.major, props.minor)),
                max_threads_per_block: Some(props.maxThreadsPerBlock as usize),
                max_shared_memory: Some(props.sharedMemPerBlock as usize),
                device_id: self.device_id as usize,
                pci_bus_id: Some(format!("{:04x}:{:02x}:{:02x}.0", 
                    props.pciDomainID, props.pciBusID, props.pciDeviceID)),
            })
        }
    }
    
    fn create_storage(
        &self,
        dimension: usize,
        metric: crate::GpuDistanceMetric,
    ) -> Result<Box<dyn GpuVectorStorage>, HiveGpuError> {
        Ok(Box::new(CudaVectorStorage::new(
            self.device_id,
            self.stream,
            self.cublas_handle,
            dimension,
            metric,
        )?))
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe {
            if !self.cublas_handle.is_null() {
                cublasDestroy_v2(self.cublas_handle);
            }
            if !self.stream.is_null() {
                cudaStreamDestroy(self.stream);
            }
        }
    }
}

// Helper functions
unsafe fn cuda_check(result: cudaError_t) -> Result<(), HiveGpuError> {
    if result != cudaError::cudaSuccess {
        let error_str = std::ffi::CStr::from_ptr(cudaGetErrorString(result))
            .to_string_lossy()
            .into_owned();
        return Err(HiveGpuError::CudaError(error_str));
    }
    Ok(())
}

unsafe fn cublas_check(result: cublasStatus_t) -> Result<(), HiveGpuError> {
    if result != cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        return Err(HiveGpuError::CublasError(format!("{:?}", result)));
    }
    Ok(())
}
```

### 2.4 CUDA Vector Storage

```rust
// src/cuda/storage.rs

use cuda_runtime_sys::*;
use std::ptr;

use crate::{GpuVector, GpuSearchResult, GpuVectorStorage, HiveGpuError, GpuDistanceMetric};

pub struct CudaVectorStorage {
    device_id: i32,
    stream: cudaStream_t,
    cublas_handle: cublasHandle_t,
    dimension: usize,
    metric: GpuDistanceMetric,
    vectors: Vec<GpuVector>,
    d_vectors: *mut f32,  // Device pointer
    capacity: usize,
}

impl CudaVectorStorage {
    pub fn new(
        device_id: i32,
        stream: cudaStream_t,
        cublas_handle: cublasHandle_t,
        dimension: usize,
        metric: GpuDistanceMetric,
    ) -> Result<Self, HiveGpuError> {
        let capacity = 10000; // Initial capacity
        let bytes = capacity * dimension * std::mem::size_of::<f32>();
        
        unsafe {
            let mut d_vectors = ptr::null_mut();
            cuda_check(cudaMalloc(&mut d_vectors as *mut *mut c_void, bytes))?;
            
            Ok(Self {
                device_id,
                stream,
                cublas_handle,
                dimension,
                metric,
                vectors: Vec::new(),
                d_vectors: d_vectors as *mut f32,
                capacity,
            })
        }
    }
    
    fn ensure_capacity(&mut self, new_size: usize) -> Result<(), HiveGpuError> {
        if new_size > self.capacity {
            let new_capacity = new_size.next_power_of_two();
            let bytes = new_capacity * self.dimension * std::mem::size_of::<f32>();
            
            unsafe {
                let mut new_d_vectors = ptr::null_mut();
                cuda_check(cudaMalloc(&mut new_d_vectors as *mut *mut c_void, bytes))?;
                
                // Copy old data
                if !self.d_vectors.is_null() && self.vectors.len() > 0 {
                    let old_bytes = self.vectors.len() * self.dimension * std::mem::size_of::<f32>();
                    cuda_check(cudaMemcpyAsync(
                        new_d_vectors as *mut c_void,
                        self.d_vectors as *const c_void,
                        old_bytes,
                        cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                        self.stream,
                    ))?;
                    cuda_check(cudaStreamSynchronize(self.stream))?;
                    cuda_check(cudaFree(self.d_vectors as *mut c_void))?;
                }
                
                self.d_vectors = new_d_vectors as *mut f32;
                self.capacity = new_capacity;
            }
        }
        Ok(())
    }
}

impl GpuVectorStorage for CudaVectorStorage {
    fn add_vector(&mut self, vector: GpuVector) -> Result<usize, HiveGpuError> {
        if vector.data.len() != self.dimension {
            return Err(HiveGpuError::DimensionMismatch {
                expected: self.dimension,
                got: vector.data.len(),
            });
        }
        
        let index = self.vectors.len();
        self.ensure_capacity(index + 1)?;
        
        unsafe {
            // Copy vector to GPU
            let offset = index * self.dimension;
            let bytes = self.dimension * std::mem::size_of::<f32>();
            cuda_check(cudaMemcpyAsync(
                self.d_vectors.add(offset) as *mut c_void,
                vector.data.as_ptr() as *const c_void,
                bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream,
            ))?;
        }
        
        self.vectors.push(vector);
        Ok(index)
    }
    
    fn add_vectors(&mut self, vectors: &[GpuVector]) -> Result<(), HiveGpuError> {
        if vectors.is_empty() {
            return Ok(());
        }
        
        let start_index = self.vectors.len();
        self.ensure_capacity(start_index + vectors.len())?;
        
        // Flatten all vectors into single buffer
        let mut flat_vectors = Vec::with_capacity(vectors.len() * self.dimension);
        for vector in vectors {
            if vector.data.len() != self.dimension {
                return Err(HiveGpuError::DimensionMismatch {
                    expected: self.dimension,
                    got: vector.data.len(),
                });
            }
            flat_vectors.extend_from_slice(&vector.data);
        }
        
        unsafe {
            // Batch copy to GPU
            let offset = start_index * self.dimension;
            let bytes = flat_vectors.len() * std::mem::size_of::<f32>();
            cuda_check(cudaMemcpyAsync(
                self.d_vectors.add(offset) as *mut c_void,
                flat_vectors.as_ptr() as *const c_void,
                bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream,
            ))?;
            cuda_check(cudaStreamSynchronize(self.stream))?;
        }
        
        self.vectors.extend_from_slice(vectors);
        Ok(())
    }
    
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<GpuSearchResult>, HiveGpuError> {
        if query.len() != self.dimension {
            return Err(HiveGpuError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }
        
        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }
        
        let k = k.min(self.vectors.len());
        
        unsafe {
            // Allocate device memory for query
            let mut d_query = ptr::null_mut();
            let query_bytes = self.dimension * std::mem::size_of::<f32>();
            cuda_check(cudaMalloc(&mut d_query as *mut *mut c_void, query_bytes))?;
            cuda_check(cudaMemcpyAsync(
                d_query,
                query.as_ptr() as *const c_void,
                query_bytes,
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream,
            ))?;
            
            // Allocate device memory for distances
            let mut d_distances = ptr::null_mut();
            let distances_bytes = self.vectors.len() * std::mem::size_of::<f32>();
            cuda_check(cudaMalloc(&mut d_distances as *mut *mut c_void, distances_bytes))?;
            
            // Compute distances using cuBLAS (matrix-vector multiply)
            match self.metric {
                GpuDistanceMetric::Cosine | GpuDistanceMetric::DotProduct => {
                    // Use SGEMV: y = alpha * A * x + beta * y
                    let alpha = 1.0f32;
                    let beta = 0.0f32;
                    
                    cublas_check(cublasSgemv_v2(
                        self.cublas_handle,
                        cublasOperation_t::CUBLAS_OP_N,
                        self.vectors.len() as i32,
                        self.dimension as i32,
                        &alpha,
                        self.d_vectors,
                        self.vectors.len() as i32,
                        d_query as *const f32,
                        1,
                        &beta,
                        d_distances as *mut f32,
                        1,
                    ))?;
                }
                GpuDistanceMetric::Euclidean => {
                    // Call custom CUDA kernel for L2 distance
                    cuda_l2_distance(
                        d_query as *const f32,
                        self.d_vectors as *const f32,
                        d_distances as *mut f32,
                        self.vectors.len(),
                        self.dimension,
                        self.stream,
                    )?;
                }
            }
            
            // Copy distances back to host
            let mut h_distances = vec![0.0f32; self.vectors.len()];
            cuda_check(cudaMemcpyAsync(
                h_distances.as_mut_ptr() as *mut c_void,
                d_distances as *const c_void,
                distances_bytes,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                self.stream,
            ))?;
            cuda_check(cudaStreamSynchronize(self.stream))?;
            
            // Free device memory
            cuda_check(cudaFree(d_query))?;
            cuda_check(cudaFree(d_distances))?;
            
            // Find top-k
            let mut results: Vec<(usize, f32)> = h_distances
                .into_iter()
                .enumerate()
                .collect();
            
            // Sort by distance (higher is better for cosine/dot, lower for euclidean)
            match self.metric {
                GpuDistanceMetric::Cosine | GpuDistanceMetric::DotProduct => {
                    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                }
                GpuDistanceMetric::Euclidean => {
                    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                }
            }
            
            results.truncate(k);
            
            Ok(results
                .into_iter()
                .map(|(idx, distance)| GpuSearchResult {
                    id: self.vectors[idx].id.clone(),
                    distance,
                    metadata: self.vectors[idx].metadata.clone(),
                })
                .collect())
        }
    }
    
    fn get_vector(&self, id: &str) -> Result<Option<GpuVector>, HiveGpuError> {
        Ok(self.vectors.iter().find(|v| v.id == id).cloned())
    }
    
    fn remove_vector(&mut self, id: &str) -> Result<bool, HiveGpuError> {
        if let Some(pos) = self.vectors.iter().position(|v| v.id == id) {
            self.vectors.remove(pos);
            // TODO: Compact GPU memory
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    fn update_vector(&mut self, vector: GpuVector) -> Result<bool, HiveGpuError> {
        if let Some(pos) = self.vectors.iter().position(|v| v.id == vector.id) {
            self.vectors[pos] = vector.clone();
            
            unsafe {
                // Update on GPU
                let offset = pos * self.dimension;
                let bytes = self.dimension * std::mem::size_of::<f32>();
                cuda_check(cudaMemcpyAsync(
                    self.d_vectors.add(offset) as *mut c_void,
                    vector.data.as_ptr() as *const c_void,
                    bytes,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                    self.stream,
                ))?;
                cuda_check(cudaStreamSynchronize(self.stream))?;
            }
            
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    fn vector_count(&self) -> usize {
        self.vectors.len()
    }
    
    fn clear(&mut self) -> Result<(), HiveGpuError> {
        self.vectors.clear();
        Ok(())
    }
}

impl Drop for CudaVectorStorage {
    fn drop(&mut self) {
        unsafe {
            if !self.d_vectors.is_null() {
                cudaFree(self.d_vectors as *mut c_void);
            }
        }
    }
}

// External CUDA kernel functions
extern "C" {
    fn cuda_l2_distance(
        query: *const f32,
        vectors: *const f32,
        distances: *mut f32,
        n_vectors: usize,
        dimension: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;
}
```

### 2.5 CUDA Kernels

```cuda
// src/cuda/kernels.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {

// Kernel for computing L2 (Euclidean) distances
__global__ void l2_distance_kernel(
    const float* query,
    const float* vectors,
    float* distances,
    int n_vectors,
    int dimension
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_vectors) {
        float sum = 0.0f;
        const float* vector = vectors + idx * dimension;
        
        for (int d = 0; d < dimension; d++) {
            float diff = query[d] - vector[d];
            sum += diff * diff;
        }
        
        distances[idx] = sqrtf(sum);
    }
}

// Wrapper function
cudaError_t cuda_l2_distance(
    const float* query,
    const float* vectors,
    float* distances,
    size_t n_vectors,
    size_t dimension,
    cudaStream_t stream
) {
    int threads_per_block = 256;
    int num_blocks = (n_vectors + threads_per_block - 1) / threads_per_block;
    
    l2_distance_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        query, vectors, distances, n_vectors, dimension
    );
    
    return cudaGetLastError();
}

} // extern "C"
```

### 2.6 Error Types

```rust
// src/error.rs

#[derive(Debug, thiserror::Error)]
pub enum HiveGpuError {
    // Existing errors...
    
    #[error("CUDA error: {0}")]
    CudaError(String),
    
    #[error("cuBLAS error: {0}")]
    CublasError(String),
    
    #[error("No CUDA device available")]
    NoDevice,
    
    #[error("Invalid device ID: {0}")]
    InvalidDeviceId(i32),
}
```

### 2.7 Testes CUDA

```rust
// tests/cuda_tests.rs

#[cfg(feature = "cuda")]
mod cuda_tests {
    use hive_gpu::cuda::CudaContext;
    use hive_gpu::{GpuContext, GpuVector, GpuDistanceMetric, GpuVectorStorage};
    
    #[test]
    fn test_cuda_availability() {
        if CudaContext::is_available() {
            println!("CUDA is available");
            println!("Device count: {}", CudaContext::device_count().unwrap());
        } else {
            println!("CUDA not available, skipping test");
        }
    }
    
    #[test]
    fn test_cuda_device_info() {
        if !CudaContext::is_available() {
            println!("CUDA not available, skipping test");
            return;
        }
        
        let ctx = CudaContext::new().expect("Failed to create CUDA context");
        let info = ctx.device_info().expect("Failed to get device info");
        
        println!("Device: {}", info.name);
        println!("VRAM: {:.2} GB", info.vram_total as f64 / 1024.0 / 1024.0 / 1024.0);
        println!("Driver: {}", info.driver_version);
        println!("Compute Capability: {:?}", info.compute_capability);
        
        assert_eq!(info.backend, "CUDA");
        assert!(info.vram_total > 0);
    }
    
    #[test]
    fn test_cuda_vector_operations() {
        if !CudaContext::is_available() {
            return;
        }
        
        let ctx = CudaContext::new().unwrap();
        let mut storage = ctx.create_storage(128, GpuDistanceMetric::Cosine).unwrap();
        
        // Add vectors
        let v1 = GpuVector {
            id: "v1".to_string(),
            data: vec![1.0; 128],
            metadata: Default::default(),
        };
        let v2 = GpuVector {
            id: "v2".to_string(),
            data: vec![0.5; 128],
            metadata: Default::default(),
        };
        
        storage.add_vector(v1).unwrap();
        storage.add_vector(v2).unwrap();
        
        assert_eq!(storage.vector_count(), 2);
        
        // Search
        let query = vec![1.0; 128];
        let results = storage.search(&query, 2).unwrap();
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "v1");
    }
    
    #[test]
    fn test_cuda_batch_operations() {
        if !CudaContext::is_available() {
            return;
        }
        
        let ctx = CudaContext::new().unwrap();
        let mut storage = ctx.create_storage(128, GpuDistanceMetric::Cosine).unwrap();
        
        // Create 1000 vectors
        let vectors: Vec<GpuVector> = (0..1000)
            .map(|i| GpuVector {
                id: format!("v{}", i),
                data: vec![i as f32 / 1000.0; 128],
                metadata: Default::default(),
            })
            .collect();
        
        // Batch add
        let start = std::time::Instant::now();
        storage.add_vectors(&vectors).unwrap();
        let duration = start.elapsed();
        
        println!("Batch add 1000 vectors: {:?}", duration);
        assert_eq!(storage.vector_count(), 1000);
        
        // Search
        let query = vec![0.5; 128];
        let start = std::time::Instant::now();
        let results = storage.search(&query, 10).unwrap();
        let duration = start.elapsed();
        
        println!("Search in 1000 vectors: {:?}", duration);
        assert_eq!(results.len(), 10);
    }
}
```

---

## 📦 Phase 3: ROCm Backend (Prioridade: ⚡ MÉDIA)

**Tempo Estimado:** 1-2 semanas  
**Complexidade:** Alta  
**Impacto:** Médio (15% do mercado)

### Especificação

Similar ao CUDA, mas usando HIP (Heterogeneous-compute Interface for Portability):

```rust
// src/rocm/mod.rs
// src/rocm/context.rs
// src/rocm/storage.rs
```

**Dependências:**
```toml
hip-runtime-sys = { version = "0.3", optional = true }
rocblas-sys = { version = "0.3", optional = true }
```

**Kernels em HIP:**
```hip
// src/rocm/kernels.hip
// Sintaxe similar a CUDA, mas usando HIP APIs
```

---

## 📦 Phase 4: WebGPU Backend (Prioridade: 📦 BAIXA)

**Tempo Estimado:** 1 semana  
**Complexidade:** Média  
**Impacto:** Baixo-Médio (10% uso geral, fallback universal)

### Especificação

```toml
# Cargo.toml
[dependencies]
wgpu = { version = "0.19", optional = true }
pollster = { version = "0.3", optional = true }

[features]
wgpu = ["dep:wgpu", "dep:pollster"]
```

```rust
// src/wgpu/context.rs

use wgpu::*;
use crate::{GpuContext, GpuDeviceInfo, HiveGpuError};

pub struct WgpuContext {
    device: Device,
    queue: Queue,
    adapter: Adapter,
}

impl WgpuContext {
    pub async fn new() -> Result<Self, HiveGpuError> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok_or(HiveGpuError::NoDevice)?;
        
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default(), None)
            .await
            .map_err(|e| HiveGpuError::InitializationFailed(e.to_string()))?;
        
        Ok(Self { device, queue, adapter })
    }
    
    pub fn is_available() -> bool {
        pollster::block_on(async {
            let instance = Instance::new(InstanceDescriptor::default());
            instance.request_adapter(&RequestAdapterOptions::default())
                .await
                .is_some()
        })
    }
}

impl GpuContext for WgpuContext {
    fn device_info(&self) -> Result<GpuDeviceInfo, HiveGpuError> {
        let info = self.adapter.get_info();
        let limits = self.device.limits();
        
        Ok(GpuDeviceInfo {
            name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            vram_total: 0, // WebGPU doesn't expose this
            vram_available: 0,
            vram_used: 0,
            driver_version: info.driver_info.clone(),
            compute_capability: None,
            max_threads_per_block: Some(limits.max_compute_workgroup_size_x as usize),
            max_shared_memory: Some(limits.max_compute_workgroup_storage_size as usize),
            device_id: 0,
            pci_bus_id: None,
        })
    }
    
    // ... rest of implementation
}
```

---

## 📦 Phase 5: Memory Pooling (Prioridade: ⚡ MÉDIA)

**Tempo Estimado:** 3-5 dias  
**Complexidade:** Média  
**Impacto:** Médio (otimização de performance)

### Especificação

```rust
// src/pool/memory_pool.rs

use std::collections::VecDeque;

/// GPU memory pool for reducing allocation overhead
pub struct GpuMemoryPool {
    buffers: VecDeque<GpuBuffer>,
    buffer_size: usize,
    max_buffers: usize,
}

pub struct GpuBuffer {
    ptr: *mut c_void,
    size: usize,
    in_use: bool,
}

impl GpuMemoryPool {
    pub fn new(buffer_size: usize, max_buffers: usize) -> Self {
        Self {
            buffers: VecDeque::new(),
            buffer_size,
            max_buffers,
        }
    }
    
    pub fn allocate(&mut self) -> Result<GpuBuffer, HiveGpuError> {
        // Try to reuse existing buffer
        if let Some(mut buffer) = self.buffers.pop_front() {
            buffer.in_use = true;
            return Ok(buffer);
        }
        
        // Create new buffer if under limit
        if self.buffers.len() < self.max_buffers {
            let mut ptr = std::ptr::null_mut();
            unsafe {
                cuda_check(cudaMalloc(&mut ptr, self.buffer_size))?;
            }
            return Ok(GpuBuffer {
                ptr,
                size: self.buffer_size,
                in_use: true,
            });
        }
        
        Err(HiveGpuError::OutOfMemory)
    }
    
    pub fn deallocate(&mut self, mut buffer: GpuBuffer) {
        buffer.in_use = false;
        self.buffers.push_back(buffer);
    }
    
    pub fn clear(&mut self) {
        for buffer in &self.buffers {
            unsafe {
                cudaFree(buffer.ptr);
            }
        }
        self.buffers.clear();
    }
}
```

---

## 📦 Phase 6: Async Operations (Prioridade: 📦 BAIXA)

**Tempo Estimado:** 3-5 dias  
**Complexidade:** Média  
**Impacto:** Baixo-Médio (melhor integração com async Rust)

### Especificação

```rust
// src/lib.rs

#[async_trait::async_trait]
pub trait GpuVectorStorageAsync: Send + Sync {
    async fn add_vector_async(&mut self, vector: GpuVector) -> Result<usize, HiveGpuError>;
    async fn search_async(&self, query: &[f32], k: usize) -> Result<Vec<GpuSearchResult>, HiveGpuError>;
    async fn add_vectors_async(&mut self, vectors: &[GpuVector]) -> Result<(), HiveGpuError>;
}
```

---

## 🧪 Testing Strategy

### Unit Tests
- ✅ Device detection per backend
- ✅ Context creation/destruction
- ✅ Memory allocation/deallocation
- ✅ Vector CRUD operations
- ✅ Search correctness

### Integration Tests
- ✅ Cross-backend consistency
- ✅ Batch operations
- ✅ Large dataset handling (1M+ vectors)
- ✅ Multi-GPU support

### Performance Tests
- ✅ Benchmark vs CPU
- ✅ Compare Metal vs CUDA vs ROCm
- ✅ Memory usage tracking
- ✅ Latency measurements

### CI/CD
```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test-metal:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - run: cargo test --features metal-native
  
  test-cuda:
    runs-on: ubuntu-latest
    container: nvidia/cuda:12.0-devel
    steps:
      - uses: actions/checkout@v3
      - run: cargo test --features cuda
  
  test-cpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: cargo test --no-default-features
```

---

## 📊 Roadmap e Prioridades

### Q1 2025 - Foundational (Critical)
- [x] Metal backend (✅ Done)
- [ ] Device Info API (🔥 Critical - 1-2 days)
- [ ] CUDA backend (🔥 Critical - 1-2 weeks)

### Q2 2025 - Expansion (High Priority)
- [ ] ROCm backend (⚡ Medium - 1-2 weeks)
- [ ] Memory Pooling (⚡ Medium - 3-5 days)
- [ ] Performance benchmarks
- [ ] Documentation improvements

### Q3 2025 - Polish (Low Priority)
- [ ] WebGPU backend (📦 Low - 1 week)
- [ ] Async operations (📦 Low - 3-5 days)
- [ ] Multi-GPU support
- [ ] Advanced optimizations

---

## 📈 Expected Impact

### Before (Current State)
| Platform | GPU Support | Market Share |
|----------|-------------|--------------|
| macOS | ✅ Metal | ~5% |
| Linux | ❌ None | ~75% |
| Windows | ❌ None | ~20% |
| **Total GPU Coverage** | **5%** | |

### After (With CUDA)
| Platform | GPU Support | Market Share |
|----------|-------------|--------------|
| macOS | ✅ Metal | ~5% |
| Linux | ✅ CUDA | ~52% (NVIDIA) |
| Windows | ✅ CUDA | ~14% (NVIDIA) |
| **Total GPU Coverage** | **~71%** | |

### After (With CUDA + ROCm)
| Platform | GPU Support | Market Share |
|----------|-------------|--------------|
| macOS | ✅ Metal | ~5% |
| Linux | ✅ CUDA + ROCm | ~67% |
| Windows | ✅ CUDA | ~14% |
| **Total GPU Coverage** | **~86%** | |

### After (Full Implementation)
| Platform | GPU Support | Market Share |
|----------|-------------|--------------|
| macOS | ✅ Metal | ~5% |
| Linux | ✅ CUDA + ROCm + WebGPU | ~75% |
| Windows | ✅ CUDA + WebGPU | ~20% |
| **Total GPU Coverage** | **~100%** | |

---

## 💰 ROI Analysis

### Investment
- **Device Info API:** 1-2 days (~$2K)
- **CUDA Backend:** 1-2 weeks (~$10-20K)
- **ROCm Backend:** 1-2 weeks (~$10-20K)
- **WebGPU Backend:** 1 week (~$5-10K)
- **Total:** 4-6 weeks (~$27-52K)

### Returns
- **10-50x speedup** on GPU operations
- **95% market coverage** with CUDA alone
- **Competitive advantage** vs CPU-only solutions
- **Reduced latency:** 10-30ms → 0.5-3ms
- **Higher throughput:** 100x more queries/second

### Break-even
- With just **10 enterprise customers** using GPU acceleration
- Or **1000+ community users** benefiting from GPU speedup
- **Estimated ROI:** 10-20x within first year

---

## 📞 Support & Contact

For questions or assistance with implementation:

- **GitHub Issues:** https://github.com/hivellm/hive-gpu/issues
- **Email:** dev@hivellm.com
- **Discord:** https://discord.gg/hivellm

---

## 📚 References

### CUDA
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- cuBLAS Documentation: https://docs.nvidia.com/cuda/cublas/
- cuda-runtime-sys crate: https://crates.io/crates/cuda-runtime-sys

### ROCm
- ROCm Documentation: https://rocmdocs.amd.com/
- HIP Programming Guide: https://rocm.docs.amd.com/projects/HIP/
- rocBLAS: https://rocm.docs.amd.com/projects/rocBLAS/

### WebGPU
- WebGPU Spec: https://www.w3.org/TR/webgpu/
- wgpu Rust: https://wgpu.rs/
- WGSL Shading Language: https://www.w3.org/TR/WGSL/

### Metal
- Metal Programming Guide: https://developer.apple.com/metal/
- Metal Shading Language: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf

---

## ✅ Checklist de Implementação

### Device Info API
- [ ] Define `GpuDeviceInfo` struct
- [ ] Add `device_info()` to `GpuContext` trait
- [ ] Implement for Metal
- [ ] Add tests
- [ ] Update documentation

### CUDA Backend
- [ ] Add CUDA dependencies to Cargo.toml
- [ ] Create build.rs for kernel compilation
- [ ] Implement `CudaContext`
- [ ] Implement `CudaVectorStorage`
- [ ] Write CUDA kernels
- [ ] Add error handling
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Benchmark performance
- [ ] Update documentation

### ROCm Backend
- [ ] Add ROCm/HIP dependencies
- [ ] Implement `RocmContext`
- [ ] Implement `RocmVectorStorage`
- [ ] Write HIP kernels
- [ ] Add tests
- [ ] Benchmark performance
- [ ] Update documentation

### WebGPU Backend
- [ ] Add wgpu dependency
- [ ] Implement `WgpuContext`
- [ ] Implement `WgpuVectorStorage`
- [ ] Write WGSL shaders
- [ ] Add tests
- [ ] Update documentation

### Memory Pooling
- [ ] Implement `GpuMemoryPool`
- [ ] Integrate with existing backends
- [ ] Add configuration options
- [ ] Benchmark improvements
- [ ] Update documentation

### Async Operations
- [ ] Define async trait
- [ ] Implement for all backends
- [ ] Add async tests
- [ ] Update examples
- [ ] Update documentation

---

**Last Updated:** 2025-01-07  
**Version:** 1.0  
**Status:** Ready for Implementation  
**Next Steps:** Review with hive-gpu team, prioritize phases, begin implementation


