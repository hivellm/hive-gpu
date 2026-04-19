# hive-gpu - Integration Guide

## Overview

This guide demonstrates how to integrate hive-gpu into various applications and workflows, including integration with vector databases, embedding pipelines, and custom applications.

---

## Table of Contents

1. [Basic Integration](#basic-integration)
2. [Vectorizer Integration](#vectorizer-integration)
3. [Custom Vector Database](#custom-vector-database)
4. [Embedding Pipeline](#embedding-pipeline)
5. [Web Service Integration](#web-service-integration)
6. [Real-time Search Application](#real-time-search-application)
7. [Best Practices](#best-practices)

---

## Basic Integration

### Quick Start

```rust
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};
use hive_gpu::types::{GpuVector, GpuDistanceMetric};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU context
    let context = MetalNativeContext::new()?;
    
    // Create vector storage
    let mut storage = context.create_storage(384, GpuDistanceMetric::Cosine)?;
    
    // Add vectors
    let vectors = vec![
        GpuVector::new("doc_1".into(), vec![0.1; 384]),
        GpuVector::new("doc_2".into(), vec![0.2; 384]),
    ];
    storage.add_vectors(&vectors)?;
    
    // Search
    let query = vec![0.15; 384];
    let results = storage.search(&query, 10)?;
    
    for result in results {
        println!("{}: {:.4}", result.id, result.score);
    }
    
    Ok(())
}
```

---

## Vectorizer Integration

### Using with Hive-Vectorizer

hive-gpu is designed to integrate seamlessly with the [hive-vectorizer](https://github.com/hivellm/vectorizer) project.

#### Installation

```toml
[dependencies]
vectorizer = { git = "https://github.com/hivellm/vectorizer.git" }
hive-gpu = "0.1"
```

#### Integration Example

```rust
use vectorizer::{VectorStore, models::*};
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};
use std::sync::Arc;

struct GpuVectorStore {
    store: VectorStore,
    gpu_storage: Box<dyn GpuVectorStorage>,
}

impl GpuVectorStore {
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize vectorizer store
        let store = VectorStore::new();
        
        // Initialize GPU context
        let context = MetalNativeContext::new()?;
        let gpu_storage = context.create_storage(512, GpuDistanceMetric::Cosine)?;
        
        Ok(Self {
            store,
            gpu_storage,
        })
    }
    
    async fn add_documents(&mut self, documents: Vec<Document>) -> Result<(), Box<dyn std::error::Error>> {
        // Generate embeddings using vectorizer
        let vectors: Vec<GpuVector> = documents
            .iter()
            .map(|doc| {
                let embedding = self.store.generate_embedding(&doc.content)?;
                Ok(GpuVector {
                    id: doc.id.clone(),
                    data: embedding,
                    metadata: doc.metadata.clone(),
                })
            })
            .collect::<Result<_, Box<dyn std::error::Error>>>()?;
        
        // Add to GPU storage
        self.gpu_storage.add_vectors(&vectors)?;
        
        Ok(())
    }
    
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<Document>, Box<dyn std::error::Error>> {
        // Generate query embedding
        let query_embedding = self.store.generate_embedding(query)?;
        
        // GPU-accelerated search
        let results = self.gpu_storage.search(&query_embedding, limit)?;
        
        // Fetch full documents
        let documents = results
            .into_iter()
            .map(|result| self.store.get_document(&result.id))
            .collect::<Result<Vec<_>, _>>()?;
        
        Ok(documents)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut store = GpuVectorStore::new().await?;
    
    // Add documents
    let documents = vec![
        Document {
            id: "doc1".into(),
            content: "Machine learning with GPUs".into(),
            metadata: HashMap::new(),
        },
        Document {
            id: "doc2".into(),
            content: "Vector similarity search".into(),
            metadata: HashMap::new(),
        },
    ];
    store.add_documents(documents).await?;
    
    // Search
    let results = store.search("GPU acceleration", 5).await?;
    for doc in results {
        println!("Found: {}", doc.content);
    }
    
    Ok(())
}
```

---

## Custom Vector Database

### Building a Custom Vector DB

```rust
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};
use hive_gpu::types::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
}

pub struct VectorDatabase {
    gpu_storage: Arc<RwLock<Box<dyn GpuVectorStorage>>>,
    documents: Arc<RwLock<HashMap<String, Document>>>,
}

impl VectorDatabase {
    pub fn new(dimension: usize, metric: GpuDistanceMetric) -> Result<Self, Box<dyn std::error::Error>> {
        let context = MetalNativeContext::new()?;
        let storage = context.create_storage(dimension, metric)?;
        
        Ok(Self {
            gpu_storage: Arc::new(RwLock::new(storage)),
            documents: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    pub fn add_document(&self, document: Document, embedding: Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let vector = GpuVector {
            id: document.id.clone(),
            data: embedding,
            metadata: document.metadata.clone(),
        };
        
        // Add to GPU storage
        let mut storage = self.gpu_storage.write().unwrap();
        storage.add_vectors(&[vector])?;
        
        // Store document
        let mut documents = self.documents.write().unwrap();
        documents.insert(document.id.clone(), document);
        
        Ok(())
    }
    
    pub fn batch_add_documents(&self, docs_with_embeddings: Vec<(Document, Vec<f32>)>) -> Result<(), Box<dyn std::error::Error>> {
        let vectors: Vec<GpuVector> = docs_with_embeddings
            .iter()
            .map(|(doc, embedding)| GpuVector {
                id: doc.id.clone(),
                data: embedding.clone(),
                metadata: doc.metadata.clone(),
            })
            .collect();
        
        // Batch add to GPU
        let mut storage = self.gpu_storage.write().unwrap();
        storage.add_vectors(&vectors)?;
        
        // Store documents
        let mut documents = self.documents.write().unwrap();
        for (doc, _) in docs_with_embeddings {
            documents.insert(doc.id.clone(), doc);
        }
        
        Ok(())
    }
    
    pub fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<(Document, f32)>, Box<dyn std::error::Error>> {
        // GPU search
        let storage = self.gpu_storage.read().unwrap();
        let results = storage.search(query_embedding, limit)?;
        
        // Fetch documents
        let documents = self.documents.read().unwrap();
        let docs_with_scores: Vec<(Document, f32)> = results
            .into_iter()
            .filter_map(|result| {
                documents.get(&result.id).map(|doc| (doc.clone(), result.score))
            })
            .collect();
        
        Ok(docs_with_scores)
    }
    
    pub fn delete_document(&self, id: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Remove from GPU storage
        let mut storage = self.gpu_storage.write().unwrap();
        storage.remove_vectors(&[id.to_string()])?;
        
        // Remove document
        let mut documents = self.documents.write().unwrap();
        documents.remove(id);
        
        Ok(())
    }
    
    pub fn count(&self) -> usize {
        let documents = self.documents.read().unwrap();
        documents.len()
    }
}

// Usage example
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = VectorDatabase::new(384, GpuDistanceMetric::Cosine)?;
    
    // Add documents
    let docs_with_embeddings = vec![
        (
            Document {
                id: "1".into(),
                content: "GPU acceleration".into(),
                metadata: HashMap::new(),
            },
            vec![0.1; 384],
        ),
        (
            Document {
                id: "2".into(),
                content: "Vector search".into(),
                metadata: HashMap::new(),
            },
            vec![0.2; 384],
        ),
    ];
    
    db.batch_add_documents(docs_with_embeddings)?;
    
    // Search
    let query = vec![0.15; 384];
    let results = db.search(&query, 5)?;
    
    for (doc, score) in results {
        println!("{}: {} (score: {:.4})", doc.id, doc.content, score);
    }
    
    println!("Total documents: {}", db.count());
    
    Ok(())
}
```

---

## Embedding Pipeline

### Integration with Embedding Models

```rust
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};
use hive_gpu::types::*;

// Mock embedding model (replace with actual model)
struct EmbeddingModel {
    dimension: usize,
}

impl EmbeddingModel {
    fn new() -> Self {
        Self { dimension: 384 }
    }
    
    async fn encode(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // In practice, use sentence-transformers, OpenAI API, etc.
        // This is a mock implementation
        Ok((0..self.dimension).map(|i| (text.len() as f32 + i as f32) * 0.01).collect())
    }
    
    async fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let mut embeddings = Vec::new();
        for text in texts {
            embeddings.push(self.encode(text).await?);
        }
        Ok(embeddings)
    }
}

pub struct EmbeddingPipeline {
    model: EmbeddingModel,
    gpu_storage: Box<dyn GpuVectorStorage>,
}

impl EmbeddingPipeline {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let model = EmbeddingModel::new();
        let context = MetalNativeContext::new()?;
        let gpu_storage = context.create_storage(model.dimension, GpuDistanceMetric::Cosine)?;
        
        Ok(Self {
            model,
            gpu_storage,
        })
    }
    
    pub async fn index_documents(&mut self, documents: Vec<(String, String)>) -> Result<(), Box<dyn std::error::Error>> {
        // Extract texts
        let texts: Vec<String> = documents.iter().map(|(_, text)| text.clone()).collect();
        
        // Generate embeddings in batch
        let embeddings = self.model.encode_batch(&texts).await?;
        
        // Create GPU vectors
        let vectors: Vec<GpuVector> = documents
            .into_iter()
            .zip(embeddings)
            .map(|((id, text), embedding)| {
                let mut metadata = HashMap::new();
                metadata.insert("text".to_string(), text);
                GpuVector::with_metadata(id, embedding, metadata)
            })
            .collect();
        
        // Add to GPU storage
        self.gpu_storage.add_vectors(&vectors)?;
        
        Ok(())
    }
    
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        // Generate query embedding
        let query_embedding = self.model.encode(query).await?;
        
        // GPU search
        let results = self.gpu_storage.search(&query_embedding, limit)?;
        
        // Return results
        Ok(results.into_iter().map(|r| SearchResult {
            id: r.id,
            score: r.score,
        }).collect())
    }
}

#[derive(Debug)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
}

// Usage
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut pipeline = EmbeddingPipeline::new().await?;
    
    // Index documents
    let documents = vec![
        ("1".into(), "GPU-accelerated vector search".into()),
        ("2".into(), "Machine learning on Apple Silicon".into()),
        ("3".into(), "HNSW graph algorithms".into()),
    ];
    
    pipeline.index_documents(documents).await?;
    
    // Search
    let results = pipeline.search("GPU vector search", 5).await?;
    for result in results {
        println!("{}: {:.4}", result.id, result.score);
    }
    
    Ok(())
}
```

---

## Web Service Integration

### REST API with Axum

```rust
use axum::{
    routing::{get, post},
    Json, Router, extract::State,
};
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};
use hive_gpu::types::*;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
use tokio::net::TcpListener;

#[derive(Clone)]
struct AppState {
    storage: Arc<RwLock<Box<dyn GpuVectorStorage>>>,
}

#[derive(Deserialize)]
struct AddVectorRequest {
    id: String,
    data: Vec<f32>,
}

#[derive(Deserialize)]
struct SearchRequest {
    query: Vec<f32>,
    limit: usize,
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<SearchResultItem>,
}

#[derive(Serialize)]
struct SearchResultItem {
    id: String,
    score: f32,
}

async fn add_vector(
    State(state): State<AppState>,
    Json(request): Json<AddVectorRequest>,
) -> Json<serde_json::Value> {
    let vector = GpuVector::new(request.id, request.data);
    
    let mut storage = state.storage.write().unwrap();
    match storage.add_vectors(&[vector]) {
        Ok(_) => Json(serde_json::json!({ "success": true })),
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })),
    }
}

async fn search(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> Json<SearchResponse> {
    let storage = state.storage.read().unwrap();
    
    let results = storage.search(&request.query, request.limit)
        .unwrap_or_default()
        .into_iter()
        .map(|r| SearchResultItem {
            id: r.id,
            score: r.score,
        })
        .collect();
    
    Json(SearchResponse { results })
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "healthy" }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU storage
    let context = MetalNativeContext::new()?;
    let storage = context.create_storage(384, GpuDistanceMetric::Cosine)?;
    
    let state = AppState {
        storage: Arc::new(RwLock::new(storage)),
    };
    
    // Build router
    let app = Router::new()
        .route("/health", get(health))
        .route("/add", post(add_vector))
        .route("/search", post(search))
        .with_state(state);
    
    // Run server
    let listener = TcpListener::bind("0.0.0.0:3000").await?;
    println!("Server running on http://0.0.0.0:3000");
    axum::serve(listener, app).await?;
    
    Ok(())
}
```

#### Testing the API

```bash
# Add vectors
curl -X POST http://localhost:3000/add \
  -H "Content-Type: application/json" \
  -d '{"id": "vec1", "data": [0.1, 0.2, 0.3]}'

# Search
curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{"query": [0.15, 0.25, 0.35], "limit": 10}'

# Health check
curl http://localhost:3000/health
```

---

## Real-time Search Application

### Live Document Search

```rust
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};
use hive_gpu::types::*;
use std::sync::{Arc, RwLock};
use tokio::sync::mpsc;

pub enum IndexCommand {
    AddDocument { id: String, embedding: Vec<f32> },
    RemoveDocument { id: String },
    Search { query: Vec<f32>, limit: usize, response_tx: mpsc::Sender<Vec<GpuSearchResult>> },
}

pub struct RealtimeSearch {
    command_tx: mpsc::Sender<IndexCommand>,
}

impl RealtimeSearch {
    pub async fn new(dimension: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let context = MetalNativeContext::new()?;
        let storage = context.create_storage(dimension, GpuDistanceMetric::Cosine)?;
        let storage = Arc::new(RwLock::new(storage));
        
        let (command_tx, mut command_rx) = mpsc::channel::<IndexCommand>(1000);
        
        // Spawn background worker
        tokio::spawn(async move {
            while let Some(command) = command_rx.recv().await {
                match command {
                    IndexCommand::AddDocument { id, embedding } => {
                        let vector = GpuVector::new(id, embedding);
                        let mut storage = storage.write().unwrap();
                        if let Err(e) = storage.add_vectors(&[vector]) {
                            eprintln!("Error adding vector: {}", e);
                        }
                    }
                    IndexCommand::RemoveDocument { id } => {
                        let mut storage = storage.write().unwrap();
                        if let Err(e) = storage.remove_vectors(&[id]) {
                            eprintln!("Error removing vector: {}", e);
                        }
                    }
                    IndexCommand::Search { query, limit, response_tx } => {
                        let storage = storage.read().unwrap();
                        if let Ok(results) = storage.search(&query, limit) {
                            let _ = response_tx.send(results).await;
                        }
                    }
                }
            }
        });
        
        Ok(Self { command_tx })
    }
    
    pub async fn add_document(&self, id: String, embedding: Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
        self.command_tx.send(IndexCommand::AddDocument { id, embedding }).await?;
        Ok(())
    }
    
    pub async fn remove_document(&self, id: String) -> Result<(), Box<dyn std::error::Error>> {
        self.command_tx.send(IndexCommand::RemoveDocument { id }).await?;
        Ok(())
    }
    
    pub async fn search(&self, query: Vec<f32>, limit: usize) -> Result<Vec<GpuSearchResult>, Box<dyn std::error::Error>> {
        let (response_tx, mut response_rx) = mpsc::channel(1);
        self.command_tx.send(IndexCommand::Search { query, limit, response_tx }).await?;
        
        response_rx.recv().await.ok_or_else(|| "No response".into())
    }
}

// Usage
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let search = RealtimeSearch::new(384).await?;
    
    // Add documents continuously
    let search_clone = search.clone();
    tokio::spawn(async move {
        for i in 0..1000 {
            let embedding = vec![i as f32 * 0.001; 384];
            search_clone.add_document(format!("doc_{}", i), embedding).await.unwrap();
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    });
    
    // Concurrent searches
    for _ in 0..10 {
        let query = vec![0.5; 384];
        let results = search.search(query, 10).await?;
        println!("Found {} results", results.len());
    }
    
    Ok(())
}
```

---

## Best Practices

### 1. Resource Management

```rust
// ✅ GOOD: Single context, multiple storages
let context = Arc::new(MetalNativeContext::new()?);

let storage1 = context.create_storage(128, GpuDistanceMetric::Cosine)?;
let storage2 = context.create_storage(384, GpuDistanceMetric::Cosine)?;

// ❌ BAD: Multiple contexts
let ctx1 = MetalNativeContext::new()?;
let ctx2 = MetalNativeContext::new()?;  // Wasteful!
```

### 2. Error Handling

```rust
// ✅ GOOD: Proper error propagation
pub async fn add_document(&self, doc: Document) -> Result<(), AppError> {
    let vector = self.create_vector(&doc)?;
    self.storage.add_vectors(&[vector])
        .map_err(|e| AppError::GpuError(e))?;
    Ok(())
}

// ❌ BAD: Swallowing errors
pub async fn add_document(&self, doc: Document) {
    let _ = self.storage.add_vectors(&[vector]);  // Ignores errors!
}
```

### 3. Batch Processing

```rust
// ✅ GOOD: Batch operations
async fn index_documents(&mut self, docs: Vec<Document>) -> Result<()> {
    let vectors: Vec<GpuVector> = docs.into_iter()
        .map(|doc| self.create_vector(&doc))
        .collect::<Result<_>>()?;
    
    self.storage.add_vectors(&vectors)?;
    Ok(())
}

// ❌ BAD: Individual operations
async fn index_documents(&mut self, docs: Vec<Document>) -> Result<()> {
    for doc in docs {
        let vector = self.create_vector(&doc)?;
        self.storage.add_vectors(&[vector])?;  // Inefficient!
    }
    Ok(())
}
```

### 4. Thread Safety

```rust
// ✅ GOOD: Thread-safe access
use std::sync::{Arc, RwLock};

struct ThreadSafeStorage {
    storage: Arc<RwLock<Box<dyn GpuVectorStorage>>>,
}

impl ThreadSafeStorage {
    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<GpuSearchResult>> {
        let storage = self.storage.read().unwrap();
        storage.search(query, limit)
    }
}
```

---

## Troubleshooting

### Common Integration Issues

#### Issue: Out of Memory

**Solution:**
```rust
// Check VRAM before adding
let stats = context.memory_stats();
if stats.utilization > 0.9 {
    return Err("VRAM nearly full".into());
}

// Or batch with memory checks
for chunk in vectors.chunks(1000) {
    storage.add_vectors(chunk)?;
    
    let stats = context.memory_stats();
    if stats.utilization > 0.95 {
        eprintln!("Warning: High VRAM usage");
        break;
    }
}
```

#### Issue: Dimension Mismatch

**Solution:**
```rust
// Validate dimensions before adding
fn validate_vector(vector: &GpuVector, expected_dim: usize) -> Result<()> {
    if vector.dimension() != expected_dim {
        return Err(format!("Expected dimension {}, got {}", 
                          expected_dim, vector.dimension()).into());
    }
    Ok(())
}
```

---

*Last Updated: 2025-01-03*


