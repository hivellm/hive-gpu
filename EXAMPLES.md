# 🚀 Hive-GPU Examples

**Practical examples for using hive-gpu v0.1.0 in real-world applications**

## 📚 Table of Contents

1. [Basic Usage](#basic-usage)
2. [Document Search](#document-search)
3. [Image Similarity](#image-similarity)
4. [Recommendation System](#recommendation-system)
5. [Vectorizer Integration](#vectorizer-integration)
6. [Performance Optimization](#performance-optimization)
7. [Error Handling](#error-handling)

## 🎯 Basic Usage

### Simple Vector Operations

```rust
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};
use hive_gpu::types::{GpuVector, GpuDistanceMetric};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU context
    let context = MetalNativeContext::new()?;
    let mut storage = context.create_storage(4, GpuDistanceMetric::Cosine)?;
    
    // Create vectors
    let vectors = vec![
        GpuVector {
            id: "vector_1".to_string(),
            data: vec![1.0, 0.0, 0.0, 0.0],
            metadata: HashMap::new(),
        },
        GpuVector {
            id: "vector_2".to_string(),
            data: vec![0.0, 1.0, 0.0, 0.0],
            metadata: HashMap::new(),
        },
        GpuVector {
            id: "vector_3".to_string(),
            data: vec![0.0, 0.0, 1.0, 0.0],
            metadata: HashMap::new(),
        },
    ];
    
    // Add vectors to GPU
    storage.add_vectors(&vectors)?;
    println!("Added {} vectors", storage.vector_count());
    
    // Search for similar vectors
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let results = storage.search(&query, 3)?;
    
    println!("Search results:");
    for (i, result) in results.iter().enumerate() {
        println!("{}. {} (similarity: {:.4})", i + 1, result.id, result.score);
    }
    
    Ok(())
}
```

### Batch Processing

```rust
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};
use hive_gpu::types::{GpuVector, GpuDistanceMetric};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = MetalNativeContext::new()?;
    let mut storage = context.create_storage(128, GpuDistanceMetric::Cosine)?;
    
    // Generate large batch of random vectors
    let batch_size = 10000;
    let mut vectors = Vec::with_capacity(batch_size);
    
    for i in 0..batch_size {
        let data = (0..128).map(|_| rand::random::<f32>()).collect();
        vectors.push(GpuVector {
            id: format!("batch_vector_{}", i),
            data,
            metadata: HashMap::new(),
        });
    }
    
    // Add vectors in chunks for efficiency
    let chunk_size = 1000;
    for chunk in vectors.chunks(chunk_size) {
        storage.add_vectors(chunk)?;
        println!("Added {} vectors", chunk.len());
    }
    
    println!("Total vectors: {}", storage.vector_count());
    
    // Batch search
    let queries = vec![
        (0..128).map(|_| rand::random::<f32>()).collect::<Vec<f32>>(),
        (0..128).map(|_| rand::random::<f32>()).collect::<Vec<f32>>(),
        (0..128).map(|_| rand::random::<f32>()).collect::<Vec<f32>>(),
    ];
    
    for (i, query) in queries.iter().enumerate() {
        let results = storage.search(query, 5)?;
        println!("Query {}: Found {} results", i + 1, results.len());
    }
    
    Ok(())
}
```

## 📄 Document Search

### Semantic Document Search

```rust
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};
use hive_gpu::types::{GpuVector, GpuDistanceMetric};
use std::collections::HashMap;

struct DocumentSearch {
    storage: Box<dyn GpuVectorStorage>,
}

impl DocumentSearch {
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let context = MetalNativeContext::new()?;
        let storage = context.create_storage(384, GpuDistanceMetric::Cosine)?;
        
        Ok(Self { storage })
    }
    
    async fn add_document(&mut self, id: &str, text: &str, embedding: Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let mut metadata = HashMap::new();
        metadata.insert("text".to_string(), text.to_string());
        metadata.insert("length".to_string(), text.len().to_string());
        
        let vector = GpuVector {
            id: id.to_string(),
            data: embedding,
            metadata,
        };
        
        self.storage.add_vectors(&[vector])?;
        Ok(())
    }
    
    async fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        let results = self.storage.search(query_embedding, limit)?;
        
        Ok(results.into_iter().map(|r| SearchResult {
            id: r.id,
            score: r.score,
        }).collect())
    }
}

#[derive(Debug)]
struct SearchResult {
    id: String,
    score: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut search = DocumentSearch::new().await?;
    
    // Add documents (in practice, use real embeddings)
    let documents = vec![
        ("doc_1", "Machine learning and artificial intelligence", generate_embedding("Machine learning and artificial intelligence")),
        ("doc_2", "Deep learning neural networks", generate_embedding("Deep learning neural networks")),
        ("doc_3", "Natural language processing", generate_embedding("Natural language processing")),
        ("doc_4", "Computer vision and image recognition", generate_embedding("Computer vision and image recognition")),
        ("doc_5", "Reinforcement learning algorithms", generate_embedding("Reinforcement learning algorithms")),
    ];
    
    for (id, text, embedding) in documents {
        search.add_document(id, text, embedding).await?;
    }
    
    // Search for similar documents
    let query = "AI and machine learning";
    let query_embedding = generate_embedding(query);
    let results = search.search(&query_embedding, 3).await?;
    
    println!("Search results for: '{}'", query);
    for (i, result) in results.iter().enumerate() {
        println!("{}. {} (similarity: {:.4})", i + 1, result.id, result.score);
    }
    
    Ok(())
}

// Mock embedding function (replace with real implementation)
fn generate_embedding(text: &str) -> Vec<f32> {
    // In practice, use sentence-transformers or similar
    (0..384).map(|i| (i as f32 + text.len() as f32) * 0.01).collect()
}
```

### Advanced Document Search with Metadata

```rust
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};
use hive_gpu::types::{GpuVector, GpuDistanceMetric};
use std::collections::HashMap;
use serde_json::Value;

struct AdvancedDocumentSearch {
    storage: Box<dyn GpuVectorStorage>,
}

impl AdvancedDocumentSearch {
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let context = MetalNativeContext::new()?;
        let storage = context.create_storage(512, GpuDistanceMetric::Cosine)?;
        
        Ok(Self { storage })
    }
    
    async fn add_document(&mut self, document: Document) -> Result<(), Box<dyn std::error::Error>> {
        let mut metadata = HashMap::new();
        metadata.insert("title".to_string(), document.title);
        metadata.insert("author".to_string(), document.author);
        metadata.insert("category".to_string(), document.category);
        metadata.insert("date".to_string(), document.date);
        metadata.insert("tags".to_string(), serde_json::to_string(&document.tags)?);
        
        let vector = GpuVector {
            id: document.id,
            data: document.embedding,
            metadata,
        };
        
        self.storage.add_vectors(&[vector])?;
        Ok(())
    }
    
    async fn search_with_filters(&self, query: &[f32], category: Option<&str>, limit: usize) -> Result<Vec<DocumentResult>, Box<dyn std::error::Error>> {
        let results = self.storage.search(query, limit * 2)?; // Get more results for filtering
        
        let mut filtered_results = Vec::new();
        for result in results {
            if let Some(cat) = category {
                if let Some(metadata_category) = result.metadata.get("category") {
                    if metadata_category != cat {
                        continue;
                    }
                }
            }
            
            filtered_results.push(DocumentResult {
                id: result.id,
                score: result.score,
                metadata: result.metadata,
            });
            
            if filtered_results.len() >= limit {
                break;
            }
        }
        
        Ok(filtered_results)
    }
}

#[derive(Debug)]
struct Document {
    id: String,
    title: String,
    author: String,
    category: String,
    date: String,
    tags: Vec<String>,
    embedding: Vec<f32>,
}

#[derive(Debug)]
struct DocumentResult {
    id: String,
    score: f32,
    metadata: HashMap<String, String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut search = AdvancedDocumentSearch::new().await?;
    
    // Add documents with rich metadata
    let documents = vec![
        Document {
            id: "doc_1".to_string(),
            title: "Introduction to Machine Learning".to_string(),
            author: "John Doe".to_string(),
            category: "AI".to_string(),
            date: "2024-01-15".to_string(),
            tags: vec!["machine-learning".to_string(), "ai".to_string()],
            embedding: generate_embedding("Introduction to Machine Learning"),
        },
        Document {
            id: "doc_2".to_string(),
            title: "Deep Learning Fundamentals".to_string(),
            author: "Jane Smith".to_string(),
            category: "AI".to_string(),
            date: "2024-01-20".to_string(),
            tags: vec!["deep-learning".to_string(), "neural-networks".to_string()],
            embedding: generate_embedding("Deep Learning Fundamentals"),
        },
        Document {
            id: "doc_3".to_string(),
            title: "Web Development Best Practices".to_string(),
            author: "Bob Johnson".to_string(),
            category: "Web".to_string(),
            date: "2024-01-25".to_string(),
            tags: vec!["web-development".to_string(), "best-practices".to_string()],
            embedding: generate_embedding("Web Development Best Practices"),
        },
    ];
    
    for document in documents {
        search.add_document(document).await?;
    }
    
    // Search with category filter
    let query = generate_embedding("machine learning algorithms");
    let results = search.search_with_filters(&query, Some("AI"), 5).await?;
    
    println!("AI category search results:");
    for (i, result) in results.iter().enumerate() {
        println!("{}. {} (similarity: {:.4})", i + 1, result.id, result.score);
        println!("   Title: {}", result.metadata.get("title").unwrap_or(&"Unknown".to_string()));
        println!("   Author: {}", result.metadata.get("author").unwrap_or(&"Unknown".to_string()));
    }
    
    Ok(())
}

fn generate_embedding(text: &str) -> Vec<f32> {
    (0..512).map(|i| (i as f32 + text.len() as f32) * 0.01).collect()
}
```

## 🖼️ Image Similarity

### Image Search System

```rust
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};
use hive_gpu::types::{GpuVector, GpuDistanceMetric};
use std::collections::HashMap;

struct ImageSearch {
    storage: Box<dyn GpuVectorStorage>,
}

impl ImageSearch {
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let context = MetalNativeContext::new()?;
        let storage = context.create_storage(2048, GpuDistanceMetric::Cosine)?; // ResNet-50 embedding size
        
        Ok(Self { storage })
    }
    
    async fn add_image(&mut self, id: &str, path: &str, embedding: Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let mut metadata = HashMap::new();
        metadata.insert("path".to_string(), path.to_string());
        metadata.insert("type".to_string(), "image".to_string());
        
        let vector = GpuVector {
            id: id.to_string(),
            data: embedding,
            metadata,
        };
        
        self.storage.add_vectors(&[vector])?;
        Ok(())
    }
    
    async fn find_similar_images(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<ImageResult>, Box<dyn std::error::Error>> {
        let results = self.storage.search(query_embedding, limit)?;
        
        Ok(results.into_iter().map(|r| ImageResult {
            id: r.id,
            score: r.score,
            path: r.metadata.get("path").unwrap_or(&"Unknown".to_string()).clone(),
        }).collect())
    }
}

#[derive(Debug)]
struct ImageResult {
    id: String,
    score: f32,
    path: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut search = ImageSearch::new().await?;
    
    // Add images (in practice, use real image embeddings)
    let images = vec![
        ("img_1", "/path/to/cat1.jpg", generate_image_embedding("cat")),
        ("img_2", "/path/to/cat2.jpg", generate_image_embedding("cat")),
        ("img_3", "/path/to/dog1.jpg", generate_image_embedding("dog")),
        ("img_4", "/path/to/dog2.jpg", generate_image_embedding("dog")),
        ("img_5", "/path/to/car1.jpg", generate_image_embedding("car")),
    ];
    
    for (id, path, embedding) in images {
        search.add_image(id, path, embedding).await?;
    }
    
    // Search for similar images
    let query_embedding = generate_image_embedding("cat");
    let results = search.find_similar_images(&query_embedding, 3).await?;
    
    println!("Similar images to 'cat':");
    for (i, result) in results.iter().enumerate() {
        println!("{}. {} (similarity: {:.4}) - {}", i + 1, result.id, result.score, result.path);
    }
    
    Ok(())
}

fn generate_image_embedding(description: &str) -> Vec<f32> {
    // In practice, use ResNet-50 or similar model
    (0..2048).map(|i| (i as f32 + description.len() as f32) * 0.01).collect()
}
```

## 🎯 Recommendation System

### Product Recommendation Engine

```rust
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};
use hive_gpu::types::{GpuVector, GpuDistanceMetric};
use std::collections::HashMap;

struct RecommendationEngine {
    storage: Box<dyn GpuVectorStorage>,
}

impl RecommendationEngine {
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let context = MetalNativeContext::new()?;
        let storage = context.create_storage(100, GpuDistanceMetric::Cosine)?;
        
        Ok(Self { storage })
    }
    
    async fn add_product(&mut self, product: Product) -> Result<(), Box<dyn std::error::Error>> {
        let mut metadata = HashMap::new();
        metadata.insert("name".to_string(), product.name);
        metadata.insert("category".to_string(), product.category);
        metadata.insert("price".to_string(), product.price.to_string());
        metadata.insert("rating".to_string(), product.rating.to_string());
        
        let vector = GpuVector {
            id: product.id,
            data: product.features,
            metadata,
        };
        
        self.storage.add_vectors(&[vector])?;
        Ok(())
    }
    
    async fn get_recommendations(&self, user_preferences: &[f32], limit: usize) -> Result<Vec<ProductRecommendation>, Box<dyn std::error::Error>> {
        let results = self.storage.search(user_preferences, limit)?;
        
        Ok(results.into_iter().map(|r| ProductRecommendation {
            product_id: r.id,
            score: r.score,
            name: r.metadata.get("name").unwrap_or(&"Unknown".to_string()).clone(),
            category: r.metadata.get("category").unwrap_or(&"Unknown".to_string()).clone(),
            price: r.metadata.get("price").unwrap_or(&"0".to_string()).parse().unwrap_or(0.0),
            rating: r.metadata.get("rating").unwrap_or(&"0".to_string()).parse().unwrap_or(0.0),
        }).collect())
    }
}

#[derive(Debug)]
struct Product {
    id: String,
    name: String,
    category: String,
    price: f32,
    rating: f32,
    features: Vec<f32>,
}

#[derive(Debug)]
struct ProductRecommendation {
    product_id: String,
    score: f32,
    name: String,
    category: String,
    price: f32,
    rating: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = RecommendationEngine::new().await?;
    
    // Add products
    let products = vec![
        Product {
            id: "prod_1".to_string(),
            name: "Wireless Headphones".to_string(),
            category: "Electronics".to_string(),
            price: 99.99,
            rating: 4.5,
            features: generate_product_features("electronics", "audio", 99.99, 4.5),
        },
        Product {
            id: "prod_2".to_string(),
            name: "Smartphone".to_string(),
            category: "Electronics".to_string(),
            price: 699.99,
            rating: 4.8,
            features: generate_product_features("electronics", "mobile", 699.99, 4.8),
        },
        Product {
            id: "prod_3".to_string(),
            name: "Running Shoes".to_string(),
            category: "Sports".to_string(),
            price: 129.99,
            rating: 4.2,
            features: generate_product_features("sports", "footwear", 129.99, 4.2),
        },
    ];
    
    for product in products {
        engine.add_product(product).await?;
    }
    
    // Get recommendations for a user
    let user_preferences = generate_user_preferences("electronics", "high_quality", 500.0);
    let recommendations = engine.get_recommendations(&user_preferences, 3).await?;
    
    println!("Product recommendations:");
    for (i, rec) in recommendations.iter().enumerate() {
        println!("{}. {} (score: {:.4})", i + 1, rec.name, rec.score);
        println!("   Category: {}, Price: ${:.2}, Rating: {:.1}", rec.category, rec.price, rec.rating);
    }
    
    Ok(())
}

fn generate_product_features(category: &str, subcategory: &str, price: f32, rating: f32) -> Vec<f32> {
    // In practice, use real product feature extraction
    let mut features = vec![0.0; 100];
    
    // Category encoding
    match category {
        "electronics" => features[0] = 1.0,
        "sports" => features[1] = 1.0,
        _ => {}
    }
    
    // Price normalization
    features[2] = price / 1000.0;
    
    // Rating normalization
    features[3] = rating / 5.0;
    
    // Add some random features
    for i in 4..100 {
        features[i] = rand::random::<f32>();
    }
    
    features
}

fn generate_user_preferences(category: &str, quality: &str, max_price: f32) -> Vec<f32> {
    let mut preferences = vec![0.0; 100];
    
    // Category preference
    match category {
        "electronics" => preferences[0] = 1.0,
        "sports" => preferences[1] = 1.0,
        _ => {}
    }
    
    // Quality preference
    match quality {
        "high_quality" => preferences[2] = 1.0,
        "budget" => preferences[2] = 0.3,
        _ => {}
    }
    
    // Price preference
    preferences[3] = max_price / 1000.0;
    
    // Add some random preferences
    for i in 4..100 {
        preferences[i] = rand::random::<f32>();
    }
    
    preferences
}
```

## 🔗 Vectorizer Integration

### Using with Hive-Vectorizer

```rust
use vectorizer::VectorStore;
use vectorizer::models::{CollectionConfig, DistanceMetric, HnswConfig, Vector, Payload};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create vectorizer store
    let mut store = VectorStore::new();
    
    // Configure collection with GPU acceleration
    let config = CollectionConfig {
        dimension: 512,
        metric: DistanceMetric::Cosine,
        hnsw_config: HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            seed: 42,
        },
    };
    
    // Create collection
    store.create_collection("documents", config)?;
    
    // Add documents
    let documents = vec![
        Vector {
            id: "doc_1".to_string(),
            data: generate_embedding("Machine learning and AI"),
            payload: Some(Payload::new(json!({
                "title": "Introduction to ML",
                "author": "John Doe",
                "category": "AI"
            }))),
        },
        Vector {
            id: "doc_2".to_string(),
            data: generate_embedding("Deep learning fundamentals"),
            payload: Some(Payload::new(json!({
                "title": "Deep Learning Guide",
                "author": "Jane Smith",
                "category": "AI"
            }))),
        },
    ];
    
    store.add_vectors("documents", documents)?;
    
    // Search
    let query = generate_embedding("artificial intelligence");
    let results = store.search("documents", &query, 5)?;
    
    println!("Search results:");
    for (i, result) in results.iter().enumerate() {
        println!("{}. {} (score: {:.4})", i + 1, result.id, result.score);
        if let Some(payload) = &result.payload {
            println!("   Title: {}", payload.data.get("title").unwrap_or(&json!("Unknown")));
            println!("   Author: {}", payload.data.get("author").unwrap_or(&json!("Unknown")));
        }
    }
    
    Ok(())
}

fn generate_embedding(text: &str) -> Vec<f32> {
    (0..512).map(|i| (i as f32 + text.len() as f32) * 0.01).collect()
}
```

## ⚡ Performance Optimization

### Efficient Batch Processing

```rust
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};
use hive_gpu::types::{GpuVector, GpuDistanceMetric};
use std::collections::HashMap;
use tokio::task;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = MetalNativeContext::new()?;
    let mut storage = context.create_storage(128, GpuDistanceMetric::Cosine)?;
    
    // Generate large dataset
    let total_vectors = 100000;
    let batch_size = 1000;
    
    println!("Processing {} vectors in batches of {}", total_vectors, batch_size);
    
    let start_time = std::time::Instant::now();
    
    for batch_num in 0..(total_vectors / batch_size) {
        let mut vectors = Vec::with_capacity(batch_size);
        
        for i in 0..batch_size {
            let global_index = batch_num * batch_size + i;
            let data = (0..128).map(|_| rand::random::<f32>()).collect();
            
            vectors.push(GpuVector {
                id: format!("vector_{}", global_index),
                data,
                metadata: HashMap::new(),
            });
        }
        
        storage.add_vectors(&vectors)?;
        
        if batch_num % 10 == 0 {
            println!("Processed {} vectors", (batch_num + 1) * batch_size);
        }
    }
    
    let elapsed = start_time.elapsed();
    println!("Added {} vectors in {:.2}s", total_vectors, elapsed.as_secs_f32());
    println!("Throughput: {:.0} vectors/sec", total_vectors as f32 / elapsed.as_secs_f32());
    
    // Benchmark search performance
    let search_start = std::time::Instant::now();
    let query = (0..128).map(|_| rand::random::<f32>()).collect::<Vec<f32>>();
    let results = storage.search(&query, 10)?;
    let search_elapsed = search_start.elapsed();
    
    println!("Search completed in {:.2}μs", search_elapsed.as_micros());
    println!("Found {} results", results.len());
    
    Ok(())
}
```

### Memory Monitoring

```rust
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};
use hive_gpu::types::{GpuVector, GpuDistanceMetric};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = MetalNativeContext::new()?;
    
    // Monitor memory usage
    let memory_stats = context.memory_stats();
    println!("Initial GPU memory: {:.2} MB used, {:.2} MB available", 
             memory_stats.used_memory_mb, memory_stats.available_memory_mb);
    
    let mut storage = context.create_storage(512, GpuDistanceMetric::Cosine)?;
    
    // Add vectors and monitor memory
    let batch_size = 1000;
    for batch in 0..10 {
        let mut vectors = Vec::with_capacity(batch_size);
        
        for i in 0..batch_size {
            let data = (0..512).map(|_| rand::random::<f32>()).collect();
            vectors.push(GpuVector {
                id: format!("batch_{}_vector_{}", batch, i),
                data,
                metadata: HashMap::new(),
            });
        }
        
        storage.add_vectors(&vectors)?;
        
        let memory_stats = context.memory_stats();
        println!("Batch {}: {:.2} MB used, {:.2} MB available", 
                 batch + 1, memory_stats.used_memory_mb, memory_stats.available_memory_mb);
    }
    
    Ok(())
}
```

## 🛡️ Error Handling

### Comprehensive Error Handling

```rust
use hive_gpu::metal::context::MetalNativeContext;
use hive_gpu::traits::{GpuContext, GpuVectorStorage};
use hive_gpu::types::{GpuVector, GpuDistanceMetric};
use hive_gpu::error::HiveGpuError;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize with error handling
    let context = match MetalNativeContext::new() {
        Ok(ctx) => {
            println!("✅ GPU context initialized successfully");
            ctx
        },
        Err(HiveGpuError::NoDeviceAvailable) => {
            println!("❌ No GPU device available, falling back to CPU");
            return fallback_to_cpu().await;
        },
        Err(e) => {
            println!("❌ Failed to initialize GPU: {}", e);
            return Err(e.into());
        }
    };
    
    // Create storage with error handling
    let mut storage = match context.create_storage(128, GpuDistanceMetric::Cosine) {
        Ok(storage) => {
            println!("✅ Vector storage created successfully");
            storage
        },
        Err(e) => {
            println!("❌ Failed to create storage: {}", e);
            return Err(e.into());
        }
    };
    
    // Add vectors with error handling
    let vectors = create_test_vectors();
    match storage.add_vectors(&vectors) {
        Ok(indices) => {
            println!("✅ Added {} vectors successfully", indices.len());
        },
        Err(HiveGpuError::InsufficientMemory) => {
            println!("❌ Insufficient GPU memory, reducing batch size");
            return add_vectors_in_smaller_batches(&mut storage, &vectors).await;
        },
        Err(e) => {
            println!("❌ Failed to add vectors: {}", e);
            return Err(e.into());
        }
    }
    
    // Search with error handling
    let query = vec![1.0; 128];
    match storage.search(&query, 10) {
        Ok(results) => {
            println!("✅ Search completed: {} results", results.len());
            for (i, result) in results.iter().enumerate() {
                println!("{}. {} (score: {:.4})", i + 1, result.id, result.score);
            }
        },
        Err(e) => {
            println!("❌ Search failed: {}", e);
            return Err(e.into());
        }
    }
    
    Ok(())
}

async fn fallback_to_cpu() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Implementing CPU fallback...");
    // Implement CPU-based vector operations
    Ok(())
}

async fn add_vectors_in_smaller_batches(
    storage: &mut Box<dyn GpuVectorStorage>, 
    vectors: &[GpuVector]
) -> Result<(), Box<dyn std::error::Error>> {
    let small_batch_size = 100;
    
    for chunk in vectors.chunks(small_batch_size) {
        storage.add_vectors(chunk)?;
        println!("Added batch of {} vectors", chunk.len());
    }
    
    Ok(())
}

fn create_test_vectors() -> Vec<GpuVector> {
    (0..1000).map(|i| GpuVector {
        id: format!("test_vector_{}", i),
        data: (0..128).map(|_| rand::random::<f32>()).collect(),
        metadata: HashMap::new(),
    }).collect()
}
```

---

**These examples demonstrate the power and flexibility of hive-gpu v0.1.0 for real-world applications! 🚀**
