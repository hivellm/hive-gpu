//! Mathematical utilities for vector operations

/// Vector math utilities
pub mod vector_math {
    /// Calculate cosine similarity between two vectors
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Calculate Euclidean distance between two vectors
    pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::INFINITY;
        }

        let sum_squares: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();

        sum_squares.sqrt()
    }

    /// Calculate dot product between two vectors
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Normalize a vector to unit length
    pub fn normalize_vector(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// Calculate vector magnitude
    pub fn vector_magnitude(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

/// Similarity calculation utilities
pub mod similarity_calculations {
    use crate::types::GpuDistanceMetric;

    use super::vector_math;

    /// Calculate similarity based on distance metric
    pub fn calculate_similarity(a: &[f32], b: &[f32], metric: GpuDistanceMetric) -> f32 {
        match metric {
            GpuDistanceMetric::Cosine => vector_math::cosine_similarity(a, b),
            GpuDistanceMetric::Euclidean => {
                // Convert distance to similarity (higher is better)
                let distance = vector_math::euclidean_distance(a, b);
                1.0 / (1.0 + distance)
            }
            GpuDistanceMetric::DotProduct => vector_math::dot_product(a, b),
        }
    }

    /// Batch similarity calculation
    pub fn batch_similarity(
        query: &[f32],
        vectors: &[Vec<f32>],
        metric: GpuDistanceMetric,
    ) -> Vec<f32> {
        vectors
            .iter()
            .map(|v| calculate_similarity(query, v, metric))
            .collect()
    }

    /// Find top-k most similar vectors
    pub fn top_k_similar(
        query: &[f32],
        vectors: &[Vec<f32>],
        k: usize,
        metric: GpuDistanceMetric,
    ) -> Vec<(usize, f32)> {
        let similarities: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, calculate_similarity(query, v, metric)))
            .collect();

        let mut sorted = similarities;
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(k);
        sorted
    }
}
