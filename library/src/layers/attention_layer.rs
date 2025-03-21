use crate::{Helper, NeuralNetwork};
use rand::Rng;
use std::ops::Range;

/// I - input/output width, M - number of heads
pub struct Attention<const I: usize, const M: usize> {
    // Query, key, and value projection weights
    q_weights: [[[f64; I]; I]; M],
    k_weights: [[[f64; I]; I]; M],
    v_weights: [[[f64; I]; I]; M],
    
    // Output projection
    o_weights: [[f64; I]; I],
    
    // Attention scaling factors
    attention_scale: [f64; M],
    
    // Whether to apply layer normalization
    use_layer_norm: bool,
    
    // Parameter for gradient updates
    head_dropout_rate: f64,
}

impl<const I: usize, const M: usize> Attention<I, M> {
    pub fn random(value_range: Range<f64>) -> impl NeuralNetwork<I, I> {
        let mut rng = rand::rng();
        
        Attention {
            q_weights: [[[rng.random_range(value_range.clone()); I]; I]; M],
            k_weights: [[[rng.random_range(value_range.clone()); I]; I]; M],
            v_weights: [[[rng.random_range(value_range.clone()); I]; I]; M],
            o_weights: [[rng.random_range(value_range.clone()); I]; I],
            attention_scale: [1.0 / (I as f64).sqrt(); M], // Initialize with standard scaling
            use_layer_norm: true,
            head_dropout_rate: 0.1,
        }
    }
    
    pub fn with_layer_norm(mut self, use_norm: bool) -> Self {
        self.use_layer_norm = use_norm;
        self
    }
    
    pub fn with_dropout_rate(mut self, rate: f64) -> Self {
        self.head_dropout_rate = rate.max(0.0).min(1.0); // Clamp between 0 and 1
        self
    }
    
    fn layer_norm(&self, x: &mut [f64; I]) {
        if !self.use_layer_norm {
            return;
        }
        
        // Calculate mean
        let mean = x.iter().sum::<f64>() / I as f64;
        
        // Calculate variance
        let var = x.iter()
            .map(|&val| (val - mean).powi(2))
            .sum::<f64>() / I as f64;
            
        // Normalize
        let std_dev = (var + 1e-8).sqrt();
        for i in 0..I {
            x[i] = (x[i] - mean) / std_dev;
        }
    }
    
    fn compute_qkv(&self, x: &[f64; I], head: usize) -> ([f64; I], [f64; I], [f64; I]) {
        let mut q = [0.0; I];
        let mut k = [0.0; I];
        let mut v = [0.0; I];
        
        // Project input to query, key, and value
        for i in 0..I {
            for j in 0..I {
                q[i] += x[j] * self.q_weights[head][i][j];
                k[i] += x[j] * self.k_weights[head][i][j];
                v[i] += x[j] * self.v_weights[head][i][j];
            }
            
            // Apply scaling factor
            q[i] *= self.attention_scale[head];
        }
        
        (q, k, v)
    }
}

impl<const I: usize, const M: usize> NeuralNetwork<I, I> for Attention<I, M> {
    fn forward(&mut self, x: &[f64; I], h: Option<&mut impl Helper>) -> [f64; I] {
        let ret = self.preprocess(x);

        if let Some(h) = h {
            h.push(x);
        }

        ret
    }

    fn reset(&mut self) {
        // No state to reset
    }
    
    fn preprocess(&mut self, x: &[f64; I]) -> [f64; I] {
        let mut multi_head_results = [[0.0; I]; M];
        
        // Process each attention head
        for head in 0..M {
            let (q, k, v) = self.compute_qkv(x, head);
            
            // Compute attention scores (Q·K^T)
            let mut scores = [0.0; I];
            for i in 0..I {
                for j in 0..I {
                    scores[i] += q[i] * k[j];
                }
            }
            
            // Apply softmax
            let mut weights = [0.0; I];
            let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut sum_exp = 0.0;
            
            for i in 0..I {
                weights[i] = (scores[i] - max_score).exp();
                sum_exp += weights[i];
            }
            
            for i in 0..I {
                weights[i] /= sum_exp;
            }
            
            // Apply attention weights to values
            for i in 0..I {
                for j in 0..I {
                    multi_head_results[head][i] += weights[j] * v[j];
                }
            }
            
            // Apply layer normalization to each head output
            if self.use_layer_norm {
                self.layer_norm(&mut multi_head_results[head]);
            }
        }
        
        // Concatenate and project all heads
        let mut result = [0.0; I];
        for i in 0..I {
            for head in 0..M {
                // Apply head dropout during training
                let head_factor = if rand::random::<f64>() < self.head_dropout_rate {
                    0.0
                } else {
                    1.0 / (1.0 - self.head_dropout_rate) // Scale to maintain expected value
                };
                
                // Combine heads with the output projection
                for j in 0..I {
                    result[i] += multi_head_results[head][j] * self.o_weights[i][j] * head_factor / M as f64;
                }
            }
        }
        
        // Final layer norm on the output
        if self.use_layer_norm {
            self.layer_norm(&mut result);
        }
        
        result
    }
    
    fn backward(&mut self, helper: &mut impl Helper, errors: [f64; I], temperature: f64) -> [f64; I] {

        let helper = helper
          .pop()
          .expect("Expected input to be pushed to helper");

        let x: &[f64; I] = helper
          .as_array()
          .expect("Expected array of right size");

        // Simplified backpropagation for attention mechanism
        let mut input_grads = [0.0; I];
        
        // For each head, compute gradients
        for head in 0..M {
            let (q, k, v) = self.compute_qkv(x, head);
            
            // Compute attention scores and gradients
            // This is a simplified implementation - a real one would compute proper gradients
            for i in 0..I {
                for j in 0..I {
                    // Update query weights
                    self.q_weights[head][i][j] += temperature * errors[i] * k[j] * 0.01;
                    
                    // Update key weights
                    self.k_weights[head][i][j] += temperature * errors[i] * q[j] * 0.01;
                    
                    // Update value weights
                    self.v_weights[head][i][j] += temperature * errors[i] * 0.01;
                    
                    // Update output projection weights
                    self.o_weights[i][j] += temperature * errors[i] * v[j] * 0.01;
                    
                    // Propagate error back to input
                    input_grads[j] += errors[i] * (self.q_weights[head][i][j] + 
                                                  self.k_weights[head][i][j] + 
                                                  self.v_weights[head][i][j]) / 3.0;
                }
            }
            
            // Update attention scale
            self.attention_scale[head] += temperature * errors.iter().sum::<f64>() * 0.001;
        }
        
        input_grads
    }
}