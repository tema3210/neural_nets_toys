use crate::*;
use std::ops::Range;
use rand::Rng;

/// DEPTH must be greater than 1
pub struct LNNLayer<const INP: usize, const OUT: usize, const DEPTH: usize>
{   
    // attention: [f64; INP],
    weights: [[f64; INP]; OUT],
    bias: [f64; OUT],

    /// this is the current step in the hidden state vector, always less than DEPTH
    /// it is used to determine the index of the hidden state vector to write to
    /// an index to the ring buffer
    current_step: usize,
    hidden_state: [[f64; INP]; DEPTH],

    /// Governs the amount of information and the way to keep it 
    /// from the hidden state vector
    alpha_vector: [f64; INP],

    activation: fn(f64) -> f64,
    activation_derivative: Option<Box<dyn Fn(f64) -> f64>>,
}

impl<const I: usize, const O: usize, const D: usize> LNNLayer<I, O, D> {
    pub fn random(activation: fn(f64) -> f64, value_range: Range<f64>) -> Self {
        let mut rng = rand::rng();
        
        LNNLayer {
            weights: [[rng.random_range(value_range.clone()); I]; O],
            bias: [rng.random_range(value_range.clone()); O],
            // attention: [1.0; I],

            // q_weights: [[rng.random_range(value_range.clone()); I]; I],
            // k_weights: [[rng.random_range(value_range.clone()); I]; I],
            // v_weights: [[rng.random_range(value_range.clone()); I]; I],

            current_step: 0,
            hidden_state: [[0.0; I]; D],

            alpha_vector: [rng.random_range(value_range.clone()); I],
            activation,
            activation_derivative: None,
        }
    }

    pub fn with_derivative(mut self, derivative: impl Fn(f64) -> f64 + 'static) -> Self {
        self.activation_derivative = Some(Box::new(derivative));
        self
    }

    fn derivative(&self, x: f64) -> f64 {
        if let Some(derivative) = &self.activation_derivative {
            return derivative(x);
        }
        let step = 0.001;
        ((self.activation)(x + step) - (self.activation)(x - step)) / (2.0 * step)
    }

    /// the absolute length of the hidden state vector
    fn alpha(&self) -> f64 {
        self.alpha_vector.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// get the indices of the ring buffer: (head, tail)
    fn get_ring_indices(&self) -> (usize, usize) {
        let from = self.current_step % D;
        let to = (self.current_step + 1) % D;
        (from, to)
    }

    /// this one of the optimized operators
    /// compress the hidden state vector from the from index to the to index
    fn merge_hidden_states(&mut self, from: usize, to: usize) {
        for inp in 0..I {
            self.hidden_state[to][inp] *= self.alpha_vector[inp];
            self.hidden_state[to][inp] += ( 1.0 - self.alpha_vector[inp] )* self.hidden_state[from][inp];
        }
    }

}

impl<const I: usize, const O: usize, const D: usize> NeuralNetwork<I, O> for LNNLayer<I, O, D> {

    fn forward(&mut self, x: &[f64; I]) -> [f64; O] {
        let x = self.preprocess(x);
        let mut sum = self.bias;

        // activate on alpha value to normalize
        let alpha = (self.activation)(self.alpha());

        for neuron in 0..O {
            for inp in 0..I {
                //here we account for the inputs
                sum[neuron] += (1.0 - alpha) * self.weights[neuron][inp] * x[inp];
                //here we account the D hidden states
                for hi in 0..D {
                    sum[neuron] += self.hidden_state[hi][inp] * alpha / D as f64;
                }
            }
            //activate
            sum[neuron] = (self.activation)(sum[neuron]);
        }

        let (head,tail) = self.get_ring_indices();
        self.merge_hidden_states(head, tail);
        self.hidden_state[head] = x;
        self.current_step +=1;
        sum
    }
    
    fn backward(&mut self, x: &[f64; I], errors: [f64; O], temperature: f64) -> [f64; I] {
        let sample = self.preprocess(x);
        let predict = self.forward(&sample);

        let mut correction = [0.0; I];
        let alpha = self.alpha(); // the magnitude of accumulation
        
        for neuron in 0..O {
            let delta = errors[neuron] * self.derivative(predict[neuron]);
            for input_no in 0..I {
                // the correction for the weigths
                let static_delta = delta * (1.0 - (self.activation)(alpha));
                // the correction for the hidden states
                let recurrent_delta = delta * (self.activation)(alpha);

                //total correction accounting
                correction[input_no] += static_delta * self.weights[neuron][input_no];
                correction[input_no] += recurrent_delta * self.alpha_vector[input_no];

                //update the weights
                self.weights[neuron][input_no] += temperature * static_delta * sample[input_no];
                self.alpha_vector[input_no] += temperature * recurrent_delta * self.alpha_vector[input_no];
            }
            self.bias[neuron] += temperature * delta;
        }
        
        correction
    }
}