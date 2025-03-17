use crate::*;
use std::ops::Range;
use rand::Rng;
use crate::layers::fc_layer::Layer as FCLayer;

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

    /// FC layer for controlling hidden state information flow
    alpha_layer: FCLayer<INP, INP>,

    alpha: f64,

    activation: fn(f64) -> f64,
    activation_derivative: Option<fn(f64) -> f64>,
}

impl<const I: usize, const O: usize, const D: usize> LNNLayer<I, O, D> {
    pub fn random(activation: fn(f64) -> f64, value_range: Range<f64>,alpha: f64) -> Self {
        let mut rng = rand::rng();
        
        LNNLayer {
            weights: [[rng.random_range(value_range.clone()); I]; O],
            bias: [rng.random_range(value_range.clone()); O],

            current_step: 0,
            hidden_state: [[0.0; I]; D],

            alpha_layer: FCLayer::random(activation, value_range.clone()),
            alpha,

            activation,
            activation_derivative: None,
        }
    }

    pub fn with_derivative(mut self, derivative: fn(f64) -> f64) -> Self {
      self.activation_derivative = Some(derivative);
      self.alpha_layer = self.alpha_layer.with_derivative(derivative);
      self
    }

    fn derivative(&self, x: f64) -> f64 {
        if let Some(derivative) = &self.activation_derivative {
            return derivative(x);
        }
        let step = 0.001;
        ((self.activation)(x + step) - (self.activation)(x - step)) / (2.0 * step)
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
        let from_state = self.hidden_state[from];
        let alpha_weights = self.alpha_layer.forward(&from_state, None::<&mut DefaultHelper>);
        
        for inp in 0..I {
            let alpha_weight = (self.activation)(alpha_weights[inp]);
            self.hidden_state[to][inp] = alpha_weight * self.hidden_state[to][inp] + 
                                        (1.0 - alpha_weight) * from_state[inp];
        }
    }
}

impl<const I: usize, const O: usize, const D: usize> NeuralNetwork<I, O> for LNNLayer<I, O, D> {

    fn forward(&mut self, x: &[f64; I], helper: Option<&mut impl Helper>) -> [f64; O] {
        let x = self.preprocess(x);
        // Push the input to the helper for use in backpropagation
        if let Some(h) = helper {
            h.push(&x);
        }
        
        let mut sum = self.bias;

        for neuron in 0..O {
            for inp in 0..I {
                //here we account for the inputs
                sum[neuron] += (1.0 - self.alpha) * self.weights[neuron][inp] * x[inp];
                //here we account the D hidden states
                for hi in 0..D {
                    sum[neuron] += self.hidden_state[hi][inp] * self.alpha / D as f64;
                }
            }
            //activate
            sum[neuron] = (self.activation)(sum[neuron]);
        }

        let (head, tail) = self.get_ring_indices();
        self.merge_hidden_states(head, tail);
        self.hidden_state[head] = x;
        self.current_step += 1;
        sum
    }
    
    fn backward(&mut self, helper_ref: &mut impl Helper, errors: [f64; O], temperature: f64) -> [f64; I] {
        let helper = helper_ref.pop().expect("Expected input to be pushed to helper");
        let sample: &[f64; I] = helper.as_array().expect("Expected array right size");

        // let sample = self.preprocess(x);
        let predict = self.forward(&sample,Some(helper_ref));

        let mut correction = [0.0; I];
        
        for neuron in 0..O {
            let delta = errors[neuron] * self.derivative(predict[neuron]);

            // the correction for the weights
            let static_delta = delta * (1.0 - self.alpha);
            for input_no in 0..I {
                // Total correction accounting
                correction[input_no] += static_delta * self.weights[neuron][input_no];
                
                // Update the weights
                self.weights[neuron][input_no] += temperature * static_delta * sample[input_no];
            }
            self.bias[neuron] += temperature * delta;
        }

        // Process corrections through the alpha layer
        let alpha_errors = correction.map(|err| err * self.alpha);
        let alpha_corrections = self.alpha_layer.backward(helper_ref, alpha_errors, temperature);
        
        // Combine corrections
        for i in 0..I {
            correction[i] += alpha_corrections[i];
        }
        
        correction
    }
}