#![feature(generic_const_exprs)]

struct Concat<L,R,const LL: usize, const RL: usize> {
    left: L,
    right: R,
}

impl<L,R, const I: usize, const LL: usize, const RL: usize> NeuralNetwork<I,{LL + RL}> for Concat<L,R,LL,RL> 
    where L: NeuralNetwork<I,LL>, R: NeuralNetwork<I,RL>
{
    fn forward(&mut self, x: &[f64; I]) -> [f64; LL + RL] {
        let mut result = [0.0; {LL + RL}];
        let left = self.left.forward(x);
        let right = self.right.forward(x);
        for i in 0..LL {
            result[i] = left[i];
        }
        for i in 0..RL {
            result[i + LL] = right[i];
        }
        result
    }
    
    fn backward(&mut self, x: &[f64; I], errors: [f64; LL + RL], temperature: f64) -> [f64; I] {
        let mut left_errors = [0.0; LL];
        let mut right_errors = [0.0; RL];
        for i in 0..LL {
            left_errors[i] = errors[i];
        }
        for i in 0..RL {
            right_errors[i] = errors[i + LL];
        }
        let left = self.left.backward(x, left_errors, temperature);
        let right = self.right.backward(x, right_errors, temperature);

        let mut result = [0.0; I];
        for i in 0..I {
            result[i] = left[i] + right[i];
        }
        result
    }
}

struct Chain<F, S, const M: usize> {
    first: F,
    second: S,
}

impl<F,S, const I: usize,const M: usize, const O: usize> NeuralNetwork<I,O> for Chain<F,S,M> where F: NeuralNetwork<I,M>, S: NeuralNetwork<M,O> {
    fn forward(&mut self, x: &[f64; I]) -> [f64; O] {
        self.second.forward(&self.first.forward(x))
    }
    fn backward(&mut self, x: &[f64; I], errors: [f64; O], temperature: f64) -> [f64; I] {
        let errors = self.second.backward(&self.first.forward(x), errors, temperature);
        self.first.backward(x, errors, temperature)
    }
}

pub trait NeuralNetwork<const INP: usize, const OUT: usize> 
    where [(); INP]:, [(); OUT]:
{
    fn forward(&mut self, x: &[f64; INP]) -> [f64; OUT];
    fn backward(&mut self, x: &[f64; INP], errors: [f64; OUT], temperature: f64) -> [f64; INP];
    /// this is for stuff like transformers, norms, etc
    /// it is called by forward pass, default returns argument
    fn preprocess(&mut self, x: &[f64; INP]) -> [f64; INP] {
        *x
    }
    fn chain<const SZ: usize>(self, other: impl NeuralNetwork<OUT, SZ>) -> impl NeuralNetwork<INP, SZ> where Self: Sized {
        Chain { first: self, second: other }
    }
    fn concat<const RSZ: usize>(self,other: impl NeuralNetwork<INP,RSZ>) -> impl NeuralNetwork<INP, {OUT + RSZ}> 
        where Self: Sized, [();OUT + RSZ]:
    {
        Concat { left: self, right: other }
    }
}

pub mod attention;

pub mod fc_layer {
    use super::*;

    pub struct Layer<const INP: usize, const WIDTH: usize> {
        weights: [[f64; INP]; WIDTH],
        bias: [f64; WIDTH],
        activate: fn(f64) -> f64,
        activate_derivative: Option<Box<dyn Fn(f64) -> f64>>,
    }    

    impl<const I: usize, const W: usize> Layer<I, W> {
        pub fn derivative(&self, x: f64) -> f64 {
            if let Some(derivative) = &self.activate_derivative {
                return derivative(x);
            }
            let step = 0.001;
            ((self.activate)(x + step) - (self.activate)(x - step)) / (2.0 * step)
        }
    
        pub fn with_derivative(mut self, derivative: impl Fn(f64) -> f64 + 'static) -> Self {
            self.activate_derivative = Some(Box::new(derivative));
            self
        }
    
        pub fn random(activate: fn(f64) -> f64, value_range: std::ops::Range<f64>) -> Layer<I, W> {
            use rand::Rng;
            let mut rng = rand::rng();
    
            Layer {
                weights: [[rng.random_range(value_range.clone()); I]; W],
                bias: [rng.random_range(value_range); W],
                activate,
                activate_derivative: None,
            }
        }
    }

    impl<const I: usize, const W: usize> NeuralNetwork<I, W> for Layer<I, W> {
        fn forward(&mut self, x: &[f64; I]) -> [f64; W] {
            let mut sum = self.bias;
    
            self.weights.iter().enumerate().for_each(|(ind, coefs)| {
                for i in 0..I {
                    sum[ind] += coefs[i] * x[i];
                }
                sum[ind] = (self.activate)(sum[ind]);
            });
    
            sum
        }
    
        fn backward(&mut self, x: &[f64; I], error: [f64; W], temperature: f64) -> [f64; I] {
            let y = self.forward(x);
            let mut correction = [0.0; I];
    
            for neuron in 0..W {
                let delta = error[neuron] * self.derivative(y[neuron]);
                for input_no in 0..I {
                    correction[input_no] += delta * self.weights[neuron][input_no];
                    self.weights[neuron][input_no] += temperature * delta * x[input_no];
                }
                self.bias[neuron] += temperature * delta;
            }
    
            correction
        }
    }
    
}

/// Module for lnn layer with weighted recap and self attention
pub mod lnn_exp {
    use super::*;
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
}


pub struct TrainParams<const O: usize> {
    pub epochs: usize,
    pub temperature: f64,
    pub cutoff: f64,
    pub fn_loss: fn(&[f64; O], &[f64; O]) -> [f64;O],
}

pub fn train<const I: usize, const O: usize>(
    model: &mut impl NeuralNetwork<I, O>,
    data: &[([f64; I], [f64; O])],
    params: TrainParams<O>,
) -> usize {
    for cnt in 0..params.epochs {
        let mut err_acc = 0.0;
        for (sample, target) in data {
            let predict = model.forward(&sample);
            let error = (params.fn_loss)(target, &predict);
            for i in 0..O {
                err_acc += error[i] * error[i];
            }
            model.backward(&sample, error, params.temperature);
        }
        if err_acc < params.cutoff {
            return cnt;
        }
    }
    params.epochs
}
