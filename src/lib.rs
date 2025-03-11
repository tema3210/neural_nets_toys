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
    fn chain<const SZ: usize>(self, other: impl NeuralNetwork<O, SZ>) -> impl NeuralNetwork<I, SZ> where Self: Sized {
        Chain { first: self, second: other }
    }
}

pub trait NeuralNetwork<const INP: usize, const OUT: usize> {
    fn forward(&mut self, x: &[f64; INP]) -> [f64; OUT];
    fn backward(&mut self, x: &[f64; INP], errors: [f64; OUT], temperature: f64) -> [f64; INP];
    fn chain<const SZ: usize>(self, other: impl NeuralNetwork<OUT, SZ>) -> impl NeuralNetwork<INP, SZ> where Self: Sized {
        Chain { first: self, second: other }
    }
}

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

pub mod lnn_layer {
    use super::*;
    use std::ops::Range;
    use rand::Rng;
    
    pub struct LNNLayer<const INP: usize, const OUT: usize> {
        weights: [[f64; INP]; OUT],
        bias: [f64; OUT],
        hidden_state: [f64; OUT],
        alpha: f64, // Governs temporal memory effect
        activation: fn(f64) -> f64,
        activation_derivative: Option<Box<dyn Fn(f64) -> f64>>,
    }
    
    impl<const I: usize, const O: usize> LNNLayer<I, O> {
        pub fn new(activation: fn(f64) -> f64, value_range: Range<f64>, alpha: f64) -> Self {
            let mut rng = rand::rng();
            
            LNNLayer {
                weights: [[rng.random_range(value_range.clone()); I]; O],
                bias: [rng.random_range(value_range); O],
                hidden_state: [0.0; O],
                alpha,
                activation,
                activation_derivative: None,
            }
        }
        
        pub fn with_derivative(mut self, derivative: impl Fn(f64) -> f64 + 'static) -> Self {
            self.activation_derivative = Some(Box::new(derivative));
            self
        }

        pub fn derivative(&self, x: f64) -> f64 {
            if let Some(derivative) = &self.activation_derivative {
                return derivative(x);
            }
            let step = 0.001;
            ((self.activation)(x + step) - (self.activation)(x - step)) / (2.0 * step)
        }
    }
    
    impl<const I: usize, const O: usize> NeuralNetwork<I, O> for LNNLayer<I, O> {
        fn forward(&mut self, x: &[f64; I]) -> [f64; O] {
            let mut sum = self.bias;
            for neuron in 0..O {
                for inp in 0..I {
                    sum[neuron] += self.weights[neuron][inp] * x[inp];
                }
                sum[neuron] = self.hidden_state[neuron] * self.alpha + (1.0 - self.alpha) * sum[neuron];
                sum[neuron] = (self.activation)(sum[neuron]);
            }
            self.hidden_state = sum;
            sum
        }
        
        fn backward(&mut self, x: &[f64; I], errors: [f64; O], temperature: f64) -> [f64; I] {
            let predict = self.forward(x);
            let mut correction = [0.0; I];
            
            for neuron in 0..O {
                let delta = errors[neuron] * self.derivative(predict[neuron]);
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


pub struct TrainParams {
    pub epochs: usize,
    pub temperature: f64,
    pub cutoff: f64,
}

pub fn train<const I: usize, const O: usize>(
    model: &mut impl NeuralNetwork<I, O>,
    data: &[([f64; I], [f64; O])],
    params: TrainParams,
) -> usize {
    for cnt in 0..params.epochs {
        let mut err_acc = 0.0;
        for (sample, target) in data {
            let predict = model.forward(&sample);
            let mut error = [0.0; O];
            for i in 0..O {
                let err =  2.0 * (target[i] - predict[i]);
                err_acc += err * err;
                error[i] = err;
            }
            model.backward(&sample, error, params.temperature);
        }
        if err_acc < params.cutoff {
            return cnt;
        }
    }
    params.epochs
}
