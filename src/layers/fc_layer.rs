use crate::*;

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