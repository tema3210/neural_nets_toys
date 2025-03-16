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

pub mod layers;

pub mod attention;

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
