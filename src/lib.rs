#![feature(slice_as_array)]

// struct Concat<L,R,const LL: usize, const RL: usize> {
//     left: L,
//     right: R,
// }

// impl<L,R, const I: usize, const LL: usize, const RL: usize> NeuralNetwork<I,{LL + RL}> for Concat<L,R,LL,RL> 
//     where L: NeuralNetwork<I,LL>, R: NeuralNetwork<I,RL>
// {
//     fn forward(&mut self, x: &[f64; I]) -> [f64; LL + RL] {
//         let mut result = [0.0; {LL + RL}];
//         let left = self.left.forward(x);
//         let right = self.right.forward(x);
//         for i in 0..LL {
//             result[i] = left[i];
//         }
//         for i in 0..RL {
//             result[i + LL] = right[i];
//         }
//         result
//     }
    
//     fn backward(&mut self, x: &[f64; I], errors: [f64; LL + RL], temperature: f64) -> [f64; I] {
//         let mut left_errors = [0.0; LL];
//         let mut right_errors = [0.0; RL];
//         for i in 0..LL {
//             left_errors[i] = errors[i];
//         }
//         for i in 0..RL {
//             right_errors[i] = errors[i + LL];
//         }
//         let left = self.left.backward(x, left_errors, temperature);
//         let right = self.right.backward(x, right_errors, temperature);

//         let mut result = [0.0; I];
//         for i in 0..I {
//             result[i] = left[i] + right[i];
//         }
//         result
//     }
// }

struct Chain<F, S, const M: usize> {
    first: F,
    second: S,
}

impl<F,S, const I: usize,const M: usize, const O: usize> NeuralNetwork<I,O> for Chain<F,S,M> where F: NeuralNetwork<I,M>, S: NeuralNetwork<M,O> {
    fn forward(&mut self, x: &[f64; I], helper: Option<&mut impl Helper>) -> [f64; O] {
        if let Some(helper) = helper {
            let fst_x = self.first.forward(x, Some(helper));
            return self.second.forward(&fst_x, Some(helper))
        }
        let fst_x = self.first.forward(x, None::<&mut DefaultHelper>);
        self.second.forward(&fst_x, None::<&mut DefaultHelper>)
    }
    fn backward(&mut self, helper: &mut impl Helper, errors: [f64; O], temperature: f64) -> [f64; I] {
        let errors = self.second.backward(helper, errors, temperature);
        self.first.backward(helper, errors, temperature)
    }
}

pub trait NeuralNetwork<const INP: usize, const OUT: usize> 
    where [(); INP]:, [(); OUT]:
{   
    fn forward(&mut self, x: &[f64; INP], helper: Option<&mut impl Helper>) -> [f64; OUT];

    fn backward(&mut self, helper: &mut impl Helper, errors: [f64; OUT], temperature: f64) -> [f64; INP];
    /// this is for stuff like transformers, norms, etc
    /// it is called by forward pass, default returns argument
    fn preprocess(&mut self, x: &[f64; INP]) -> [f64; INP] {
        *x
    }
    fn chain<const SZ: usize>(self, other: impl NeuralNetwork<OUT, SZ>) -> impl NeuralNetwork<INP, SZ> where Self: Sized {
        Chain { first: self, second: other }
    }
    // fn concat<const RSZ: usize>(self,other: impl NeuralNetwork<INP,RSZ>) -> impl NeuralNetwork<INP, {OUT + RSZ}> 
    //     where Self: Sized, [();OUT + RSZ]:
    // {
    //     Concat { left: self, right: other }
    // }
}

pub struct TrainParams<const O: usize> {
    pub epochs: usize,
    pub temperature: f64,
    pub cutoff: f64,
    pub fn_loss: fn(&[f64; O], &[f64; O]) -> [f64;O],
}

pub trait Helper {
    fn push<const I: usize>(&mut self, sample: &[f64; I]);
    fn peek(&self) -> Option<&[f64]>;
    fn pop(&mut self) -> Option<Box<[f64]>>;
    fn clear(&mut self);
}

#[derive(Debug)]
pub struct DefaultHelper {
    samples: Vec<Box<[f64]>>,
}

impl DefaultHelper {
    pub fn with_capacity(capacity: usize) -> Self {
        Self { samples: Vec::with_capacity(capacity) }
    }
}

impl Helper for DefaultHelper {
    fn push<const I: usize>(&mut self, sample: &[f64; I]) {
        let sample = Box::new(*sample);
        self.samples.push(sample);
    }

    fn peek(&self) -> Option<&[f64]> {
        self.samples.last().map(|x| &**x)
    }

    fn pop(&mut self) -> Option<Box<[f64]>> {
        self.samples.pop()
    }

    fn clear(&mut self) {
        self.samples.clear();
    }
}

impl<const I: usize, const O: usize> NNExt<I, O> for DefaultHelper {

    fn train(model: &mut impl NeuralNetwork<I, O>, data: &[([f64; I], [f64; O])], params: TrainParams<O>) -> usize {
        let mut helper = Self::with_capacity(data.len());

        for cnt in 0..params.epochs {
            let mut err_acc = 0.0;
            helper.clear();
            
            for (sample, target) in data {
                let prediction = model.forward(sample, Some(&mut helper));
                let error = (params.fn_loss)(target, &prediction);
                
                for i in 0..O {
                    err_acc += error[i] * error[i];
                }
                
                model.backward(&mut helper, error, params.temperature);

                // Helper should be empty after backprop if all activations were popped
                debug_assert!(helper.peek().is_none(), "Helper should be empty after backprop");
            }
            
            if err_acc < params.cutoff {
                return cnt;
            }
        }
        
        params.epochs
    }
}

pub trait NNExt<const I: usize, const O: usize> {

    fn train(model: &mut impl NeuralNetwork<I, O>, data: &[([f64; I], [f64; O])], params: TrainParams<O>) -> usize;
}

pub mod layers;

pub fn train<const I: usize, const O: usize>(
    model: &mut impl NeuralNetwork<I, O>,
    data: &[([f64; I], [f64; O])], 
    params: TrainParams<O>,
) -> usize {
    DefaultHelper::train(model, data, params)
}
