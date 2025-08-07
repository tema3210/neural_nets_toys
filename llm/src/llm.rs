use burn::{prelude::Backend, tensor::{activation::sigmoid, Int, Tensor, TensorPrimitive}};

use crate::{model::HRMModel};

pub struct LLM<'d, B: Backend> {
    model: HRMModel<'d, B>,
}

impl<'d, B: Backend> LLM<'d, B> {
    pub fn new(device: &'d B::Device) -> Self {
        let model = HRMModel::<B>::new(device);
        Self { model }
    }

    pub fn run(&mut self, input: Tensor<B, 2, Int>, max_steps: usize, halt_threshold: f32) -> Tensor<B, 2, Int> {
        self.reasoning_loop(input, max_steps, halt_threshold)
    }


    fn reasoning_loop(&mut self, input: Tensor<B, 2, Int>, max_steps: usize, halt_threshold: f32) -> Tensor<B, 2, Int> 
        where TensorPrimitive<B>: PartialOrd<f32>
    {
        let mut byte;
        let mut latent = self.model.initial_latent(input.clone()); // Initialize latent state
        for _ in 0..max_steps {
            let (new_latent, byte_logits, q_logits) = self.model.forward(input.clone(), latent);
            byte = byte_logits.argmax(2); // Get predicted byte indices

            // Q-head output interpreted as continue probability
            let continue_prob = sigmoid(q_logits);
            if continue_prob.mean().into_primitive() < halt_threshold {
                break; // halt reasoning early
            }

            latent = new_latent;
            // Optionally update input or context with predictions for autoregressive generation
        }
        byte
    }

}

