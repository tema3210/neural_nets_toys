mod data;
mod model;
mod llm;
use crate::data::{DataSample, JsonlDataset, MyBatcher};
use burn::backend;
use burn::optim::AdamConfig;
use burn::data::dataloader::DataLoaderBuilder;
use burn::tensor::Tensor;


type BackendType = backend::wgpu::Wgpu;


pub fn main() {
    let device = Default::default();

    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
        &device,
        Default::default(),
    );

    // Initialize the model, optimizer, and input data
    let model = llm::LLM::<BackendType>::new(&device);

    let optimizer = AdamConfig::new().init();
    
    // Create dataset and batcher
    let dataset = JsonlDataset::<DataSample>::new("data.jsonl");
    let batcher = MyBatcher::new(100);

    // Create dataloader
    let dataloader = DataLoaderBuilder::new(batcher)
        .set_device(device.clone())
        .batch_size(32)
        .shuffle(3214382438)
        .num_workers(4)
        .build(dataset);

    // Sample training loop
    for batch in dataloader.iter() {
        let input = &batch.input;
        let [_batch_size, seq_length] = input.dims();

        let mask = Tensor::<BackendType, 2>::ones([seq_length, seq_length], &input.device())
            .tril(0)
            .bool();

        // Forward pass
        let result = reasoning_loop(
            &model,
            input.clone(), // Pass input tensor
            10,
            0.5,
            Some(mask),
        );
        // Compute loss, backward, optimizer step, etc. (pseudo-code)
        // let loss = compute_loss(result, label);
        // optimizer.backward_step(&model, &loss);
        println!("Batch result shape: {:?}", result.shape());
    }
}
