use burn::backend::wgpu::WgpuDevice;
use burn::backend::{self, Autodiff};
use burn::optim::{Adam, AdamConfig};
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::Int;
use burn::tensor::{Tensor, backend::Backend};
use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::nn::transformer::{TransformerDecoder, TransformerDecoderConfig};

#[derive(Module, Debug)]
struct Expert<B: Backend> {
    block: TransformerDecoder<B>,
    memory: Tensor<B, 3>,
}

impl<B: Backend> Expert<B> {
    pub fn new(device: &<B as Backend>::Device) -> Self {
        let block = TransformerDecoderConfig::new(512, 8, 2, 4).init(device);
        let memory = Tensor::<B, 3>::empty([1, 1, 512], device); // Initialize memory as empty
        Self { block, memory }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let data = burn::nn::transformer::TransformerDecoderInput::new(input, self.memory.clone());
        self.block.forward(data)
    }
}

struct HRMModel<'d,B: Backend> {
    byte_embedding: Embedding<B>,           // vocab_size=256, emb_dim=512
    pos_embedding: Embedding<B>,            // seq_len, emb_dim=512
    experts: Vec<Expert<B>>,                 // 4 experts
    moe_gating: Linear<B>,                   // maps latent → 4 logits (per token or pooled)
    q_head: Linear<B>,                       // scalar halt/continue logits
    output_head: Linear<B>,
    device: &'d <B as Backend>::Device,         // maps emb_dim → 256 logits for byte prediction
}

impl<'d,B: Backend> HRMModel<'d,B> {

    pub fn new(device: &'d <B as Backend>::Device) -> Self {
        let byte_embedding = EmbeddingConfig::new(256, 512).init(device);
        let pos_embedding = EmbeddingConfig::new(512, 512).init(device);
        let experts = (0..4).map(|_| Expert::new(device)).collect();
        let moe_gating = LinearConfig::new(512, 4).init(device);
        let q_head = LinearConfig::new(512, 1).init(device);
        let output_head = LinearConfig::new(512, 256).init(device);

        Self {
            byte_embedding,
            pos_embedding,
            experts,
            moe_gating,
            q_head,
            output_head,
            device,
        }
    }

    pub fn forward(&self, input_bytes: Tensor<B, 2, Int>) -> (Tensor<B, 3>, Tensor<B, 2>, Tensor<B, 2>) {
        // input_bytes shape: [batch, seq_len], dtype=u8 but as Tensor<B, 2>
        let batch = input_bytes.shape().dims[0];
        let seq_len = input_bytes.shape().dims[1];

        // Embed bytes and positions
        let byte_emb = self.byte_embedding.forward(input_bytes.clone());        // [batch, seq_len, emb_dim]
        let pos_ids = Tensor::<B, 2, Int>::zeros([batch, seq_len], self.device);
        let pos_emb = self.pos_embedding.forward(pos_ids);

        // Shared workspace latent
        let mut latent = byte_emb + pos_emb;  // [batch, seq_len, emb_dim]

        // MoE routing logits (could be pooled or per-token, here pooled for simplicity)
        let pooled = latent.mean_dim(1);  // [batch, emb_dim]
        let gating_logits = self.moe_gating.forward(pooled);  // [batch, 4]
        let gating_probs = softmax(gating_logits, 1);  // [batch, 4]

        // Run experts weighted by gating probs and sum outputs
        let mut expert_outputs = Vec::new();
        for (i, expert) in self.experts.iter().enumerate() {
            let expert_out = expert.forward(latent.clone()); // [batch, seq_len, emb_dim]
            expert_outputs.push(expert_out * gating_probs.slice::<2,_>([..,0..i]));
            //.unsqueeze(-1).unsqueeze(-1));
        }
        latent = expert_outputs.iter().sum(); // weighted sum of experts

        // Q-head for halt/continue decision (use pooled latent)
        let q_logits = self.q_head.forward(latent.mean_dim(1));  // [batch, 1]
        // You can apply sigmoid or softmax in training/inference loop

        // Output logits for byte prediction
        let byte_logits = self.output_head.forward(latent); // [batch, seq_len, 256]

        (latent, byte_logits, q_logits)
    }
}


fn reasoning_loop<B: Backend>(model: &HRMModel<B>, input: Tensor<B, 2, Int>, max_steps: usize, halt_threshold: f32) -> Tensor<B, 2> {
    // let input = input.unsqueeze::<2>(); // [1, seq_len, 1] for batch processing

    let mut latent = model.byte_embedding.forward(input.clone()) + model.pos_embedding.forward(input.clone());
    for step in 0..max_steps {
        let (new_latent, byte_logits, q_logits) = model.forward(input.clone());

        // Q-head output interpreted as continue probability
        let continue_prob = sigmoid(q_logits);
        if continue_prob.mean() < halt_threshold {
            break; // halt reasoning early
        }

        latent = new_latent;
        // Optionally update input or context with predictions for autoregressive generation
    }
    latent
}

type BackendType = backend::wgpu::Wgpu;


pub fn main() {
    let device = Default::default();

    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Vulkan>(
        &device,
        Default::default(),
    );

    // Initialize the model, optimizer, and input data
    let model = HRMModel::<BackendType>::new(&device);

    let optimizer = AdamConfig::new().init();

    let mut input_data = std::io::BufReader::new(std::fs::File::open("data.jsonl").unwrap());
    
    let data = jsonl::read(&mut input_data).unwrap();

    let input_data = Tensor::<BackendType, 2>::from_data(vec![vec![0u8; 128]; 32], &device);
    let result = reasoning_loop(&model, input_data, 10, 0.5);

    println!("Reasoning result shape: {:?}", result.shape());
}
