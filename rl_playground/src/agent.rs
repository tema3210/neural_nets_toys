use bevy::prelude::*;
use burn::{
    backend::{wgpu::WgpuRuntime, Autodiff}, module::Module, nn::{attention::{MultiHeadAttention, MultiHeadAttentionConfig}, gru::{Gru, GruConfig}, loss::{MseLoss, Reduction}, Linear, LinearConfig, Lstm, LstmConfig, LstmState}, optim::{AdamConfig, GradientsParams, Optimizer}, tensor::{activation::*, backend::Backend, Tensor, TensorData}
};
use crate::{MyAutodiffBackend};

pub const AGENT_SIZE: f32 = 2.0;

const SENSOR_NUMBER: usize = 8;

// N sensors x 2 params + fuel + rewards collected + legs vector
pub const INPUT_PARAMS: usize = SENSOR_NUMBER * 2 + 2 + 2;

// 2 movement actions
pub const OUTPUT_PARAMS: usize = 4;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    inp: Linear<B>,
    prefix: Linear<B>,
    mem1: Lstm<B>,
    inner: Linear<B>,
    mem2: Lstm<B>,
    suffix: Linear<B>,
    out: Linear<B>,
    // activation: ReLU,

}

impl Model<MyAutodiffBackend> {
    pub fn new(device: &<Autodiff<burn_fusion::Fusion<burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>>> as Backend>::Device) -> Self {
        Self {
            inp: LinearConfig::new(INPUT_PARAMS, 32).init(device),
            prefix: LinearConfig::new(32, 16).init(device),
            mem1: LstmConfig::new(16, 64, true).init(device),
            inner: LinearConfig::new(64, 64).init(device),
            mem2: LstmConfig::new(64, 32, true).init(device),
            suffix: LinearConfig::new(32, 16).init(device),
            out: LinearConfig::new(16, OUTPUT_PARAMS).init(device),
        }
    }

    /// input: batch_size, seq_l, INPUT_PARAMS;
    /// 
    /// output: batch_size, seq_l, OUTPUT_PARAMS;
    pub fn forward(&mut self, input: Tensor<MyAutodiffBackend, 3>, brain: &mut AgentBrain) -> Tensor<MyAutodiffBackend, 3> {
        let x = self.inp.forward(input);
        let x = relu(x);
        let x = self.prefix.forward(x);
        let x = relu(x);
        // Forward pass through the LSTM layers
        let (x, st) = self.mem1.forward(x, brain.mem1.take());
        brain.mem1 = Some(st);

        let x = self.inner.forward(x);

        let (x, st) = self.mem2.forward(x, brain.mem2.take());
        brain.mem2 = Some(st);
        let x = self.suffix.forward(x);
        let x = relu(x);
        let x = self.out.forward(x);
        relu(x)
    }
}

unsafe impl<B: Backend> Sync for Model<B> where B: Sync {}

unsafe impl<B: Backend> Send for Model<B> where B: Send {}

// Neural network brain for the agent
pub struct AgentBrain {
    model: Option<Model<MyAutodiffBackend>>,
    device: <MyAutodiffBackend as Backend>::Device,

    mem1: Option<LstmState<MyAutodiffBackend, 2>>,
    mem2: Option<LstmState<MyAutodiffBackend, 2>>,
}

impl AgentBrain {
    pub fn new(device: &<MyAutodiffBackend as Backend>::Device) -> Self {
        let model = Model::new(device);
        // let optimizer = burn::optim::Adam::new(&model);
        Self { model: Some(model), device: device.clone(), mem1: None, mem2: None }
    }

    pub fn reset(&mut self) {
        self.model = Some(Model::new(&self.device));
        // self.optimizer = burn::optim::Adam::new(&self.model);
    }

    pub fn forward(&mut self, input: &[f64; INPUT_PARAMS]) -> [f64; OUTPUT_PARAMS] {
        let input_data = TensorData::new(
            input.to_vec(),
            [1, 1, INPUT_PARAMS],
        );
        let input_tensor = Tensor::<MyAutodiffBackend, 3>::from_data(input_data, &self.device)
            .reshape([1, 1, INPUT_PARAMS]);

        // Forward pass through the neural network
        let Some(mut model) = self.model.take() else {
            panic!("Impossible! F");
        };

        let output = model.forward(input_tensor, self);

        self.model = Some(model);

        let output_data = output.into_data();
        let result = output_data.iter().collect::<Vec<_>>().try_into().unwrap();
        result
    }

    pub fn train<O: Optimizer<Model<MyAutodiffBackend>, MyAutodiffBackend>>(
        &mut self,
        input: Tensor<MyAutodiffBackend, 3>,
        target: Tensor<MyAutodiffBackend, 3>,
        optimizer: &mut O
    ) {
        let Some(mut model) = self.model.take() else {
            panic!("Impossible! B")
        };
        let output = model.forward(input.clone(), self);
        let loss = MseLoss::new().forward(output, target, Reduction::Mean);
        // let grad = loss.forward_no_reduction(output, target);
        let grads = loss.backward();

        // Gradients linked to each parameter of the model.
        let grads = GradientsParams::from_grads(grads, &model);

        model = optimizer.step(0.001f64, model, grads);

        self.model = Some(model)
    }
}

// Agent component
#[derive(Component)]
pub struct Agent {
    pub name: String,
    pub sight_distance: f32,

    pub brain: AgentBrain,
    // Neural network brain
    pub legs: Vec2,
    // Movement direction
    pub fuel: f32,
    // Fuel for the agent
    pub rewards: usize,
    // Number of rewards collected
    pub sensors: [[f64; 2]; 8], // 8 directional sensors
}

impl Agent {
    pub fn reset(&mut self) {
        self.fuel = 100.0;
        self.rewards = 0;
        self.sensors = [[0.0; 2]; 8];
        // self.brain.reset();
        self.legs = Vec2::ZERO;
    }

    pub fn new(name: String, sight_distance: f32, device: &<MyAutodiffBackend as Backend>::Device) -> Self {
        Self {
            sight_distance,
            name,
            brain: AgentBrain::new(&device),
            sensors: [[0.0; 2]; 8],
            fuel: 100.0,
            rewards: 0,
            legs: Vec2::ZERO,
        }
    }

    pub fn sense(&mut self, mut query_world: impl FnMut(Vec2) -> [Option<f64>; 2]) {
        // Cast rays in SENSOR_NUMBER directions to detect objects
        for (i, angle) in (0..SENSOR_NUMBER)
            .map(|i| i as f32 * (2.0 * std::f32::consts::PI / SENSOR_NUMBER as f32))
            .enumerate()
        {
            let direction = Vec2::new(angle.cos(), angle.sin());
            let q = query_world(direction);
            self.sensors[i] = [
                q[0].map(|x| x / self.sight_distance as f64)
                    .unwrap_or(1.0),
                q[1].map(|x| x / self.sight_distance as f64)
                    .unwrap_or(1.0),
            ];
        }
    }

    pub fn encode(&self) -> [f64; INPUT_PARAMS] {
        let mut input = [0.0f64; INPUT_PARAMS];
        input[0] = self.fuel as f64;
        input[1] = self.rewards as f64;
        for s in 0..SENSOR_NUMBER {
            input[2 * (s + 1)] = self.sensors[s][0];
            input[2 * (s + 1) + 1] = self.sensors[s][1];
        }
        input[18] = self.legs.x as f64;
        input[19] = self.legs.y as f64;

        input
    }

    pub fn think(&mut self, input: &[f64; INPUT_PARAMS]) -> [f64; OUTPUT_PARAMS] {
        // Forward pass through neural network
        let output = self.brain.forward(input);

        println!("I: {:?}", input);
        println!("O: {:?}", output);
        output
    }

    /// Sleep to train the agent
    pub fn sleep(&mut self, data: &Vec<Vec<([f64; INPUT_PARAMS], [f64; OUTPUT_PARAMS])>>) {

        let optimizer = AdamConfig::new();
        let mut optimizer = optimizer.init();

        // let mut optimizer_state: Option<_> = None;

        let max_seq_len = data.iter().map(|x| x.len()).max().unwrap_or(0);
        // Prepare input and output tensors
        // input: batch_size, seq_l, INPUT_PARAMS;
        // output: batch_size, seq_l, OUTPUT_PARAMS;

        let mut input = vec![];
        let mut output = vec![];

        for (_, story) in data.iter().enumerate() {
            for (input_data, output_data) in story.iter() {
                input.extend(input_data.iter().copied());
                output.extend(output_data.iter().copied());
            }
            let input_pad = (max_seq_len - story.len() ) * INPUT_PARAMS;
            let output_pad = (max_seq_len - story.len() ) * OUTPUT_PARAMS;

            input.extend(vec![0.0; input_pad].iter().copied());
            output.extend(vec![0.0; output_pad].iter().copied());
        }

        let output_tensor = Tensor::<MyAutodiffBackend, 3>::from_data(
            TensorData::new(
                output,
                [data.len(), max_seq_len, OUTPUT_PARAMS],
            ),
            &self.brain.device,
        );

        let input_tensor = Tensor::<MyAutodiffBackend, 3>::from_data(
            TensorData::new(
                input,
                [data.len(), max_seq_len, INPUT_PARAMS],
            ),
            &self.brain.device,
        );

        self.brain.train(input_tensor, output_tensor, &mut optimizer);
        
    }
}