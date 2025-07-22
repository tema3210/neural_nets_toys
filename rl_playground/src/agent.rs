use bevy::prelude::*;
use burn::{
    backend::{wgpu::WgpuRuntime, Autodiff}, module::Module, nn::{loss::MseLoss, Linear, LinearConfig}, optim::{Adam, AdamConfig, SimpleOptimizer}, tensor::{activation::*, backend::Backend, Tensor, TensorData}
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
    // inner: Lstm<B> ,
    out: Linear<B>,
    // activation: ReLU,
}

unsafe impl<B: Backend> Sync for Model<B> where B: Sync {}

unsafe impl<B: Backend> Send for Model<B> where B: Send {}

impl Model<MyAutodiffBackend> {
    pub fn new(device: &<Autodiff<burn_fusion::Fusion<burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>>> as Backend>::Device) -> Self {
        let linear1 = LinearConfig::new(INPUT_PARAMS, 32).init(device);
        // let lstm = LstmConfig::new(32, 32, true).init(device);
        let linear2 = LinearConfig::new(32, OUTPUT_PARAMS).init(device);
        Self {
            inp: linear1,
            // inner: lstm,
            out: linear2,
        }
    }

    pub fn forward(&self, input: Tensor<MyAutodiffBackend, 2>) -> Tensor<MyAutodiffBackend, 2> {
        let x = self.inp.forward(input);
        let x = relu(x);
        // let (x, st) = self.inner.forward(x, brain.memory);
        // brain.memory = Some(st);
        let x = self.out.forward(x);
        relu(x)
    }

    ///TODO: finish / review
    pub fn train<O: SimpleOptimizer<MyAutodiffBackend>>(
        &mut self,
        input: Tensor<MyAutodiffBackend, 2>,
        target: Tensor<MyAutodiffBackend, 2>,
        optimizer: &mut O,
        opt_state: &mut Option<O::State<2>>
    ) {
        let output = self.forward(input.clone());
        let loss = MseLoss::new();
        let grad = loss.forward_no_reduction(output, target);

        let (model, state) = optimizer.step(0.001f64, input, grad, None);
        self.model = model;
        *opt_state = Some(state);
    }
}

// Neural network brain for the agent
pub struct AgentBrain {
    pub model: Model<MyAutodiffBackend>,
    // optimizer: burn::optim::Adam,
    device: <MyAutodiffBackend as Backend>::Device,

    // memory: Option<LstmState<MyAutodiffBackend, 2>>,
}

impl AgentBrain {
    pub fn new(device: &<MyAutodiffBackend as Backend>::Device) -> Self {
        let model = Model::new(device);
        // let optimizer = burn::optim::Adam::new(&model);
        Self { model, device: device.clone() }
    }

    pub fn reset(&mut self) {
        self.model = Model::new(&self.device);
        // self.optimizer = burn::optim::Adam::new(&self.model);
    }

    pub fn forward(&mut self, input: &[f64; INPUT_PARAMS]) -> [f64; OUTPUT_PARAMS] {
        let input_data = TensorData::new(
            input.to_vec(),
            [1, INPUT_PARAMS],
        );
        let input_tensor = Tensor::<MyAutodiffBackend, 1>::from_data(input_data, &self.device)
            .reshape([1, INPUT_PARAMS]);

        // Forward pass through the neural network
        let output = self.model.forward(input_tensor);
        let output_data = output.into_data();
        let result = output_data.iter().collect::<Vec<_>>().try_into().unwrap();
        result
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
    pub fn sleep(&mut self, data: &[([f64; INPUT_PARAMS], [f64; OUTPUT_PARAMS])]) {

        let optimizer = Adam::for_model(&self.brain.model);
        let mut optimizer = optimizer.init(&self.brain.device);

        let mut optimizer_state: Option<_> = None;

        let input_tensor = Tensor::<MyAutodiffBackend, 2>::from_data(
            TensorData::new(
                data.iter().map(|(input, _)| input.iter()).flatten().copied().collect(),
                [data.len(), INPUT_PARAMS],
            ),
            &self.brain.device,
        );

        let output_tensor = Tensor::<MyAutodiffBackend, 2>::from_data(
            TensorData::new(
                data.iter().map(|(_, output)| output.iter()).flatten().copied().collect(),
                [data.len(), OUTPUT_PARAMS],
            ),
            &self.brain.device,
        );
        
        for (input, output) in data {
            self.brain.model.train(input_tensor, output_tensor, &mut optimizer, &mut optimizer_state);
        }
        
    }
}