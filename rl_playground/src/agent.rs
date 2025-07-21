use bevy::prelude::*;
use burn::{
    backend::{Autodiff, Wgpu},
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, TensorData, Tensor, activation::relu},
};

pub const AGENT_SIZE: f32 = 2.0;

const SENSOR_NUMBER: usize = 8;

// N sensors x 2 params + fuel + rewards collected + legs vector
pub const INPUT_PARAMS: usize = SENSOR_NUMBER * 2 + 2 + 2;

// 2 movement actions
pub const OUTPUT_PARAMS: usize = 4;

type MyBackend = Wgpu;
type MyAutodiffBackend = Autodiff<MyBackend>;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    // activation: ReLU,
}

unsafe impl<B: Backend> Sync for Model<B>  where B: Sync {}

unsafe impl<B: Backend> Send for Model<B>  where B: Send {}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        let linear1 = LinearConfig::new(INPUT_PARAMS, 32).init(device);
        let linear2 = LinearConfig::new(32, OUTPUT_PARAMS).init(device);
        Self {
            linear1,
            linear2,
            // activation: ReLU::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = relu(x);
        self.linear2.forward(x)
    }
}

// Neural network brain for the agent
pub struct AgentBrain {
    model: Model<MyAutodiffBackend>,
    device: <MyAutodiffBackend as Backend>::Device,
}

impl AgentBrain {
    pub fn new(device: &<MyAutodiffBackend as Backend>::Device) -> Self {
        let model = Model::new(device);
        Self { model, device: device.clone() }
    }

    pub fn reset(&mut self) {
        self.model = Model::new(&self.device);
    }

    pub fn forward(&self, input: &[f64; INPUT_PARAMS]) -> [f64; OUTPUT_PARAMS] {
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
        self.brain.reset();
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
}