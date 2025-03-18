use bevy::prelude::*;
use neural_nets_toys::{layers, DefaultHelper, NeuralNetwork};

const SENSOR_NUMBER: usize = 8;

// N sensors x 2 params + fuel + rewards collected
pub const INPUT_PARAMS: usize = SENSOR_NUMBER * 2 + 2;

// 4 movement actions
pub const OUTPUT_PARAMS: usize = 4;

pub type NN = impl NeuralNetwork<INPUT_PARAMS, OUTPUT_PARAMS>;

// Neural network brain for the agent
pub struct AgentBrain {
  pub network: NN,
}

impl AgentBrain {
  #[define_opaque(NN)]
  pub fn new() -> Self {
    let network = layers::attention_layer::Attention::<INPUT_PARAMS, 2>::random(-1.0..1.0)
    .chain(
        layers::lnn_exp_layer::LNNLayer::<INPUT_PARAMS, 8, 4>::random(
            |x| 1.0 / (1.0 + (-x).exp()), // sigmoid
            -1.0..1.0,
            0.7,
        )
    ).chain(layers::attention_layer::Attention::<8, 2>::random(-1.0..1.0))
    .chain(
        layers::lnn_exp_layer::LNNLayer::<8, OUTPUT_PARAMS, 10>::random(
        |x| 1.0 / (1.0 + (-x).exp()), // sigmoid
        -1.0..1.0,
        0.7,
        )
    );
      Self { network }
  }

}

// Agent component
#[derive(Component)]
pub struct Agent {
  pub brain: AgentBrain, // Neural network brain
  pub fuel: f32, // Fuel for the agent
  pub rewards: usize, // Number of rewards collected
  pub sensors: [[f64;2]; 8],  // 8 directional sensors
}

impl Agent {
    pub fn reset(&mut self) {
        self.fuel = 100.0;
        self.rewards = 0;
        self.sensors = [[0.0;2]; 8];
    }

  pub fn new() -> Self {
      Self {
          brain: AgentBrain::new(),
          sensors: [[0.0;2]; 8],
          fuel: 100.0,
          rewards: 0,
      }
  }

  pub fn sense(&mut self, mut query_world: impl FnMut(Vec2) -> [f64;2]) {
      // Cast rays in SENSOR_NUMBER directions to detect objects
      for (i, angle) in (0..SENSOR_NUMBER).map(|i| i as f32 * (2.0 * std::f32::consts::PI / SENSOR_NUMBER as f32)).enumerate() {
          let direction = Vec2::new(angle.cos(), angle.sin());
          self.sensors[i] = query_world(direction);
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
      
      input
  }

  pub fn think(&mut self, input: &[f64; INPUT_PARAMS]) -> [f64; OUTPUT_PARAMS] {
      
      // Forward pass through neural network
      let output = self.brain.network.forward(input, None::<&mut DefaultHelper>);

      println!("I-O: {:?} {:?}", input, output);
      output
  }
}