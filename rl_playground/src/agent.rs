use std::f32::consts::PI;

use bevy::prelude::*;
use neural_nets_toys::{layers, DefaultHelper, NeuralNetwork};
use rand::Rng;

pub const AGENT_SIZE: f32 = 2.0;

const SENSOR_NUMBER: usize = 8;

// N sensors x 2 params + fuel + rewards collected + legs vector
pub const INPUT_PARAMS: usize = SENSOR_NUMBER * 2 + 2 + 2;

// 2 movement actions
pub const OUTPUT_PARAMS: usize = 4;

pub type NN = impl NeuralNetwork<INPUT_PARAMS, OUTPUT_PARAMS>;

// Neural network brain for the agent
pub struct AgentBrain {
  pub network: NN,
}

impl AgentBrain {
  #[define_opaque(NN)]
  pub fn new() -> Self {
    let network = layers::attention_layer::
    Attention::<INPUT_PARAMS, 3>::random(
        -1.0..1.0
    )
    .chain(
        layers::lnn_exp_layer::
        LNNLayer::<INPUT_PARAMS, 32, 8>::random(
            |x| 1.0 / (1.0 + (-x).exp()), // sigmoid
            -1.0..1.0,
            0.7,
        )
    )
    .chain(layers::attention_layer::
        Attention::<32, 3>::random(
            -1.0..1.0
        )
    )
    .chain(
        layers::lnn_exp_layer::
        LNNLayer::<32, OUTPUT_PARAMS, 4>::random(
        |x| 1.0 / (1.0 + (-x).exp()), // sigmoid
        -1.0..1.0,
        0.7,
        )
    )
    ;
      Self { network }
  }

}

// Agent component
#[derive(Component)]
pub struct Agent {
  pub name: String,
  pub sight_distance: f32,


  pub brain: AgentBrain, // Neural network brain
  pub legs: Vec2, // Movement direction
  pub fuel: f32, // Fuel for the agent
  pub rewards: usize, // Number of rewards collected
  pub sensors: [[f64;2]; 8],  // 8 directional sensors
}

impl Agent {
    pub fn reset(&mut self) {
        self.fuel = 100.0;
        self.rewards = 0;
        self.sensors = [[0.0;2]; 8];
        self.brain.network.reset();
        self.legs = Vec2::ZERO;
    }

  pub fn new(name: String, sight_distance: f32) -> Self {
      Self {
          sight_distance,
          name,
          brain: AgentBrain::new(),
          sensors: [[0.0;2]; 8],
          fuel: 100.0,
          rewards: 0,
          legs: Vec2::ZERO,
      }
  }

  pub fn sense(&mut self, mut query_world: impl FnMut(Vec2) -> [Option<f64>;2]) {
      // Cast rays in SENSOR_NUMBER directions to detect objects
      for (i, angle) in (0..SENSOR_NUMBER).map(|i| i as f32 * (2.0 * std::f32::consts::PI / SENSOR_NUMBER as f32)).enumerate() {
          let direction = Vec2::new(angle.cos(), angle.sin());
          let q = query_world(direction);
          self.sensors[i] = [
            q[0].map(|x| x / self.sight_distance as f64).unwrap_or(1.0),
            q[1].map(|x| x / self.sight_distance as f64).unwrap_or(1.0), 
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
      let mut output = self.brain.network.forward(input, None::<&mut DefaultHelper>);

      // compute the direction from output
      let mut direction = Vec2::ZERO;
      direction.x = output[0] as f32; //(output[0] - output[1]) as f32;
      direction.y = output[1] as f32; //(output[2] - output[3]) as f32;
    
      // now add a random angle step to the direction, but keep it in range of +- 5 deg
      let mut rng = rand::thread_rng();
      let deg = 5.0 / PI * 180.0;
      let angle: f32 = rng.gen_range(-deg..deg); // -15 to 15 degrees
      let angle_vec = Vec2::new(angle.cos(), angle.sin());
      let direction = angle_vec.rotate(direction).normalize_or_zero() - direction;

      // Update the output
      output[1] -= direction.x as f64;
      output[0] += direction.x as f64;
      output[3] -= direction.y as f64;
      output[2] += direction.y as f64;


      println!("I: {:?}", input);
      println!("O: {:?}", output);
      output
  }
}