use bevy::prelude::*;
use neural_nets_toys::{layers, DefaultHelper, NeuralNetwork};

type NN = impl NeuralNetwork<11, 4>;

// Neural network brain for the agent
pub struct AgentBrain {
  pub network: NN, // 8 sensors + 3 internal state, 4 movement actions
}

impl AgentBrain {
  #[define_opaque(NN)]
  pub fn new() -> Self {
      let network = layers::attention_layer::Attention::<11, 8>::random(-1.0..1.0)
      .chain(layers::lnn_exp_layer::LNNLayer::<11, 4, 10>::random(
          |x| 1.0 / (1.0 + (-x).exp()), // sigmoid
          -1.0..1.0,
          0.7,
      ));
      Self { network }
  }

}

// Agent component
#[derive(Component)]
pub struct Agent {
  pub brain: AgentBrain,
  pub velocity: Vec2,
  pub sensors: [f64; 8],  // 8 directional sensors

  pub score: f64,
}

impl Agent {
  pub fn new() -> Self {
      Self {
          brain: AgentBrain::new(),
          score: 0.0,
          velocity: Vec2::ZERO,
          sensors: [0.0; 8],
      }
  }

  pub fn sense(&mut self, mut query_world: impl FnMut(Vec2, f32) -> f32) {
      // Cast rays in 8 directions to detect objects
      for (i, angle) in (0..8).map(|i| i as f32 * std::f32::consts::PI / 4.0).enumerate() {
          let direction = Vec2::new(angle.cos(), angle.sin());
          self.sensors[i] = query_world(direction, 5.0) as f64;
      }
  }

  pub fn encode(&self) -> [f64; 11] {
      let mut input = [0.0; 11];
      for i in 0..8 {
          input[i] = self.sensors[i];
      }
      input[8] = self.velocity.x as f64;
      input[9] = self.velocity.y as f64;
      input[10] = self.score;
      input
  }

  pub fn think(&mut self) -> [f32; 4] {
      // Combine sensors and position into input vector
      let input = self.encode();
      
      // Forward pass through neural network
      let output = self.brain.network.forward(&input, None::<&mut DefaultHelper>);

      [output[0] as f32, output[1] as f32, output[2] as f32, output[3] as f32]
  }
}