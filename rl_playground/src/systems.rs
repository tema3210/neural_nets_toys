use crate::{agent::AGENT_SIZE, utils::spawn_reward, EndRoundEvent, EnvironmentConfig, RlEvent, Experience, Obstacle, Reward, TrainingState};
// use rand::prelude::*;

use super::agent::Agent;
use bevy::prelude::*;
use burn::tensor::backend::AutodiffBackend;

// const PROB_OF_REWARD_DETECT: f64 = 0.33;

// pub struct RNG(rand::distributions::Bernoulli, rand::rngs::StdRng);

// impl Default for RNG {
//   fn default() -> Self {
//       Self(
//         rand::distributions::Bernoulli::new(PROB_OF_REWARD_DETECT).unwrap(),
//         rand::rngs::StdRng::from_entropy(),
//       )
//   }
// }

pub fn agent_sensing<B: AutodiffBackend>(
  mut agents: Query<(&mut Agent<B>, &Transform)>,
  obstacles: Query<&Transform, With<Obstacle>>,
  rewards: Query<&Transform, With<Reward>>,
  // mut rng: Local<RNG>,
) {
  for (mut agent, agent_transform) in agents.iter_mut() {
      let agent_pos = agent_transform.translation.truncate();
      
      agent.sense(|direction| {

          // Cast ray to find closest obstacle
          let mut obst_distance: f64 = 0.0;  // 1.0 means nothing detected (normalized)
          let mut obst_found = 0;
          for obstacle_transform in obstacles.iter() {
              let obstacle_pos = obstacle_transform.translation.truncate();
              let to_obstacle = obstacle_pos - agent_pos;
              
              // Check if obstacle is in this direction
              let dot = direction.dot(to_obstacle.normalize());
              if dot > 0.7 {  // Within ~45 degrees of ray direction
                obst_found += 1;
                let dist = to_obstacle.length();
                obst_distance += obst_distance.min(dist as f64);
              }
          }
          

          // Cast ray to find closest reward
          let mut reward_distance: f64 = 1.0;  // 1.0 means nothing detected (normalized)
          let mut rew_found = 0;
          for reward in rewards.iter() {
              let reward_pos = reward.translation.truncate();
              let to_reward = reward_pos - agent_pos;
              
              // Check if reward is in this direction
              let dot = direction.dot(to_reward.normalize());
              if dot > 0.7 {  // Within ~45 degrees of ray direction
                  rew_found += 1;
                  let dist = to_reward.length(); //also normalized
                  reward_distance = reward_distance.min(dist as f64);
                  // let RNG(ref dst, ref mut rng) = *rng;
                  // if dst.sample(rng) {
                    
                  // }
              }
          }

          [
            if obst_found > 0 {Some(obst_distance)} else {None},
            if rew_found > 0 {Some(reward_distance)} else {None},
          ]
      });
  }
}

pub struct Decoded {
  pub movement: Vec2,
}

pub fn agent_thinking<B: AutodiffBackend>(
  mut agents: Query<(&mut Agent<B>, &mut Transform)>,
  mut training: ResMut<TrainingState>,
  mut ew: EventWriter<EndRoundEvent>,
  cfg: Res<EnvironmentConfig>,
  time: Res<Time>,
) {
  let current_time = time.elapsed_secs_f64();
  for (mut agent, mut transform_mut) in agents.iter_mut() {
      let state = agent.encode();
      let action = agent.think(&state);

      let decoded = crate::utils::decode_action(&action);

      // Log action
      training.current_episode.push(Experience {
        timestamp: current_time,
        fuel: agent.fuel as f64,
        event: RlEvent::Action {
            state: state.clone(),
            action
        },
      });

      agent.legs += decoded.movement;
      
      // Apply movement
      transform_mut.translation += agent.legs.extend(0.0);

      println!("Agent: {:?} at: {:?}", agent.name, transform_mut.translation);

      agent.fuel -= cfg.fuel_cost_modifier * decoded.movement.length();  // Fuel consumption

      if agent.fuel <= 0.0001 {
          training.current_episode.push(Experience {
              timestamp: current_time,
              fuel: agent.fuel as f64,
              event: RlEvent::End {
                  reason: crate::EndReason::OutOfFuel,
                  rewards_collected: agent.rewards,
              },
          });
          ew.write(EndRoundEvent {
            reason: crate::EndReason::OutOfFuel,
          });
      }

  }
}

pub fn collision_detection<B: AutodiffBackend>(
  agents: Query<(&Agent<B>, &Transform)>,
  obstacles: Query<&Transform, With<Obstacle>>,
  config: Res<EnvironmentConfig>,
  mut training: ResMut<TrainingState>,
  time: Res<Time>,
  mut ew: EventWriter<EndRoundEvent>
) {
  let current_time = time.elapsed_secs_f64();
  for (agent, agent_transform) in agents.iter() {
      let agent_pos = agent_transform.translation.truncate();

      // Check obstacle collisions
      for obstacle_transform in obstacles.iter() {
        let obstacle_pos = obstacle_transform.translation.truncate();
        if (agent_pos - obstacle_pos).length() < AGENT_SIZE {
          training.current_episode.push(Experience {
            timestamp: current_time,
            fuel: agent.fuel as f64,
            event: RlEvent::Collision,
          });
        }
      }
      
      // Check boundary collisions
      if agent_pos.x.abs() > config.width/2.0 || agent_pos.y.abs() > config.height/2.0 {
          // Penalty for hitting boundaries is death
          training.current_episode.push(Experience {
              timestamp: current_time,
              fuel: agent.fuel as f64,
              event: RlEvent::End {
                  reason: crate::EndReason::OutOfBounds,
                  rewards_collected: agent.rewards,
              },
          });
          ew.write(EndRoundEvent {
            reason: crate::EndReason::OutOfBounds,
          });
      }

  }
}

pub fn reward_collection<B: AutodiffBackend>(
  mut commands: Commands,
  mut agents: Query<(&mut Agent<B>, &Transform)>,
  rewards: Query<(Entity, &Transform), With<Reward>>,
  config: Res<EnvironmentConfig>, 
  mut training: ResMut<TrainingState>,
  time: Res<Time>,

  mut meshes: ResMut<Assets<Mesh>>,
  mut materials: ResMut<Assets<StandardMaterial>>,
) {
  let current_time = time.elapsed_secs_f64();
  for (mut agent, agent_transform) in agents.iter_mut() {
      let agent_pos = agent_transform.translation.truncate();
      
      // Check reward collection
      for (reward_entity, reward_transform) in rewards.iter() {
          let reward_pos = reward_transform.translation.truncate();
          
          if (agent_pos - reward_pos).length() < AGENT_SIZE {
              // Collect reward
              agent.rewards += 1;
              agent.fuel += config.reward_fuel_bonus;
              agent.fuel = agent.fuel.min(100.0);  // Cap fuel at 100

              training.current_episode.push(Experience {
                timestamp: current_time,
                fuel: agent.fuel as f64,
                event: RlEvent::Reward,
              });
              
              commands.entity(reward_entity).despawn();
              
              // Spawn new reward
              spawn_reward(&mut commands, &config, &mut meshes, &mut materials);
          }
      }
  }
}

