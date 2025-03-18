use crate::{utils::spawn_reward, EndRoundEvent, EnvironmentConfig, Event, Experience, Obstacle, Reward, TrainingState};
use rand::prelude::*;

use super::agent::Agent;
use bevy::prelude::*;

const PROB_OF_REWARD_DETECT: f64 = 0.33;

const AGENT_SIZE: f32 = 1.5;

pub struct RNG(rand::distributions::Bernoulli, rand::rngs::StdRng);

impl Default for RNG {
  fn default() -> Self {
      Self(
        rand::distributions::Bernoulli::new(PROB_OF_REWARD_DETECT).unwrap(),
        rand::rngs::StdRng::from_entropy(),
      )
  }
}

pub fn agent_sensing(
  mut agents: Query<(&mut Agent, &Transform)>,
  obstacles: Query<&Transform, With<Obstacle>>,
  rewards: Query<&Transform, With<Reward>>,
  cfg: Res<EnvironmentConfig>,
  mut rng: Local<RNG>,
) {
  let max_distance = cfg.sight_distance;
  for (mut agent, agent_transform) in agents.iter_mut() {
      let agent_pos = agent_transform.translation.truncate();
      
      agent.sense(|direction| {
          // Cast ray to find closest obstacle
          let mut obst_distance: f64 = 1.0;  // 1.0 means nothing detected (normalized)
          
          for obstacle_transform in obstacles.iter() {
              let obstacle_pos = obstacle_transform.translation.truncate();
              let to_obstacle = obstacle_pos - agent_pos;
              
              // Check if obstacle is in this direction
              let dot = direction.dot(to_obstacle.normalize());
              if dot > 0.7 {  // Within ~45 degrees of ray direction

                  let dist = to_obstacle.length() / max_distance; //also normalized
                  obst_distance = obst_distance.min(dist as f64);
              }
          }

          let mut reward_distance: f64 = 1.0;  // 1.0 means nothing detected (normalized)
          for reward in rewards.iter() {
              let reward_pos = reward.translation.truncate();
              let to_reward = reward_pos - agent_pos;
              
              // Check if reward is in this direction
              let dot = direction.dot(to_reward.normalize());
              if dot > 0.7 {  // Within ~45 degrees of ray direction
                  let RNG(ref dst, ref mut rng) = *rng;
                  if dst.sample(rng) {
                    let dist = to_reward.length() / max_distance; //also normalized
                    reward_distance = reward_distance.min(dist as f64);
                  }
              }
          }
          
          [obst_distance,reward_distance]
      });
  }
}

pub struct Decoded {
  pub movement: Vec2,
}

pub fn agent_thinking(
  mut agents: Query<(&mut Agent, &mut Transform)>,
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

      training.current_episode.push(Experience {
        timestamp: current_time,
        fuel: agent.fuel as f64,
        event: Event::Action {
            state: state.clone(),
            action: [action[0], action[1], action[2], action[3]],
        },
      });
      
      // Apply movement
      transform_mut.translation += decoded.movement.extend(0.0);
      agent.fuel -= cfg.fuel_cost_modifier;  // Fuel consumption

      if agent.fuel <= 0.0001 {
          training.current_episode.push(Experience {
              timestamp: current_time,
              fuel: agent.fuel as f64,
              event: Event::End {
                  reason: crate::EndReason::OutOfFuel,
                  rewards_collected: agent.rewards,
              },
          });
          ew.send(EndRoundEvent {
            reason: crate::EndReason::OutOfFuel,
          });
      }

  }
}

pub fn collision_detection(
  agents: Query<(&Agent, &Transform)>,
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
            event: Event::Collision,
          });
        }
      }
      
      // Check boundary collisions
      if agent_pos.x.abs() > config.width/2.0 || agent_pos.y.abs() > config.height/2.0 {
          // Penalty for hitting boundaries is death
          training.current_episode.push(Experience {
              timestamp: current_time,
              fuel: agent.fuel as f64,
              event: Event::End {
                  reason: crate::EndReason::OutOfBounds,
                  rewards_collected: agent.rewards,
              },
          });
          ew.send(EndRoundEvent {
            reason: crate::EndReason::OutOfBounds,
          });
      }

  }
}

pub fn reward_collection(
  mut commands: Commands,
  mut agents: Query<(&mut Agent, &Transform)>,
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
                event: Event::Reward,
              });
              
              commands.entity(reward_entity).despawn();
              
              // Spawn new reward
              spawn_reward(&mut commands, &config, &mut meshes, &mut materials);
          }
      }
  }
}

