use crate::{utils::spawn_reward, EnvironmentConfig, Obstacle, Reward};

use super::agent::Agent;
use bevy::prelude::*;

pub fn agent_sensing(
  mut agents: Query<(&mut Agent, &Transform)>,
  obstacles: Query<&Transform, With<Obstacle>>,
) {
  for (mut agent, agent_transform) in agents.iter_mut() {
      let agent_pos = agent_transform.translation.truncate();
      
      agent.sense(|direction, max_distance| {
          // Cast ray to find closest obstacle
          let mut distance: f32 = 1.0;  // 1.0 means nothing detected (normalized)
          
          for obstacle_transform in obstacles.iter() {
              let obstacle_pos = obstacle_transform.translation.truncate();
              let to_obstacle = obstacle_pos - agent_pos;
              
              // Check if obstacle is in this direction
              let dot = direction.dot(to_obstacle.normalize());
              if dot > 0.7 {  // Within ~45 degrees of ray direction
                  let dist = to_obstacle.length() / max_distance;
                  distance = distance.min(dist);
              }
          }
          
          distance
      });
  }
}

pub fn agent_thinking(
  mut agents: Query<(&mut Agent, &mut Transform)>,
  time: Res<Time>,
) {
  for (mut agent, mut transform_mut) in agents.iter_mut() {
      let movement = agent.think();

      let movement = Vec2::new(
          (movement[0] - movement[1]) as f32,  // Left vs Right
          (movement[2] - movement[3]) as f32,  // Up vs Down
      ).normalize_or_zero() * 2.0; // Speed factor
      
      // Update velocity
      agent.velocity = agent.velocity * 0.8 + movement * 0.2;
      
      // Apply movement
      transform_mut.translation += Vec3::new(
          agent.velocity.x * time.delta_secs() * 5.0,
          agent.velocity.y * time.delta_secs() * 5.0,
          0.0,
      );
  }
}

pub fn collision_detection(
  mut agents: Query<(&mut Agent, &Transform)>,
  obstacles: Query<&Transform, With<Obstacle>>,
  config: Res<EnvironmentConfig>,
) {
  for (mut agent, agent_transform) in agents.iter_mut() {
      let agent_pos = agent_transform.translation.truncate();
      
      // Check boundary collisions
      if agent_pos.x.abs() > config.width/2.0 || agent_pos.y.abs() > config.height/2.0 {
          agent.score -= 0.1;  // Penalty for hitting boundaries
      }
      
      // Check obstacle collisions
      for obstacle_transform in obstacles.iter() {
          let obstacle_pos = obstacle_transform.translation.truncate();
          if (agent_pos - obstacle_pos).length() < 1.5 {
              agent.score -= 0.5;  // Bigger penalty for hitting obstacles
          }
      }
  }
}

pub fn reward_collection(
  mut commands: Commands,
  mut agents: Query<(&mut Agent, &Transform)>,
  rewards: Query<(Entity, &Transform, &Reward)>,
  config: Res<EnvironmentConfig>, 
  mut meshes: ResMut<Assets<Mesh>>,
  mut materials: ResMut<Assets<StandardMaterial>>,
) {
  for (mut agent, agent_transform) in agents.iter_mut() {
      let agent_pos = agent_transform.translation.truncate();
      
      // Check reward collection
      for (reward_entity, reward_transform, reward) in rewards.iter() {
          let reward_pos = reward_transform.translation.truncate();
          
          if (agent_pos - reward_pos).length() < 1.5 {
              // Collect reward
              agent.score += reward.value;
              commands.entity(reward_entity).despawn();
              
              // Spawn new reward
              spawn_reward(&mut commands, &config, &mut meshes, &mut materials);
          }
      }
  }
}

