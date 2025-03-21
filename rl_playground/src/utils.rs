use crate::agent::{AGENT_SIZE, OUTPUT_PARAMS};
use crate::systems::Decoded;
use crate::{EnvironmentConfig, Obstacle, Reward};

use bevy::prelude::*;
use bevy::math::primitives::Sphere;
use rand::prelude::*;

pub fn spawn_reward(
  commands: &mut Commands,
  config: &EnvironmentConfig,
  meshes: &mut Assets<Mesh>,
  materials: &mut Assets<StandardMaterial>,
) {
  let mut rng = thread_rng();
  let x = rng.gen_range(-config.width/2.0..config.width/2.0);
  let y = rng.gen_range(-config.height/2.0..config.height/2.0);
  
  commands.spawn((
      Reward { value: 1.0 },
      Mesh3d(meshes.add(Sphere::new(0.5).mesh())),
      MeshMaterial3d(materials.add(StandardMaterial {
          base_color: Color::srgb(0.9, 0.9, 0.1),
          ..default()
      })),
      Transform::from_xyz(x, y, -0.5),
  ));
}

pub fn spawn_obstacle(
  commands: &mut Commands,
  config: &EnvironmentConfig,
  meshes: &mut Assets<Mesh>,
  materials: &mut Assets<StandardMaterial>,
) {
  let mut rng = thread_rng();
  let x = rng.gen_range(-config.width/2.0..config.width/2.0);
  let y = rng.gen_range(-config.height/2.0..config.height/2.0);
  
  commands.spawn((
      Obstacle,
      Mesh3d(meshes.add(Sphere::new(AGENT_SIZE).mesh())),
      MeshMaterial3d(materials.add(StandardMaterial {
          base_color: Color::srgb(0.9, 0.1, 0.1),
          ..default()
      })),
      Transform::from_xyz(x, y, -0.5),
  ));
}

pub fn decode_action(action: &[f64; OUTPUT_PARAMS]) -> Decoded {
  let movement = Vec2::new(
    (action[0] - action[1]) as f32,  // Left vs Right
    (action[2] - action[3]) as f32,  // Up vs Down
  );

  // let movement = Vec2::new(action[0] as f32, action[1] as f32);

  Decoded {
    movement
  }
}

pub fn new_round(
  mut commands: Commands,
  config: Res<EnvironmentConfig>,
  mut materials: ResMut<Assets<StandardMaterial>>,
  mut meshes: ResMut<Assets<Mesh>>,
  obstacles: Query<Entity, With<Obstacle>>,
  rewards: Query<Entity, With<Reward>>,
) {
  let mut rng = thread_rng();

  // Clear the scene
  for entity in obstacles.iter().chain(rewards.iter()) {
    commands.entity(entity).despawn_recursive();
  };

  // Spawn rewards
  let rew_c = rng.gen_range(config.reward_count.clone());
  for _ in 0..rew_c {
      spawn_reward(&mut commands, &config, &mut meshes, &mut materials);
  }

  //spawn obstacles
  let obs_c = rng.gen_range(config.obstacle_count.clone());
  for _ in 0..obs_c {
      spawn_obstacle(&mut commands, &config, &mut meshes, &mut materials);
  }
}