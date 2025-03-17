use crate::{EnvironmentConfig, Reward};

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
      Transform::from_xyz(x, y, 0.5),
  ));
}