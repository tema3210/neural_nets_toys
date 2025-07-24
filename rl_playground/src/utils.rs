use bevy::{
    math::primitives::Sphere,
    prelude::*,
};
use rand::prelude::*;

use crate::{
    agent::{AGENT_SIZE, INPUT_PARAMS, OUTPUT_PARAMS}, systems::Decoded, EndReason, EnvironmentConfig, Experience, Obstacle, Reward, RlEvent
};

pub fn spawn_reward(
    commands: &mut Commands,
    config: &EnvironmentConfig,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let mut rng = thread_rng();
    let x = rng.gen_range(-config.width / 2.0..config.width / 2.0);
    let y = rng.gen_range(-config.height / 2.0..config.height / 2.0);

    commands.spawn((
        Reward { value: 1.0 },
        Mesh3d (
            meshes.add(
              Sphere::new(0.5).mesh().build(),
            )
        ),
        MeshMaterial3d (
            materials.add(
              StandardMaterial {
                  base_color: Color::srgb(0.9, 0.9, 0.1),
                  ..default()
              },
            )
        ),
        Transform::from_xyz(x, y, -0.5)
    ));
}

pub fn spawn_obstacle(
    commands: &mut Commands,
    config: &EnvironmentConfig,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let mut rng = thread_rng();
    let x = rng.gen_range(-config.width / 2.0..config.width / 2.0);
    let y = rng.gen_range(-config.height / 2.0..config.height / 2.0);

    commands.spawn((
        Obstacle,
        Mesh3d (
            meshes.add(
              Sphere::new(AGENT_SIZE).mesh().build(),
            )
        ),
        MeshMaterial3d (
            materials.add(
              StandardMaterial {
                  base_color: Color::srgb(0.1, 0.1, 0.9),
                  ..default()
              },
            )
        ),
        Transform::from_xyz(x, y, -0.5),
    ));
}

pub fn decode_action(action: &[f64; OUTPUT_PARAMS]) -> Decoded {
    let movement = Vec2::new(
        (action[0] - action[1]) as f32, // Left vs Right
        (action[2] - action[3]) as f32, // Up vs Down
    );

    Decoded { movement }
}

//TODO: make it to return a tensor
pub fn form_training_data<'i>(
    data: impl Iterator<Item = &'i Experience>,
    reward_at_timestamp: &impl Fn(f64) -> f64,
    agent_state: [f64; INPUT_PARAMS],
) -> Vec<([f64; INPUT_PARAMS], [f64; OUTPUT_PARAMS])> {
    data.filter_map(|x| {
        // Calculate reward
        let base_reward = reward_at_timestamp(x.timestamp) - 0.5;

        let fuel_eff = x.fuel / 100.0; // fuel expended, %

        let (state, action) = match &x.event {
            RlEvent::Action { state, action } => (state, action),
            RlEvent::End {
                reason,
                rewards_collected,
            } => {
                let action = [0.0; OUTPUT_PARAMS]; // at the end we don't move
                let state = agent_state; // state at the end is the same as the last state
                let reward = match reason {
                    EndReason::OutOfBounds => -1.0,
                    EndReason::OutOfFuel => *rewards_collected as f64 * 0.5,
                    EndReason::ManualReset => 0.0,
                };
                return Some((state, action, reward, x.timestamp));
            }
            _ => return None,
        };

        let combined_reward = if base_reward > 0.0 {
            base_reward * (1.0 + fuel_eff) // Up to 2x reward when fuel is 100%
        } else {
            base_reward // Keep negative feedback unchanged
        };

        Some((*state, *action, combined_reward, x.timestamp))
    })
    .fold(
        vec![], // training data
        |mut data, (state, action, reward, _at)| {
            // 0, 1 -> left, right; 2, 3 -> up, down
            let mut new_action = [0.0; OUTPUT_PARAMS];
            // Scale action by reward
            for i in 0..OUTPUT_PARAMS {
                new_action[i] = action[i] as f64 * reward.abs();
            }

            // inverse target movement if reward is negative
            if reward < 0.005 {
                new_action.swap(0, 1);
                new_action.swap(2, 3);
            }

            data.push((state, new_action));
            data
        },
    )
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
        commands.entity(entity).despawn();
    }

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