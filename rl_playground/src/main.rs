#![feature(type_alias_impl_trait)]
use std::ops::Range;

use agent::Agent;
use bevy::prelude::*;
use bevy::math::primitives::{Capsule3d, Cuboid};
use neural_nets_toys::{train, TrainParams};
use rand::prelude::*;
use utils::spawn_reward;

mod agent;

mod utils;

mod systems;

// Bevy plugin for RL
pub struct RLPlugin;

impl Plugin for RLPlugin {
    fn build(&self, app: &mut App) {
        app
            .insert_resource(EnvironmentConfig {
                width: 300.0,
                height: 300.0,
                obstacle_count: 10..14,
                reward_count: 3..6,
            })
            .insert_resource(ArenaState::default())
            .add_systems(Startup, setup_environment)
            .add_systems(Update, (
                systems::agent_sensing,
                systems::agent_thinking,
                systems::collision_detection,
                systems::reward_collection,
                training_system,
            ));
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(RLPlugin)
        .run();
}

// Environment configuration
#[derive(Resource)]
pub struct EnvironmentConfig {
    pub width: f32,
    pub height: f32,
    pub obstacle_count: Range<usize>,
    pub reward_count: Range<usize>,
}

fn setup_environment(
    mut commands: Commands,
    config: Res<EnvironmentConfig>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut rng = thread_rng();
    let plane_z_level = -0.5;

    let agent_start = Transform::from_xyz(0.0, 0.0, plane_z_level);

    let camera_distance = 6.0;

    // Spawn agent
    commands.spawn((
        Agent::new(),
        Mesh3d(meshes.add(Capsule3d::default().mesh())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.0, 0.5, 0.5),
            ..default()
        })),
        agent_start,
    )).with_child(( // Camera bound to agent
        Camera3d::default(),
        Transform::from_xyz(0.0, camera_distance / 2.0_f32.sqrt(), camera_distance / 2.0_f32.sqrt()).looking_at(agent_start.translation, Vec3::Z),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Plane3d::new(Vec3::Z, Vec2::new(config.width, config.height)).mesh())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::linear_rgba(0.5, 0.5, 0.5, 0.9),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, plane_z_level)
    ));

    let obst_c = rng.gen_range(config.obstacle_count.clone());

    // Spawn obstacles
    for _ in 0..obst_c {
        let x = rng.gen_range(-config.width/2.0..config.width/2.0);
        let y = rng.gen_range(-config.height/2.0..config.height/2.0);
        
        commands.spawn((
            Obstacle,
            Mesh3d(meshes.add(Cuboid::new(1.0,1.0,1.0).mesh())),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.8, 0.2, 0.2),
                ..default()
            })),
            Transform::from_xyz(x, y, plane_z_level),
        ));
    }

    let rew_c = rng.gen_range(config.reward_count.clone());

    // Spawn rewards
    for _ in 0..rew_c {
        spawn_reward(&mut commands, &config, &mut meshes, &mut materials);
    }
}

// Environment components
#[derive(Component)]
pub struct Obstacle;

#[derive(Component)]
pub struct Reward {
    pub value: f64,
}


// Environment configuration
#[derive(Resource, Default)]
pub struct ArenaState {
    pub round: usize,
    pub states: Vec<[f64; 11]>,
    pub actions: Vec<[f64; 4]>,
    pub rewards: Vec<f64>,
}

#[derive(Default)]
struct TrainingState {
    timer: f32,
}

fn training_system(
    mut arena: ResMut<ArenaState>,
    mut agents: Query<&mut Agent>,
    time: Res<Time>,
    mut state: Local<TrainingState>,
) {
    // Update timer
    state.timer += time.delta_secs();
    
    // Train every second
    if state.timer >= 1.0 {
        state.timer = 0.0;
        
        for mut agent in agents.iter_mut() {
            println!("Agent score: {} at {}", agent.score, arena.round );

            let agent_state = agent.encode();
            arena.states.push(agent_state);



            // Prepare training data
            let mut training_data = Vec::new();
            
            // For now, we'll use a simple reward-based learning approach
            // In a more complex system, you'd implement proper RL algorithms
            
            // Simple training rule: if score positive, reinforce current behavior
            if agent.score > 0.0 {
                // Create some training examples based on recent experience
                // This is highly simplified - real RL would use state-action-reward tuples
                
                // We'll implement this in the next section
            }
            
        
            let train_params = TrainParams {
                epochs: 1,
                temperature: agent.score.abs() - 0.5,
                cutoff: 0.00001,
                fn_loss: |x, y| (x - y).powi(2),
            };

            // Train the agent
            train(&mut agent.brain.network, &training_data, train_params);
            agent.score = 0.0; // Reset score for next training period
        }

        arena.round += 1;
    }
}

