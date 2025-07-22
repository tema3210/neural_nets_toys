#![feature(type_alias_impl_trait)]
#![recursion_limit = "256"]
use std::collections::VecDeque;
use std::ops::Range;

use agent::{Agent, INPUT_PARAMS, OUTPUT_PARAMS};
use bevy::asset::RenderAssetUsages;
use bevy::ecs::system::SystemId;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
// use neural_nets_toys::{train, TrainParams};

use burn::{
    backend::{Autodiff, Wgpu},
    tensor::backend::Backend,
};


mod agent;

mod utils;

mod systems;

pub type MyBackend = Wgpu;

pub type MyAutodiffBackend = Autodiff<MyBackend>;

#[derive(Resource)]
struct Device {
    device: <MyAutodiffBackend as Backend>::Device,
}

#[derive(Resource)]
struct UtilSystems {
    new_round: SystemId,
}

impl FromWorld for UtilSystems {
    fn from_world(world: &mut World) -> Self {
        let id = world.register_system(utils::new_round);

        Self {
            new_round: id,
        }
    }
}

// Bevy plugin for RL
pub struct RLPlugin;

mod app_ui {
    use super::*;

    // Add near your other component definitions
    #[derive(Component)]
    pub struct TrainingInfoText;

    pub fn setup_ui(mut commands: Commands) {
        // UI camera
        // commands.spawn(Camera2dBundle::default());
        
        // Training info text
        // Root node
        commands
            .spawn(Node {
                width: Val::Vw(50.0),
                height: Val::Vh(25.0),
                position_type: PositionType::Absolute,
                ..Default::default()
            })
            .with_children(|parent| {
                // Training info text container
                parent.spawn((
                    Text::new("Info"),
                    TrainingInfoText,
                    Label
                ));
            });
    }


    pub fn update_training_ui(
        training_state: Res<TrainingState>,
        agents: Query<&Agent>,
        mut text_query: Query<&mut Text, With<TrainingInfoText>>,
    ) {
        if let Ok(mut text) = text_query.single_mut() {
            let mut info = String::new();
            
            // Basic training info
            info.push_str(&format!("Round: {}\n", training_state.round));
            info.push_str(&format!("Episodes: {}\n", training_state.episode_histories.len()));
            info.push_str(&format!("Current episode events: {}\n", training_state.current_episode.len()));
            
            // Agent info
            if let Ok(agent) = agents.single() {
                info.push_str(&format!("Agent fuel: {:.1}\n", agent.fuel));
                info.push_str(&format!("Rewards collected: {}\n", agent.rewards));
            }
            
            text.0 = info;
        }
    }
}

fn setup_environment(
    tr_sys: Res<UtilSystems>,
    _config: Res<EnvironmentConfig>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    device: Res<Device>,
) {
    let plane_z_level = -0.5;

    let agent_start = Transform::from_xyz(0.0, 0.0, plane_z_level);

    let camera_distance = 80.0;

    // Spawn agent
    commands.spawn((
        Agent::new("John".to_string(),_config.sight_distance, &device.device),
        Mesh3d(meshes.add(Cuboid::new(
            agent::AGENT_SIZE,
            agent::AGENT_SIZE,
            agent::AGENT_SIZE
        ).mesh())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.0, 0.5, 0.5),
            ..default()
        })),
        agent_start,
    )).with_child(( // Camera bound to agent
        Camera3d::default(),
        Transform::from_xyz(0.0, camera_distance / 2.0_f32.sqrt(), camera_distance / 2.0_f32.sqrt()).looking_at(agent_start.translation, Vec3::Z),
    ));

    // Spawn plane
    let texture = {
        let texture_size = 256u32;
        let mut texture_data = vec![0; (texture_size * texture_size * 4) as usize];
        
        // Generate pavement pattern
        for y in 0..texture_size {
            for x in 0..texture_size {
                let idx = ((y * texture_size + x) * 4) as usize;
                
                // Create tile grid pattern
                let tile_size = 32;
                let tile_x = x / tile_size;
                let tile_y = y / tile_size;
                
                // Grid lines
                let is_grid_line = x % tile_size <= 1 || y % tile_size <= 1;
                
                // Alternate tile colors
                let is_dark_tile = (tile_x + tile_y) % 2 == 0;
                
                // Add some noise for texture
                let noise = ((x * y) % 13) as u8;
                
                // Base color values
                let (r, g, b) = if is_grid_line {
                    (50, 50, 50) // Dark grid lines
                } else if is_dark_tile {
                    (140 + (noise % 20), 140 + (noise % 10), 140 + (noise % 15)) // Dark gray tile
                } else {
                    (180 + (noise % 15), 180 + (noise % 20), 180 + (noise % 10)) // Light gray tile
                };
                
                // Set RGBA values
                texture_data[idx] = r;
                texture_data[idx + 1] = g;
                texture_data[idx + 2] = b;
                texture_data[idx + 3] = 255; // Full opacity
            }
        }
        
        // Create the texture asset
        let pavement_texture = Image::new(
            Extent3d {
                width: texture_size,
                height: texture_size,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            texture_data,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::MAIN_WORLD,
        );

        pavement_texture
    };

    // Add the texture to assets
    let texture_handle = images.add(texture);

    // Create the material with the texture
    let pavement_material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(texture_handle),
        perceptual_roughness: 0.9,
        metallic: 0.01,
        unlit: true,
        ..default()
    });

    let plane_transform = Transform::from_xyz(0.0, 0.0, plane_z_level);

    commands.spawn((
        Mesh3d(meshes.add(Plane3d::new(Vec3::Z, Vec2::new(_config.width, _config.height)).mesh())),
        MeshMaterial3d(pavement_material),
        plane_transform
    )).with_child((
        DirectionalLight {
            illuminance: 10000.0,
            // range: 100.0,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 100.0)
            .looking_at(plane_transform.translation, Vec3::Z),
    ));

    commands.run_system(tr_sys.new_round);

}

#[derive(Event)]
pub struct EndRoundEvent {
    pub reason: EndReason,
}

impl Plugin for RLPlugin {
    fn build(&self, app: &mut App) {
        app
            .insert_resource(EnvironmentConfig {
                width: 100.0,
                height: 100.0,
                obstacle_count: 10..14,
                reward_count: 30..45,
                reward_fuel_bonus: 8.0,
                retries_per_round: 3,
                fuel_cost_modifier: 2.0,
                sight_distance: 16.6,
                max_episodes: 100,
            })
            .insert_resource(TrainingState::default())
            .init_resource::<UtilSystems>()
            .insert_resource(Device {
                device: <MyAutodiffBackend as Backend>::Device::default(),
            })

            // .add_event::<ResetSimulationEvent>()
            .add_event::<EndRoundEvent>() // Register the reset event
            .add_systems(Startup, (setup_environment,app_ui::setup_ui))
            .add_systems(Update, (
                systems::agent_sensing,
                systems::agent_thinking,
                systems::collision_detection,
                systems::reward_collection,
                training_system,
                app_ui::update_training_ui,
                // reset_simulation, // Add our reset system
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
    pub retries_per_round: usize,

    pub fuel_cost_modifier: f32,
    pub reward_fuel_bonus: f32,
    pub sight_distance: f32,
    pub max_episodes: usize,
}

// Environment components
#[derive(Component)]
pub struct Obstacle;

#[derive(Component)]
pub struct Reward {
    pub value: f64,
}

#[derive(Clone)]
pub enum RlEvent {
    // Agent took an action (state, action chosen)
    Action {
        state: [f64; INPUT_PARAMS],
        action: [f64; OUTPUT_PARAMS]
    },
    // Agent received reward
    Reward,
    // Agent collided with obstacle
    Collision,
    // Episode ended
    End { 
        reason: EndReason,
        rewards_collected: usize
    },
}

#[derive(Clone, PartialEq, Debug)]
pub enum EndReason {
    OutOfFuel,
    OutOfBounds,
    ManualReset,
}

#[derive(Clone)]
pub struct Experience {
    timestamp: f64,
    fuel: f64,
    event: RlEvent,
}



// Environment configuration
#[derive(Resource, Default)]
pub struct TrainingState {
    pub round: usize,
    pub episode_histories: VecDeque<Vec<Experience>>,
    pub current_episode: Vec<Experience>,
}

fn training_system(
    mut commands: Commands,
    tr_sys: Res<UtilSystems>,

    mut training_state: ResMut<TrainingState>,
    env_config: Res<EnvironmentConfig>,
    mut agents: Query<(&mut Agent,&mut Transform)>,

    mut retry_count: Local<usize>,
    // mut reset_writer: EventWriter<ResetSimulationEvent>,
    mut round_end_reader: EventReader<EndRoundEvent>,
) {

    println!("Training system {}", training_state.round + 1);

    match round_end_reader.read().next() {
        Some(ev) => {
            println!("Round ended: {:?}", ev.reason);
        },
        None => {
            return;
        }
    }

    *retry_count += 1;
    // Reset agent position and state
    for (mut agent, mut transform) in agents.iter_mut() {
        // Reset position to start
        *transform = Transform::from_xyz(0.0, 0.0, -0.5);
        
        agent.reset();
    }

    if *retry_count > env_config.retries_per_round {
        *retry_count = 0;
        println!("Resetting simulation");
        commands.run_system(tr_sys.new_round);
        return;
    }
    
    //this one is assumed to be singular
    for (mut agent,_) in agents.iter_mut() {

        // Extract all experiences from current episode and history
        let get_exprt_iter = || {
            let TrainingState { 
                // episode_histories, 
                current_episode, .. 
            } = &*training_state;
            current_episode.iter()
                // .chain(episode_histories.iter().flat_map(|x| x.iter()))
        };

        let agent_state = agent.encode();

        // Reward interpolation function
        let reward_at_timestamp = {
            let exprs: Vec<(f64,f64)> = get_exprt_iter()
            .filter_map(|x| match x.event {
                RlEvent::Reward => Some((x.timestamp, 1.0)),
                RlEvent::Collision => Some((x.timestamp, -1.0)),
                _ => None,
            }).collect();

            // Interpolate reward at timestamp
            fn compare(a: &&(f64,f64), b: &&(f64,f64)) -> std::cmp::Ordering {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            }
            
            move |at: f64| {
                let max_greater = exprs
                    .iter()
                    .filter(|(x,_)| *x > at)
                    .min_by(compare);
                let min_lower = exprs
                    .iter()
                    .filter(|(x,_)| *x <= at)
                    .max_by(compare);
                match (min_lower, max_greater) {
                    (Some((l_at,l_rew)), Some((r_at,r_rew))) => {
                        
                        let l = l_rew / ((at - l_at).abs() + 0.01);
                        let r = r_rew / ((r_at - at).abs() + 0.01);
                        // Geometrical mean
                        (l * r).sqrt().tanh()
                    },
                    (Some((_, v)), None) => v.tanh(),
                    (None, Some((_, v))) => v.tanh(),
                    _ => 0.0,
                }
            }
        };

        let training_data = crate::utils::form_training_data(
            get_exprt_iter(),
            &reward_at_timestamp,
            agent_state
        );
        
        // Turbo sleep - this is the training
        agent.sleep(&training_data[..])
    }
    // Clear current episode
    let episode = std::mem::take(&mut training_state.current_episode);
    training_state.episode_histories.push_back(episode);

    training_state.round += 1;
}

