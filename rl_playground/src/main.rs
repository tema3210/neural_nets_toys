#![feature(type_alias_impl_trait)]
use std::collections::VecDeque;
use std::ops::Range;

use agent::{Agent, INPUT_PARAMS};
use bevy::ecs::system::SystemId;
use bevy::prelude::*;
use bevy::math::primitives::Capsule3d;
use neural_nets_toys::{train, TrainParams};

mod agent;

mod utils;

mod systems;

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
        if let Ok(mut text) = text_query.get_single_mut() {
            let mut info = String::new();
            
            // Basic training info
            info.push_str(&format!("Round: {}\n", training_state.round));
            info.push_str(&format!("Episodes: {}\n", training_state.episode_histories.len()));
            info.push_str(&format!("Current episode events: {}\n", training_state.current_episode.len()));
            
            // Agent info
            if let Ok(agent) = agents.get_single() {
                info.push_str(&format!("Agent fuel: {:.1}\n", agent.fuel));
                info.push_str(&format!("Rewards collected: {}\n", agent.rewards));
            }
            
            text.0 = info;
        }
    }
}

fn setup_environment(
    tr_sys: Res<UtilSystems>,
    config: Res<EnvironmentConfig>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
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

    // Spawn plane
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::new(Vec3::Z, Vec2::new(config.width, config.height)).mesh())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::linear_rgba(0.5, 0.5, 0.5, 0.9),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, plane_z_level)
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
                width: 300.0,
                height: 300.0,
                obstacle_count: 10..14,
                reward_count: 3..6,
                reward_fuel_bonus: 8.0,
                sleep_interval: 30,
                fuel_cost_modifier: 1.0,
                sight_distance: 5.0,
                max_episodes: 100,
            })
            .insert_resource(TrainingState::default())
            .init_resource::<UtilSystems>()

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

    /// retries per round
    pub sleep_interval: usize,

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
pub enum Event {
    // Agent took an action (state, action chosen)
    Action {
        state: [f64; INPUT_PARAMS],
        action: [f64; 4]
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

#[derive(Clone, PartialEq)]
pub enum EndReason {
    OutOfFuel,
    OutOfBounds,
    ManualReset,
}

#[derive(Clone)]
pub struct Experience {
    timestamp: f64,
    fuel: f64,
    event: Event,
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

    println!("Training system {}", training_state.round);

    if round_end_reader.read().next().is_none() {
        // if the round has not ended skip training
        // println!("Round has not ended");
        return;
    }
    *retry_count += 1;
    // Reset agent position and state
    for (mut agent, mut transform) in agents.iter_mut() {
        // Reset position to start
        *transform = Transform::from_xyz(0.0, 0.0, -0.5);
        
        agent.reset();
    }

    if *retry_count > env_config.sleep_interval {
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
                episode_histories, 
                current_episode, .. 
            } = &*training_state;
            current_episode.iter()
                .chain(episode_histories.iter().flat_map(|x| x.iter()))   
        };

        let agent_state = agent.encode();

        // Reward interpolation function
        let reward_at_timestamp = {
            let exprs: Vec<(f64,f64)> = get_exprt_iter()
            .filter_map(|x| match x.event {
                Event::Reward => Some((x.timestamp, 1.0)),
                Event::Collision => Some((x.timestamp, -1.0)),
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
                        
                        let l = l_rew / ((at - l_at).abs() + 0.0001);
                        let r = r_rew / ((r_at - at).abs() + 0.0001);
                        // Geometrical mean
                        (l * r).sqrt().tanh()
                    },
                    (Some((_, v)), None) => v.tanh(),
                    (None, Some((_, v))) => v.tanh(),
                    _ => 0.0,
                }
            }
        };

        let training_data= get_exprt_iter()
            .filter_map(|x| {
                let base_reward = reward_at_timestamp(x.timestamp);
                let fuel_eff = x.fuel / 100.0; // fuel expended, %

                let (state,action) = match &x.event {
                    Event::Action { state, action } => (state, action),
                    Event::End { reason, rewards_collected } => {
                        let action = [0.0; 4]; // at the end we don't move
                        let state = agent_state; // state at the end is the same as the last state
                        let reward = match reason {
                            EndReason::OutOfBounds => -1.0,
                            EndReason::OutOfFuel => *rewards_collected as f64 * 0.5,
                            EndReason::ManualReset => 0.0,
                        };
                        return Some((state,action,reward,x.timestamp));
                    },
                    _ => return None,
                };

                let combined_reward = if base_reward > 0.0 {
                    base_reward * (1.0 + fuel_eff) // Up to 2x reward when fuel is 100%
                } else {
                    base_reward // Keep negative feedback unchanged
                };

                Some((*state,*action,combined_reward,x.timestamp))
            })
            .fold((
                vec![], // training data
                f64::NEG_INFINITY, // lowest timetamp
                f64::INFINITY, // highest timestamp
                
            ),|(mut data,min,max),(state,action,reward,at)| {

                // 0, 1 -> left, right; 2, 3 -> up, down
                let mut new_action =  [0.0;4]; 
                // Scale action by reward
                for i in 0..4 {
                    new_action[i] = action[i] as f64 * reward.abs();
                }

                // inverse target movement if reward is negative
                if reward < 0.0001 {
                    new_action.swap(0, 1);
                    new_action.swap(2, 3);
                }
                
                data.push((state,new_action));
                (   
                    data,
                    min.min(at),
                    max.max(at),
                )
            });
        
        let temperature = training_data.0.len() as f64 * (training_data.2 - training_data.1 / training_data.2);
    
        let train_params = TrainParams {
            epochs: 1,
            temperature,
            cutoff: 0.0001,
            fn_loss: |t, p| {
                let tx = t[0] - t[1];
                let ty = t[2] - t[3];
                let px = p[0] - p[1];
                let py = p[2] - p[3];
                let tv = Vec2::new(tx as f32, ty as f32);
                let pv = Vec2::new(px as f32, py as f32);
                let loss = tv.dot(pv).sqrt() as f64;
                [
                    (t[0] - p[0]) * loss,
                    (t[1] - p[1]) * loss,
                    (t[2] - p[2]) * loss,
                    (t[3] - p[3]) * loss,
                ]
            },
        };

        // Train the agent
        println!("Training agent with {} samples", training_data.0.len());
        train(&mut agent.brain.network, &training_data.0, train_params);
    }
    training_state.round += 1;
}

