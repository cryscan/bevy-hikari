use bevy::{
    core_pipeline::bloom::BloomSettings,
    input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel},
    prelude::*,
    render::camera::CameraRenderGraph,
};
use bevy_hikari::prelude::*;
use bevy_inspector_egui::WorldInspectorPlugin;
use bevy_mod_raycast::{
    DefaultRaycastingPlugin, Intersection, RaycastMesh, RaycastMethod, RaycastSource, RaycastSystem,
};
use smooth_bevy_cameras::{
    controllers::orbit::{
        ControlEvent, OrbitCameraBundle, OrbitCameraController, OrbitCameraPlugin,
    },
    LookTransformPlugin,
};
use std::f32::consts::PI;

fn main() {
    App::new()
        .register_type::<EmissiveSphere>()
        .insert_resource(LoadTimer(Timer::from_seconds(1.0, TimerMode::Repeating)))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            // window: WindowDescriptor {
            //     width: 800.0,
            //     height: 600.0,
            //     ..Default::default()
            // },
            ..Default::default()
        }))
        .add_plugin(WorldInspectorPlugin::new())
        .add_plugin(LookTransformPlugin)
        .add_plugin(OrbitCameraPlugin::new(false))
        .add_plugin(DefaultRaycastingPlugin::<RaycastSet>::default())
        .add_plugin(HikariPlugin)
        .add_startup_system(setup)
        .add_system(load_models)
        .add_system(camera_input_map)
        .add_system(sphere_rotate_system)
        .add_system_to_stage(
            CoreStage::First,
            control_directional_light.before(RaycastSystem::BuildRays::<RaycastSet>),
        )
        .run();
}

pub struct RaycastSet;

#[derive(Default, Component, Reflect)]
#[reflect(Component)]
pub struct EmissiveSphere {
    pub speed: f32,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(shape::Plane::default().into()),
            material: materials.add(StandardMaterial {
                base_color: Color::rgb(0.8, 0.7, 0.6).into(),
                perceptual_roughness: 0.9,
                ..Default::default()
            }),
            transform: Transform {
                translation: Vec3::new(0.0, 0.0, 0.0),
                scale: Vec3::new(100.0, 1.0, 100.0),
                ..Default::default()
            },
            ..Default::default()
        },
        RaycastMesh::<RaycastSet>::default(),
    ));

    // Sphere
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Mesh::from(shape::UVSphere {
                radius: 0.5,
                ..Default::default()
            })),
            material: materials.add(StandardMaterial {
                base_color_texture: Some(asset_server.load("models/Earth/earth_daymap.jpg")),
                emissive: Color::rgba(1.0, 1.0, 1.0, 0.5),
                emissive_texture: Some(asset_server.load("models/Earth/earth_daymap.jpg")),
                ..Default::default()
            }),
            transform: Transform {
                translation: Vec3::new(0.0, 1.0, 0.0),
                rotation: Quat::from_rotation_x(-PI / 2.0),
                ..Default::default()
            },
            ..Default::default()
        },
        EmissiveSphere { speed: 0.2 },
        Name::new("Emissive Sphere"),
    ));

    // Only directional light is supported
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 10000.0,
            ..Default::default()
        },
        transform: Transform {
            translation: Vec3::new(0.0, 5.0, 0.0),
            rotation: Quat::from_euler(EulerRot::XYZ, -PI / 4.0, PI / 4.0, 0.0),
            ..Default::default()
        },
        ..Default::default()
    });

    // Camera
    commands
        .spawn((
            Camera3dBundle {
                camera_render_graph: CameraRenderGraph::new(bevy_hikari::graph::NAME),
                camera: Camera {
                    hdr: true,
                    ..Default::default()
                },
                transform: Transform::from_xyz(0.0, 2.5, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
                ..Default::default()
            },
            HikariSettings::default(),
            BloomSettings::default(),
            RaycastSource::<RaycastSet>::default(),
        ))
        .insert(OrbitCameraBundle::new(
            OrbitCameraController::default(),
            Vec3::new(-20.0, 10.0, 20.0),
            Vec3::ZERO,
        ));
}

#[derive(Resource, Deref, DerefMut)]
pub struct LoadTimer(Timer);

fn load_models(
    mut commands: Commands,
    time: Res<Time>,
    asset_server: Res<AssetServer>,
    mut timer: ResMut<LoadTimer>,
    mut counter: Local<usize>,
) {
    if timer.tick(time.delta()).just_finished() {
        match *counter {
            0 => {
                let handle = asset_server.load("models/Low Poly/Big House 2.glb#Scene0");
                for location in [-3, -1, 1, 3] {
                    commands.spawn(SceneBundle {
                        scene: handle.clone(),
                        transform: Transform::from_translation(Vec3::new(
                            4.0 * location as f32,
                            0.0,
                            0.0,
                        )),
                        ..default()
                    });
                }
            }
            1 => {
                let handle = asset_server.load("models/Low Poly/Big House 3.glb#Scene0");
                for (id, location) in [-3, -1, 1, 3].iter().enumerate() {
                    let sign = if id % 2 == 0 { 1.0 } else { -1.0 };
                    commands.spawn(SceneBundle {
                        scene: handle.clone(),
                        transform: Transform::from_translation(Vec3::new(
                            4.0 * (*location as f32),
                            0.0,
                            8.0 * sign,
                        )),
                        ..default()
                    });
                }
            }
            2 => {
                let handle = asset_server.load("models/Low Poly/Big House.glb#Scene0");
                for (id, location) in [-3, -1, 1, 3].iter().enumerate() {
                    let sign = if id % 2 == 0 { -1.0 } else { 1.0 };
                    commands.spawn(SceneBundle {
                        scene: handle.clone(),
                        transform: Transform::from_translation(Vec3::new(
                            4.0 * (*location as f32),
                            0.0,
                            8.0 * sign,
                        )),
                        ..default()
                    });
                }
            }
            _ => {}
        }

        *counter += 1;
    }
}

pub fn camera_input_map(
    mut events: EventWriter<ControlEvent>,
    mut mouse_wheel_reader: EventReader<MouseWheel>,
    mut mouse_motion_events: EventReader<MouseMotion>,
    mouse_buttons: Res<Input<MouseButton>>,
    controllers: Query<&OrbitCameraController>,
) {
    // Can only control one camera at a time.
    let controller = if let Some(controller) = controllers.iter().next() {
        controller
    } else {
        return;
    };
    let OrbitCameraController {
        enabled,
        mouse_rotate_sensitivity,
        mouse_translate_sensitivity,
        mouse_wheel_zoom_sensitivity,
        pixels_per_line,
        ..
    } = *controller;

    if !enabled {
        return;
    }

    let mut cursor_delta = Vec2::ZERO;
    for event in mouse_motion_events.iter() {
        cursor_delta += event.delta;
    }

    if mouse_buttons.pressed(MouseButton::Left) {
        events.send(ControlEvent::Orbit(mouse_rotate_sensitivity * cursor_delta));
    }

    if mouse_buttons.pressed(MouseButton::Right) {
        events.send(ControlEvent::TranslateTarget(
            mouse_translate_sensitivity * cursor_delta,
        ));
    }

    let mut scalar = 1.0;
    for event in mouse_wheel_reader.iter() {
        // scale the event magnitude per pixel or per line
        let scroll_amount = match event.unit {
            MouseScrollUnit::Line => event.y,
            MouseScrollUnit::Pixel => event.y / pixels_per_line,
        };
        scalar *= 1.0 - scroll_amount * mouse_wheel_zoom_sensitivity;
    }
    events.send(ControlEvent::Zoom(scalar));
}

pub fn control_directional_light(
    time: Res<Time>,
    mut cursor: EventReader<CursorMoved>,
    keys: Res<Input<KeyCode>>,
    mut queries: ParamSet<(
        Query<&mut Transform, With<DirectionalLight>>,
        Query<&mut RaycastSource<RaycastSet>>,
        Query<&Intersection<RaycastSet>>,
    )>,
    mut target: Local<Vec3>,
) {
    let cursor_position = match cursor.iter().last() {
        Some(cursor_moved) => cursor_moved.position,
        None => return,
    };

    for mut pick_source in &mut queries.p1() {
        pick_source.cast_method = RaycastMethod::Screenspace(cursor_position);
    }

    if let Ok(intersection) = queries.p2().get_single() {
        if let Some(position) = intersection.position() {
            *target = target.lerp(*position, 1.0 - (-10.0 * time.delta_seconds()).exp());
        }
    }

    if keys.pressed(KeyCode::LShift) {
        if let Ok(mut transform) = queries.p0().get_single_mut() {
            transform.look_at(*target, Vec3::Z);
        }
    }
}

fn sphere_rotate_system(time: Res<Time>, mut query: Query<(&mut Transform, &EmissiveSphere)>) {
    for (mut transform, emissive) in &mut query {
        transform.rotate_local_z(emissive.speed * time.delta_seconds());
    }
}
