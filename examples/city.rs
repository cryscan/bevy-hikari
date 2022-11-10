use bevy::{
    input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel},
    prelude::*,
    render::camera::CameraRenderGraph,
};
use bevy_hikari::prelude::*;
use bevy_inspector_egui::WorldInspectorPlugin;
use bevy_mod_raycast::{
    DefaultRaycastingPlugin, Intersection, RayCastMesh, RayCastMethod, RayCastSource, RaycastSystem,
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
        // .insert_resource(WindowDescriptor {
        //     width: 400.,
        //     height: 300.,
        //     ..Default::default()
        // })
        .insert_resource(LoadTimer(Timer::from_seconds(1.0, true)))
        .add_plugins(DefaultPlugins)
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

#[derive(Component)]
pub struct EmissiveSphere;

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
        RayCastMesh::<RaycastSet>::default(),
    ));

    // Sphere
    commands
        .spawn_bundle(PbrBundle {
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
            transform: Transform::from_xyz(0.0, 0.5, 0.0),
            ..Default::default()
        })
        .insert(EmissiveSphere)
        .insert(Name::new("Emissive Sphere"));

    // Only directional light is supported
    commands.spawn_bundle(DirectionalLightBundle {
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
        .spawn_bundle(Camera3dBundle {
            camera_render_graph: CameraRenderGraph::new(bevy_hikari::graph::NAME),
            transform: Transform::from_xyz(-10.0, 2.5, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..Default::default()
        })
        .insert_bundle(OrbitCameraBundle::new(
            OrbitCameraController::default(),
            Vec3::new(-50.0, 25.0, 100.0),
            Vec3::new(0., 0., 0.),
        ))
        .insert(RayCastSource::<RaycastSet>::default());
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
                    commands.spawn_bundle(SceneBundle {
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
                    commands.spawn_bundle(SceneBundle {
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
                    commands.spawn_bundle(SceneBundle {
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
        Query<&mut RayCastSource<RaycastSet>>,
        Query<&Intersection<RaycastSet>>,
    )>,
    mut target: Local<Vec3>,
) {
    let cursor_position = match cursor.iter().last() {
        Some(cursor_moved) => cursor_moved.position,
        None => return,
    };

    for mut pick_source in &mut queries.p1() {
        pick_source.cast_method = RayCastMethod::Screenspace(cursor_position);
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

fn sphere_rotate_system(time: Res<Time>, mut query: Query<&mut Transform, With<EmissiveSphere>>) {
    for mut transform in &mut query {
        transform.rotate_y(0.1 * time.delta_seconds());
    }
}
