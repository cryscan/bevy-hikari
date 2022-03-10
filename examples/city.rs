use bevy::{
    input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel},
    prelude::*,
    scene::InstanceId,
};
use bevy_full_throttle::FullThrottlePlugin;
use bevy_hikari::{NotGiReceiver, Volume, VoxelConeTracingPlugin};
use smooth_bevy_cameras::{
    controllers::orbit::{
        ControlEvent, OrbitCameraBundle, OrbitCameraController, OrbitCameraPlugin,
    },
    LookTransform, LookTransformBundle, LookTransformPlugin, Smoother,
};

#[derive(Default)]
struct SceneInstance(Option<InstanceId>);

fn main() {
    let mut app = App::new();

    app.insert_resource(ClearColor(Color::BLACK))
        .insert_resource(Msaa { samples: 4 })
        .init_resource::<SceneInstance>()
        .add_plugins(DefaultPlugins)
        .add_plugin(LookTransformPlugin)
        .add_plugin(OrbitCameraPlugin::new(true))
        .add_plugin(VoxelConeTracingPlugin)
        .add_plugin(FullThrottlePlugin)
        .add_startup_system(setup)
        .add_system(scene_update)
        .add_system(light_rotate_system)
        .add_system(camera_input_map);

    app.run();
}

#[derive(Component)]
struct Controller;

#[derive(Component)]
struct DirectionalLightTarget;

/// Set up a simple 3D scene
fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scene_spawner: ResMut<SceneSpawner>,
    mut scene_instance: ResMut<SceneInstance>,
) {
    // Target
    commands
        .spawn_bundle(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Icosphere {
                radius: 0.1,
                ..Default::default()
            })),
            material: materials.add(StandardMaterial {
                base_color: Color::rgb(1.0, 1.0, 1.0),
                perceptual_roughness: 1.0,
                emissive: Color::rgb(1.0, 1.0, 1.0),
                ..Default::default()
            }),
            transform: Transform::from_xyz(2.0, 0.0, -2.0),
            ..Default::default()
        })
        .insert(DirectionalLightTarget);

    const HALF_SIZE: f32 = 20.0;
    commands
        .spawn_bundle(DirectionalLightBundle {
            directional_light: DirectionalLight {
                illuminance: 200000.0,
                shadow_projection: OrthographicProjection {
                    left: -HALF_SIZE,
                    right: HALF_SIZE,
                    bottom: -HALF_SIZE,
                    top: HALF_SIZE,
                    near: -10.0 * HALF_SIZE,
                    far: 10.0 * HALF_SIZE,
                    ..Default::default()
                },
                shadows_enabled: true,
                ..Default::default()
            },
            ..Default::default()
        })
        .insert_bundle(LookTransformBundle {
            transform: LookTransform {
                eye: Vec3::new(0.0, 5.0, 0.0),
                target: Vec3::ZERO,
            },
            smoother: Smoother::new(0.8),
        });

    let parent = commands
        .spawn()
        .insert_bundle((
            Transform {
                translation: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::new(0.01, 0.01, 0.01),
            },
            GlobalTransform::default(),
        ))
        .id();

    let scene_id =
        scene_spawner.spawn_as_child(asset_server.load("models/City/scene.gltf#Scene0"), parent);
    scene_instance.0 = Some(scene_id);

    // Camera
    commands
        .spawn_bundle(OrbitCameraBundle::new(
            OrbitCameraController::default(),
            PerspectiveCameraBundle {
                transform: Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
                ..Default::default()
            },
            Vec3::new(0.0, 30.0, 30.0),
            Vec3::ZERO,
        ))
        .insert(Volume::new(
            Vec3::new(-15.0, -5.0, -15.0),
            Vec3::new(15.0, 25.0, 15.0),
        ));
}

fn scene_update(
    mut commands: Commands,
    query: Query<(&Name, &Children)>,
    scene_spawner: Res<SceneSpawner>,
    scene_instance: Res<SceneInstance>,
    mut done: Local<bool>,
) {
    if !*done {
        let names = vec![
            Name::new("CityTree_T_Leaves_D_0"),
            Name::new("Tree_T_Leaves_D_0"),
        ];

        if let Some(instance_id) = scene_instance.0 {
            if let Some(entities) = scene_spawner.iter_instance_entities(instance_id) {
                for entity in entities {
                    if let Ok((name, children)) = query.get(entity) {
                        if names.contains(name) {
                            for child in children.iter() {
                                commands.entity(*child).insert(NotGiReceiver);
                            }
                        }
                    }
                }
                *done = true;
            }
        }
    }
}

#[allow(clippy::type_complexity)]
fn light_rotate_system(
    keyboard_input: Res<Input<KeyCode>>,
    time: Res<Time>,
    mut queries: QuerySet<(
        QueryState<&LookTransform, With<Camera>>,
        QueryState<&mut LookTransform, With<DirectionalLight>>,
        QueryState<&mut Transform, With<DirectionalLightTarget>>,
    )>,
) {
    let query = queries.q0();
    let look = query.single();

    let mut forward = look.target - look.eye;
    forward.y = 0.0;
    forward = forward.normalize_or_zero();
    let right = forward.cross(Vec3::Y);

    let speed = if keyboard_input.pressed(KeyCode::LShift) {
        4.0
    } else {
        1.0
    };

    let mut query = queries.q2();
    let mut target = query.single_mut();

    if keyboard_input.pressed(KeyCode::Up) || keyboard_input.pressed(KeyCode::W) {
        target.translation += forward * speed * time.delta_seconds();
    }
    if keyboard_input.pressed(KeyCode::Down) || keyboard_input.pressed(KeyCode::S) {
        target.translation -= forward * speed * time.delta_seconds();
    }
    if keyboard_input.pressed(KeyCode::Left) || keyboard_input.pressed(KeyCode::A) {
        target.translation -= right * speed * time.delta_seconds();
    }
    if keyboard_input.pressed(KeyCode::Right) || keyboard_input.pressed(KeyCode::D) {
        target.translation += right * speed * time.delta_seconds();
    }
    let target = target.translation;

    let mut query = queries.q1();
    let mut light = query.single_mut();

    light.target = target;
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
