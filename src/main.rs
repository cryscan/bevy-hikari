use crate::voxel_cone_tracing::Volume;
use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    prelude::*,
};
use std::f32::consts::PI;

mod voxel_cone_tracing;

fn main() {
    let mut app = App::new();

    app.insert_resource(Msaa { samples: 4 })
        .add_plugins(DefaultPlugins)
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(voxel_cone_tracing::VoxelConeTracingPlugin)
        .add_startup_system(setup)
        .add_system(controller_system)
        .add_system(light_rotate_system);

    // bevy_mod_debugdump::print_render_graph(&mut app);

    app.run();
}

#[derive(Component)]
struct Controller;

/// Set up a simple 3D scene
fn setup(
    mut commands: Commands,
    _asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Ground
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Cube::default())),
        material: materials.add(StandardMaterial {
            base_color: Color::rgb(0.6, 1.0, 0.6),
            perceptual_roughness: 1.0,
            ..Default::default()
        }),
        transform: Transform {
            translation: Vec3::new(0.0, -2.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::new(5.0, 0.1, 5.0),
        },
        ..Default::default()
    });
    // Right
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Cube::default())),
        material: materials.add(StandardMaterial {
            base_color: Color::rgb(1.0, 0.5, 0.5),
            perceptual_roughness: 1.0,
            ..Default::default()
        }),
        transform: Transform {
            translation: Vec3::new(2.0, 0.0, 0.0),
            rotation: Quat::from_rotation_z(PI / 2.0),
            scale: Vec3::new(4.0, 0.1, 4.0),
        },
        ..Default::default()
    });
    // Back
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Cube::default())),
        material: materials.add(StandardMaterial {
            base_color: Color::rgb(0.2, 0.8, 1.0),
            perceptual_roughness: 1.0,
            ..Default::default()
        }),
        transform: Transform {
            translation: Vec3::new(0.0, 0.0, -2.0),
            rotation: Quat::from_rotation_x(PI / 2.0),
            scale: Vec3::new(4.0, 0.1, 5.0),
        },
        ..Default::default()
    });
    // Top
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Cube::default())),
        material: materials.add(StandardMaterial {
            base_color: Color::rgb(0.9, 0.9, 0.7),
            perceptual_roughness: 1.0,
            ..Default::default()
        }),
        transform: Transform {
            translation: Vec3::new(1.5, 1.5, -0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::new(1.0, 0.1, 4.0),
        },
        ..Default::default()
    });

    // Cube
    commands
        .spawn_bundle(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Torus {
                radius: 0.5,
                ring_radius: 0.25,
                ..Default::default()
            })),
            material: materials.add(StandardMaterial {
                base_color: Color::rgb(1.0, 1.0, 1.0),
                perceptual_roughness: 1.0,
                ..Default::default()
            }),
            transform: Transform::from_xyz(0.0, 0.5, 0.0),
            ..Default::default()
        })
        .insert(Controller);

    // Emissive
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Icosphere {
            radius: 0.4,
            subdivisions: 2,
        })),
        material: materials.add(StandardMaterial {
            base_color: Color::rgb(0.8, 0.7, 0.6),
            perceptual_roughness: 1.0,
            emissive: Color::rgb(0.8, 0.7, 0.6),
            ..Default::default()
        }),
        transform: Transform::from_xyz(1.2, 0.6, 1.0),
        ..Default::default()
    });

    const HALF_SIZE: f32 = 5.0;
    commands.spawn_bundle(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 10000.0,
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
        transform: Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            -PI / 8.0,
            -PI / 4.0,
            0.0,
        )),
        ..Default::default()
    });

    // commands.spawn_scene(asset_server.load("models/FlightHelmet/FlightHelmet.gltf#Scene0"));

    // Camera
    commands
        .spawn_bundle(PerspectiveCameraBundle {
            transform: Transform::from_xyz(-2.0, 0.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..Default::default()
        })
        .insert(Volume::new(
            Vec3::new(-2.5, -2.5, -2.5),
            Vec3::new(2.5, 2.5, 2.5),
        ));

    commands.spawn_bundle(UiCameraBundle::default());
}

fn controller_system(
    keyboard_input: Res<Input<KeyCode>>,
    time: Res<Time>,
    mut query: Query<&mut Transform, With<Controller>>,
) {
    for mut transform in query.iter_mut() {
        if keyboard_input.pressed(KeyCode::W) {
            transform.translation -= Vec3::Z * time.delta_seconds();
        }
        if keyboard_input.pressed(KeyCode::A) {
            transform.translation -= Vec3::X * time.delta_seconds();
        }
        if keyboard_input.pressed(KeyCode::S) {
            transform.translation += Vec3::Z * time.delta_seconds();
        }
        if keyboard_input.pressed(KeyCode::D) {
            transform.translation += Vec3::X * time.delta_seconds();
        }
        if keyboard_input.pressed(KeyCode::E) {
            transform.translation += Vec3::Y * time.delta_seconds();
        }
        if keyboard_input.pressed(KeyCode::Q) {
            transform.translation -= Vec3::Y * time.delta_seconds();
        }

        let speed = 0.7;
        transform.rotation *= Quat::from_euler(
            EulerRot::XYZ,
            speed * time.delta_seconds(),
            speed * time.delta_seconds(),
            speed * time.delta_seconds(),
        );
    }
}

fn light_rotate_system(
    keyboard_input: Res<Input<KeyCode>>,
    time: Res<Time>,
    mut query: Query<&mut Transform, With<DirectionalLight>>,
    mut euler: Local<Vec3>,
) {
    let speed = 1.0;
    if keyboard_input.pressed(KeyCode::Up) {
        euler.x -= speed * time.delta_seconds();
    }
    if keyboard_input.pressed(KeyCode::Down) {
        euler.x += speed * time.delta_seconds();
    }
    if keyboard_input.pressed(KeyCode::Left) {
        euler.y -= speed * time.delta_seconds();
    }
    if keyboard_input.pressed(KeyCode::Right) {
        euler.y += speed * time.delta_seconds();
    }

    for mut transform in query.iter_mut() {
        transform.rotation = Quat::from_euler(EulerRot::ZYX, euler.z, euler.y, euler.x);
    }
}
