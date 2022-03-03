use bevy::prelude::*;
use bevy_hikari::{Volume, VoxelConeTracingPlugin};
use std::f32::consts::PI;

fn main() {
    App::new()
        .insert_resource(Msaa { samples: 4 })
        .add_plugins(DefaultPlugins)
        // Add the voxel cone tracing plugin
        .add_plugin(VoxelConeTracingPlugin)
        .add_startup_system(setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Plane
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Plane { size: 5.0 })),
        material: materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
        ..Default::default()
    });
    // Cube
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
        material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
        transform: Transform::from_xyz(0.0, 0.5, 0.0),
        ..Default::default()
    });

    // Only directional light is supported
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
        transform: Transform {
            translation: Vec3::new(0.0, 5.0, 0.0),
            rotation: Quat::from_euler(EulerRot::XYZ, -PI / 8.0, -PI / 4.0, 0.0),
            ..Default::default()
        },
        ..Default::default()
    });

    // Camera
    commands
        .spawn_bundle(PerspectiveCameraBundle {
            transform: Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..Default::default()
        })
        // Insert the voxel volume as a component to the main 3D camera
        // The volume would better be a cube encapsules the whole scene
        .insert(Volume::new(
            Vec3::new(-2.5, -2.5, -2.5),
            Vec3::new(2.5, 2.5, 2.5),
        ));
}
