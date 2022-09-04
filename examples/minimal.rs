use bevy::{pbr::PbrPlugin, prelude::*, render::camera::CameraRenderGraph};
use bevy_hikari::prelude::*;
use std::f32::consts::PI;

fn main() {
    App::new()
        .insert_resource(Msaa { samples: 4 })
        .add_plugins(DefaultPlugins)
        .add_plugin(PbrPlugin)
        .add_plugin(HikariPlugin)
        .add_startup_system(setup)
        .add_system(rotate_camera)
        .run();
}

fn setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    // Plane
    commands
        .spawn_bundle(TransformBundle {
            local: Transform::from_rotation(Quat::from_rotation_y(PI / 4.0)),
            ..Default::default()
        })
        .insert_bundle(VisibilityBundle::default())
        .insert(meshes.add(Mesh::from(shape::Plane { size: 5.0 })));
    // Sphere
    commands
        .spawn_bundle(TransformBundle {
            local: Transform::from_xyz(0.0, 0.5, 0.0),
            ..Default::default()
        })
        .insert_bundle(VisibilityBundle::default())
        .insert(meshes.add(Mesh::from(shape::UVSphere {
            radius: 0.5,
            ..Default::default()
        })));

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
    commands.spawn_bundle(Camera3dBundle {
        camera_render_graph: CameraRenderGraph::new(bevy_hikari::graph::NAME),
        transform: Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..Default::default()
    });
}

fn rotate_camera(time: Res<Time>, mut query: Query<&mut Transform, With<Camera3d>>) {
    let radius = Vec2::new(-2.0, 5.0).length();
    let sin = time.delta_seconds().sin();
    let cos = time.delta_seconds().cos();
    for mut transform in &mut query {
        *transform =
            Transform::from_xyz(radius * cos, 2.5, radius * sin).looking_at(Vec3::ZERO, Vec3::Y);
    }
}
