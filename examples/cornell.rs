use bevy::{
    input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel},
    prelude::*,
    render::camera::CameraRenderGraph,
};
use bevy_hikari::prelude::*;
// use bevy_inspector_egui::WorldInspectorPlugin;
use smooth_bevy_cameras::{
    controllers::orbit::{
        ControlEvent, OrbitCameraBundle, OrbitCameraController, OrbitCameraPlugin,
    },
    LookTransformPlugin,
};

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins(DefaultPlugins.build().set(WindowPlugin {
            // window: WindowDescriptor {
            //     width: 800.0,
            //     height: 600.0,
            //     ..Default::default()
            // },
            ..default()
        }))
        // .add_plugin(WorldInspectorPlugin::new())
        .add_plugin(LookTransformPlugin)
        .add_plugin(OrbitCameraPlugin::new(false))
        .add_plugin(HikariPlugin)
        .add_startup_system(setup)
        .add_system(camera_input_map)
        .run();
}

pub struct RaycastSet;

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Model
    commands.spawn(SceneBundle {
        scene: asset_server.load("models/cornell.glb#Scene0"),
        ..default()
    });

    // Camera
    commands
        .spawn((
            Camera3dBundle {
                camera_render_graph: CameraRenderGraph::new(bevy_hikari::graph::NAME),
                transform: Transform::from_xyz(0.0, 1.0, 4.0)
                    .looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
                ..Default::default()
            },
            HikariSettings::default(),
            // RayCastSource::<RaycastSet>::default(),
        ))
        .insert(OrbitCameraBundle::new(
            OrbitCameraController::default(),
            Vec3::new(0.0, 1.0, 4.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::Y,
        ));
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
