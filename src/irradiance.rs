use bevy::{
    prelude::*,
    render::{
        camera::{CameraTypePlugin, RenderTarget},
        render_resource::*,
        texture::BevyDefault,
    },
    transform::TransformSystem,
};

use crate::utils::update_custom_camera;

pub struct IrradiancePlugin;
impl Plugin for IrradiancePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(CameraTypePlugin::<IrradianceCamera>::default())
            .add_startup_system(setup_irradiance_camera)
            .add_system_to_stage(
                CoreStage::PostUpdate,
                update_custom_camera::<IrradianceCamera>
                    .before(TransformSystem::TransformPropagate),
            );
    }
}

#[derive(Default, Component)]
pub struct IrradianceCamera;

pub fn setup_irradiance_camera(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    windows: Res<Windows>,
) {
    let size = Extent3d {
        width: windows.primary().width() as u32 >> 1,
        height: windows.primary().height() as u32 >> 1,
        ..default()
    };
    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::bevy_default(),
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST,
        },
        ..default()
    };
    image.resize(size);
    let image_handle = images.add(image);

    let camera = Camera {
        target: RenderTarget::Image(image_handle),
        ..default()
    };

    commands.spawn_bundle(PerspectiveCameraBundle::<IrradianceCamera> {
        camera,
        perspective_projection: default(),
        visible_entities: default(),
        frustum: default(),
        transform: default(),
        global_transform: default(),
        marker: default(),
    });
}
