use crate::{VOXEL_COUNT, VOXEL_MIPMAP_LEVEL_COUNT, VOXEL_SIZE};
use bevy::{
    core_pipeline::{AlphaMask3d, Opaque3d, Transparent3d},
    prelude::*,
    render::{
        camera::{ActiveCamera, CameraProjection, CameraTypePlugin, RenderTarget},
        primitives::Frustum,
        render_phase::RenderPhase,
        render_resource::{std140::AsStd140, std430::AsStd430, *},
        renderer::{RenderDevice, RenderQueue},
        texture::{BevyDefault, CachedTexture, TextureCache},
        RenderApp, RenderStage,
    },
};
use std::f32::consts::FRAC_PI_2;

pub struct VolumePlugin;
impl Plugin for VolumePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Volume>()
            .add_plugin(CameraTypePlugin::<VolumeCamera<0>>::default())
            .add_plugin(CameraTypePlugin::<VolumeCamera<1>>::default())
            .add_plugin(CameraTypePlugin::<VolumeCamera<2>>::default())
            .add_startup_system(setup_volume);

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<VolumeMeta>()
            .add_system_to_stage(RenderStage::Extract, extract_volume_camera_phase::<0>)
            .add_system_to_stage(RenderStage::Extract, extract_volume_camera_phase::<1>)
            .add_system_to_stage(RenderStage::Extract, extract_volume_camera_phase::<2>)
            .add_system_to_stage(RenderStage::Extract, extract_volume)
            .add_system_to_stage(RenderStage::Prepare, prepare_volume);
    }
}

#[derive(Clone)]
pub struct Volume {
    pub enabled: bool,
    pub min: Vec3,
    pub max: Vec3,
    pub views: Option<[Entity; 3]>,
}

impl Default for Volume {
    fn default() -> Self {
        Self {
            enabled: true,
            min: Vec3::new(-5.0, -5.0, -5.0),
            max: Vec3::new(5.0, 5.0, 5.0),
            views: None,
        }
    }
}

pub struct VolumeMeta {
    pub volume_uniform: UniformVec<GpuVolume>,
    pub voxel_buffer: Buffer,
    pub voxel_texture: CachedTexture,
    pub anisotropic_textures: [CachedTexture; 6],
    pub sampler: Sampler,
}

impl FromWorld for VolumeMeta {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.remove_resource::<RenderDevice>().unwrap();
        let mut texture_cache = world.resource_mut::<TextureCache>();

        let voxel_buffer = render_device.create_buffer(&BufferDescriptor {
            label: None,
            size: GpuVoxelBuffer::std430_size_static() as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let voxel_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: VOXEL_SIZE as u32,
                    height: VOXEL_SIZE as u32,
                    depth_or_array_layers: VOXEL_SIZE as u32,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            },
        );

        let size = (VOXEL_SIZE >> 1) as u32;
        let anisotropic_textures = [(); 6].map(|_| {
            texture_cache.get(
                &render_device,
                TextureDescriptor {
                    label: None,
                    size: Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: size,
                    },
                    mip_level_count: VOXEL_MIPMAP_LEVEL_COUNT as u32,
                    sample_count: 1,
                    dimension: TextureDimension::D3,
                    format: TextureFormat::Rgba16Float,
                    usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                },
            )
        });

        let sampler = render_device.create_sampler(&SamplerDescriptor {
            label: None,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });

        world.insert_resource(render_device);

        Self {
            volume_uniform: Default::default(),
            voxel_buffer,
            voxel_texture,
            anisotropic_textures,
            sampler,
        }
    }
}

#[derive(AsStd140)]
pub struct GpuVolume {
    pub min: Vec3,
    pub max: Vec3,
}

#[derive(AsStd430)]
pub struct GpuVoxelBuffer {
    data: [u32; VOXEL_COUNT],
}

#[derive(Component, Default)]
pub struct VolumeCamera<const I: usize>;

/// Setup cameras for the volume.
pub fn setup_volume(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut volume: ResMut<Volume>,
) {
    let size = Extent3d {
        width: VOXEL_SIZE as u32,
        height: VOXEL_SIZE as u32,
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
            usage: TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT,
        },
        ..default()
    };
    image.resize(size);

    let image_handle = images.add(image);

    let center = (volume.max + volume.min) / 2.0;
    let extent = (volume.max - volume.min) / 2.0;

    let camera = Camera {
        target: RenderTarget::Image(image_handle),
        ..default()
    };

    let projection = OrthographicProjection {
        left: -extent.x,
        right: extent.x,
        bottom: -extent.y,
        top: extent.y,
        near: -extent.z,
        far: extent.z,
        ..default()
    };

    let frustum = Frustum::from_view_projection(
        &projection.get_projection_matrix(),
        &Vec3::ZERO,
        &Vec3::Z,
        projection.far,
    );

    volume.views = Some([
        commands
            .spawn_bundle(OrthographicCameraBundle {
                camera: camera.clone(),
                orthographic_projection: projection.clone(),
                visible_entities: default(),
                frustum: frustum.clone(),
                transform: Transform {
                    translation: center,
                    rotation: Quat::IDENTITY,
                    ..default()
                },
                global_transform: default(),
                marker: VolumeCamera::<0>,
            })
            .id(),
        commands
            .spawn_bundle(OrthographicCameraBundle {
                camera: camera.clone(),
                orthographic_projection: projection.clone(),
                visible_entities: default(),
                frustum: frustum.clone(),
                transform: Transform {
                    translation: center,
                    rotation: Quat::from_rotation_y(FRAC_PI_2),
                    ..default()
                },
                global_transform: default(),
                marker: VolumeCamera::<1>,
            })
            .id(),
        commands
            .spawn_bundle(OrthographicCameraBundle {
                camera,
                orthographic_projection: projection.clone(),
                visible_entities: default(),
                frustum,
                transform: Transform {
                    translation: center,
                    rotation: Quat::from_rotation_x(FRAC_PI_2),
                    ..default()
                },
                global_transform: default(),
                marker: VolumeCamera::<2>,
            })
            .id(),
    ]);
}

pub fn extract_volume_camera_phase<const I: usize>(
    mut commands: Commands,
    active: Res<ActiveCamera<VolumeCamera<I>>>,
) {
    if let Some(entity) = active.get() {
        commands.get_or_spawn(entity).insert_bundle((
            RenderPhase::<Opaque3d>::default(),
            RenderPhase::<AlphaMask3d>::default(),
            RenderPhase::<Transparent3d>::default(),
        ));
    }
}

pub fn extract_volume(mut commands: Commands, volume: Res<Volume>) {
    commands.insert_resource(volume.clone());
}

pub fn prepare_volume(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    volume: Res<Volume>,
    mut volume_meta: ResMut<VolumeMeta>,
) {
    volume_meta.volume_uniform.clear();
    volume_meta.volume_uniform.push(GpuVolume {
        min: volume.min,
        max: volume.max,
    });
    volume_meta
        .volume_uniform
        .write_buffer(&render_device, &render_queue);
}
