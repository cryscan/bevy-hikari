//! # bevy-hikari
//!
//! An implementation of Voxel Cone Tracing Global Illumination for [bevy].
//!

use self::{deferred::*, overlay::*, tracing::*, voxel::*};
use bevy::{
    core_pipeline::{self, AlphaMask3d, Opaque3d, Transparent3d},
    prelude::*,
    reflect::TypeUuid,
    render::{
        render_graph::RenderGraph,
        render_phase::RenderPhase,
        render_resource::{std140::AsStd140, *},
        renderer::{RenderDevice, RenderQueue},
        texture::{BevyDefault, CachedTexture, TextureCache},
        RenderApp, RenderStage,
    },
};

mod deferred;
mod overlay;
mod storage_vec;
mod tracing;
mod voxel;

pub const VOXEL_SIZE: usize = 256;
pub const VOXEL_ANISOTROPIC_MIPMAP_LEVEL_COUNT: usize = 8;
pub const VOXEL_COUNT: usize = 16777216;

pub const VOXEL_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 14750151725749984740);
pub const TRACING_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 14750151725749984840);
pub const OVERLAY_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 14750151725749984940);
pub const ALBEDO_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 14750151725749984640);

pub mod draw_3d_graph {
    pub mod node {
        pub const VOXEL_PASS: &str = "voxel_pass";
        pub const CLEAR_PASS: &str = "voxel_clear_pass";
        pub const MIPMAP_PASS: &str = "mipmap_pass";
        pub const TRACING_PASS: &str = "tracing_pass";
        pub const OVERLAY_PASS: &str = "overlay_pass";
        pub const DEFERRED_PASS: &str = "deferred_pass";
    }
}

pub use deferred::DeferredMaterialPlugin;
pub use storage_vec::StorageVec;
pub use tracing::TracingMaterialPlugin;
pub use voxel::VoxelMaterialPlugin;

/// The main plugin, registers required systems and resources.
/// The only material registered is [`StandardMaterial`].
/// To register custom [`Material`], add [`VoxelMaterialPlugin`] to the app.
#[derive(Default)]
pub struct VoxelConeTracingPlugin;

impl Plugin for VoxelConeTracingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(VoxelPlugin)
            .add_plugin(DeferredPlugin)
            .add_plugin(TracingPlugin)
            .add_plugin(OverlayPlugin)
            .add_plugin(DeferredMaterialPlugin::<StandardMaterial>::default())
            .add_plugin(VoxelMaterialPlugin::<StandardMaterial>::default())
            .add_plugin(TracingMaterialPlugin::<StandardMaterial>::default())
            .add_system_to_stage(CoreStage::PostUpdate, add_volume_overlay.exclusive_system())
            .add_system_to_stage(CoreStage::PostUpdate, add_volume_views.exclusive_system())
            .add_system_to_stage(CoreStage::PostUpdate, check_visibility);

        let mut shaders = app.world.get_resource_mut::<Assets<Shader>>().unwrap();
        shaders.set_untracked(
            VOXEL_SHADER_HANDLE,
            Shader::from_wgsl(include_str!("shaders/voxel.wgsl").replace("\r\n", "\n")),
        );
        shaders.set_untracked(
            TRACING_SHADER_HANDLE,
            Shader::from_wgsl(include_str!("shaders/tracing.wgsl").replace("\r\n", "\n")),
        );
        shaders.set_untracked(
            OVERLAY_SHADER_HANDLE,
            Shader::from_wgsl(include_str!("shaders/overlay.wgsl").replace("\r\n", "\n")),
        );
        shaders.set_untracked(
            ALBEDO_SHADER_HANDLE,
            Shader::from_wgsl(include_str!("shaders/albedo.wgsl").replace("\r\n", "\n")),
        );

        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => return,
        };

        let voxel_pass_node = VoxelPassNode::new(&mut render_app.world);
        let clear_pass_node = VoxelClearPassNode::new(&mut render_app.world);
        let mipmap_pass_node = MipmapPassNode::new(&mut render_app.world);
        let tracing_pass_node = TracingPassNode::new(&mut render_app.world);
        let overlay_pass_node = OverlayPassNode::new(&mut render_app.world);
        let deferred_pass_node = DeferredPassNode::new(&mut render_app.world);

        render_app
            .init_resource::<VolumeMeta>()
            .add_system_to_stage(
                RenderStage::Extract,
                extract_volumes.label(VoxelConeTracingSystems::ExtractVolumes),
            )
            .add_system_to_stage(
                RenderStage::Extract,
                extract_views.label(VoxelConeTracingSystems::ExtractViews),
            )
            .add_system_to_stage(
                RenderStage::Prepare,
                prepare_volumes.label(VoxelConeTracingSystems::PrepareVolumes),
            )
            .add_system_to_stage(
                RenderStage::Queue,
                queue_volume_view_bind_groups
                    .label(VoxelConeTracingSystems::QueueVolumeViewBindGroups),
            );

        let mut render_graph = render_app.world.get_resource_mut::<RenderGraph>().unwrap();

        let clear_graph = render_graph
            .get_sub_graph_mut(core_pipeline::clear_graph::NAME)
            .unwrap();
        clear_graph.add_node(draw_3d_graph::node::CLEAR_PASS, clear_pass_node);

        let draw_3d_graph = render_graph
            .get_sub_graph_mut(core_pipeline::draw_3d_graph::NAME)
            .unwrap();

        draw_3d_graph.add_node(draw_3d_graph::node::VOXEL_PASS, voxel_pass_node);
        draw_3d_graph.add_node(draw_3d_graph::node::MIPMAP_PASS, mipmap_pass_node);
        draw_3d_graph.add_node(draw_3d_graph::node::TRACING_PASS, tracing_pass_node);
        draw_3d_graph.add_node(draw_3d_graph::node::OVERLAY_PASS, overlay_pass_node);
        draw_3d_graph.add_node(draw_3d_graph::node::DEFERRED_PASS, deferred_pass_node);

        draw_3d_graph
            .add_slot_edge(
                draw_3d_graph.input_node().unwrap().id,
                core_pipeline::draw_3d_graph::input::VIEW_ENTITY,
                draw_3d_graph::node::VOXEL_PASS,
                VoxelPassNode::IN_VIEW,
            )
            .unwrap();
        draw_3d_graph
            .add_node_edge(
                bevy::pbr::draw_3d_graph::node::SHADOW_PASS,
                draw_3d_graph::node::VOXEL_PASS,
            )
            .unwrap();

        draw_3d_graph
            .add_node_edge(
                draw_3d_graph::node::VOXEL_PASS,
                draw_3d_graph::node::MIPMAP_PASS,
            )
            .unwrap();
        draw_3d_graph
            .add_node_edge(
                draw_3d_graph::node::MIPMAP_PASS,
                core_pipeline::draw_3d_graph::node::MAIN_PASS,
            )
            .unwrap();

        draw_3d_graph
            .add_slot_edge(
                draw_3d_graph.input_node().unwrap().id,
                core_pipeline::draw_3d_graph::input::VIEW_ENTITY,
                draw_3d_graph::node::TRACING_PASS,
                VoxelPassNode::IN_VIEW,
            )
            .unwrap();
        draw_3d_graph
            .add_node_edge(
                core_pipeline::draw_3d_graph::node::MAIN_PASS,
                draw_3d_graph::node::TRACING_PASS,
            )
            .unwrap();

        draw_3d_graph
            .add_slot_edge(
                draw_3d_graph.input_node().unwrap().id,
                core_pipeline::draw_3d_graph::input::VIEW_ENTITY,
                draw_3d_graph::node::DEFERRED_PASS,
                VoxelPassNode::IN_VIEW,
            )
            .unwrap();

        draw_3d_graph
            .add_slot_edge(
                draw_3d_graph.input_node().unwrap().id,
                core_pipeline::draw_3d_graph::input::VIEW_ENTITY,
                draw_3d_graph::node::OVERLAY_PASS,
                VoxelPassNode::IN_VIEW,
            )
            .unwrap();
        draw_3d_graph
            .add_node_edge(
                draw_3d_graph::node::TRACING_PASS,
                draw_3d_graph::node::OVERLAY_PASS,
            )
            .unwrap();
        draw_3d_graph
            .add_node_edge(
                draw_3d_graph::node::DEFERRED_PASS,
                draw_3d_graph::node::OVERLAY_PASS,
            )
            .unwrap();
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
pub enum VoxelConeTracingSystems {
    ExtractVolumes,
    ExtractViews,
    ExtractReceiverFilter,
    PrepareVolumes,
    QueueVolumeViewBindGroups,
    QueueVoxelBindGroups,
    QueueVoxel,
    QueueMipmapBindGroups,
    QueueTracing,
    QueueTracingBindGroups,
    QueueDeferred,
}

/// Marker component for meshes not casting GI.
#[derive(Component)]
pub struct NotGiCaster;

/// Marker component for meshes not receiving GI.
#[derive(Component)]
pub struct NotGiReceiver;

/// A component attached to [`Camera`] to indicate the volume of voxelization.
#[derive(Component, Clone)]
pub struct Volume {
    pub min: Vec3,
    pub max: Vec3,
    views: Vec<Entity>,
}

impl Volume {
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self {
            min,
            max,
            views: vec![],
        }
    }
}

impl Default for Volume {
    fn default() -> Self {
        Self::new(Vec3::new(-5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0))
    }
}

#[derive(Component, Clone)]
pub struct VolumeOverlay {
    pub irradiance_size: Extent3d,
    pub irradiance: Handle<Image>,
    pub irradiance_resolve: Handle<Image>,

    pub albedo_size: Extent3d,
    pub albedo: Handle<Image>,
    pub albedo_resolve: Handle<Image>,
}

#[derive(Component)]
pub struct VolumeView;

#[derive(Component, Clone)]
pub struct VolumeUniformOffset {
    pub offset: u32,
}

#[derive(Component, Clone)]
pub struct VoxelBufferOffset {
    pub offset: u32,
}

#[derive(Component)]
pub struct VolumeColorAttachment {
    pub texture: CachedTexture,
}

#[derive(Component)]
pub struct VolumeBindings {
    pub irradiance_depth_texture: CachedTexture,
    pub albedo_depth_texture: CachedTexture,
    pub voxel_texture: CachedTexture,
    pub anisotropic_textures: Vec<CachedTexture>,
    pub texture_sampler: Sampler,
}

#[derive(Clone, AsStd140)]
pub struct GpuVolume {
    min: Vec3,
    max: Vec3,
}

#[derive(Clone, AsStd140)]
pub struct GpuVoxel {
    color: [u32; 2],
}

#[derive(AsStd140)]
pub struct GpuVoxelBuffer {
    data: [GpuVoxel; VOXEL_COUNT],
}

#[derive(Default)]
pub struct VolumeMeta {
    volume_uniforms: DynamicUniformVec<GpuVolume>,
    voxel_buffers: StorageVec<GpuVoxelBuffer>,
}

pub fn add_volume_overlay(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<OverlayMaterial>>,
    mut images: ResMut<Assets<Image>>,
    msaa: Res<Msaa>,
    windows: Res<Windows>,
    volumes: Query<(Entity, &Volume), Without<VolumeOverlay>>,
) {
    if let Some(window) = windows.get_primary() {
        let width = window.width() as u32;
        let height = window.height() as u32;

        for (entity, _) in volumes.iter() {
            let irradiance_size = Extent3d {
                width: width >> 1,
                height: height >> 1,
                depth_or_array_layers: 1,
            };
            let mut image = Image::new_fill(
                irradiance_size,
                TextureDimension::D2,
                &[0, 0, 0, 255],
                TextureFormat::bevy_default(),
            );
            image.texture_descriptor.usage = TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING;

            image.texture_descriptor.sample_count = msaa.samples;
            let irradiance = images.add(image.clone());

            image.texture_descriptor.sample_count = 1;
            let irradiance_resolve = images.add(image);

            let albedo_size = Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            };
            let mut image = Image::new_fill(
                albedo_size,
                TextureDimension::D2,
                &[0, 0, 0, 255],
                TextureFormat::bevy_default(),
            );
            image.texture_descriptor.usage = TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING;
            image.sampler_descriptor.mag_filter = FilterMode::Linear;
            image.sampler_descriptor.min_filter = FilterMode::Linear;

            image.texture_descriptor.sample_count = msaa.samples;
            let albedo = images.add(image.clone());

            image.texture_descriptor.sample_count = 1;
            let albedo_resolve = images.add(image);

            commands.spawn_bundle(MaterialMeshBundle::<OverlayMaterial> {
                mesh: meshes.add(shape::Quad::new(Vec2::ZERO).into()),
                material: materials.add(OverlayMaterial {
                    irradiance_image: irradiance_resolve.clone(),
                    albedo_image: albedo_resolve.clone(),
                }),
                ..Default::default()
            });

            commands.entity(entity).insert(VolumeOverlay {
                irradiance_size,
                irradiance,
                irradiance_resolve,
                albedo_size,
                albedo,
                albedo_resolve,
            });
        }
    }
}

pub fn extract_volumes(mut commands: Commands, volumes: Query<(Entity, &Volume, &VolumeOverlay)>) {
    for (entity, volume, overlay) in volumes.iter() {
        commands
            .get_or_spawn(entity)
            .insert_bundle((volume.clone(), overlay.clone()));
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_volumes(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    msaa: Res<Msaa>,
    mut texture_cache: ResMut<TextureCache>,
    mut volumes: Query<(Entity, &Volume, &VolumeOverlay)>,
    mut volume_meta: ResMut<VolumeMeta>,
) {
    volume_meta.volume_uniforms.clear();
    volume_meta.voxel_buffers.clear();

    for (entity, volume, overlay) in volumes.iter_mut() {
        let volume_uniform_offset = VolumeUniformOffset {
            offset: volume_meta.volume_uniforms.push(GpuVolume {
                min: volume.min,
                max: volume.max,
            }),
        };

        let voxel_buffer_offset = VoxelBufferOffset {
            offset: volume_meta.voxel_buffers.push(&render_device),
        };

        let voxel_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("voxel_texture"),
                size: Extent3d {
                    width: VOXEL_SIZE as u32,
                    height: VOXEL_SIZE as u32,
                    depth_or_array_layers: VOXEL_SIZE as u32,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            },
        );

        let anisotropic_textures = (0..6)
            .map(|_| {
                texture_cache.get(
                    &render_device,
                    TextureDescriptor {
                        label: None,
                        size: Extent3d {
                            width: (VOXEL_SIZE / 2) as u32,
                            height: (VOXEL_SIZE / 2) as u32,
                            depth_or_array_layers: (VOXEL_SIZE / 2) as u32,
                        },
                        mip_level_count: VOXEL_ANISOTROPIC_MIPMAP_LEVEL_COUNT as u32,
                        sample_count: 1,
                        dimension: TextureDimension::D3,
                        format: TextureFormat::Rgba8Unorm,
                        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                    },
                )
            })
            .collect();

        let texture_sampler = render_device.create_sampler(&SamplerDescriptor {
            label: None,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });

        for view in volume.views.iter().cloned() {
            let color_texture = texture_cache.get(
                &render_device,
                TextureDescriptor {
                    label: Some("voxel_volume_texture"),
                    size: Extent3d {
                        width: VOXEL_SIZE as u32,
                        height: VOXEL_SIZE as u32,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::bevy_default(),
                    usage: TextureUsages::RENDER_ATTACHMENT,
                },
            );

            commands.entity(view).insert_bundle((
                volume_uniform_offset.clone(),
                voxel_buffer_offset.clone(),
                VolumeColorAttachment {
                    texture: color_texture,
                },
                RenderPhase::<Voxel>::default(),
            ));
        }

        let irradiance_depth_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("volume_overlay_depth_texture"),
                size: overlay.irradiance_size,
                mip_level_count: 1,
                sample_count: msaa.samples,
                dimension: TextureDimension::D2,
                format: TextureFormat::Depth32Float,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            },
        );

        let albedo_depth_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("volume_overlay_depth_texture"),
                size: overlay.albedo_size,
                mip_level_count: 1,
                sample_count: msaa.samples,
                dimension: TextureDimension::D2,
                format: TextureFormat::Depth32Float,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            },
        );

        commands.entity(entity).insert_bundle((
            volume_uniform_offset.clone(),
            voxel_buffer_offset.clone(),
            VolumeBindings {
                irradiance_depth_texture,
                albedo_depth_texture,
                voxel_texture,
                anisotropic_textures,
                texture_sampler: texture_sampler.clone(),
            },
            RenderPhase::<Tracing<Opaque3d>>::default(),
            RenderPhase::<Tracing<AlphaMask3d>>::default(),
            RenderPhase::<Tracing<Transparent3d>>::default(),
            RenderPhase::<AmbientOcclusion>::default(),
            RenderPhase::<Deferred<Opaque3d>>::default(),
            RenderPhase::<Deferred<AlphaMask3d>>::default(),
            RenderPhase::<Deferred<Transparent3d>>::default(),
        ));
    }

    volume_meta
        .volume_uniforms
        .write_buffer(&render_device, &render_queue);
}
