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
    utils::HashMap,
};

mod deferred;
mod overlay;
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
pub use tracing::TracingMaterialPlugin;
pub use voxel::VoxelMaterialPlugin;

/// The main plugin, registers required systems and resources.
/// The only material registered is [`StandardMaterial`].
/// To register custom [`Material`], add [`VoxelMaterialPlugin`] to the app.
#[derive(Default)]
pub struct VoxelConeTracingPlugin;

impl Plugin for VoxelConeTracingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GiConfig>()
            .add_plugin(VoxelPlugin)
            .add_plugin(DeferredPlugin)
            .add_plugin(TracingPlugin)
            .add_plugin(OverlayPlugin)
            .add_plugin(DeferredMaterialPlugin::<StandardMaterial>::default())
            .add_plugin(VoxelMaterialPlugin::<StandardMaterial>::default())
            .add_plugin(TracingMaterialPlugin::<StandardMaterial>::default());

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
            .add_system_to_stage(RenderStage::Extract, extract_config)
            .add_system_to_stage(RenderStage::Extract, extract_volumes)
            .add_system_to_stage(RenderStage::Prepare, prepare_volumes);

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

#[derive(Clone)]
pub struct GiConfig {
    pub enabled: bool,
}

impl Default for GiConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
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
pub struct VolumeUniformOffset {
    pub offset: u32,
}

#[derive(Component)]
pub struct VolumeColorAttachment {
    pub texture: CachedTexture,
}

#[derive(Component)]
pub struct VolumeBindings {
    pub voxel_texture: CachedTexture,
    pub anisotropic_textures: Vec<CachedTexture>,
    pub texture_sampler: Sampler,
}

#[derive(Clone, AsStd140)]
pub struct GpuVolume {
    min: Vec3,
    max: Vec3,
}

#[derive(AsStd140)]
pub struct GpuVoxelBuffer {
    data: [u32; VOXEL_COUNT],
}

#[derive(Default)]
pub struct VolumeMeta {
    volume_uniforms: DynamicUniformVec<GpuVolume>,
    voxel_buffers: HashMap<Entity, Buffer>,
}

fn extract_config(mut commands: Commands, config: Res<GiConfig>) {
    commands.insert_resource(config.clone());
}

fn extract_volumes(mut commands: Commands, volumes: Query<(Entity, &Volume)>) {
    for (entity, volume) in volumes.iter() {
        commands.get_or_spawn(entity).insert(volume.clone());
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_volumes(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut texture_cache: ResMut<TextureCache>,
    mut volumes: Query<(Entity, &Volume)>,
    mut volume_meta: ResMut<VolumeMeta>,
    config: Res<GiConfig>,
) {
    if !config.enabled {
        return;
    }

    volume_meta.volume_uniforms.clear();

    for (entity, volume) in volumes.iter_mut() {
        let volume_uniform_offset = VolumeUniformOffset {
            offset: volume_meta.volume_uniforms.push(GpuVolume {
                min: volume.min,
                max: volume.max,
            }),
        };

        if volume_meta.voxel_buffers.get(&entity).is_none() {
            let buffer = render_device.create_buffer(&BufferDescriptor {
                label: None,
                size: GpuVoxelBuffer::std140_size_static() as u64,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            // TODO: clear unused buffers.
            volume_meta.voxel_buffers.insert(entity, buffer);
        }

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
                format: TextureFormat::Rgba16Float,
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
                        format: TextureFormat::Rgba16Float,
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
            let texture = texture_cache.get(
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
                VolumeColorAttachment { texture },
                RenderPhase::<Voxel>::default(),
            ));
        }

        commands.entity(entity).insert_bundle((
            volume_uniform_offset.clone(),
            VolumeBindings {
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
