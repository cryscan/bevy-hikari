use self::{tracing::*, voxel::*};
use bevy::{
    core_pipeline,
    prelude::*,
    reflect::TypeUuid,
    render::{
        render_graph::RenderGraph,
        render_phase::RenderPhase,
        render_resource::{std140::AsStd140, *},
        renderer::{RenderDevice, RenderQueue},
        texture::{CachedTexture, TextureCache},
        RenderApp, RenderStage,
    },
};

mod tracing;
mod voxel;

pub const VOXEL_SIZE: usize = 256;
pub const VOXEL_ANISOTROPIC_MIPMAP_LEVEL_COUNT: usize = 8;

pub const VOXEL_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 14750151725749984738);
pub const TRACING_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 14750151725749984840);

pub mod draw_3d_graph {
    pub mod node {
        pub const VOXEL_PASS: &str = "voxel_pass";
        pub const CLEAR_PASS: &str = "voxel_clear_pass";
        pub const MIPMAP_PASS: &str = "mipmap_pass";
        pub const TRACING_PASS: &str = "tracing_pass";
    }
}

#[derive(Default)]
pub struct VoxelConeTracingPlugin;

impl Plugin for VoxelConeTracingPlugin {
    fn build(&self, app: &mut App) {
        app
            //.add_plugin(MaterialPlugin::<OverlayMaterial>::default())
            .add_plugin(VoxelPlugin)
            .add_plugin(TracingPlugin)
            .add_plugin(VoxelMaterialPlugin::<StandardMaterial>::default())
            .add_system_to_stage(CoreStage::PostUpdate, add_volume_overlay.exclusive_system())
            .add_system_to_stage(CoreStage::PostUpdate, add_volume_views.exclusive_system())
            .add_system_to_stage(CoreStage::PostUpdate, check_visibility);

        let mut shaders = app.world.get_resource_mut::<Assets<Shader>>().unwrap();
        shaders.set_untracked(
            VOXEL_SHADER_HANDLE,
            Shader::from_wgsl(include_str!("../shaders/voxel.wgsl").replace("\r\n", "\n")),
        );
        shaders.set_untracked(
            TRACING_SHADER_HANDLE,
            Shader::from_wgsl(include_str!("../shaders/tracing.wgsl").replace("\r\n", "\n")),
        );

        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => return,
        };

        let voxel_pass_node = VoxelPassNode::new(&mut render_app.world);
        let clear_pass_node = VoxelClearPassNode::new(&mut render_app.world);
        let mipmap_pass_node = MipmapPassNode::new(&mut render_app.world);
        let tracing_pass_node = TracingPassNode::new(&mut render_app.world);

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
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
pub enum VoxelConeTracingSystems {
    ExtractVolumes,
    ExtractViews,
    PrepareVolumes,
    QueueVolumeViewBindGroups,
    QueueVoxelBindGroups,
    QueueVoxel,
    QueueMipmapBindGroups,
    QueueTracing,
    QueueTracingBindGroups,
}

#[derive(Component, Clone)]
pub struct Volume {
    pub min: Vec3,
    pub max: Vec3,
    views: Vec<Entity>,
}

impl Volume {
    pub fn new(min: Vec3, max: Vec3) -> Self {
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
    view: Handle<Image>,
    resolve_target: Handle<Image>,
}

#[derive(Component)]
pub struct VolumeView;

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
    pub anisotropic_texture: CachedTexture,

    pub texture_sampler: Sampler,
}

#[derive(Clone, AsStd140)]
struct GpuVolume {
    min: Vec3,
    max: Vec3,
}

#[derive(Default)]
pub struct VolumeMeta {
    volume_uniforms: DynamicUniformVec<GpuVolume>,
}

pub fn add_volume_overlay(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    msaa: Res<Msaa>,
    windows: Res<Windows>,
    volumes: Query<(Entity, &Volume), Without<VolumeOverlay>>,
) {
    if let Some(window) = windows.get_primary() {
        let width = window.width() as u32;
        let height = window.height() as u32;

        for (entity, _) in volumes.iter() {
            let mut image = Image::new_fill(
                Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                &[0, 0, 0, 255],
                TextureFormat::Bgra8UnormSrgb,
            );
            image.texture_descriptor.sample_count = msaa.samples;
            image.texture_descriptor.usage = TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING;
            let view = images.add(image.clone());

            image.texture_descriptor.sample_count = 1;
            let image = images.add(image);

            commands.entity(entity).insert(VolumeOverlay {
                view,
                resolve_target: image.clone(),
            });

            commands.spawn_bundle(NodeBundle {
                style: Style {
                    size: Size::new(Val::Percent(100.0), Val::Percent(100.0)),
                    ..Default::default()
                },
                image: UiImage(image),
                ..Default::default()
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

pub fn prepare_volumes(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut texture_cache: ResMut<TextureCache>,
    mut volumes: Query<(Entity, &Volume)>,
    mut volume_meta: ResMut<VolumeMeta>,
) {
    volume_meta.volume_uniforms.clear();

    for (entity, volume) in volumes.iter_mut() {
        let volume_uniform_offset = VolumeUniformOffset {
            offset: volume_meta.volume_uniforms.push(GpuVolume {
                min: volume.min,
                max: volume.max,
            }),
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

        let anisotropic_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("voxel_anisotropic_texture"),
                size: Extent3d {
                    width: (VOXEL_SIZE / 2) as u32,
                    height: (VOXEL_SIZE / 2) as u32,
                    depth_or_array_layers: 6 * (VOXEL_SIZE / 2) as u32,
                },
                mip_level_count: VOXEL_ANISOTROPIC_MIPMAP_LEVEL_COUNT as u32,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            },
        );

        let texture_sampler = render_device.create_sampler(&SamplerDescriptor {
            label: None,
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
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
                    sample_count: 4,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Bgra8UnormSrgb,
                    usage: TextureUsages::RENDER_ATTACHMENT,
                },
            );

            commands.entity(view).insert_bundle((
                volume_uniform_offset.clone(),
                VolumeColorAttachment {
                    texture: color_texture,
                },
                RenderPhase::<Voxel>::default(),
            ));
        }

        commands.entity(entity).insert_bundle((
            volume_uniform_offset.clone(),
            VolumeBindings {
                voxel_texture,
                anisotropic_texture,
                texture_sampler,
            },
            RenderPhase::<Tracing>::default(),
        ));
    }

    volume_meta
        .volume_uniforms
        .write_buffer(&render_device, &render_queue);
}
