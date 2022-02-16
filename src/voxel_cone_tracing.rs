use bevy::{
    core_pipeline::Transparent3d,
    ecs::system::{lifetimeless::SRes, SystemParamItem},
    math::Vec3Swizzles,
    pbr::{MeshPipeline, MeshPipelineKey},
    prelude::*,
    reflect::TypeUuid,
    render::{
        camera::CameraProjection,
        render_phase::{EntityRenderCommand, RenderCommandResult, RenderPhase, TrackedRenderPass},
        render_resource::{std140::AsStd140, *},
        renderer::RenderDevice,
        view::ExtractedView,
        RenderApp, RenderStage,
    },
};

pub const VOXEL_SIZE: usize = 256;
pub const MAX_FRAGMENTS: usize = 4194304;
pub const MAX_OCTREE_SIZE: usize = 524288;

pub const VOXEL_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 14750151725749984738);

#[derive(Default)]
pub struct VoxelConeTracingPlugin;

impl Plugin for VoxelConeTracingPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Volume {
            min: Vec3::new(-5.0, -5.0, -5.0),
            max: Vec3::new(5.0, 5.0, 5.0),
        });

        let mut shaders = app.world.get_resource_mut::<Assets<Shader>>().unwrap();
        shaders.set_untracked(
            VOXEL_SHADER_HANDLE,
            Shader::from_wgsl(include_str!("shaders/voxel.wgsl")),
        );

        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => return,
        };

        render_app
            .init_resource::<VoxelPipeline>()
            .init_resource::<SpecializedPipelines<VoxelPipeline>>()
            .init_resource::<VoxelMeta>()
            .add_system_to_stage(
                RenderStage::Extract,
                extract_volume.label(VoxelConeTracingSystems::ExtractVolume),
            )
            .add_system_to_stage(
                RenderStage::Prepare,
                prepare_volume
                    .exclusive_system()
                    .label(VoxelConeTracingSystems::PrepareVolume),
            );
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
pub enum VoxelConeTracingSystems {
    ExtractVolume,
    PrepareVolume,
    QueueVoxelBindGroup,
}

#[derive(Component)]
pub struct VolumeView;

#[derive(Clone, Copy)]
pub struct Volume {
    min: Vec3,
    max: Vec3,
}

#[derive(AsStd140)]
struct GpuVolume {
    min: Vec3,
    max: Vec3,
}

#[derive(AsStd140)]
struct GpuList {
    data: [u32; MAX_FRAGMENTS],
    counter: u32,
}

#[derive(AsStd140, Clone, Copy)]
struct GpuNode {
    children: u32,
}

#[derive(AsStd140)]
struct GpuOctree {
    nodes: [GpuNode; MAX_OCTREE_SIZE],
    levels: [u32; 8],
    node_counter: u32,
    level_counter: u32,
}

#[derive(Default)]
struct VoxelMeta {
    bind_group: Option<BindGroup>,
}

pub struct VoxelPipeline {
    shader: Handle<Shader>,
    bind_group_layout: BindGroupLayout,

    mesh_pipeline: MeshPipeline,

    init_pipeline: ComputePipeline,
    mark_nodes_pipeline: ComputePipeline,
    expand_nodes_pipeline: ComputePipeline,
    expand_final_level_nodes_pipeline: ComputePipeline,
    build_mipmap_pipeline: ComputePipeline,
}

impl FromWorld for VoxelPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();
        let bind_group_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: BufferSize::new(
                                GpuVolume::std140_size_static() as u64
                            ),
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: BufferSize::new(GpuList::std140_size_static() as u64),
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: BufferSize::new(
                                GpuOctree::std140_size_static() as u64
                            ),
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::ReadWrite,
                            format: TextureFormat::Rgba8Unorm,
                            view_dimension: TextureViewDimension::D1,
                        },
                        count: None,
                    },
                ],
            });

        let shader = render_device.create_shader_module(&ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(include_str!("shaders/octree.wgsl").into()),
        });
        let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("octree_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let make_compute_pipeline = |entry_point: &str| -> ComputePipeline {
            let label = format!("{entry_point}_pipeline");

            render_device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(&label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point,
            })
        };

        let init_pipeline = make_compute_pipeline("init");
        let mark_nodes_pipeline = make_compute_pipeline("mark_nodes");
        let expand_nodes_pipeline = make_compute_pipeline("expand_nodes");
        let expand_final_level_nodes_pipeline = make_compute_pipeline("expand_final_level_nodes");
        let build_mipmap_pipeline = make_compute_pipeline("build_mipmap");

        let asset_sever = world.get_resource::<AssetServer>().unwrap();
        let shader = asset_sever.get_handle(VOXEL_SHADER_HANDLE);

        let mesh_pipeline = world.get_resource::<MeshPipeline>().unwrap().clone();

        Self {
            shader,
            bind_group_layout,
            mesh_pipeline,
            init_pipeline,
            mark_nodes_pipeline,
            expand_nodes_pipeline,
            expand_final_level_nodes_pipeline,
            build_mipmap_pipeline,
        }
    }
}

impl SpecializedPipeline for VoxelPipeline {
    type Key = MeshPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut descriptor = self.mesh_pipeline.specialize(key);
        descriptor.vertex.shader = self.shader.clone();
        descriptor.fragment.as_mut().unwrap().shader = self.shader.clone();
        descriptor.layout = Some(vec![
            self.mesh_pipeline.view_layout.clone(),
            self.mesh_pipeline.mesh_layout.clone(),
            self.bind_group_layout.clone(),
        ]);
        descriptor.depth_stencil = None;

        descriptor
    }
}

fn extract_volume(mut commands: Commands, volume: Res<Volume>) {
    commands.insert_resource(volume.clone());
}

fn prepare_volume(mut commands: Commands, volume: Res<Volume>) {
    let center = (volume.max + volume.min) / 2.0;
    let extend = (volume.max - volume.min) / 2.0;

    let make_view = |extend: Vec3| -> ExtractedView {
        ExtractedView {
            width: VOXEL_SIZE as u32,
            height: VOXEL_SIZE as u32,
            transform: GlobalTransform::from_translation(center),
            projection: OrthographicProjection {
                left: -extend.x,
                right: extend.x,
                bottom: -extend.y,
                top: extend.y,
                near: -extend.z,
                far: extend.z,
                ..Default::default()
            }
            .get_projection_matrix(),
            near: 0.0,
            far: 2.0 * extend.z,
        }
    };

    commands.spawn().insert_bundle((
        VolumeView,
        make_view(extend),
        RenderPhase::<Transparent3d>::default(),
    ));
    commands.spawn().insert_bundle((
        VolumeView,
        make_view(extend.yzx()),
        RenderPhase::<Transparent3d>::default(),
    ));
    commands.spawn().insert_bundle((
        VolumeView,
        make_view(extend.zxy()),
        RenderPhase::<Transparent3d>::default(),
    ));
}

struct SetVolumeBindGroup<const I: usize>;
impl<const I: usize> EntityRenderCommand for SetVolumeBindGroup<I> {
    type Param = SRes<VoxelMeta>;

    fn render<'w>(
        _view: Entity,
        _item: Entity,
        meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let bind_group = meta.into_inner().bind_group.as_ref().unwrap();

        RenderCommandResult::Success
    }
}
