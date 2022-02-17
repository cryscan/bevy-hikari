use bevy::{
    core::FloatOrd,
    core_pipeline,
    ecs::system::{
        lifetimeless::{Read, SQuery, SRes},
        SystemParamItem,
    },
    pbr::{DrawMesh, MeshPipeline, MeshPipelineKey, SetMeshBindGroup},
    prelude::*,
    reflect::TypeUuid,
    render::{
        camera::CameraProjection,
        primitives::{Aabb, Frustum, Plane},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph},
        render_phase::{
            sort_phase_system, AddRenderCommand, CachedPipelinePhaseItem, DrawFunctionId,
            DrawFunctions, EntityPhaseItem, EntityRenderCommand, PhaseItem, RenderCommandResult,
            RenderPhase, SetItemPipeline, TrackedRenderPass,
        },
        render_resource::{std140::AsStd140, *},
        renderer::{RenderDevice, RenderQueue},
        texture::TextureCache,
        view::{ExtractedView, ViewUniform, ViewUniformOffset, ViewUniforms},
        RenderApp, RenderStage,
    },
    transform::TransformSystem,
};
use std::f32::consts::FRAC_PI_2;

pub const VOXEL_SIZE: usize = 256;
pub const MAX_FRAGMENTS: usize = 1048576;
pub const MAX_OCTREE_SIZE: usize = 131072;
pub const MAX_RADIANCE_SIZE: usize = 2097152;

pub const VOXEL_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 14750151725749984738);

pub mod node {
    pub const INIT: &str = "voxel_init";
}

pub mod draw_3d_graph {
    pub mod node {
        pub const VOXEL_PASS: &str = "voxel_pass";
    }
}

#[derive(Default)]
pub struct VoxelConeTracingPlugin;

impl Plugin for VoxelConeTracingPlugin {
    fn build(&self, app: &mut App) {
        app.add_system_to_stage(
            CoreStage::PostUpdate,
            check_volume_visiblilty.after(TransformSystem::TransformPropagate),
        )
        .init_resource::<VolumeVisibileEntities>()
        .insert_resource(Volume {
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

        let voxel_pass_node = VoxelPassNode::new(&mut render_app.world);

        render_app
            .init_resource::<VoxelPipeline>()
            .init_resource::<SpecializedPipelines<VoxelPipeline>>()
            .init_resource::<VolumeViewMeta>()
            .init_resource::<VoxelMeta>()
            .init_resource::<DrawFunctions<Voxel>>()
            .add_render_command::<Voxel, DrawVoxelMesh>()
            .add_system_to_stage(
                RenderStage::Extract,
                extract_volume.label(VoxelConeTracingSystems::ExtractVolume),
            )
            .add_system_to_stage(
                RenderStage::Prepare,
                prepare_volume
                    .exclusive_system()
                    .label(VoxelConeTracingSystems::PrepareVolume),
            )
            .add_system_to_stage(
                RenderStage::Prepare,
                prepare_voxel.label(VoxelConeTracingSystems::PrepareVoxel),
            )
            .add_system_to_stage(
                RenderStage::Queue,
                queue_volume_view_bind_group
                    .label(VoxelConeTracingSystems::QueueVolumeViewBindGroup),
            )
            .add_system_to_stage(
                RenderStage::Queue,
                queue_voxel_bind_group.label(VoxelConeTracingSystems::QueueVoxelBindGroup),
            )
            .add_system_to_stage(
                RenderStage::Queue,
                queue_voxel.label(VoxelConeTracingSystems::QueueVoxel),
            )
            .add_system_to_stage(RenderStage::PhaseSort, sort_phase_system::<Voxel>);

        let mut render_graph = render_app.world.get_resource_mut::<RenderGraph>().unwrap();
        render_graph.add_node(node::INIT, DispatchVoxelInit);
        render_graph
            .add_node_edge(node::INIT, core_pipeline::node::MAIN_PASS_DEPENDENCIES)
            .unwrap();

        let draw_3d_graph = render_graph
            .get_sub_graph_mut(core_pipeline::draw_3d_graph::NAME)
            .unwrap();
        draw_3d_graph.add_node(draw_3d_graph::node::VOXEL_PASS, voxel_pass_node);
        draw_3d_graph
            .add_node_edge(
                draw_3d_graph::node::VOXEL_PASS,
                core_pipeline::draw_3d_graph::node::MAIN_PASS,
            )
            .unwrap();
        draw_3d_graph
            .add_node_edge(
                draw_3d_graph.input_node().unwrap().id,
                draw_3d_graph::node::VOXEL_PASS,
            )
            .unwrap();
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
pub enum VoxelConeTracingSystems {
    ExtractVolume,
    PrepareVolume,
    PrepareVoxel,
    QueueVolumeViewBindGroup,
    QueueVoxelBindGroup,
    QueueVoxel,
}

#[derive(Clone, Copy)]
pub struct Volume {
    min: Vec3,
    max: Vec3,
}

impl From<Volume> for Frustum {
    fn from(volume: Volume) -> Self {
        Self {
            planes: [
                Plane {
                    normal_d: Vec4::new(1.0, 0.0, 0.0, volume.min.x),
                },
                Plane {
                    normal_d: Vec4::new(-1.0, 0.0, 0.0, volume.max.x),
                },
                Plane {
                    normal_d: Vec4::new(0.0, 1.0, 0.0, volume.min.y),
                },
                Plane {
                    normal_d: Vec4::new(0.0, -1.0, 0.0, volume.max.y),
                },
                Plane {
                    normal_d: Vec4::new(0.0, 0.0, 1.0, volume.min.z),
                },
                Plane {
                    normal_d: Vec4::new(0.0, 0.0, -1.0, volume.max.z),
                },
            ],
        }
    }
}

#[derive(Default, Clone)]
pub struct VolumeVisibileEntities {
    entities: Vec<Entity>,
}

#[derive(Component)]
pub struct VolumeView {
    texture_view: TextureView,
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

#[derive(AsStd140)]
struct GpuRadiance {
    data: [u32; MAX_RADIANCE_SIZE],
}

#[derive(Default)]
struct VolumeViewMeta {
    bind_group: Option<BindGroup>,
}

#[derive(Default)]
struct VoxelMeta {
    volume: UniformVec<GpuVolume>,
    fragments: Option<Buffer>,
    octree: Option<Buffer>,
    radiance: Option<Buffer>,

    bind_group: Option<BindGroup>,
}

pub struct VoxelPipeline {
    volume_view_layout: BindGroupLayout,
    voxel_layout: BindGroupLayout,

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
        let volume_view_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: BufferSize::new(ViewUniform::std140_size_static() as u64),
                    },
                    count: None,
                }],
            });
        let voxel_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(GpuVolume::std140_size_static() as u64),
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
                        min_binding_size: BufferSize::new(GpuOctree::std140_size_static() as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(GpuRadiance::std140_size_static() as u64),
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
            bind_group_layouts: &[&voxel_layout],
            push_constant_ranges: &[],
        });

        let make_compute_pipeline = |entry_point: &str| -> ComputePipeline {
            let label = format!("octree_{entry_point}_pipeline");

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

        let mesh_pipeline = world.get_resource::<MeshPipeline>().unwrap().clone();

        Self {
            volume_view_layout,
            voxel_layout,
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
        let shader = VOXEL_SHADER_HANDLE.typed::<Shader>();

        let mut descriptor = self.mesh_pipeline.specialize(key);
        descriptor.vertex.shader = shader.clone();
        descriptor.fragment.as_mut().unwrap().shader = shader.clone();
        descriptor.layout = Some(vec![
            self.volume_view_layout.clone(),
            self.mesh_pipeline.mesh_layout.clone(),
            self.voxel_layout.clone(),
        ]);
        descriptor.depth_stencil = None;

        descriptor
    }
}

fn check_volume_visiblilty(
    volume: Res<Volume>,
    mut volume_visible_entities: ResMut<VolumeVisibileEntities>,
    mut visible_entity_query: Query<(Entity, &Visibility, Option<&Aabb>, Option<&GlobalTransform>)>,
) {
    let frustum: Frustum = volume.into_inner().clone().into();

    for (entity, visibility, maybe_aabb, maybe_transform) in visible_entity_query.iter_mut() {
        if !visibility.is_visible {
            continue;
        }

        if let (Some(aabb), Some(transform)) = (maybe_aabb, maybe_transform) {
            if !frustum.intersects_obb(aabb, &transform.compute_matrix()) {
                continue;
            }
        }

        volume_visible_entities.entities.push(entity);
    }
}

fn extract_volume(
    mut commands: Commands,
    volume: Res<Volume>,
    volume_visible_entities: Res<VolumeVisibileEntities>,
) {
    commands.insert_resource(volume.clone());
    commands.insert_resource(volume_visible_entities.clone());
}

fn prepare_volume(
    mut commands: Commands,
    volume: Res<Volume>,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
) {
    let center = (volume.max + volume.min) / 2.0;
    let extend = (volume.max - volume.min) / 2.0;

    let create_view = |rotation: GlobalTransform| -> ExtractedView {
        ExtractedView {
            width: VOXEL_SIZE as u32,
            height: VOXEL_SIZE as u32,
            transform: GlobalTransform::from_translation(center) * rotation,
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

    let texture_view = texture_cache
        .get(
            &render_device,
            TextureDescriptor {
                label: Some("voxel_volume_texture"),
                size: Extent3d {
                    width: 256,
                    height: 256,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 4,
                dimension: TextureDimension::D2,
                format: TextureFormat::Bgra8UnormSrgb,
                usage: TextureUsages::RENDER_ATTACHMENT,
            },
        )
        .texture
        .create_view(&TextureViewDescriptor {
            label: Some("voxel_volume_texture_view"),
            format: None,
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });

    commands.spawn().insert_bundle((
        create_view(GlobalTransform::identity()),
        VolumeView {
            texture_view: texture_view.clone(),
        },
        RenderPhase::<Voxel>::default(),
    ));
    commands.spawn().insert_bundle((
        create_view(GlobalTransform::from_rotation(Quat::from_rotation_y(
            FRAC_PI_2,
        ))),
        VolumeView {
            texture_view: texture_view.clone(),
        },
        RenderPhase::<Voxel>::default(),
    ));
    commands.spawn().insert_bundle((
        create_view(GlobalTransform::from_rotation(Quat::from_rotation_x(
            FRAC_PI_2,
        ))),
        VolumeView {
            texture_view: texture_view.clone(),
        },
        RenderPhase::<Voxel>::default(),
    ));
}

fn prepare_voxel(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut voxel_meta: ResMut<VoxelMeta>,
    volume: Res<Volume>,
) {
    voxel_meta.volume.clear();
    voxel_meta.volume.push(GpuVolume {
        min: volume.min,
        max: volume.max,
    });
    voxel_meta
        .volume
        .write_buffer(&render_device, &render_queue);

    if voxel_meta.fragments.is_none() {
        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("voxel_fragments_buffer"),
            size: GpuList::std140_size_static() as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        voxel_meta.fragments = Some(buffer);
    }

    if voxel_meta.octree.is_none() {
        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("voxel_octree_buffer"),
            size: GpuOctree::std140_size_static() as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        voxel_meta.octree = Some(buffer);
    }

    if voxel_meta.radiance.is_none() {
        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("voxel_radiance_buffer"),
            size: GpuRadiance::std140_size_static() as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        voxel_meta.radiance = Some(buffer);
    }
}

fn queue_volume_view_bind_group(
    render_device: Res<RenderDevice>,
    voxel_pipeline: Res<VoxelPipeline>,
    view_uniforms: Res<ViewUniforms>,
    mut volume_view_meta: ResMut<VolumeViewMeta>,
) {
    if let Some(view_binding) = view_uniforms.uniforms.binding() {
        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: Some("volume_view_bind_group"),
            layout: &voxel_pipeline.volume_view_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: view_binding,
            }],
        });
        volume_view_meta.bind_group = Some(bind_group);
    }
}

fn queue_voxel_bind_group(
    render_device: Res<RenderDevice>,
    voxel_pipeline: Res<VoxelPipeline>,
    mut voxel_meta: ResMut<VoxelMeta>,
) {
    let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
        label: Some("voxel_bind_group"),
        layout: &voxel_pipeline.voxel_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: voxel_meta.volume.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: voxel_meta.fragments.as_ref().unwrap(),
                    offset: 0,
                    size: None,
                }),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: voxel_meta.octree.as_ref().unwrap(),
                    offset: 0,
                    size: None,
                }),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: voxel_meta.radiance.as_ref().unwrap(),
                    offset: 0,
                    size: None,
                }),
            },
        ],
    });

    voxel_meta.bind_group = Some(bind_group);
}

fn queue_voxel(
    voxel_draw_functions: Res<DrawFunctions<Voxel>>,
    voxel_pipeline: Res<VoxelPipeline>,
    meshes: Query<&Handle<Mesh>>,
    render_meshes: Res<RenderAssets<Mesh>>,
    mut pipelines: ResMut<SpecializedPipelines<VoxelPipeline>>,
    mut pipeline_cache: ResMut<RenderPipelineCache>,
    volume_visible_entities: Res<VolumeVisibileEntities>,
    mut voxel_phases: Query<&mut RenderPhase<Voxel>>,
) {
    let draw_mesh = voxel_draw_functions
        .read()
        .get_id::<DrawVoxelMesh>()
        .unwrap();

    for mut phase in voxel_phases.iter_mut() {
        for entity in volume_visible_entities.entities.iter().copied() {
            if let Ok(mesh_handle) = meshes.get(entity) {
                let mut key = MeshPipelineKey::empty();
                if let Some(mesh) = render_meshes.get(mesh_handle) {
                    if mesh.has_tangents {
                        key |= MeshPipelineKey::VERTEX_TANGENTS;
                    }
                    key |= MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);
                    key |= MeshPipelineKey::from_msaa_samples(4)
                }

                let pipeline_id = pipelines.specialize(&mut pipeline_cache, &voxel_pipeline, key);
                phase.add(Voxel {
                    draw_function: draw_mesh,
                    pipeline: pipeline_id,
                    entity,
                    distance: 0.0,
                });
            }
        }
    }
}

struct Voxel {
    distance: f32,
    entity: Entity,
    pipeline: CachedPipelineId,
    draw_function: DrawFunctionId,
}

impl PhaseItem for Voxel {
    type SortKey = FloatOrd;

    fn sort_key(&self) -> Self::SortKey {
        FloatOrd(self.distance)
    }

    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }
}

impl EntityPhaseItem for Voxel {
    fn entity(&self) -> Entity {
        self.entity
    }
}

impl CachedPipelinePhaseItem for Voxel {
    fn cached_pipeline(&self) -> CachedPipelineId {
        self.pipeline
    }
}

pub type DrawVoxelMesh = (
    SetItemPipeline,
    SetVolumeViewBindGroup<0>,
    SetMeshBindGroup<1>,
    SetVoxelBindGroup<2>,
    DrawMesh,
);

struct SetVolumeViewBindGroup<const I: usize>;
impl<const I: usize> EntityRenderCommand for SetVolumeViewBindGroup<I> {
    type Param = (SRes<VolumeViewMeta>, SQuery<Read<ViewUniformOffset>>);

    fn render<'w>(
        view: Entity,
        _item: Entity,
        (volume_view_meta, view_query): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let view_uniform_offset = view_query.get(view).unwrap();
        pass.set_bind_group(
            I,
            volume_view_meta.into_inner().bind_group.as_ref().unwrap(),
            &[view_uniform_offset.offset],
        );

        RenderCommandResult::Success
    }
}

struct SetVoxelBindGroup<const I: usize>;
impl<const I: usize> EntityRenderCommand for SetVoxelBindGroup<I> {
    type Param = SRes<VoxelMeta>;

    fn render<'w>(
        _view: Entity,
        _item: Entity,
        voxel_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(I, voxel_meta.into_inner().bind_group.as_ref().unwrap(), &[]);
        RenderCommandResult::Success
    }
}

pub struct VoxelPassNode {
    volume_view_query: QueryState<(Entity, &'static VolumeView, &'static RenderPhase<Voxel>)>,
}

impl VoxelPassNode {
    pub fn new(world: &mut World) -> Self {
        let volume_view_query = QueryState::new(world);
        Self { volume_view_query }
    }
}

impl render_graph::Node for VoxelPassNode {
    fn update(&mut self, world: &mut World) {
        self.volume_view_query.update_archetypes(world);
    }

    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        for (entity, volume_view, phase) in self.volume_view_query.iter_manual(world) {
            let descriptor = RenderPassDescriptor {
                label: None,
                color_attachments: &[RenderPassColorAttachment {
                    view: &volume_view.texture_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK.into()),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            };

            let draw_functions = world.get_resource::<DrawFunctions<Voxel>>().unwrap();
            let render_pass = render_context
                .command_encoder
                .begin_render_pass(&descriptor);
            let mut draw_functions = draw_functions.write();
            let mut tracked_pass = TrackedRenderPass::new(render_pass);
            for item in &phase.items {
                let draw_function = draw_functions.get_mut(item.draw_function).unwrap();
                draw_function.draw(world, &mut tracked_pass, entity, item);
            }
        }

        Ok(())
    }
}
pub struct DispatchVoxelInit;

impl render_graph::Node for DispatchVoxelInit {
    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let pipeline = world.get_resource::<VoxelPipeline>().unwrap();
        let voxel_meta = world.get_resource::<VoxelMeta>().unwrap();
        let bind_group = voxel_meta.bind_group.as_ref().unwrap();

        let mut pass = render_context
            .command_encoder
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("voxel_init_pass"),
            });

        pass.set_pipeline(&pipeline.init_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch(1, 1, 1);

        Ok(())
    }
}
