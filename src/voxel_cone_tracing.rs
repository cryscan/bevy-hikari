use bevy::{
    core::FloatOrd,
    core_pipeline::{self},
    ecs::system::{
        lifetimeless::{Read, SQuery},
        SystemParamItem,
    },
    pbr::{
        DrawMesh, MeshPipeline, MeshPipelineKey, SetMaterialBindGroup, SetMeshBindGroup,
        SetMeshViewBindGroup, SpecializedMaterial,
    },
    prelude::*,
    reflect::TypeUuid,
    render::{
        camera::CameraProjection,
        primitives::Frustum,
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
        view::{ExtractedView, VisibleEntities},
        RenderApp, RenderStage,
    },
    transform::TransformSystem,
};
use std::f32::consts::FRAC_PI_2;

pub const VOXEL_SIZE: usize = 256;

pub const VOXEL_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 14750151725749984738);

pub mod draw_3d_graph {
    pub mod node {
        pub const VOXEL_PASS: &str = "voxel_pass";
    }
}

#[derive(Default)]
pub struct VoxelConeTracingPlugin;

impl Plugin for VoxelConeTracingPlugin {
    fn build(&self, app: &mut App) {
        app.add_system_to_stage(CoreStage::PostUpdate, add_volume_views.exclusive_system());

        let mut shaders = app.world.get_resource_mut::<Assets<Shader>>().unwrap();
        shaders.set_untracked(
            VOXEL_SHADER_HANDLE,
            Shader::from_wgsl(include_str!("shaders/voxel_3d.wgsl")),
        );

        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => return,
        };

        let voxel_pass_node = VoxelPassNode::new(&mut render_app.world);

        render_app
            .init_resource::<VoxelPipeline>()
            .init_resource::<SpecializedPipelines<VoxelPipeline>>()
            .init_resource::<VoxelMeta>()
            .init_resource::<DrawFunctions<Voxel>>()
            .add_render_command::<Voxel, DrawVoxelMesh>()
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
                prepare_volumes
                    .exclusive_system()
                    .label(VoxelConeTracingSystems::PrepareVolumes),
            )
            .add_system_to_stage(
                RenderStage::Queue,
                queue_voxel_bind_groups.label(VoxelConeTracingSystems::QueueVoxelBindGroup),
            )
            .add_system_to_stage(
                RenderStage::Queue,
                queue_voxel.label(VoxelConeTracingSystems::QueueVoxel),
            )
            .add_system_to_stage(RenderStage::PhaseSort, sort_phase_system::<Voxel>);

        let mut render_graph = render_app.world.get_resource_mut::<RenderGraph>().unwrap();

        let draw_3d_graph = render_graph
            .get_sub_graph_mut(core_pipeline::draw_3d_graph::NAME)
            .unwrap();

        /*
        draw_3d_graph.add_node(draw_3d_graph::node::VOXEL_PASS, voxel_pass_node);
        draw_3d_graph
            .add_node_edge(
                bevy::pbr::draw_3d_graph::node::SHADOW_PASS,
                draw_3d_graph::node::VOXEL_PASS,
            )
            .unwrap();
        draw_3d_graph
            .add_node_edge(
                draw_3d_graph::node::VOXEL_PASS,
                core_pipeline::draw_3d_graph::node::MAIN_PASS,
            )
            .unwrap();
         */
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
pub enum VoxelConeTracingSystems {
    ExtractVolumes,
    ExtractViews,
    PrepareVolumes,
    QueueVoxelBindGroup,
    QueueVoxel,
}

#[derive(Component, Clone)]
pub struct Volume {
    pub min: Vec3,
    pub max: Vec3,
    views: Vec<Entity>,
}

#[derive(Component)]
pub struct VolumeView;

#[derive(Bundle, Clone)]
pub struct VolumeBundle {
    pub volume: Volume,
}

impl Default for VolumeBundle {
    fn default() -> Self {
        Self {
            volume: Volume {
                min: Vec3::new(-5.0, -5.0, -5.0),
                max: Vec3::new(5.0, 5.0, 5.0),
                views: vec![],
            },
        }
    }
}

#[derive(Component)]
pub struct VolumeUniformOffset {
    pub offset: u32,
}

#[derive(Component)]
pub struct VolumeTexture {
    pub texture_view: TextureView,
}

#[derive(Component)]
pub struct VoxelBindings {
    voxel_texture: Texture,
    voxel_texture_view: TextureView,
}

#[derive(Clone, AsStd140)]
struct GpuVolume {
    min: Vec3,
    max: Vec3,
}

#[derive(Default)]
struct VoxelMeta {
    volume_uniforms: DynamicUniformVec<GpuVolume>,
}

#[derive(Component)]
struct VoxelBindGroup {
    bind_group: BindGroup,
}

pub struct VoxelPipeline {
    material_layout: BindGroupLayout,
    voxel_layout: BindGroupLayout,
    mesh_pipeline: MeshPipeline,
}

impl FromWorld for VoxelPipeline {
    fn from_world(world: &mut World) -> Self {
        let mesh_pipeline = world.get_resource::<MeshPipeline>().unwrap().clone();

        let render_device = world.get_resource::<RenderDevice>().unwrap();

        let material_layout = StandardMaterial::bind_group_layout(render_device);

        let voxel_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("voxel_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: BufferSize::new(GpuVolume::std140_size_static() as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });

        Self {
            material_layout,
            voxel_layout,
            mesh_pipeline,
        }
    }
}

impl SpecializedPipeline for VoxelPipeline {
    type Key = MeshPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let shader = VOXEL_SHADER_HANDLE.typed::<Shader>();

        let mut descriptor = self.mesh_pipeline.specialize(key);
        descriptor.fragment.as_mut().unwrap().shader = shader.clone();
        descriptor.layout = Some(vec![
            self.mesh_pipeline.view_layout.clone(),
            self.material_layout.clone(),
            self.mesh_pipeline.mesh_layout.clone(),
            self.voxel_layout.clone(),
        ]);
        descriptor.primitive.cull_mode = None;
        descriptor.primitive.conservative = true;
        descriptor.depth_stencil = None;

        descriptor
    }
}

fn add_volume_views(mut commands: Commands, mut query: Query<&mut Volume>) {
    for mut volume in query.iter_mut() {
        if !volume.views.is_empty() {
            continue;
        }

        let center = (volume.max + volume.min) / 2.0;
        let extend = (volume.max - volume.min) / 2.0;

        for rotation in [
            Quat::IDENTITY,
            Quat::from_rotation_y(FRAC_PI_2),
            Quat::from_rotation_x(FRAC_PI_2),
        ] {
            let transform = GlobalTransform::from_translation(center)
                * GlobalTransform::from_rotation(rotation);

            let projection = OrthographicProjection {
                left: -extend.x,
                right: extend.x,
                bottom: -extend.y,
                top: extend.y,
                near: -extend.z,
                far: extend.z,
                ..Default::default()
            };

            let entity = commands
                .spawn_bundle((
                    VolumeView,
                    transform,
                    projection,
                    Frustum::default(),
                    VisibleEntities::default(),
                ))
                .id();
            volume.views.push(entity);
        }
    }
}

fn extract_views(
    mut commands: Commands,
    query: Query<
        (
            Entity,
            &GlobalTransform,
            &OrthographicProjection,
            &VisibleEntities,
        ),
        With<VolumeView>,
    >,
) {
    for (entity, transform, projection, visible_entities) in query.iter() {
        commands.get_or_spawn(entity).insert_bundle((
            ExtractedView {
                projection: projection.get_projection_matrix(),
                transform: *transform,
                width: VOXEL_SIZE as u32,
                height: VOXEL_SIZE as u32,
                near: projection.near,
                far: projection.far,
            },
            visible_entities.clone(),
            VolumeView,
        ));
    }
}

fn extract_volumes(mut commands: Commands, query: Query<(Entity, &Volume)>) {
    for (entity, volume) in query.iter() {
        commands.get_or_spawn(entity).insert(volume.clone());
    }
}

fn prepare_volumes(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut texture_cache: ResMut<TextureCache>,
    mut query: Query<(Entity, &Volume)>,
    mut voxel_meta: ResMut<VoxelMeta>,
) {
    voxel_meta.volume_uniforms.clear();

    for (entity, volume) in query.iter_mut() {
        let texture_view = texture_cache
            .get(
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

        for view in &volume.views {
            let texture_view = texture_view.clone();
            commands.entity(*view).insert_bundle((
                VolumeTexture { texture_view },
                RenderPhase::<Voxel>::default(),
            ));
        }

        let volume_uniform_offset = VolumeUniformOffset {
            offset: voxel_meta.volume_uniforms.push(GpuVolume {
                min: volume.min,
                max: volume.max,
            }),
        };

        let voxel_texture = texture_cache
            .get(
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
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                },
            )
            .texture;

        let voxel_texture_view = voxel_texture.create_view(&TextureViewDescriptor {
            label: Some(&format!("voxel_texture_view_{}", entity.id())),
            format: None,
            dimension: Some(TextureViewDimension::D3),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });

        let voxel_bindings = VoxelBindings {
            voxel_texture,
            voxel_texture_view,
        };

        commands
            .entity(entity)
            .insert(volume_uniform_offset)
            .insert(voxel_bindings);
    }

    voxel_meta
        .volume_uniforms
        .write_buffer(&render_device, &render_queue);
}

fn queue_voxel_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    voxel_pipeline: Res<VoxelPipeline>,
    voxel_meta: Res<VoxelMeta>,
    view_query: Query<(Entity, &VoxelBindings)>,
) {
    for (entity, bingings) in view_query.iter() {
        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: Some("voxel_bind_group"),
            layout: &voxel_pipeline.voxel_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: voxel_meta.volume_uniforms.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&bingings.voxel_texture_view),
                },
            ],
        });

        commands
            .entity(entity)
            .insert(VoxelBindGroup { bind_group });
    }
}

fn queue_voxel(
    voxel_draw_functions: Res<DrawFunctions<Voxel>>,
    voxel_pipeline: Res<VoxelPipeline>,
    meshes: Query<&Handle<Mesh>>,
    render_meshes: Res<RenderAssets<Mesh>>,
    mut pipelines: ResMut<SpecializedPipelines<VoxelPipeline>>,
    mut pipeline_cache: ResMut<RenderPipelineCache>,
    volume_query: Query<&Volume, Without<VolumeView>>,
    mut volume_view_query: Query<(&VisibleEntities, &mut RenderPhase<Voxel>), With<VolumeView>>,
) {
    let draw_mesh = voxel_draw_functions
        .read()
        .get_id::<DrawVoxelMesh>()
        .unwrap();

    for volume in volume_query.iter() {
        for view in volume.views.iter().cloned() {
            let (visible_entities, mut phase) = volume_view_query.get_mut(view).unwrap();
            for entity in visible_entities.entities.iter().cloned() {
                if let Ok(mesh_handle) = meshes.get(entity) {
                    let mut key = MeshPipelineKey::empty();
                    if let Some(mesh) = render_meshes.get(mesh_handle) {
                        if mesh.has_tangents {
                            key |= MeshPipelineKey::VERTEX_TANGENTS;
                        }
                        key |= MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);
                        key |= MeshPipelineKey::from_msaa_samples(1);
                    }

                    let pipeline_id =
                        pipelines.specialize(&mut pipeline_cache, &voxel_pipeline, key);
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
    SetMeshViewBindGroup<0>,
    SetMaterialBindGroup<StandardMaterial, 1>,
    SetMeshBindGroup<2>,
    SetVoxelBindGroup<3>,
    DrawMesh,
);

struct SetVoxelBindGroup<const I: usize>;
impl<const I: usize> EntityRenderCommand for SetVoxelBindGroup<I> {
    type Param = SQuery<(Read<VolumeUniformOffset>, Read<VoxelBindGroup>)>;

    fn render<'w>(
        view: Entity,
        _item: Entity,
        query: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let (volume_uniform_offset, bind_group) = query.get(view).unwrap();
        pass.set_bind_group(I, &bind_group.bind_group, &[volume_uniform_offset.offset]);
        RenderCommandResult::Success
    }
}

pub struct VoxelPassNode {
    volume_view_query: QueryState<(Entity, &'static VolumeTexture, &'static RenderPhase<Voxel>)>,
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
