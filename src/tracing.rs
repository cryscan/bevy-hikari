use super::{
    GpuVolume, Volume, VolumeBindings, VolumeMeta, VolumeOverlay, VolumeUniformOffset,
    VoxelConeTracingSystems, TRACING_SHADER_HANDLE,
};
use bevy::{
    core::FloatOrd,
    ecs::system::lifetimeless::{Read, SQuery},
    pbr::{
        DrawMesh, MeshPipeline, MeshPipelineKey, MeshUniform, SetMaterialBindGroup,
        SetMeshBindGroup, SetMeshViewBindGroup, SpecializedMaterial,
    },
    prelude::*,
    render::{
        render_asset::RenderAssets,
        render_graph::{self, SlotInfo, SlotType},
        render_phase::{
            sort_phase_system, AddRenderCommand, CachedPipelinePhaseItem, DrawFunctionId,
            DrawFunctions, EntityPhaseItem, EntityRenderCommand, PhaseItem, RenderCommandResult,
            RenderPhase, SetItemPipeline, TrackedRenderPass,
        },
        render_resource::{std140::AsStd140, *},
        renderer::RenderDevice,
        view::{ExtractedView, VisibleEntities},
        RenderApp, RenderStage,
    },
};
use std::marker::PhantomData;

pub struct TracingPlugin;
impl Plugin for TracingPlugin {
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<TracingPipeline>()
                .init_resource::<SpecializedPipelines<TracingPipeline>>()
                .init_resource::<DrawFunctions<Tracing>>()
                .add_system_to_stage(
                    RenderStage::Queue,
                    queue_tracing_bind_groups
                        .label(VoxelConeTracingSystems::QueueTracingBindGroups),
                )
                .add_system_to_stage(RenderStage::PhaseSort, sort_phase_system::<Tracing>);
        }
    }
}

/// The plugin registers the GI draw functions/systems for a [`SpecializedMaterial`].
#[derive(Default)]
pub struct TracingMaterialPlugin<M: SpecializedMaterial>(PhantomData<M>);
impl<M: SpecializedMaterial> Plugin for TracingMaterialPlugin<M> {
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<Tracing, DrawTracingMesh<M>>()
                .add_system_to_stage(
                    RenderStage::Queue,
                    queue_tracing_meshes::<M>.label(VoxelConeTracingSystems::QueueTracing),
                );
        }
    }
}

pub struct TracingPipeline {
    pub material_layout: BindGroupLayout,
    pub tracing_layout: BindGroupLayout,
    pub mesh_pipeline: MeshPipeline,
}

impl FromWorld for TracingPipeline {
    fn from_world(world: &mut World) -> Self {
        let mesh_pipeline = world.get_resource::<MeshPipeline>().unwrap().clone();

        let render_device = world.get_resource::<RenderDevice>().unwrap();

        let material_layout = StandardMaterial::bind_group_layout(render_device);

        let anisotropic_layout_entries = (0..6).map(|direction| BindGroupLayoutEntry {
            binding: direction,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        });
        let tracing_layout_entries = vec![
            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: BufferSize::new(GpuVolume::std140_size_static() as u64),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 7,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 8,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ];

        let tracing_layout_entries = anisotropic_layout_entries
            .chain(tracing_layout_entries)
            .collect::<Vec<_>>();

        let tracing_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("tracing_layout"),
            entries: tracing_layout_entries.as_slice(),
        });

        Self {
            material_layout,
            tracing_layout,
            mesh_pipeline,
        }
    }
}

impl SpecializedPipeline for TracingPipeline {
    type Key = MeshPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let shader = TRACING_SHADER_HANDLE.typed::<Shader>();

        let mut descriptor = self.mesh_pipeline.specialize(key);
        descriptor.fragment.as_mut().unwrap().shader = shader;
        // descriptor.fragment.as_mut().unwrap().targets[0].blend = Some(BlendState::ALPHA_BLENDING);
        descriptor.depth_stencil = Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: CompareFunction::Greater,
            stencil: StencilState::default(),
            bias: DepthBiasState {
                constant: 0,
                slope_scale: 0.0,
                clamp: 0.0,
            },
        });
        descriptor.layout = Some(vec![
            self.mesh_pipeline.view_layout.clone(),
            self.material_layout.clone(),
            self.mesh_pipeline.mesh_layout.clone(),
            self.tracing_layout.clone(),
        ]);

        descriptor
    }
}

#[derive(Component)]
pub struct TracingBindGroup {
    value: BindGroup,
}

#[allow(clippy::too_many_arguments)]
pub fn queue_tracing_meshes<M: SpecializedMaterial>(
    tracing_draw_functions: Res<DrawFunctions<Tracing>>,
    tracing_pipeline: Res<TracingPipeline>,
    material_meshes: Query<(&Handle<M>, &Handle<Mesh>, &MeshUniform)>,
    render_meshes: Res<RenderAssets<Mesh>>,
    render_materials: Res<RenderAssets<M>>,
    mut pipelines: ResMut<SpecializedPipelines<TracingPipeline>>,
    mut pipeline_cache: ResMut<RenderPipelineCache>,
    msaa: Res<Msaa>,
    mut view_query: Query<(&ExtractedView, &VisibleEntities, &mut RenderPhase<Tracing>)>,
) {
    let draw_mesh = tracing_draw_functions
        .read()
        .get_id::<DrawTracingMesh<M>>()
        .unwrap();

    for (view, visible_entities, mut phase) in view_query.iter_mut() {
        let inverse_view_matrix = view.transform.compute_matrix().inverse();
        let inverse_view_row_2 = inverse_view_matrix.row(2);

        for entity in visible_entities.entities.iter().cloned() {
            if let Ok((material_handle, mesh_handle, mesh_uniform)) = material_meshes.get(entity) {
                if !render_materials.contains_key(material_handle) {
                    continue;
                }

                let mut key = MeshPipelineKey::from_msaa_samples(msaa.samples);
                if let Some(mesh) = render_meshes.get(mesh_handle) {
                    if mesh.has_tangents {
                        key |= MeshPipelineKey::VERTEX_TANGENTS;
                    }
                    key |= MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);
                }

                let mesh_z = inverse_view_row_2.dot(mesh_uniform.transform.col(3));
                let pipeline_id = pipelines.specialize(&mut pipeline_cache, &tracing_pipeline, key);
                phase.add(Tracing {
                    draw_function: draw_mesh,
                    pipeline: pipeline_id,
                    entity,
                    distance: -mesh_z,
                });
            }
        }
    }
}

pub fn queue_tracing_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    tracing_pipeline: Res<TracingPipeline>,
    volume_meta: Res<VolumeMeta>,
    volume_query: Query<(Entity, &VolumeBindings), With<Volume>>,
) {
    for (view, volume_bindings) in volume_query.iter() {
        let anisotropic_bindings = (0..6).map(|direction| {
            let view = &volume_bindings.anisotropic_textures[direction].default_view;
            BindGroupEntry {
                binding: direction as u32,
                resource: BindingResource::TextureView(view),
            }
        });
        let tracing_bindings = vec![
            BindGroupEntry {
                binding: 6,
                resource: volume_meta.volume_uniforms.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 7,
                resource: BindingResource::TextureView(&volume_bindings.voxel_texture.default_view),
            },
            BindGroupEntry {
                binding: 8,
                resource: BindingResource::Sampler(&volume_bindings.texture_sampler),
            },
        ];

        let tracing_bindings = anisotropic_bindings
            .chain(tracing_bindings)
            .collect::<Vec<_>>();

        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: Some("tracing_bind_group"),
            layout: &tracing_pipeline.tracing_layout,
            entries: tracing_bindings.as_slice(),
        });

        commands
            .entity(view)
            .insert(TracingBindGroup { value: bind_group });
    }
}

pub struct Tracing {
    distance: f32,
    entity: Entity,
    pipeline: CachedPipelineId,
    draw_function: DrawFunctionId,
}

impl PhaseItem for Tracing {
    type SortKey = FloatOrd;

    fn sort_key(&self) -> Self::SortKey {
        FloatOrd(self.distance)
    }

    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }
}

impl EntityPhaseItem for Tracing {
    fn entity(&self) -> Entity {
        self.entity
    }
}

impl CachedPipelinePhaseItem for Tracing {
    fn cached_pipeline(&self) -> CachedPipelineId {
        self.pipeline
    }
}

pub type DrawTracingMesh<M> = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMaterialBindGroup<M, 1>,
    SetMeshBindGroup<2>,
    SetTracingBindGroup<3>,
    DrawMesh,
);

pub struct SetTracingBindGroup<const I: usize>;
impl<const I: usize> EntityRenderCommand for SetTracingBindGroup<I> {
    type Param = SQuery<(Read<VolumeUniformOffset>, Read<TracingBindGroup>)>;

    fn render<'w>(
        view: Entity,
        _item: Entity,
        query: bevy::ecs::system::SystemParamItem<'w, '_, Self::Param>,
        pass: &mut bevy::render::render_phase::TrackedRenderPass<'w>,
    ) -> bevy::render::render_phase::RenderCommandResult {
        let (volume_uniform_offset, bind_group) = query.get(view).unwrap();
        pass.set_bind_group(I, &bind_group.value, &[volume_uniform_offset.offset]);
        RenderCommandResult::Success
    }
}

pub struct TracingPassNode {
    query: QueryState<
        (
            &'static RenderPhase<Tracing>,
            &'static VolumeOverlay,
            &'static VolumeBindings,
        ),
        With<ExtractedView>,
    >,
}

impl TracingPassNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl render_graph::Node for TracingPassNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(Self::IN_VIEW, SlotType::Entity)]
    }

    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
    }

    fn run(
        &self,
        graph: &mut render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let view_entity = graph.get_input_entity(Self::IN_VIEW)?;
        let (phase, overlay, bindings) = match self.query.get_manual(world, view_entity) {
            Ok(query) => query,
            Err(_) => return Ok(()),
        };

        let images = world.get_resource::<RenderAssets<Image>>().unwrap();
        let color_attachment = &images[&overlay.color_attachment].texture_view;
        let resolve_target = &images[&overlay.resolve_target].texture_view;

        let pass_descriptor = RenderPassDescriptor {
            label: Some("tracing_pass"),
            color_attachments: &[RenderPassColorAttachment {
                view: color_attachment,
                resolve_target: Some(resolve_target),
                ops: Operations {
                    load: LoadOp::Clear(Color::NONE.into()),
                    store: true,
                },
            }],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &bindings.overlay_depth_texture.default_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(0.0),
                    store: false,
                }),
                stencil_ops: None,
            }),
        };

        let draw_functions = world.get_resource::<DrawFunctions<Tracing>>().unwrap();
        let render_pass = render_context
            .command_encoder
            .begin_render_pass(&pass_descriptor);
        let mut draw_functions = draw_functions.write();
        let mut tracked_pass = TrackedRenderPass::new(render_pass);
        for item in &phase.items {
            let draw_function = draw_functions.get_mut(item.draw_function).unwrap();
            draw_function.draw(world, &mut tracked_pass, view_entity, item);
        }

        Ok(())
    }
}
