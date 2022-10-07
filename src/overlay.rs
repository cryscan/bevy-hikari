use crate::{light::LightPassTarget, OVERLAY_SHADER_HANDLE, QUAD_HANDLE};
use bevy::{
    core_pipeline::clear_color::ClearColorConfig,
    ecs::system::{
        lifetimeless::{Read, SQuery},
        SystemParamItem,
    },
    pbr::{DrawMesh, MeshPipelineKey},
    prelude::{shape::Quad, *},
    render::{
        camera::ExtractedCamera,
        mesh::MeshVertexBufferLayout,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
        render_phase::{
            AddRenderCommand, CachedRenderPipelinePhaseItem, DrawFunctionId, DrawFunctions,
            EntityPhaseItem, EntityRenderCommand, PhaseItem, RenderCommandResult, RenderPhase,
            SetItemPipeline, TrackedRenderPass,
        },
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
        texture::BevyDefault,
        view::{ExtractedView, ViewTarget},
        Extract, RenderApp, RenderStage,
    },
    utils::FloatOrd,
};

pub struct OverlayPlugin;
impl Plugin for OverlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_startup_system(setup);

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<DrawFunctions<Overlay>>()
                .init_resource::<OverlayPipeline>()
                .init_resource::<SpecializedMeshPipelines<OverlayPipeline>>()
                .add_render_command::<Overlay, DrawOverlay>()
                .add_system_to_stage(RenderStage::Extract, extract_overlay_camera_phases)
                .add_system_to_stage(RenderStage::Queue, queue_overlay_bind_groups)
                .add_system_to_stage(RenderStage::Queue, queue_overlay_mesh);
        }
    }
}

fn setup(mut meshes: ResMut<Assets<Mesh>>) {
    let mesh: Mesh = Quad::new(Vec2::new(2.0, 2.0)).into();
    meshes.set_untracked(QUAD_HANDLE, mesh);
}

pub struct OverlayPipeline {
    pub overlay_layout: BindGroupLayout,
}

impl FromWorld for OverlayPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let overlay_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        Self { overlay_layout }
    }
}

impl SpecializedMeshPipeline for OverlayPipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayout,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let vertex_attributes = vec![Mesh::ATTRIBUTE_POSITION.at_shader_location(0)];
        let vertex_buffer_layout = layout.get_layout(&vertex_attributes)?;
        let bind_group_layout = vec![self.overlay_layout.clone()];

        Ok(RenderPipelineDescriptor {
            label: None,
            layout: Some(bind_group_layout),
            vertex: VertexState {
                shader: OVERLAY_SHADER_HANDLE.typed::<Shader>(),
                shader_defs: vec![],
                entry_point: "vertex".into(),
                buffers: vec![vertex_buffer_layout],
            },
            fragment: Some(FragmentState {
                shader: OVERLAY_SHADER_HANDLE.typed::<Shader>(),
                shader_defs: vec![],
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::bevy_default(),
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: key.primitive_topology(),
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: key.msaa_samples(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
        })
    }
}

fn extract_overlay_camera_phases(
    mut commands: Commands,
    cameras_3d: Extract<Query<(Entity, &Camera), With<Camera3d>>>,
) {
    for (entity, camera) in cameras_3d.iter() {
        if camera.is_active {
            commands
                .get_or_spawn(entity)
                .insert(RenderPhase::<Overlay>::default());
        }
    }
}

#[derive(Component)]
pub struct OverlayBindGroup(pub BindGroup);

fn queue_overlay_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline: Res<OverlayPipeline>,
    query: Query<(Entity, &LightPassTarget)>,
) {
    for (entity, target) in &query {
        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.overlay_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &target.direct_render_texture.texture_view,
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&target.direct_render_texture.sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(
                        &target.indirect_render_texture.texture_view,
                    ),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(&target.indirect_render_texture.sampler),
                },
            ],
        });
        commands.entity(entity).insert(OverlayBindGroup(bind_group));
    }
}

#[allow(clippy::too_many_arguments)]
fn queue_overlay_mesh(
    mut commands: Commands,
    msaa: Res<Msaa>,
    draw_functions: Res<DrawFunctions<Overlay>>,
    render_meshes: Res<RenderAssets<Mesh>>,
    overlay_pipeline: Res<OverlayPipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<OverlayPipeline>>,
    mut pipeline_cache: ResMut<PipelineCache>,
    mut views: Query<&mut RenderPhase<Overlay>>,
) {
    let draw_function = draw_functions.read().get_id::<DrawOverlay>().unwrap();
    for mut overlay_phase in &mut views {
        let mesh_handle = QUAD_HANDLE.typed::<Mesh>();
        if let Some(mesh) = render_meshes.get(&mesh_handle) {
            let key = MeshPipelineKey::from_msaa_samples(msaa.samples)
                | MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);
            let pipeline_id =
                pipelines.specialize(&mut pipeline_cache, &overlay_pipeline, key, &mesh.layout);
            let pipeline_id = match pipeline_id {
                Ok(id) => id,
                Err(err) => {
                    error!("{}", err);
                    return;
                }
            };
            let entity = commands.spawn().insert(mesh_handle.clone()).id();
            overlay_phase.add(Overlay {
                distance: 0.0,
                entity,
                pipeline: pipeline_id,
                draw_function,
            });
        }
    }
}

pub struct Overlay {
    pub distance: f32,
    pub entity: Entity,
    pub pipeline: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
}

impl PhaseItem for Overlay {
    type SortKey = FloatOrd;

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        FloatOrd(self.distance)
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }
}

impl EntityPhaseItem for Overlay {
    #[inline]
    fn entity(&self) -> Entity {
        self.entity
    }
}

impl CachedRenderPipelinePhaseItem for Overlay {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline
    }
}

type DrawOverlay = (SetItemPipeline, SetOverlayBindGroup<0>, DrawMesh);

pub struct SetOverlayBindGroup<const I: usize>;
impl<const I: usize> EntityRenderCommand for SetOverlayBindGroup<I> {
    type Param = SQuery<Read<OverlayBindGroup>>;

    fn render<'w>(
        view: Entity,
        _item: Entity,
        query: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let bind_group = query.get_inner(view).unwrap();
        pass.set_bind_group(I, &bind_group.0, &[]);

        RenderCommandResult::Success
    }
}

pub struct OverlayPassNode {
    query: QueryState<
        (
            &'static ExtractedCamera,
            &'static RenderPhase<Overlay>,
            &'static Camera3d,
            &'static ViewTarget,
        ),
        With<ExtractedView>,
    >,
}

impl OverlayPassNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            query: world.query_filtered(),
        }
    }
}

impl Node for OverlayPassNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(Self::IN_VIEW, SlotType::Entity)]
    }

    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
    }

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let entity = graph.get_input_entity(Self::IN_VIEW)?;
        let (camera, overlay_phase, camera_3d, target) = match self.query.get_manual(world, entity)
        {
            Ok(query) => query,
            Err(_) => return Ok(()),
        };

        {
            #[cfg(feature = "trace")]
            let _main_prepass_span = info_span!("main_prepass").entered();
            let pass_descriptor = RenderPassDescriptor {
                label: Some("main_prepass"),
                color_attachments: &[Some(target.get_color_attachment(Operations {
                    load: match camera_3d.clear_color {
                        ClearColorConfig::Default => {
                            LoadOp::Clear(world.resource::<ClearColor>().0.into())
                        }
                        ClearColorConfig::Custom(color) => LoadOp::Clear(color.into()),
                        ClearColorConfig::None => LoadOp::Load,
                    },
                    store: true,
                }))],
                depth_stencil_attachment: None,
            };

            let draw_functions = world.resource::<DrawFunctions<Overlay>>();

            let render_pass = render_context
                .command_encoder
                .begin_render_pass(&pass_descriptor);
            let mut draw_functions = draw_functions.write();
            let mut tracked_pass = TrackedRenderPass::new(render_pass);
            if let Some(viewport) = camera.viewport.as_ref() {
                tracked_pass.set_camera_viewport(viewport);
            }
            for item in &overlay_phase.items {
                let draw_function = draw_functions.get_mut(item.draw_function).unwrap();
                draw_function.draw(world, &mut tracked_pass, entity, item);
            }
        }

        Ok(())
    }
}
