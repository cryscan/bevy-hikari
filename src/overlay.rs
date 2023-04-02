use crate::{
    light::LightTextures, post_process::PostProcessTextures, prepass::PrepassBindGroup,
    HikariSettings, Taa, Upscale, OVERLAY_SHADER_HANDLE, QUAD_MESH_HANDLE,
};
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
            PhaseItem, RenderCommand, RenderCommandResult, RenderPhase, SetItemPipeline,
            TrackedRenderPass,
        },
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
        texture::BevyDefault,
        view::{ExtractedView, ViewTarget},
        Extract, RenderApp, RenderSet,
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
                .add_system(
                    extract_overlay_camera_phases
                        .in_schedule(ExtractSchedule)
                        .in_set(RenderSet::ExtractCommands),
                )
                .add_system(queue_overlay_meshes.in_set(RenderSet::Queue))
                .add_system(queue_overlay_bind_groups.in_set(RenderSet::Queue));
        }
    }
}

fn setup(mut meshes: ResMut<Assets<Mesh>>) {
    let mesh: Mesh = Quad::new(Vec2::new(2.0, 2.0)).into();
    meshes.set_untracked(QUAD_MESH_HANDLE, mesh);
}

#[derive(Resource)]
pub struct OverlayPipeline {
    pub input_layout: BindGroupLayout,
}

impl FromWorld for OverlayPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let input_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        Self { input_layout }
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
        let bind_group_layout = vec![self.input_layout.clone()];

        let mut shader_defs = vec![
            ShaderDefVal::Int("MAX_CASCADES_PER_LIGHT".into(), 0),
            ShaderDefVal::Int("MAX_DIRECTIONAL_LIGHTS".into(), 0),
        ];
        let mut format = TextureFormat::bevy_default();
        if key.contains(MeshPipelineKey::HDR) {
            shader_defs.push("HDR".into());
            format = ViewTarget::TEXTURE_FORMAT_HDR;
        }

        Ok(RenderPipelineDescriptor {
            label: None,
            layout: bind_group_layout,
            vertex: VertexState {
                shader: OVERLAY_SHADER_HANDLE.typed::<Shader>(),
                shader_defs: vec![],
                entry_point: "vertex".into(),
                buffers: vec![vertex_buffer_layout],
            },
            fragment: Some(FragmentState {
                shader: OVERLAY_SHADER_HANDLE.typed::<Shader>(),
                shader_defs,
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format,
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
            multisample: MultisampleState::default(),
            // TODO: Does this default value make sense?
            push_constant_ranges: default(),
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

#[allow(clippy::too_many_arguments)]
fn queue_overlay_meshes(
    mut commands: Commands,
    msaa: Res<Msaa>,
    draw_functions: Res<DrawFunctions<Overlay>>,
    render_meshes: Res<RenderAssets<Mesh>>,
    overlay_pipeline: Res<OverlayPipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<OverlayPipeline>>,
    mut pipeline_cache: ResMut<PipelineCache>,
    mut views: Query<(&mut RenderPhase<Overlay>, &ExtractedView)>,
) {
    let draw_function = draw_functions.read().get_id::<DrawOverlay>().unwrap();
    for (mut overlay_phase, view) in &mut views {
        let mesh_handle = QUAD_MESH_HANDLE.typed::<Mesh>();
        if let Some(mesh) = render_meshes.get(&mesh_handle) {
            let mut key = MeshPipelineKey::from_msaa_samples(msaa.samples())
                | MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);

            if view.hdr {
                key |= MeshPipelineKey::HDR;
            }

            let pipeline_id =
                pipelines.specialize(&mut pipeline_cache, &overlay_pipeline, key, &mesh.layout);
            let pipeline_id = match pipeline_id {
                Ok(id) => id,
                Err(err) => {
                    error!("{}", err);
                    return;
                }
            };
            let entity = commands.spawn(mesh_handle.clone()).id();
            overlay_phase.add(Overlay {
                distance: 0.0,
                entity,
                pipeline: pipeline_id,
                draw_function,
            });
        }
    }
}

#[derive(Component)]
pub struct OverlayBindGroup(pub BindGroup);

fn queue_overlay_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline: Res<OverlayPipeline>,
    query: Query<
        (
            Entity,
            &LightTextures,
            &PostProcessTextures,
            &HikariSettings,
        ),
        With<ExtractedCamera>,
    >,
) {
    for (entity, light, post_process, settings) in &query {
        let current = post_process.head;

        let input_texture = match (settings.upscale, settings.taa) {
            (Upscale::Fsr1 { .. }, _) => &post_process.upscale_output[1],
            (Upscale::SmaaTu4x { .. }, Taa::None) => &post_process.upscale_output[0],
            (Upscale::SmaaTu4x { .. }, Taa::Jasmine) => &post_process.taa_output[current],
        };

        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.input_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(input_texture),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&light.albedo),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&post_process.linear_sampler),
                },
            ],
        });
        commands.entity(entity).insert(OverlayBindGroup(bind_group));
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
impl<const I: usize, P: PhaseItem> RenderCommand<P> for SetOverlayBindGroup<I> {
    type Param = SQuery<Read<OverlayBindGroup>>;
    type ViewWorldQuery = Read<OverlayBindGroup>;
    type ItemWorldQuery = ();

    fn render<'w>(
        _item: &P,
        overlay_bind_group: &'w OverlayBindGroup,
        _entity: (),
        _param: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(0, &overlay_bind_group.0, &[]);
        RenderCommandResult::Success
    }
}

pub struct OverlayNode {
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

impl OverlayNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            query: world.query_filtered(),
        }
    }
}

impl Node for OverlayNode {
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

        if !world.contains_resource::<PrepassBindGroup>() {
            return Ok(());
        }

        {
            #[cfg(feature = "trace")]
            let _main_overlay_span = info_span!("main_overlay").entered();
            let pass_descriptor = RenderPassDescriptor {
                label: Some("main_overlay"),
                color_attachments: &[Some(target.get_unsampled_color_attachment(Operations {
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

            let render_device = render_context.render_device().clone();
            let render_pass = render_context
                .command_encoder()
                .begin_render_pass(&pass_descriptor);
            let mut draw_functions = draw_functions.write();
            let mut tracked_pass = TrackedRenderPass::new(&render_device, render_pass);
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
