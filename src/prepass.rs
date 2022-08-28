use crate::PREPASS_SHADER_HANDLE;
use bevy::{
    pbr::{MeshPipeline, ShadowPipelineKey, SHADOW_FORMAT},
    prelude::*,
    render::{
        camera::ExtractedCamera,
        mesh::MeshVertexBufferLayout,
        render_graph::{Node, NodeRunError, RenderGraphContext},
        render_phase::{
            CachedRenderPipelinePhaseItem, DrawFunctionId, DrawFunctions, EntityPhaseItem,
            PhaseItem, RenderPhase, TrackedRenderPass,
        },
        render_resource::{
            BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
            BufferBindingType, CachedRenderPipelineId, ColorTargetState, ColorWrites,
            CompareFunction, DepthBiasState, DepthStencilState, Extent3d, FragmentState, FrontFace,
            LoadOp, MultisampleState, Operations, PolygonMode, PrimitiveState,
            RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor,
            RenderPipelineDescriptor, ShaderStages, ShaderType, SpecializedMeshPipeline,
            SpecializedMeshPipelineError, SpecializedMeshPipelines, StencilFaceState, StencilState,
            TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
            VertexState,
        },
        renderer::{RenderContext, RenderDevice},
        texture::TextureCache,
        view::{ExtractedView, ViewUniform},
        Extract, RenderApp, RenderStage,
    },
    utils::FloatOrd,
};

pub struct PrepassPlugin;
impl Plugin for PrepassPlugin {
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<DrawFunctions<Prepass>>()
                .init_resource::<PrepassPipeline>()
                .init_resource::<SpecializedMeshPipelines<PrepassPipeline>>()
                .add_system_to_stage(RenderStage::Extract, extract_prepass_camera_phases)
                .add_system_to_stage(RenderStage::Prepare, prepare_prepass_targets)
                .add_system_to_stage(RenderStage::Queue, queue_prepass_meshes);
        }
    }
}

pub struct PrepassPipeline {
    pub view_layout: BindGroupLayout,
    pub mesh_layout: BindGroupLayout,
}

impl FromWorld for PrepassPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let view_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(ViewUniform::min_size()),
                },
                count: None,
            }],
        });

        let mesh_pipeline = world.resource::<MeshPipeline>();
        let mesh_layout = mesh_pipeline.mesh_layout.clone();

        Self {
            view_layout,
            mesh_layout,
        }
    }
}

impl SpecializedMeshPipeline for PrepassPipeline {
    type Key = ShadowPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayout,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let vertex_attributes = vec![
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
        ];
        let bind_group_layout = vec![self.view_layout.clone(), self.mesh_layout.clone()];

        let vertex_buffer_layout = layout.get_layout(&vertex_attributes)?;

        Ok(RenderPipelineDescriptor {
            label: None,
            layout: Some(bind_group_layout),
            vertex: VertexState {
                shader: PREPASS_SHADER_HANDLE.typed::<Shader>(),
                shader_defs: vec![],
                entry_point: "vertex".into(),
                buffers: vec![vertex_buffer_layout],
            },
            fragment: Some(FragmentState {
                shader: PREPASS_SHADER_HANDLE.typed::<Shader>(),
                shader_defs: vec![],
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rg16Float,
                    blend: None,
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
            depth_stencil: Some(DepthStencilState {
                format: SHADOW_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::GreaterEqual,
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: MultisampleState::default(),
        })
    }
}

#[derive(Component)]
pub struct PrepassTarget {
    pub color_view: TextureView,
    pub depth_view: TextureView,
}

fn extract_prepass_camera_phases(
    mut commands: Commands,
    cameras_3d: Extract<Query<(Entity, &Camera), With<Camera3d>>>,
) {
    for (entity, camera) in cameras_3d.iter() {
        if camera.is_active {
            commands
                .get_or_spawn(entity)
                .insert(RenderPhase::<Prepass>::default());
        }
    }
}

fn prepare_prepass_targets(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    cameras: Query<(Entity, &ExtractedCamera), With<RenderPhase<Prepass>>>,
) {
    for (entity, camera) in &cameras {
        if let Some(target_size) = camera.physical_target_size {
            let size = Extent3d {
                width: target_size.x,
                height: target_size.y,
                depth_or_array_layers: 1,
            };

            let color_view = texture_cache
                .get(
                    &render_device,
                    TextureDescriptor {
                        label: Some("prepass_color_attachment_texture"),
                        size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: TextureDimension::D2,
                        format: TextureFormat::Rg16Float,
                        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
                    },
                )
                .default_view;

            let depth_view = texture_cache
                .get(
                    &render_device,
                    TextureDescriptor {
                        label: Some("prepass_depth_stencil_attachment_texture"),
                        size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: TextureDimension::D2,
                        format: SHADOW_FORMAT,
                        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
                    },
                )
                .default_view;

            commands.entity(entity).insert(PrepassTarget {
                color_view,
                depth_view,
            });
        }
    }
}

fn queue_prepass_meshes() {}

pub struct Prepass {
    pub distance: f32,
    pub entity: Entity,
    pub pipeline: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
}

impl PhaseItem for Prepass {
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

impl EntityPhaseItem for Prepass {
    #[inline]
    fn entity(&self) -> Entity {
        self.entity
    }
}

impl CachedRenderPipelinePhaseItem for Prepass {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline
    }
}

pub struct PrepassNode {
    query: QueryState<
        (
            &'static ExtractedCamera,
            &'static RenderPhase<Prepass>,
            &'static Camera3d,
            &'static PrepassTarget,
        ),
        With<ExtractedView>,
    >,
}

impl PrepassNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            query: world.query_filtered(),
        }
    }
}

impl Node for PrepassNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let entity = graph.get_input_entity(Self::IN_VIEW)?;
        let (camera, prepass_phase, camera_3d, target) = match self.query.get_manual(world, entity)
        {
            Ok(query) => query,
            Err(_) => return Ok(()),
        };

        {
            #[cfg(feature = "trace")]
            let _main_prepass_span = info_span!("main_prepass").entered();
            let pass_descriptor = RenderPassDescriptor {
                label: Some("main_prepass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &target.color_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::NONE.into()),
                        store: true,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(Operations {
                        load: camera_3d.depth_load_op.clone().into(),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            };

            let draw_functions = world.resource::<DrawFunctions<Prepass>>();

            let render_pass = render_context
                .command_encoder
                .begin_render_pass(&pass_descriptor);
            let mut draw_functions = draw_functions.write();
            let mut tracked_pass = TrackedRenderPass::new(render_pass);
            if let Some(viewport) = camera.viewport.as_ref() {
                tracked_pass.set_camera_viewport(viewport);
            }
            for item in &prepass_phase.items {
                let draw_function = draw_functions.get_mut(item.draw_function).unwrap();
                draw_function.draw(world, &mut tracked_pass, entity, item);
            }
        }

        Ok(())
    }
}
