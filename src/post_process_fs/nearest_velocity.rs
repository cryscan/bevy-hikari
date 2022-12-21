use super::{PostProcessSamplers, SamplerBindGroup, NEAREST_VELOCITY_SHADER_HANDLE};
use crate::{
    prepass::{DeferredBindGroup, PrepassBindGroups, PrepassPipeline, PrepassTextures},
    view::{FrameUniform, PreviousViewUniformOffset},
    HikariSettings,
};
use bevy::{
    core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state,
    pbr::ViewLightsUniformOffset,
    prelude::*,
    render::{
        camera::ExtractedCamera,
        extract_component::DynamicUniformIndex,
        render_graph::{Node, NodeRunError, SlotInfo, SlotType},
        render_phase::TrackedRenderPass,
        render_resource::*,
        renderer::RenderDevice,
        texture::TextureCache,
        view::ViewUniformOffset,
    },
};

pub const NEAREST_VELOCITY_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rg32Float;

pub struct NearestVelocityNode {
    query: QueryState<(
        &'static ExtractedCamera,
        &'static DynamicUniformIndex<FrameUniform>,
        &'static ViewUniformOffset,
        &'static PreviousViewUniformOffset,
        &'static ViewLightsUniformOffset,
        &'static NearestVelocityTexture,
        &'static DeferredBindGroup,
    )>,
}

impl NearestVelocityNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl Node for NearestVelocityNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(Self::IN_VIEW, SlotType::Entity)]
    }

    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
    }

    fn run(
        &self,
        graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline = world.resource::<NearestVelocityPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let samplers_bind_group = match world.get_resource::<SamplerBindGroup>() {
            Some(bind_group) => &bind_group.0,
            None => return Ok(()),
        };
        let view_bind_group = match world.get_resource::<PrepassBindGroups>() {
            Some(bind_group) => &bind_group.view,
            None => return Ok(()),
        };

        let view_entity = graph.get_input_entity(Self::IN_VIEW)?;
        let (
            camera,
            frame_uniform,
            view_uniform,
            previous_view_uniform,
            view_lights,
            texture,
            deferred_bind_group,
        ) = match self.query.get_manual(world, view_entity) {
            Ok(result) => result,
            Err(_) => return Ok(()),
        };

        let view_bind_group_indices = [
            frame_uniform.index(),
            view_uniform.offset,
            previous_view_uniform.offset,
            view_lights.offset,
        ];

        if let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline.0) {
            let mut pass =
                TrackedRenderPass::new(render_context.command_encoder.begin_render_pass(
                    &RenderPassDescriptor {
                        label: Some("hikari_nearest_velocity_pass"),
                        color_attachments: &[Some(RenderPassColorAttachment {
                            view: &texture.0,
                            resolve_target: None,
                            ops: Operations::default(),
                        })],
                        depth_stencil_attachment: None,
                    },
                ));
            pass.set_render_pipeline(pipeline);
            pass.set_bind_group(0, view_bind_group, &view_bind_group_indices);
            pass.set_bind_group(1, &deferred_bind_group.0, &[]);
            pass.set_bind_group(2, &samplers_bind_group, &[]);
            if let Some(viewport) = camera.viewport.as_ref() {
                pass.set_camera_viewport(viewport);
            }
            pass.draw(0..3, 0..1);
        }

        Ok(())
    }
}

#[derive(Resource)]
pub struct NearestVelocityPipeline(pub CachedRenderPipelineId);

impl FromWorld for NearestVelocityPipeline {
    fn from_world(world: &mut World) -> Self {
        let view_layout = world.resource::<PrepassPipeline>().view_layout.clone();

        let render_device = world.resource::<RenderDevice>();
        let deferred_layout = PrepassTextures::bind_group_layout(render_device);

        let sampler_layout = PostProcessSamplers::bind_group_layout(render_device);

        let mut pipeline_cache = world.resource_mut::<PipelineCache>();
        let pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: None,
            layout: Some(vec![
                view_layout.clone(),
                deferred_layout.clone(),
                sampler_layout.clone(),
            ]),
            vertex: fullscreen_shader_vertex_state(),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                shader: NEAREST_VELOCITY_SHADER_HANDLE.typed(),
                shader_defs: vec![],
                entry_point: "nearest_velocity".into(),
                targets: vec![Some(ColorTargetState {
                    format: NEAREST_VELOCITY_TEXTURE_FORMAT,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
        });

        Self(pipeline)
    }
}

#[derive(Component)]
pub struct NearestVelocityTexture(pub TextureView);

pub(super) fn prepare_nearest_velocity_texture(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    cameras: Query<(Entity, &ExtractedCamera), With<HikariSettings>>,
) {
    for (entity, camera) in &cameras {
        if let Some(size) = camera.physical_target_size {
            let extent = Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            };
            let view = texture_cache
                .get(
                    &render_device,
                    TextureDescriptor {
                        label: None,
                        size: extent,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: TextureDimension::D2,
                        format: NEAREST_VELOCITY_TEXTURE_FORMAT,
                        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
                    },
                )
                .default_view;
            commands.entity(entity).insert(NearestVelocityTexture(view));
        }
    }
}
