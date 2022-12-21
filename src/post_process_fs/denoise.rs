use super::{
    PostProcessSamplers, SamplerBindGroup, DEMODULATION_SHADER_HANDLE, DENOISE_SHADER_HANDLE,
    MODULATION_SHADER_HANDLE,
};
use crate::{
    light::{LightTextures, VARIANCE_TEXTURE_FORMAT},
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
        render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
        render_phase::TrackedRenderPass,
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
        texture::TextureCache,
        view::{ViewTarget, ViewUniformOffset},
    },
};

pub struct DenoiseNode {
    query: QueryState<(
        &'static ExtractedCamera,
        &'static ViewTarget,
        &'static DynamicUniformIndex<FrameUniform>,
        &'static ViewUniformOffset,
        &'static PreviousViewUniformOffset,
        &'static ViewLightsUniformOffset,
        &'static DenoiseTextures,
        &'static DeferredBindGroup,
        &'static DenoiseBindGroups,
    )>,
}

impl DenoiseNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl Node for DenoiseNode {
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
        let pipelines = world.resource::<DenoisePipelines>();
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
            view_target,
            frame_uniform,
            view_uniform,
            previous_view_uniform,
            view_lights,
            textures,
            deferred_bind_group,
            denoise_bind_groups,
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

        for id in 0..3 {
            if let Some(pipeline) =
                pipeline_cache.get_render_pipeline(pipelines.demodulation_pipeline)
            {
                let mut pass =
                    TrackedRenderPass::new(render_context.command_encoder.begin_render_pass(
                        &RenderPassDescriptor {
                            label: Some("hikari_demodulation_pass"),
                            color_attachments: &[
                                Some(RenderPassColorAttachment {
                                    view: &textures.radiance[0],
                                    resolve_target: None,
                                    ops: Operations::default(),
                                }),
                                Some(RenderPassColorAttachment {
                                    view: &textures.variance,
                                    resolve_target: None,
                                    ops: Operations::default(),
                                }),
                            ],
                            depth_stencil_attachment: None,
                        },
                    ));
                pass.set_render_pipeline(pipeline);
                pass.set_bind_group(0, view_bind_group, &view_bind_group_indices);
                pass.set_bind_group(1, &deferred_bind_group.0, &[]);
                pass.set_bind_group(2, &samplers_bind_group, &[]);
                pass.set_bind_group(3, &denoise_bind_groups.demodulation[id], &[]);
                if let Some(viewport) = camera.viewport.as_ref() {
                    pass.set_camera_viewport(viewport);
                }
                pass.draw(0..3, 0..1);
            }

            for (level, pipeline) in (0..4)
                .filter_map(|level| {
                    pipeline_cache.get_render_pipeline(pipelines.denoise_pipelines[id][level])
                })
                .enumerate()
            {
                let mut pass =
                    TrackedRenderPass::new(render_context.command_encoder.begin_render_pass(
                        &RenderPassDescriptor {
                            label: Some("hikari_denoise_pass"),
                            color_attachments: &[Some(RenderPassColorAttachment {
                                view: &textures.radiance[level + 1],
                                resolve_target: None,
                                ops: match (id, level) {
                                    (id, 3) if id > 0 => Operations {
                                        load: LoadOp::Load,
                                        store: true,
                                    },
                                    _ => Operations::default(),
                                },
                            })],
                            depth_stencil_attachment: None,
                        },
                    ));
                pass.set_render_pipeline(pipeline);
                pass.set_bind_group(0, view_bind_group, &view_bind_group_indices);
                pass.set_bind_group(1, &deferred_bind_group.0, &[]);
                pass.set_bind_group(2, &samplers_bind_group, &[]);
                pass.set_bind_group(3, &denoise_bind_groups.denoise[level], &[]);
                if let Some(viewport) = camera.viewport.as_ref() {
                    pass.set_camera_viewport(viewport);
                }
                pass.draw(0..3, 0..1);
            }
        }

        if let Some(pipeline) = pipeline_cache.get_render_pipeline(pipelines.modulate_pipeline) {
            let mut pass =
                TrackedRenderPass::new(render_context.command_encoder.begin_render_pass(
                    &RenderPassDescriptor {
                        label: Some("hikari_modulation_pass"),
                        color_attachments: &[Some(
                            view_target.get_unsampled_color_attachment(Operations::default()),
                        )],
                        depth_stencil_attachment: None,
                    },
                ));
            pass.set_render_pipeline(pipeline);
            pass.set_bind_group(0, view_bind_group, &view_bind_group_indices);
            pass.set_bind_group(1, &deferred_bind_group.0, &[]);
            pass.set_bind_group(2, &samplers_bind_group, &[]);
            pass.set_bind_group(3, &denoise_bind_groups.modulation, &[]);
            if let Some(viewport) = camera.viewport.as_ref() {
                pass.set_camera_viewport(viewport);
            }
            pass.draw(0..3, 0..1);
        }

        Ok(())
    }
}

#[derive(Component)]
pub struct DenoiseTextures {
    pub radiance: [TextureView; 5],
    pub variance: TextureView,
}

pub(super) fn prepare_denoise_textures(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    cameras: Query<(Entity, &ExtractedCamera, &HikariSettings)>,
) {
    for (entity, camera, settings) in &cameras {
        if let Some(size) = camera.physical_target_size {
            let mut create_texture = |texture_format, scale: f32| {
                let extent = Extent3d {
                    width: (size.x as f32 * scale).ceil() as u32,
                    height: (size.y as f32 * scale).ceil() as u32,
                    depth_or_array_layers: 1,
                };
                texture_cache
                    .get(
                        &render_device,
                        TextureDescriptor {
                            label: None,
                            size: extent,
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: TextureDimension::D2,
                            format: texture_format,
                            usage: TextureUsages::TEXTURE_BINDING
                                | TextureUsages::RENDER_ATTACHMENT,
                        },
                    )
                    .default_view
            };

            macro_rules! create_texture_array {
                [$texture_format:expr, $scale:ident; $count:literal] => {
                    [(); $count].map(|_| create_texture($texture_format, $scale))
                };
            }

            let scale = settings.upscale.ratio().recip();
            let radiance = create_texture_array![ViewTarget::TEXTURE_FORMAT_HDR, scale; 5];
            let variance = create_texture(VARIANCE_TEXTURE_FORMAT, scale);

            commands
                .entity(entity)
                .insert(DenoiseTextures { radiance, variance });
        }
    }
}

#[derive(Component)]
pub struct DenoiseBindGroups {
    pub demodulation: [BindGroup; 3],
    pub denoise: [BindGroup; 4],
    pub modulation: BindGroup,
}

pub(super) fn queue_denoise_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipelines: Res<DenoisePipelines>,
    cameras: Query<(Entity, &LightTextures, &DenoiseTextures)>,
) {
    for (entity, light, denoise) in &cameras {
        let demodulation = [0, 1, 2].map(|id| {
            render_device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipelines.demodulation_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&light.render[id]),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&light.albedo[light.head]),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(&light.variance[id]),
                    },
                ],
            })
        });

        let modulation = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipelines.modulation_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&denoise.radiance[4]),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&light.albedo[light.head]),
                },
            ],
        });

        let denoise = [0, 1, 2, 3].map(|level| {
            render_device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipelines.denoise_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&denoise.radiance[level]),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&denoise.variance),
                    },
                ],
            })
        });

        commands.entity(entity).insert(DenoiseBindGroups {
            demodulation,
            denoise,
            modulation,
        });
    }
}

#[derive(Resource)]
pub struct DenoisePipelines {
    pub demodulation_pipeline: CachedRenderPipelineId,
    pub denoise_pipelines: [[CachedRenderPipelineId; 4]; 3],
    pub modulate_pipeline: CachedRenderPipelineId,
    pub demodulation_layout: BindGroupLayout,
    pub denoise_layout: BindGroupLayout,
    pub modulation_layout: BindGroupLayout,
}

impl FromWorld for DenoisePipelines {
    fn from_world(world: &mut World) -> Self {
        let view_layout = world.resource::<PrepassPipeline>().view_layout.clone();

        let render_device = world.resource::<RenderDevice>();
        let deferred_layout = PrepassTextures::bind_group_layout(render_device);

        let sampler_layout = PostProcessSamplers::bind_group_layout(render_device);

        let demodulation_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    // Render
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
                    // Albedo
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
                    // Variance
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
                ],
            });

        let denoise_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Previous Level Radiance
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
                // Filtered Variance
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
            ],
        });

        let modulation_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    // Radiance
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
                    // Albedo
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
                ],
            });

        let mut pipeline_cache = world.resource_mut::<PipelineCache>();

        let demodulation_pipeline =
            pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
                label: None,
                layout: Some(vec![
                    view_layout.clone(),
                    deferred_layout.clone(),
                    sampler_layout.clone(),
                    demodulation_layout.clone(),
                ]),
                vertex: fullscreen_shader_vertex_state(),
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    shader: DEMODULATION_SHADER_HANDLE.typed(),
                    shader_defs: vec![],
                    entry_point: "demodulation".into(),
                    targets: vec![
                        Some(ColorTargetState {
                            format: ViewTarget::TEXTURE_FORMAT_HDR,
                            blend: None,
                            write_mask: ColorWrites::ALL,
                        }),
                        Some(ColorTargetState {
                            format: VARIANCE_TEXTURE_FORMAT,
                            blend: None,
                            write_mask: ColorWrites::ALL,
                        }),
                    ],
                }),
            });

        let direct_denoise_pipelines = [0, 1, 2, 3].map(|level| {
            let shader_defs = vec![format!("DENOISE_LEVEL_{}", level)];
            let blend = match level {
                3 => Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent::REPLACE,
                }),
                _ => Some(BlendState::REPLACE),
            };

            pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
                label: None,
                layout: Some(vec![
                    view_layout.clone(),
                    deferred_layout.clone(),
                    sampler_layout.clone(),
                    denoise_layout.clone(),
                ]),
                vertex: fullscreen_shader_vertex_state(),
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    shader: DENOISE_SHADER_HANDLE.typed(),
                    shader_defs,
                    entry_point: "denoise".into(),
                    targets: vec![Some(ColorTargetState {
                        format: ViewTarget::TEXTURE_FORMAT_HDR,
                        blend,
                        write_mask: ColorWrites::ALL,
                    })],
                }),
            })
        });

        let emissive_denoise_pipelines = [0, 1, 2, 3].map(|level| {
            let shader_defs = vec![
                format!("DENOISE_LEVEL_{}", level),
                "FIREFLY_FILTERING".into(),
            ];
            let blend = match level {
                3 => Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent::REPLACE,
                }),
                _ => Some(BlendState::REPLACE),
            };

            pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
                label: None,
                layout: Some(vec![
                    view_layout.clone(),
                    deferred_layout.clone(),
                    sampler_layout.clone(),
                    denoise_layout.clone(),
                ]),
                vertex: fullscreen_shader_vertex_state(),
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    shader: DENOISE_SHADER_HANDLE.typed(),
                    shader_defs,
                    entry_point: "denoise".into(),
                    targets: vec![Some(ColorTargetState {
                        format: ViewTarget::TEXTURE_FORMAT_HDR,
                        blend,
                        write_mask: ColorWrites::ALL,
                    })],
                }),
            })
        });

        let indirect_denoise_pipelines = emissive_denoise_pipelines;

        let modulation_pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: None,
            layout: Some(vec![
                view_layout.clone(),
                deferred_layout.clone(),
                sampler_layout.clone(),
                modulation_layout.clone(),
            ]),
            vertex: fullscreen_shader_vertex_state(),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                shader: MODULATION_SHADER_HANDLE.typed(),
                shader_defs: vec![],
                entry_point: "modulation".into(),
                targets: vec![Some(ColorTargetState {
                    format: ViewTarget::TEXTURE_FORMAT_HDR,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
        });

        Self {
            demodulation_pipeline,
            denoise_pipelines: [
                direct_denoise_pipelines,
                emissive_denoise_pipelines,
                indirect_denoise_pipelines,
            ],
            modulate_pipeline: modulation_pipeline,
            demodulation_layout,
            denoise_layout,
            modulation_layout,
        }
    }
}
