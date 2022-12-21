use super::{PostProcessSamplers, SMAA_SHADER_HANDLE};
use crate::{
    prepass::{PrepassPipeline, PrepassTextures},
    view::{FrameUniform, PreviousViewUniformOffset},
};
use bevy::{
    core_pipeline::fullscreen_vertex_shader::{self, fullscreen_shader_vertex_state},
    pbr::ViewLightsUniformOffset,
    prelude::*,
    render::{
        camera::ExtractedCamera,
        extract_component::DynamicUniformIndex,
        render_resource::*,
        renderer::RenderDevice,
        view::{ViewTarget, ViewUniformOffset},
    },
};

pub struct SmaaNode {
    query: QueryState<(
        &'static ExtractedCamera,
        &'static ViewTarget,
        &'static DynamicUniformIndex<FrameUniform>,
        &'static ViewUniformOffset,
        &'static PreviousViewUniformOffset,
        &'static ViewLightsUniformOffset,
    )>,
}

impl SmaaNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

#[derive(Resource)]
pub struct SmaaPipelines {
    pub smaa_pipeline: CachedComputePipelineId,
    pub smaa_extrapolation_pipeline: CachedComputePipelineId,
    pub blit_pipeline: CachedRenderPipelineId,
    pub smaa_layout: BindGroupLayout,
    pub smaa_output_layout: BindGroupLayout,
    pub blit_layout: BindGroupLayout,
}

impl FromWorld for SmaaPipelines {
    fn from_world(world: &mut World) -> Self {
        let view_layout = world.resource::<PrepassPipeline>().view_layout.clone();

        let render_device = world.resource::<RenderDevice>();
        let deferred_layout = PrepassTextures::bind_group_layout(render_device);

        let sampler_layout = PostProcessSamplers::bind_group_layout(render_device);

        let smaa_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Previous Render
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Render
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Nearest Velocity
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let smaa_output_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadWrite,
                        format: ViewTarget::TEXTURE_FORMAT_HDR,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });

        let blit_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });

        let mut pipeline_cache = world.resource_mut::<PipelineCache>();

        let smaa_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![
                view_layout.clone(),
                deferred_layout.clone(),
                sampler_layout.clone(),
                smaa_layout.clone(),
                smaa_output_layout.clone(),
            ]),
            shader: SMAA_SHADER_HANDLE.typed(),
            shader_defs: vec![],
            entry_point: "smaa_tu4x".into(),
        });
        let smaa_extrapolation_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: None,
                layout: Some(vec![
                    view_layout.clone(),
                    deferred_layout.clone(),
                    sampler_layout.clone(),
                    smaa_layout.clone(),
                    smaa_output_layout.clone(),
                ]),
                shader: SMAA_SHADER_HANDLE.typed(),
                shader_defs: vec![],
                entry_point: "smaa_tu4x_extrapolation".into(),
            });

        let blit_pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: None,
            layout: Some(vec![
                view_layout.clone(),
                deferred_layout.clone(),
                sampler_layout.clone(),
                blit_layout.clone(),
            ]),
            vertex: fullscreen_shader_vertex_state(),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                shader: SMAA_SHADER_HANDLE.typed(),
                shader_defs: vec!["BLIT".into()],
                entry_point: "blit".into(),
                targets: vec![Some(ColorTargetState {
                    format: ViewTarget::TEXTURE_FORMAT_HDR,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
        });

        Self {
            smaa_pipeline,
            smaa_extrapolation_pipeline,
            blit_pipeline,
            smaa_layout,
            smaa_output_layout,
            blit_layout,
        }
    }
}
