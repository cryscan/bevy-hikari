use crate::{
    light::LightPassTextures,
    prepass::{PrepassBindGroup, PrepassPipeline, PrepassTextures},
    view::{FrameCounter, PreviousViewUniformOffset},
    HikariConfig, DENOISE_SHADER_HANDLE, TAA_SHADER_HANDLE, TONE_MAPPING_SHADER_HANDLE,
    WORKGROUP_SIZE,
};
use bevy::{
    pbr::ViewLightsUniformOffset,
    prelude::*,
    render::{
        camera::ExtractedCamera,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
        texture::{CachedTexture, FallbackImage, TextureCache},
        view::ViewUniformOffset,
        RenderApp, RenderStage,
    },
};
use serde::Serialize;

pub const DENOISE_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
pub const POST_PROCESS_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

pub struct PostProcessPlugin;
impl Plugin for PostProcessPlugin {
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<PostProcessPipeline>()
                .init_resource::<SpecializedComputePipelines<PostProcessPipeline>>()
                .add_system_to_stage(RenderStage::Prepare, prepare_post_process_textures)
                .add_system_to_stage(RenderStage::Queue, queue_post_process_pipelines)
                .add_system_to_stage(RenderStage::Queue, queue_post_process_bind_groups);
        }
    }
}

pub struct PostProcessPipeline {
    pub view_layout: BindGroupLayout,
    pub deferred_layout: BindGroupLayout,
    pub sampler_layout: BindGroupLayout,
    pub denoise_internal_layout: BindGroupLayout,
    pub denoise_render_layout: BindGroupLayout,
    pub tone_mapping_layout: BindGroupLayout,
    pub taa_layout: BindGroupLayout,
    pub output_layout: BindGroupLayout,
}

impl FromWorld for PostProcessPipeline {
    fn from_world(world: &mut World) -> Self {
        let view_layout = world.resource::<PrepassPipeline>().view_layout.clone();

        let render_device = world.resource::<RenderDevice>();
        let deferred_layout = PrepassTextures::bind_group_layout(render_device);

        let sampler_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let denoise_internal_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    // Albedo
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
                    // Internal 0
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::ReadWrite,
                            format: DENOISE_TEXTURE_FORMAT,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Internal 1
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::ReadWrite,
                            format: DENOISE_TEXTURE_FORMAT,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
        let denoise_render_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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
                    // Output
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::ReadWrite,
                            format: DENOISE_TEXTURE_FORMAT,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let tone_mapping_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    // Direct Render
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
                    // Emissive Render
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
                    // Indirect Render
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

        let taa_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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
                // Current Render
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
                // TAA Output
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::all(),
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadWrite,
                        format: POST_PROCESS_TEXTURE_FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let output_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::all(),
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: POST_PROCESS_TEXTURE_FORMAT,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        Self {
            view_layout,
            deferred_layout,
            sampler_layout,
            denoise_internal_layout,
            denoise_render_layout,
            tone_mapping_layout,
            taa_layout,
            output_layout,
        }
    }
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, FromPrimitive)]
#[serde(rename_all = "snake_case")]
pub enum PostProcessEntryPoint {
    #[default]
    ToneMapping = 0,
    Taa = 1,
    JasmineTaa = 2,
    Denoise = 3,
}

bitflags::bitflags! {
    #[repr(transparent)]
    pub struct PostProcessPipelineKey: u32 {
        const ENTRY_POINT_BITS = PostProcessPipelineKey::ENTRY_POINT_MASK_BITS;
        const DENOISE_LEVEL_BITS = PostProcessPipelineKey::DENOISE_LEVEL_MASK_BITS << PostProcessPipelineKey::DENOISE_LEVEL_SHIFT_BITS;
    }
}

impl PostProcessPipelineKey {
    const ENTRY_POINT_MASK_BITS: u32 = 0b11;
    const DENOISE_LEVEL_MASK_BITS: u32 = 0b11;
    const DENOISE_LEVEL_SHIFT_BITS: u32 = 32 - 2;

    pub fn from_entry_point(entry_point: PostProcessEntryPoint) -> Self {
        let entry_point_bits = (entry_point as u32) & Self::ENTRY_POINT_MASK_BITS;
        Self::from_bits(entry_point_bits).unwrap()
    }

    pub fn entry_point(&self) -> PostProcessEntryPoint {
        let entry_point_bits = self.bits & Self::ENTRY_POINT_MASK_BITS;
        num_traits::FromPrimitive::from_u32(entry_point_bits).unwrap()
    }

    pub fn from_denoise_level(level: u32) -> Self {
        let denoise_level_bits =
            (level & Self::DENOISE_LEVEL_MASK_BITS) << Self::DENOISE_LEVEL_SHIFT_BITS;
        Self::from_bits(denoise_level_bits).unwrap()
    }

    pub fn denoise_level(&self) -> u32 {
        (self.bits >> Self::DENOISE_LEVEL_SHIFT_BITS) & Self::DENOISE_LEVEL_MASK_BITS
    }
}

impl SpecializedComputePipeline for PostProcessPipeline {
    type Key = PostProcessPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        let entry_point = serde_variant::to_variant_name(&key.entry_point())
            .unwrap()
            .into();

        let (layout, shader) = match key.entry_point() {
            PostProcessEntryPoint::Denoise => {
                let layout = vec![
                    self.view_layout.clone(),
                    self.deferred_layout.clone(),
                    self.sampler_layout.clone(),
                    self.denoise_internal_layout.clone(),
                    self.denoise_render_layout.clone(),
                ];
                let shader = DENOISE_SHADER_HANDLE.typed();
                (layout, shader)
            }
            PostProcessEntryPoint::ToneMapping => {
                let layout = vec![
                    self.view_layout.clone(),
                    self.deferred_layout.clone(),
                    self.sampler_layout.clone(),
                    self.tone_mapping_layout.clone(),
                    self.output_layout.clone(),
                ];
                let shader = TONE_MAPPING_SHADER_HANDLE.typed();
                (layout, shader)
            }
            PostProcessEntryPoint::Taa | PostProcessEntryPoint::JasmineTaa => {
                let layout = vec![
                    self.view_layout.clone(),
                    self.deferred_layout.clone(),
                    self.sampler_layout.clone(),
                    self.taa_layout.clone(),
                    self.output_layout.clone(),
                ];
                let shader = TAA_SHADER_HANDLE.typed();
                (layout, shader)
            }
        };

        ComputePipelineDescriptor {
            label: None,
            layout: Some(layout),
            shader,
            shader_defs: vec![],
            entry_point,
        }
    }
}

#[derive(Component)]
pub struct PostProcessTextures {
    pub head: usize,
    pub denoise_internal: [CachedTexture; 2],
    pub denoise_render: [CachedTexture; 6],
    pub tone_mapping_output: CachedTexture,
    pub taa_internal: [CachedTexture; 2],
    pub taa_output: CachedTexture,
    pub nearest_sampler: Sampler,
    pub linear_sampler: Sampler,
}

fn prepare_post_process_textures(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    frame_counter: Res<FrameCounter>,
    mut texture_cache: ResMut<TextureCache>,
    cameras: Query<(Entity, &ExtractedCamera)>,
) {
    for (entity, camera) in &cameras {
        if let Some(size) = camera.physical_target_size {
            let texture_usage = TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
            let mut create_texture = |texture_format| {
                let extent = Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: 1,
                };
                texture_cache.get(
                    &render_device,
                    TextureDescriptor {
                        label: None,
                        size: extent,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: TextureDimension::D2,
                        format: texture_format,
                        usage: texture_usage,
                    },
                )
            };

            let denoise_internal = [
                create_texture(DENOISE_TEXTURE_FORMAT),
                create_texture(DENOISE_TEXTURE_FORMAT),
            ];
            let denoise_render = [(); 6].map(|_| create_texture(DENOISE_TEXTURE_FORMAT));

            let tone_mapping_output = create_texture(POST_PROCESS_TEXTURE_FORMAT);

            let taa_internal = [
                create_texture(POST_PROCESS_TEXTURE_FORMAT),
                create_texture(POST_PROCESS_TEXTURE_FORMAT),
            ];
            let taa_output = create_texture(POST_PROCESS_TEXTURE_FORMAT);

            let nearest_sampler = render_device.create_sampler(&SamplerDescriptor {
                mag_filter: FilterMode::Nearest,
                min_filter: FilterMode::Nearest,
                mipmap_filter: FilterMode::Nearest,
                ..Default::default()
            });
            let linear_sampler = render_device.create_sampler(&SamplerDescriptor {
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: FilterMode::Linear,
                ..Default::default()
            });

            commands.entity(entity).insert(PostProcessTextures {
                head: frame_counter.0 % 2,
                denoise_internal,
                denoise_render,
                tone_mapping_output,
                taa_internal,
                taa_output,
                nearest_sampler,
                linear_sampler,
            });
        }
    }
}

pub struct CachedPostProcessPipelines {
    denoise: [CachedComputePipelineId; 4],
    tone_mapping: CachedComputePipelineId,
    taa: CachedComputePipelineId,
    taa_jasmine: CachedComputePipelineId,
}

fn queue_post_process_pipelines(
    mut commands: Commands,
    pipeline: Res<PostProcessPipeline>,
    mut pipelines: ResMut<SpecializedComputePipelines<PostProcessPipeline>>,
    mut pipeline_cache: ResMut<PipelineCache>,
) {
    let denoise = [0, 1, 2, 3].map(|level| {
        let mut key = PostProcessPipelineKey::from_entry_point(PostProcessEntryPoint::Denoise);
        key |= PostProcessPipelineKey::from_denoise_level(level);
        pipelines.specialize(&mut pipeline_cache, &pipeline, key)
    });

    let key = PostProcessPipelineKey::from_entry_point(PostProcessEntryPoint::ToneMapping);
    let tone_mapping = pipelines.specialize(&mut pipeline_cache, &pipeline, key);

    let key = PostProcessPipelineKey::from_entry_point(PostProcessEntryPoint::Taa);
    let taa = pipelines.specialize(&mut pipeline_cache, &pipeline, key);

    let key = PostProcessPipelineKey::from_entry_point(PostProcessEntryPoint::JasmineTaa);
    let taa_jasmine = pipelines.specialize(&mut pipeline_cache, &pipeline, key);

    commands.insert_resource(CachedPostProcessPipelines {
        denoise,
        tone_mapping,
        taa,
        taa_jasmine,
    })
}

#[derive(Component, Clone)]
pub struct PostProcessBindGroup {
    pub deferred: BindGroup,
    pub sampler: BindGroup,
    pub denoise_internal: BindGroup,
    pub denoise_render: [BindGroup; 3],
    pub tone_mapping: BindGroup,
    pub tone_mapping_output: BindGroup,
    pub taa: BindGroup,
    pub taa_output: BindGroup,
}

fn queue_post_process_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    config: Res<HikariConfig>,
    pipeline: Res<PostProcessPipeline>,
    images: Res<RenderAssets<Image>>,
    fallback: Res<FallbackImage>,
    query: Query<
        (
            Entity,
            &PrepassTextures,
            &LightPassTextures,
            &PostProcessTextures,
        ),
        With<ExtractedCamera>,
    >,
) {
    for (entity, prepass, light, post_process) in &query {
        let current = post_process.head;
        let previous = 1 - current;

        let deferred = match prepass.as_bind_group(
            &pipeline.deferred_layout,
            &render_device,
            &images,
            &fallback,
        ) {
            Ok(deferred) => deferred,
            Err(_) => continue,
        }
        .bind_group;

        let sampler = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.sampler_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&post_process.nearest_sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&post_process.linear_sampler),
                },
            ],
        });

        let denoise_internal = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.denoise_internal_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&light.albedo.default_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &post_process.denoise_internal[0].default_view,
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(
                        &post_process.denoise_internal[1].default_view,
                    ),
                },
            ],
        });
        let denoise_render = [0, 1, 2].map(|id| {
            render_device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.denoise_render_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(
                            &post_process.denoise_render[previous + 2 * id].default_view,
                        ),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&light.render[id].default_view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(
                            &post_process.denoise_render[current + 2 * id].default_view,
                        ),
                    },
                ],
            })
        });

        let (direct_render, emissive_render, indirect_render) = match config.denoise {
            true => (
                &light.render[0].default_view,
                &light.render[1].default_view,
                &light.render[2].default_view,
            ),
            false => (
                &post_process.denoise_render[current].default_view,
                &post_process.denoise_render[current + 2].default_view,
                &post_process.denoise_render[current + 4].default_view,
            ),
        };
        let tone_mapping = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.tone_mapping_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(direct_render),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(emissive_render),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(indirect_render),
                },
            ],
        });
        let tone_mapping_output = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.output_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(
                    &post_process.tone_mapping_output.default_view,
                ),
            }],
        });

        let taa = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.taa_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &post_process.taa_internal[previous].default_view,
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &post_process.tone_mapping_output.default_view,
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(
                        &post_process.taa_internal[current].default_view,
                    ),
                },
            ],
        });
        let taa_output = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.output_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&post_process.taa_output.default_view),
            }],
        });

        commands.entity(entity).insert(PostProcessBindGroup {
            deferred,
            sampler,
            denoise_internal,
            denoise_render,
            tone_mapping,
            tone_mapping_output,
            taa,
            taa_output,
        });
    }
}

pub struct PostProcessPassNode {
    query: QueryState<(
        &'static ExtractedCamera,
        &'static ViewUniformOffset,
        &'static PreviousViewUniformOffset,
        &'static ViewLightsUniformOffset,
        &'static PostProcessBindGroup,
    )>,
}

impl PostProcessPassNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            query: world.query_filtered(),
        }
    }
}

impl Node for PostProcessPassNode {
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
        let (camera, view_uniform, previous_view_uniform, view_lights, post_process_bind_group) =
            match self.query.get_manual(world, entity) {
                Ok(query) => query,
                Err(_) => return Ok(()),
            };
        let view_bind_group = match world.get_resource::<PrepassBindGroup>() {
            Some(bind_group) => &bind_group.view,
            None => return Ok(()),
        };

        let pipelines = world.resource::<CachedPostProcessPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let config = world.resource::<HikariConfig>();

        let mut pass = render_context
            .command_encoder
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(
            0,
            view_bind_group,
            &[
                view_uniform.offset,
                previous_view_uniform.offset,
                view_lights.offset,
            ],
        );
        pass.set_bind_group(1, &post_process_bind_group.deferred, &[]);
        pass.set_bind_group(2, &post_process_bind_group.sampler, &[]);

        if config.denoise {
            pass.set_bind_group(3, &post_process_bind_group.denoise_internal, &[]);

            for render_bind_group in &post_process_bind_group.denoise_render {
                pass.set_bind_group(4, render_bind_group, &[]);

                for pipeline in pipelines
                    .denoise
                    .iter()
                    .filter_map(|pipeline| pipeline_cache.get_compute_pipeline(*pipeline))
                {
                    pass.set_pipeline(pipeline);

                    let size = camera.physical_target_size.unwrap();
                    let count = (size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                    pass.dispatch_workgroups(count.x, count.y, 1);
                }
            }
        }

        pass.set_bind_group(3, &post_process_bind_group.tone_mapping, &[]);
        pass.set_bind_group(4, &post_process_bind_group.tone_mapping_output, &[]);

        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.tone_mapping) {
            pass.set_pipeline(pipeline);

            let size = camera.physical_target_size.unwrap();
            let count = (size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(count.x, count.y, 1);
        }

        if let Some(taa_version) = config.temporal_anti_aliasing {
            let pipeline = match taa_version {
                crate::TaaVersion::Cryscan => pipelines.taa,
                crate::TaaVersion::Jasmine => pipelines.taa_jasmine,
            };

            pass.set_bind_group(3, &post_process_bind_group.taa, &[]);
            pass.set_bind_group(4, &post_process_bind_group.taa_output, &[]);

            if let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline) {
                pass.set_pipeline(pipeline);

                let size = camera.physical_target_size.unwrap();
                let count = (size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                pass.dispatch_workgroups(count.x, count.y, 1);
            }
        }

        Ok(())
    }
}
