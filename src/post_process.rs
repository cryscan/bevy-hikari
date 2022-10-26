use crate::{
    light::LightPassTextures,
    prepass::{PrepassBindGroup, PrepassPipeline, PrepassTextures},
    view::{FrameCounter, PreviousViewUniformOffset},
    HikariConfig, POST_PROCESS_SHADER_HANDLE, WORKGROUP_SIZE,
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
        texture::{FallbackImage, GpuImage, TextureCache},
        view::ViewUniformOffset,
        RenderApp, RenderStage,
    },
};
use serde::Serialize;

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
    pub tone_mapping_layout: BindGroupLayout,
    pub taa_layout: BindGroupLayout,
    pub output_layout: BindGroupLayout,
}

impl FromWorld for PostProcessPipeline {
    fn from_world(world: &mut World) -> Self {
        let view_layout = world.resource::<PrepassPipeline>().view_layout.clone();

        let render_device = world.resource::<RenderDevice>();
        let deferred_layout = PrepassTextures::bind_group_layout(render_device);

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
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Emissive Render
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
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Indirect Render
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 5,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Sampler(SamplerBindingType::Filtering),
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
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                // Current Render
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
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                // TAA Output
                BindGroupLayoutEntry {
                    binding: 4,
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
}

bitflags::bitflags! {
    #[repr(transparent)]
    pub struct PostProcessPipelineKey: u32 {
        const ENTRY_POINT_BITS = PostProcessPipelineKey::ENTRY_POINT_MASK_BITS;
    }
}

impl PostProcessPipelineKey {
    const ENTRY_POINT_MASK_BITS: u32 = 0b1;

    pub fn from_entry_point(entry_point: PostProcessEntryPoint) -> Self {
        let entry_point_bits = (entry_point as u32) & Self::ENTRY_POINT_MASK_BITS;
        Self::from_bits(entry_point_bits).unwrap()
    }

    pub fn entry_point(&self) -> PostProcessEntryPoint {
        let entry_point_bits = self.bits & Self::ENTRY_POINT_MASK_BITS;
        num_traits::FromPrimitive::from_u32(entry_point_bits).unwrap()
    }
}

impl SpecializedComputePipeline for PostProcessPipeline {
    type Key = PostProcessPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        let entry_point = serde_variant::to_variant_name(&key.entry_point())
            .unwrap()
            .into();

        ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![
                self.view_layout.clone(),
                self.deferred_layout.clone(),
                self.tone_mapping_layout.clone(),
                self.taa_layout.clone(),
                self.output_layout.clone(),
            ]),
            shader: POST_PROCESS_SHADER_HANDLE.typed::<Shader>(),
            shader_defs: vec![],
            entry_point,
        }
    }
}

#[derive(Component)]
pub struct PostProcessTextures {
    pub head: usize,
    pub temporal: [GpuImage; 2],
    pub tone_mapping: GpuImage,
    pub taa: GpuImage,
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
            let mut create_texture = |texture_format| -> GpuImage {
                let extent = Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: 1,
                };
                let sampler = render_device.create_sampler(&SamplerDescriptor {
                    label: None,
                    address_mode_u: AddressMode::ClampToEdge,
                    address_mode_v: AddressMode::ClampToEdge,
                    address_mode_w: AddressMode::ClampToEdge,
                    mag_filter: FilterMode::Linear,
                    min_filter: FilterMode::Linear,
                    mipmap_filter: FilterMode::Linear,
                    ..Default::default()
                });
                let texture = texture_cache.get(
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
                );
                GpuImage {
                    texture: texture.texture,
                    texture_view: texture.default_view,
                    texture_format,
                    sampler,
                    size: size.as_vec2(),
                }
            };

            let temporal = [
                create_texture(POST_PROCESS_TEXTURE_FORMAT),
                create_texture(POST_PROCESS_TEXTURE_FORMAT),
            ];
            let tone_mapping = create_texture(POST_PROCESS_TEXTURE_FORMAT);
            let taa = create_texture(POST_PROCESS_TEXTURE_FORMAT);

            commands.entity(entity).insert(PostProcessTextures {
                head: frame_counter.0 % 2,
                temporal,
                tone_mapping,
                taa,
            });
        }
    }
}

pub struct CachedPostProcessPipelines {
    tone_mapping: CachedComputePipelineId,
    taa: CachedComputePipelineId,
}

fn queue_post_process_pipelines(
    mut commands: Commands,
    pipeline: Res<PostProcessPipeline>,
    mut pipelines: ResMut<SpecializedComputePipelines<PostProcessPipeline>>,
    mut pipeline_cache: ResMut<PipelineCache>,
) {
    let key = PostProcessPipelineKey::from_entry_point(PostProcessEntryPoint::ToneMapping);
    let tone_mapping = pipelines.specialize(&mut pipeline_cache, &pipeline, key);

    let key = PostProcessPipelineKey::from_entry_point(PostProcessEntryPoint::Taa);
    let taa = pipelines.specialize(&mut pipeline_cache, &pipeline, key);

    commands.insert_resource(CachedPostProcessPipelines { tone_mapping, taa })
}

#[derive(Component, Clone)]
pub struct PostProcessBindGroup {
    pub deferred: BindGroup,
    pub tone_mapping: BindGroup,
    pub taa_no_render: BindGroup,
    pub taa: BindGroup,
    pub tone_mapping_output: BindGroup,
    pub taa_output: BindGroup,
}

fn queue_post_process_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
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
    for (entity, prepass, light_pass, post_process) in &query {
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

        let tone_mapping = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.tone_mapping_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&*light_pass.direct_render.texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&light_pass.direct_render.sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(
                        &*light_pass.emissive_render.texture_view,
                    ),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(&light_pass.emissive_render.sampler),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(
                        &*light_pass.indirect_render.texture_view,
                    ),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::Sampler(&light_pass.indirect_render.sampler),
                },
            ],
        });

        let taa_no_render = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.taa_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &*post_process.temporal[previous].texture_view,
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&post_process.temporal[previous].sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&*fallback.texture_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(&fallback.sampler),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(
                        &*post_process.temporal[current].texture_view,
                    ),
                },
            ],
        });
        let taa = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.taa_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &*post_process.temporal[previous].texture_view,
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&post_process.temporal[previous].sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(
                        &*post_process.tone_mapping.texture_view,
                    ),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(&post_process.tone_mapping.sampler),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(
                        &*post_process.temporal[current].texture_view,
                    ),
                },
            ],
        });

        let tone_mapping_output = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.output_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&*post_process.tone_mapping.texture_view),
            }],
        });
        let taa_output = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.output_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&*post_process.taa.texture_view),
            }],
        });

        commands.entity(entity).insert(PostProcessBindGroup {
            deferred,
            tone_mapping,
            taa_no_render,
            taa,
            tone_mapping_output,
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
        pass.set_bind_group(2, &post_process_bind_group.tone_mapping, &[]);

        pass.set_bind_group(3, &post_process_bind_group.taa_no_render, &[]);
        pass.set_bind_group(4, &post_process_bind_group.tone_mapping_output, &[]);

        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.tone_mapping) {
            pass.set_pipeline(pipeline);

            let size = camera.physical_target_size.unwrap();
            let count = (size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(count.x, count.y, 1);
        }

        if !config.temporal_anti_aliasing {
            return Ok(());
        }

        pass.set_bind_group(3, &post_process_bind_group.taa, &[]);
        pass.set_bind_group(4, &post_process_bind_group.taa_output, &[]);

        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.taa) {
            pass.set_pipeline(pipeline);

            let size = camera.physical_target_size.unwrap();
            let count = (size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(count.x, count.y, 1);
        }

        Ok(())
    }
}
