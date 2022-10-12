use crate::{
    mesh_material::{MeshMaterialBindGroup, MeshMaterialBindGroupLayout, TextureBindGroupLayout},
    prepass::PrepassTarget,
    view::{FrameUniform, GpuFrame},
    NoiseTexture, LIGHT_SHADER_HANDLE, NOISE_TEXTURE_COUNT, WORKGROUP_SIZE,
};
use bevy::{
    ecs::system::{
        lifetimeless::{Read, SQuery},
        SystemParamItem,
    },
    pbr::{GpuLights, LightMeta, MeshPipeline, ViewLightsUniformOffset},
    prelude::*,
    render::{
        camera::ExtractedCamera,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
        render_phase::{EntityRenderCommand, RenderCommandResult, TrackedRenderPass},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::{GpuImage, TextureCache},
        view::{ExtractedView, ViewUniform, ViewUniformOffset, ViewUniforms},
        Extract, RenderApp, RenderStage,
    },
    utils::HashMap,
};
use std::num::NonZeroU32;

pub const ALBEDO_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
pub const RENDER_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
pub const RESERVOIR_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba32Float;
pub const RADIANCE_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
pub const POSITION_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba32Float;
pub const NORMAL_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba8Snorm;
pub const ID_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rg16Uint;
pub const RANDOM_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

pub struct LightPlugin;
impl Plugin for LightPlugin {
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<ReservoirCache>()
                .init_resource::<LightPipeline>()
                .init_resource::<SpecializedComputePipelines<LightPipeline>>()
                .add_system_to_stage(RenderStage::Extract, extract_noise_texture)
                .add_system_to_stage(RenderStage::Prepare, prepare_light_pass_targets)
                .add_system_to_stage(RenderStage::Queue, queue_view_bind_groups)
                .add_system_to_stage(RenderStage::Queue, queue_light_bind_groups)
                .add_system_to_stage(RenderStage::Queue, queue_light_pipelines);
        }
    }
}

fn extract_noise_texture(mut commands: Commands, noise_texture: Extract<Res<NoiseTexture>>) {
    commands.insert_resource(noise_texture.clone());
}

#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct GpuPackedReservoir {
    pub radiance: UVec2,
    pub random: UVec2,
    pub visible_position: Vec4,
    pub sample_position: Vec4,
    pub visible_normal: u32,
    pub sample_normal: u32,
    pub reservoir: UVec2,
}

#[derive(Default, Clone, ShaderType)]
pub struct GpuReservoirBuffer {
    #[size(runtime)]
    pub data: Vec<GpuPackedReservoir>,
}

#[derive(Default, Deref, DerefMut)]
pub struct ReservoirBuffer(Vec<StorageBuffer<GpuReservoirBuffer>>);

#[derive(Default, Deref, DerefMut)]
pub struct ReservoirCache(HashMap<Entity, ReservoirBuffer>);

pub struct LightPipeline {
    pub view_layout: BindGroupLayout,
    pub deferred_layout: BindGroupLayout,
    pub mesh_material_layout: BindGroupLayout,
    pub texture_layout: Option<BindGroupLayout>,
    pub frame_layout: BindGroupLayout,
    pub render_layout: BindGroupLayout,
    pub reservoir_layout: BindGroupLayout,
    pub dummy_white_gpu_image: GpuImage,
}

impl FromWorld for LightPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let mesh_pipeline = world.resource::<MeshPipeline>();
        let mesh_material_layout = world.resource::<MeshMaterialBindGroupLayout>().0.clone();

        let view_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                // View
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(ViewUniform::min_size()),
                    },
                    count: None,
                },
                // Lights
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(GpuLights::min_size()),
                    },
                    count: None,
                },
            ],
            label: None,
        });

        let deferred_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Position Buffer
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Normal Buffer
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Depth Gradient Buffer
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // UV Buffer
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Velocity Buffer
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Instance-material Buffer
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Uint,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Albedo Texture
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadWrite,
                        format: ALBEDO_TEXTURE_FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let frame_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Frame Uniform
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuFrame::min_size()),
                    },
                    count: None,
                },
                // Blue Noise Texture
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: NonZeroU32::new(NOISE_TEXTURE_COUNT as u32),
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let render_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Denoised Textures
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadWrite,
                        format: RENDER_TEXTURE_FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadWrite,
                        format: RENDER_TEXTURE_FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadWrite,
                        format: RENDER_TEXTURE_FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadWrite,
                        format: RENDER_TEXTURE_FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Render Texture
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadWrite,
                        format: RENDER_TEXTURE_FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let reservoir_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Previous Reservoir
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuReservoirBuffer::min_size()),
                    },
                    count: None,
                },
                // Current Reservoir
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuReservoirBuffer::min_size()),
                    },
                    count: None,
                },
            ],
        });

        Self {
            view_layout,
            deferred_layout,
            mesh_material_layout,
            texture_layout: None,
            frame_layout,
            render_layout,
            reservoir_layout,
            dummy_white_gpu_image: mesh_pipeline.dummy_white_gpu_image.clone(),
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct LightPipelineKey {
    pub entry_point: String,
    pub texture_count: usize,
    pub filter_level: usize,
}

impl SpecializedComputePipeline for LightPipeline {
    type Key = LightPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        let mut shader_defs = vec![];
        if key.texture_count < 1 {
            shader_defs.push("NO_TEXTURE".into());
        }
        shader_defs.push(format!("DENOISER_LEVEL_{}", key.filter_level));

        ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![
                self.view_layout.clone(),
                self.deferred_layout.clone(),
                self.mesh_material_layout.clone(),
                self.texture_layout.clone().unwrap(),
                self.frame_layout.clone(),
                self.render_layout.clone(),
                self.reservoir_layout.clone(),
            ]),
            shader: LIGHT_SHADER_HANDLE.typed::<Shader>(),
            shader_defs,
            entry_point: key.entry_point.into(),
        }
    }
}

#[derive(Component)]
pub struct LightPassTarget {
    pub denoise_textures: [GpuImage; 3],
    pub albedo_texture: GpuImage,
    pub direct_render_texture: GpuImage,
    pub direct_denoised_texture: GpuImage,
    pub indirect_render_texture: GpuImage,
    pub indirect_denoised_texture: GpuImage,
}

fn prepare_light_pass_targets(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut texture_cache: ResMut<TextureCache>,
    mut reservoir_cache: ResMut<ReservoirCache>,
    cameras: Query<(Entity, &ExtractedCamera)>,
) {
    for (entity, camera) in &cameras {
        if let Some(size) = camera.physical_target_size {
            let texture_usage = TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
            let mut create_texture = |size: UVec2, texture_format| -> GpuImage {
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
                    mag_filter: FilterMode::Nearest,
                    min_filter: FilterMode::Nearest,
                    mipmap_filter: FilterMode::Nearest,
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

            let albedo_texture = create_texture(size, ALBEDO_TEXTURE_FORMAT);
            let direct_render_texture = create_texture(size, RENDER_TEXTURE_FORMAT);
            let indirect_render_texture = create_texture(size, RENDER_TEXTURE_FORMAT);

            let denoise_textures = [(); 3].map(|_| create_texture(size, RENDER_TEXTURE_FORMAT));
            let direct_denoised_texture = create_texture(size, RENDER_TEXTURE_FORMAT);
            let indirect_denoised_texture = create_texture(size, RENDER_TEXTURE_FORMAT);

            // let mut create_reservoir = |size| -> Reservoir {
            //     Reservoir {
            //         reservoir: create_texture(size, RESERVOIR_TEXTURE_FORMAT),
            //         radiance: create_texture(size, RADIANCE_TEXTURE_FORMAT),
            //         random: create_texture(size, RANDOM_TEXTURE_FORMAT),
            //         visible_position: create_texture(size, POSITION_TEXTURE_FORMAT),
            //         visible_normal: create_texture(size, NORMAL_TEXTURE_FORMAT),
            //         visible_id: create_texture(size, ID_TEXTURE_FORMAT),
            //         sample_position: create_texture(size, POSITION_TEXTURE_FORMAT),
            //         sample_normal: create_texture(size, NORMAL_TEXTURE_FORMAT),
            //     }
            // };

            // let direct_reservoirs = [(); 2].map(|_| create_reservoir(size));
            // let indirect_reservoirs = [(); 2].map(|_| create_reservoir(size >> INDIRECT_LOG_SCALE));

            if match reservoir_cache.get(&entity) {
                Some(reservoirs) => {
                    let len = (size.x * size.y) as usize;
                    reservoirs
                        .iter()
                        .any(|buffer| buffer.get().data.len() != len)
                }
                None => true,
            } {
                // Reservoirs of this entity should be updated.
                let len = (size.x * size.y) as usize;
                let buffer = GpuReservoirBuffer {
                    data: vec![GpuPackedReservoir::default(); len],
                };
                let mut reservoirs = ReservoirBuffer(vec![
                    StorageBuffer::from(buffer.clone()),
                    StorageBuffer::from(buffer.clone()),
                    StorageBuffer::from(buffer.clone()),
                    StorageBuffer::from(buffer),
                ]);
                for buffer in reservoirs.iter_mut() {
                    buffer.write_buffer(&render_device, &render_queue);
                }
                reservoir_cache.insert(entity, reservoirs);
            }

            // Swap the double buffers.
            if let Some(reservoirs) = reservoir_cache.get_mut(&entity) {
                reservoirs.swap(0, 1);
                reservoirs.swap(2, 3);
            }

            commands.entity(entity).insert(LightPassTarget {
                albedo_texture,
                denoise_textures,
                direct_render_texture,
                direct_denoised_texture,
                indirect_render_texture,
                indirect_denoised_texture,
            });
        }
    }
}

#[allow(dead_code)]
pub struct CachedLightPipelines {
    direct_lit: CachedComputePipelineId,
    indirect_lit_ambient: CachedComputePipelineId,
    denoise: [CachedComputePipelineId; 4],
}

fn queue_light_pipelines(
    mut commands: Commands,
    layout: Res<TextureBindGroupLayout>,
    mut pipeline: ResMut<LightPipeline>,
    mut pipelines: ResMut<SpecializedComputePipelines<LightPipeline>>,
    mut pipeline_cache: ResMut<PipelineCache>,
) {
    pipeline.texture_layout = Some(layout.layout.clone());

    let direct_lit = pipelines.specialize(
        &mut pipeline_cache,
        &pipeline,
        LightPipelineKey {
            entry_point: "direct_lit".into(),
            texture_count: layout.texture_count,
            filter_level: 0,
        },
    );

    let indirect_lit_ambient = pipelines.specialize(
        &mut pipeline_cache,
        &pipeline,
        LightPipelineKey {
            entry_point: "indirect_lit_ambient".into(),
            texture_count: layout.texture_count,
            filter_level: 0,
        },
    );

    let denoise = [0, 1, 2, 3].map(|level| {
        pipelines.specialize(
            &mut pipeline_cache,
            &pipeline,
            LightPipelineKey {
                entry_point: "denoise_atrous".into(),
                texture_count: layout.texture_count,
                filter_level: level,
            },
        )
    });

    commands.insert_resource(CachedLightPipelines {
        direct_lit,
        indirect_lit_ambient,
        denoise,
    })
}

#[derive(Component)]
pub struct ViewBindGroup(pub BindGroup);

#[allow(clippy::too_many_arguments)]
pub fn queue_view_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline: Res<LightPipeline>,
    light_meta: Res<LightMeta>,
    view_uniforms: Res<ViewUniforms>,
    views: Query<Entity, With<ExtractedView>>,
) {
    if let (Some(view_binding), Some(light_binding)) = (
        view_uniforms.uniforms.binding(),
        light_meta.view_gpu_lights.binding(),
    ) {
        for entity in &views {
            let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: view_binding.clone(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: light_binding.clone(),
                    },
                ],
                label: None,
                layout: &pipeline.view_layout,
            });

            commands.entity(entity).insert(ViewBindGroup(bind_group));
        }
    }
}

#[derive(Component, Clone)]
pub struct LightBindGroup {
    pub deferred: BindGroup,
    pub frame: BindGroup,

    pub direct_render: BindGroup,
    pub direct_reservoir: BindGroup,

    pub indirect_render: BindGroup,
    pub indirect_reservoir: BindGroup,
}

#[allow(clippy::too_many_arguments)]
fn queue_light_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline: Res<LightPipeline>,
    frame_uniform: Res<FrameUniform>,
    noise_texture: Res<NoiseTexture>,
    images: Res<RenderAssets<Image>>,
    reservoir_cache: Res<ReservoirCache>,
    query: Query<(Entity, &PrepassTarget, &LightPassTarget), With<ExtractedCamera>>,
) {
    let mut noise_texture_views = vec![];
    for handle in noise_texture.iter() {
        let image = match images.get(handle) {
            Some(image) => image,
            None => {
                return;
            }
        };
        noise_texture_views.push(&*image.texture_view);
    }

    let noise_sampler = render_device.create_sampler(&SamplerDescriptor {
        label: None,
        address_mode_u: AddressMode::Repeat,
        address_mode_v: AddressMode::Repeat,
        address_mode_w: AddressMode::Repeat,
        mag_filter: FilterMode::Nearest,
        min_filter: FilterMode::Nearest,
        mipmap_filter: FilterMode::Nearest,
        ..Default::default()
    });

    for (entity, prepass, light_pass) in &query {
        let reservoirs = reservoir_cache.get(&entity).unwrap();
        if let (
            Some(frame_binding),
            Some(reservoir_binding_0),
            Some(reservoir_binding_1),
            Some(reservoir_binding_2),
            Some(reservoir_binding_3),
        ) = (
            frame_uniform.buffer.binding(),
            reservoirs[0].binding(),
            reservoirs[1].binding(),
            reservoirs[2].binding(),
            reservoirs[3].binding(),
        ) {
            let deferred = render_device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.deferred_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&prepass.position.texture_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&prepass.normal.texture_view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(
                            &prepass.depth_gradient.texture_view,
                        ),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::TextureView(&prepass.uv.texture_view),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: BindingResource::TextureView(&prepass.velocity.texture_view),
                    },
                    BindGroupEntry {
                        binding: 5,
                        resource: BindingResource::TextureView(
                            &prepass.instance_material.texture_view,
                        ),
                    },
                    BindGroupEntry {
                        binding: 6,
                        resource: BindingResource::TextureView(
                            &light_pass.albedo_texture.texture_view,
                        ),
                    },
                ],
            });

            let frame = render_device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.frame_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: frame_binding,
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureViewArray(&noise_texture_views),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Sampler(&noise_sampler),
                    },
                ],
            });

            let direct_render = render_device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.render_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(
                            &light_pass.denoise_textures[0].texture_view,
                        ),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(
                            &light_pass.denoise_textures[1].texture_view,
                        ),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(
                            &light_pass.denoise_textures[2].texture_view,
                        ),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::TextureView(
                            &light_pass.direct_denoised_texture.texture_view,
                        ),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: BindingResource::TextureView(
                            &light_pass.direct_render_texture.texture_view,
                        ),
                    },
                ],
            });
            let indirect_render = render_device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.render_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(
                            &light_pass.denoise_textures[0].texture_view,
                        ),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(
                            &light_pass.denoise_textures[1].texture_view,
                        ),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(
                            &light_pass.denoise_textures[2].texture_view,
                        ),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::TextureView(
                            &light_pass.indirect_denoised_texture.texture_view,
                        ),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: BindingResource::TextureView(
                            &light_pass.indirect_render_texture.texture_view,
                        ),
                    },
                ],
            });

            let direct_reservoir = render_device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.reservoir_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: reservoir_binding_0,
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: reservoir_binding_1,
                    },
                ],
            });
            let indirect_reservoir = render_device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.reservoir_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: reservoir_binding_2,
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: reservoir_binding_3,
                    },
                ],
            });

            commands.entity(entity).insert(LightBindGroup {
                deferred,
                frame,
                direct_render,
                direct_reservoir,
                indirect_render,
                indirect_reservoir,
            });
        }
    }
}

pub struct SetDeferredBindGroup<const I: usize>;
impl<const I: usize> EntityRenderCommand for SetDeferredBindGroup<I> {
    type Param = SQuery<Read<LightBindGroup>>;

    fn render<'w>(
        view: Entity,
        _item: Entity,
        query: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let bind_group = query.get_inner(view).unwrap();
        pass.set_bind_group(I, &bind_group.deferred, &[]);

        RenderCommandResult::Success
    }
}

pub struct LightPassNode {
    query: QueryState<(
        &'static ExtractedCamera,
        &'static ViewUniformOffset,
        &'static ViewLightsUniformOffset,
        &'static ViewBindGroup,
        &'static LightBindGroup,
    )>,
}

impl LightPassNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            query: world.query_filtered(),
        }
    }
}

impl Node for LightPassNode {
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
        let (camera, view_uniform, view_lights, view_bind_group, light_bind_group) =
            match self.query.get_manual(world, entity) {
                Ok(query) => query,
                Err(_) => return Ok(()),
            };
        let mesh_material_bind_group = match world.get_resource::<MeshMaterialBindGroup>() {
            Some(bind_group) => bind_group,
            None => return Ok(()),
        };
        let pipelines = world.resource::<CachedLightPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let mut pass = render_context
            .command_encoder
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(
            0,
            &view_bind_group.0,
            &[view_uniform.offset, view_lights.offset],
        );
        pass.set_bind_group(1, &light_bind_group.deferred, &[]);
        pass.set_bind_group(2, &mesh_material_bind_group.mesh_material, &[]);
        pass.set_bind_group(3, &mesh_material_bind_group.texture, &[]);
        pass.set_bind_group(4, &light_bind_group.frame, &[]);

        pass.set_bind_group(5, &light_bind_group.direct_render, &[]);
        pass.set_bind_group(6, &light_bind_group.direct_reservoir, &[]);

        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.direct_lit) {
            pass.set_pipeline(pipeline);

            let size = camera.physical_target_size.unwrap();
            let count = (size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(count.x, count.y, 1);
        }

        for id in 0..4 {
            if let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.denoise[id]) {
                pass.set_pipeline(pipeline);

                let size = camera.physical_target_size.unwrap();
                let count = (size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                pass.dispatch_workgroups(count.x, count.y, 1);
            }
        }

        pass.set_bind_group(5, &light_bind_group.indirect_render, &[]);
        pass.set_bind_group(6, &light_bind_group.indirect_reservoir, &[]);

        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.indirect_lit_ambient)
        {
            pass.set_pipeline(pipeline);

            let size = camera.physical_target_size.unwrap();
            let count = (size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(count.x, count.y, 1);
        }

        for id in 0..4 {
            if let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.denoise[id]) {
                pass.set_pipeline(pipeline);

                let size = camera.physical_target_size.unwrap();
                let count = (size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                pass.dispatch_workgroups(count.x, count.y, 1);
            }
        }

        Ok(())
    }
}
