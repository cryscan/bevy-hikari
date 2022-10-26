use crate::{
    mesh_material::{
        MeshMaterialBindGroup, MeshMaterialBindGroupLayout, MeshMaterialSystems,
        TextureBindGroupLayout,
    },
    prepass::{PrepassBindGroup, PrepassPipeline, PrepassTextures},
    view::{FrameCounter, PreviousViewUniformOffset},
    HikariConfig, NoiseTextures, LIGHT_SHADER_HANDLE, WORKGROUP_SIZE,
};
use bevy::{
    pbr::ViewLightsUniformOffset,
    prelude::*,
    render::{
        camera::ExtractedCamera,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::{FallbackImage, GpuImage, TextureCache},
        view::ViewUniformOffset,
        RenderApp, RenderStage,
    },
    utils::HashMap,
};
use serde::Serialize;

pub const ALBEDO_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
pub const RENDER_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

pub struct LightPlugin;
impl Plugin for LightPlugin {
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<ReservoirCache>()
                .init_resource::<SpecializedComputePipelines<LightPipeline>>()
                .add_system_to_stage(
                    RenderStage::Prepare,
                    prepare_light_pipeline.after(MeshMaterialSystems::PrepareAssets),
                )
                .add_system_to_stage(RenderStage::Prepare, prepare_light_pass_textures)
                .add_system_to_stage(RenderStage::Queue, queue_light_bind_groups)
                .add_system_to_stage(RenderStage::Queue, queue_light_pipelines);
        }
    }
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
pub struct ReservoirCache(HashMap<Entity, Vec<StorageBuffer<GpuReservoirBuffer>>>);

pub struct LightPipeline {
    pub view_layout: BindGroupLayout,
    pub deferred_layout: BindGroupLayout,
    pub mesh_material_layout: BindGroupLayout,

    pub texture_count: u32,
    pub texture_layout: BindGroupLayout,

    pub noise_layout: BindGroupLayout,
    pub render_layout: BindGroupLayout,
    pub reservoir_layout: BindGroupLayout,
    pub cache_reservoir_layout: BindGroupLayout,
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, FromPrimitive)]
#[serde(rename_all = "snake_case")]
pub enum LightEntryPoint {
    #[default]
    DirectLit = 0,
    IndirectLitAmbient = 1,
    SpatialReuse = 2,
}

bitflags::bitflags! {
    #[repr(transparent)]
    pub struct LightPipelineKey: u32 {
        const ENTRY_POINT_BITS      = LightPipelineKey::ENTRY_POINT_MASK_BITS;
        const INCLUDE_EMISSIVE_BIT  = 1 << LightPipelineKey::INCLUDE_EMISSIVE_SHIFT_BITS;
        const TEXTURE_COUNT_BITS    = LightPipelineKey::TEXTURE_COUNT_MASK_BITS << LightPipelineKey::TEXTURE_COUNT_SHIFT_BITS;
    }
}

impl LightPipelineKey {
    const ENTRY_POINT_MASK_BITS: u32 = 0b11;
    const INCLUDE_EMISSIVE_SHIFT_BITS: u32 = 4;
    const TEXTURE_COUNT_MASK_BITS: u32 = 0xFFFF;
    const TEXTURE_COUNT_SHIFT_BITS: u32 = 32 - 16;

    pub fn from_entry_point(entry_point: LightEntryPoint) -> Self {
        let entry_point_bits = (entry_point as u32) & Self::ENTRY_POINT_MASK_BITS;
        Self::from_bits(entry_point_bits).unwrap()
    }

    pub fn entry_point(&self) -> LightEntryPoint {
        let entry_point_bits = self.bits & Self::ENTRY_POINT_MASK_BITS;
        num_traits::FromPrimitive::from_u32(entry_point_bits).unwrap()
    }

    pub fn from_texture_count(texture_count: u32) -> Self {
        let texture_count_bits =
            (texture_count & Self::TEXTURE_COUNT_MASK_BITS) << Self::TEXTURE_COUNT_SHIFT_BITS;
        Self::from_bits(texture_count_bits).unwrap()
    }

    pub fn texture_count(&self) -> u32 {
        (self.bits >> Self::TEXTURE_COUNT_SHIFT_BITS) & Self::TEXTURE_COUNT_MASK_BITS
    }
}

impl SpecializedComputePipeline for LightPipeline {
    type Key = LightPipelineKey;

    fn specialize(&self, key: Self::Key) -> ComputePipelineDescriptor {
        let mut shader_defs = vec![];

        if key.texture_count() == 0 {
            shader_defs.push("NO_TEXTURE".into());
        }
        if key.contains(LightPipelineKey::INCLUDE_EMISSIVE_BIT) {
            shader_defs.push("INCLUDE_EMISSIVE".into());
        }

        let entry_point = serde_variant::to_variant_name(&key.entry_point())
            .unwrap()
            .into();

        ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![
                self.view_layout.clone(),
                self.deferred_layout.clone(),
                self.mesh_material_layout.clone(),
                self.texture_layout.clone(),
                self.noise_layout.clone(),
                self.render_layout.clone(),
                self.reservoir_layout.clone(),
                self.cache_reservoir_layout.clone(),
            ]),
            shader: LIGHT_SHADER_HANDLE.typed::<Shader>(),
            shader_defs,
            entry_point,
        }
    }
}

fn prepare_light_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mesh_material_layout: Res<MeshMaterialBindGroupLayout>,
    texture_layout: Res<TextureBindGroupLayout>,
    prepass_pipeline: Res<PrepassPipeline>,
) {
    if !texture_layout.is_changed() {
        return;
    }

    let view_layout = prepass_pipeline.view_layout.clone();
    let mesh_material_layout = mesh_material_layout.clone();

    let texture_count = texture_layout.texture_count;
    let texture_layout = texture_layout.layout.clone();

    let deferred_layout = PrepassTextures::bind_group_layout(&render_device);
    let noise_layout = NoiseTextures::bind_group_layout(&render_device);

    let render_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            // Albedo Texture
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: ALBEDO_TEXTURE_FORMAT,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            // Render Texture
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
            // Previous Spatial Reservoir
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(GpuReservoirBuffer::min_size()),
                },
                count: None,
            },
            // Current Spatial Reservoir
            BindGroupLayoutEntry {
                binding: 3,
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

    let cache_reservoir_layout =
        render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Direct Cache Reservoir
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
                // Emissive Cache Reservoir
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuReservoirBuffer::min_size()),
                    },
                    count: None,
                },
            ],
        });

    commands.insert_resource(LightPipeline {
        view_layout,
        deferred_layout,
        mesh_material_layout,
        texture_count,
        texture_layout,
        noise_layout,
        render_layout,
        reservoir_layout,
        cache_reservoir_layout,
    });
}

#[derive(Component)]
pub struct LightPassTextures {
    /// Index of the current frame's output denoised texture.
    pub head: usize,
    pub albedo: GpuImage,
    pub direct_render: GpuImage,
    pub emissive_render: GpuImage,
    pub indirect_render: GpuImage,
}

fn prepare_light_pass_textures(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    frame_counter: Res<FrameCounter>,
    mut texture_cache: ResMut<TextureCache>,
    mut reservoir_cache: ResMut<ReservoirCache>,
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

            let albedo = create_texture(ALBEDO_TEXTURE_FORMAT);
            let direct_render = create_texture(RENDER_TEXTURE_FORMAT);
            let emissive_render = create_texture(RENDER_TEXTURE_FORMAT);
            let indirect_render = create_texture(RENDER_TEXTURE_FORMAT);

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
                let reservoirs = (0..10)
                    .map(|_| {
                        let mut buffer = StorageBuffer::from(GpuReservoirBuffer {
                            data: vec![GpuPackedReservoir::default(); len],
                        });
                        buffer.write_buffer(&render_device, &render_queue);
                        buffer
                    })
                    .collect();
                reservoir_cache.insert(entity, reservoirs);
            }

            commands.entity(entity).insert(LightPassTextures {
                head: frame_counter.0 % 2,
                albedo,
                direct_render,
                emissive_render,
                indirect_render,
            });
        }
    }
}

#[allow(dead_code)]
pub struct CachedLightPipelines {
    direct_lit: CachedComputePipelineId,
    direct_emissive: CachedComputePipelineId,
    indirect_lit_ambient: CachedComputePipelineId,
    emissive_spatial_reuse: CachedComputePipelineId,
    indirect_spatial_reuse: CachedComputePipelineId,
}

fn queue_light_pipelines(
    mut commands: Commands,
    pipeline: Res<LightPipeline>,
    mut pipelines: ResMut<SpecializedComputePipelines<LightPipeline>>,
    mut pipeline_cache: ResMut<PipelineCache>,
) {
    let texture_count_key = LightPipelineKey::from_texture_count(pipeline.texture_count);
    let key = texture_count_key | LightPipelineKey::from_entry_point(LightEntryPoint::DirectLit);

    let direct_lit = pipelines.specialize(&mut pipeline_cache, &pipeline, key);
    let direct_emissive = pipelines.specialize(
        &mut pipeline_cache,
        &pipeline,
        key | LightPipelineKey::INCLUDE_EMISSIVE_BIT,
    );

    let key =
        texture_count_key | LightPipelineKey::from_entry_point(LightEntryPoint::IndirectLitAmbient);
    let indirect_lit_ambient = pipelines.specialize(&mut pipeline_cache, &pipeline, key);

    let key = texture_count_key | LightPipelineKey::from_entry_point(LightEntryPoint::SpatialReuse);
    let emissive_spatial_reuse = pipelines.specialize(
        &mut pipeline_cache,
        &pipeline,
        key | LightPipelineKey::INCLUDE_EMISSIVE_BIT,
    );
    let indirect_spatial_reuse = pipelines.specialize(&mut pipeline_cache, &pipeline, key);

    commands.insert_resource(CachedLightPipelines {
        direct_lit,
        direct_emissive,
        indirect_lit_ambient,
        emissive_spatial_reuse,
        indirect_spatial_reuse,
    })
}

#[derive(Component, Clone)]
pub struct LightBindGroup {
    pub deferred: BindGroup,
    pub noise: BindGroup,

    pub direct_render: BindGroup,
    pub direct_reservoir: BindGroup,

    pub emissive_render: BindGroup,
    pub emissive_reservoir: BindGroup,

    pub indirect_render: BindGroup,
    pub indirect_reservoir: BindGroup,

    pub cache_reservoir: BindGroup,
}

#[allow(clippy::too_many_arguments)]
fn queue_light_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline: Res<LightPipeline>,
    noise: Res<NoiseTextures>,
    images: Res<RenderAssets<Image>>,
    fallback: Res<FallbackImage>,
    reservoir_cache: Res<ReservoirCache>,
    query: Query<(Entity, &PrepassTextures, &LightPassTextures), With<ExtractedCamera>>,
) {
    for (entity, prepass, light_pass) in &query {
        let reservoirs = reservoir_cache.get(&entity).unwrap();
        if let Some(reservoir_bindings) = reservoirs
            .iter()
            .map(|buffer| buffer.binding())
            .collect::<Option<Vec<_>>>()
        {
            let current = light_pass.head;
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

            let noise = match noise.as_bind_group(
                &pipeline.noise_layout,
                &render_device,
                &images,
                &fallback,
            ) {
                Ok(noise) => noise,
                Err(_) => continue,
            }
            .bind_group;

            let direct_render = render_device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.render_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&light_pass.albedo.texture_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(
                            &light_pass.direct_render.texture_view,
                        ),
                    },
                ],
            });
            let emissive_render = render_device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.render_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&light_pass.albedo.texture_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(
                            &light_pass.emissive_render.texture_view,
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
                        resource: BindingResource::TextureView(&light_pass.albedo.texture_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(
                            &light_pass.indirect_render.texture_view,
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
                        resource: reservoir_bindings[current].clone(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: reservoir_bindings[previous].clone(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: reservoir_bindings[4 + current].clone(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: reservoir_bindings[4 + previous].clone(),
                    },
                ],
            });
            let emissive_reservoir = render_device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.reservoir_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: reservoir_bindings[2 + current].clone(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: reservoir_bindings[2 + previous].clone(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: reservoir_bindings[4 + current].clone(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: reservoir_bindings[4 + previous].clone(),
                    },
                ],
            });
            let indirect_reservoir = render_device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.reservoir_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: reservoir_bindings[6 + current].clone(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: reservoir_bindings[6 + previous].clone(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: reservoir_bindings[8 + current].clone(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: reservoir_bindings[8 + previous].clone(),
                    },
                ],
            });

            let cache_reservoir = render_device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.cache_reservoir_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: reservoir_bindings[current].clone(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: reservoir_bindings[2 + current].clone(),
                    },
                ],
            });

            commands.entity(entity).insert(LightBindGroup {
                deferred,
                noise,
                direct_render,
                direct_reservoir,
                emissive_render,
                emissive_reservoir,
                indirect_render,
                indirect_reservoir,
                cache_reservoir,
            });
        }
    }
}

pub struct LightPassNode {
    query: QueryState<(
        &'static ExtractedCamera,
        &'static ViewUniformOffset,
        &'static PreviousViewUniformOffset,
        &'static ViewLightsUniformOffset,
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
        let (camera, view_uniform, previous_view_uniform, view_lights, light_bind_group) =
            match self.query.get_manual(world, entity) {
                Ok(query) => query,
                Err(_) => return Ok(()),
            };
        let view_bind_group = match world.get_resource::<PrepassBindGroup>() {
            Some(bind_group) => &bind_group.view,
            None => return Ok(()),
        };
        let mesh_material_bind_group = match world.get_resource::<MeshMaterialBindGroup>() {
            Some(bind_group) => bind_group,
            None => return Ok(()),
        };

        let pipelines = world.resource::<CachedLightPipelines>();
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
        pass.set_bind_group(1, &light_bind_group.deferred, &[]);
        pass.set_bind_group(2, &mesh_material_bind_group.mesh_material, &[]);
        pass.set_bind_group(3, &mesh_material_bind_group.texture, &[]);
        pass.set_bind_group(4, &light_bind_group.noise, &[]);

        pass.set_bind_group(5, &light_bind_group.direct_render, &[]);
        pass.set_bind_group(6, &light_bind_group.direct_reservoir, &[]);
        pass.set_bind_group(7, &light_bind_group.cache_reservoir, &[]);

        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.direct_lit) {
            pass.set_pipeline(pipeline);

            let size = camera.physical_target_size.unwrap();
            let count = (size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(count.x, count.y, 1);
        }

        pass.set_bind_group(5, &light_bind_group.emissive_render, &[]);
        pass.set_bind_group(6, &light_bind_group.emissive_reservoir, &[]);

        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.direct_emissive) {
            pass.set_pipeline(pipeline);

            let size = camera.physical_target_size.unwrap();
            let count = (size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(count.x, count.y, 1);
        }

        if config.spatial_reuse {
            if let Some(pipeline) =
                pipeline_cache.get_compute_pipeline(pipelines.emissive_spatial_reuse)
            {
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

        if config.spatial_reuse {
            if let Some(pipeline) =
                pipeline_cache.get_compute_pipeline(pipelines.indirect_spatial_reuse)
            {
                pass.set_pipeline(pipeline);

                let size = camera.physical_target_size.unwrap();
                let count = (size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                pass.dispatch_workgroups(count.x, count.y, 1);
            }
        }

        Ok(())
    }
}
