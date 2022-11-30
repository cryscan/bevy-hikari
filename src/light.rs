use crate::{
    mesh_material::{
        MeshMaterialBindGroup, MeshMaterialBindGroupLayout, MeshMaterialSystems,
        TextureBindGroupLayout,
    },
    prepass::{PrepassBindGroup, PrepassPipeline, PrepassTextures},
    view::{FrameCounter, FrameUniform, PreviousViewUniformOffset},
    HikariSettings, NoiseTextures, LIGHT_SHADER_HANDLE, WORKGROUP_SIZE,
};
use bevy::{
    pbr::{ViewLightsUniformOffset, MAX_DIRECTIONAL_LIGHTS},
    prelude::*,
    render::{
        camera::ExtractedCamera,
        extract_component::DynamicUniformIndex,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::{FallbackImage, TextureCache},
        view::ViewUniformOffset,
        RenderApp, RenderStage,
    },
    utils::HashMap,
};
use itertools::multizip;
use serde::Serialize;

pub const ALBEDO_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
pub const VARIANCE_TEXTURE_FORMAT: TextureFormat = TextureFormat::R32Float;
pub const RENDER_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
pub const DEBUG_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

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
                .add_system_to_stage(RenderStage::Prepare, prepare_light_textures)
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

#[derive(Default, Resource, Clone, ShaderType)]
pub struct GpuReservoirBuffer {
    #[size(runtime)]
    pub data: Vec<GpuPackedReservoir>,
}

#[derive(Default, Resource, Deref, DerefMut)]
pub struct ReservoirCache(HashMap<Entity, Vec<StorageBuffer<GpuReservoirBuffer>>>);

#[derive(Resource)]
pub struct LightPipeline {
    pub view_layout: BindGroupLayout,
    pub deferred_layout: BindGroupLayout,
    pub mesh_material_layout: BindGroupLayout,

    pub texture_count: u32,
    pub texture_layout: BindGroupLayout,

    pub noise_layout: BindGroupLayout,
    pub render_layout: BindGroupLayout,
    pub reservoir_layout: BindGroupLayout,
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
        const MULTIPLE_BOUNCES_BIT  = 1 << LightPipelineKey::MULTIPLE_BOUNCES_SHIFT_BITS;
        const TEXTURE_COUNT_BITS    = LightPipelineKey::TEXTURE_COUNT_MASK_BITS << LightPipelineKey::TEXTURE_COUNT_SHIFT_BITS;
    }
}

impl LightPipelineKey {
    const ENTRY_POINT_MASK_BITS: u32 = 0b11;
    const INCLUDE_EMISSIVE_SHIFT_BITS: u32 = 4;
    const MULTIPLE_BOUNCES_SHIFT_BITS: u32 = 8;
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
        shader_defs.push(ShaderDefVal::Int(
            "MAX_DIRECTIONAL_LIGHTS".into(),
            MAX_DIRECTIONAL_LIGHTS as i32,
        ));
        shader_defs.push(ShaderDefVal::Int(
            "TEXTURE_COUNT".into(),
            key.texture_count() as i32,
        ));
        if key.contains(LightPipelineKey::INCLUDE_EMISSIVE_BIT) {
            shader_defs.push("INCLUDE_EMISSIVE".into());
        }
        if key.contains(LightPipelineKey::MULTIPLE_BOUNCES_BIT) {
            shader_defs.push("MULTIPLE_BOUNCES".into());
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
            // Variance Texture
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: VARIANCE_TEXTURE_FORMAT,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            // Render Texture
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
            // Debug Texture
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: DEBUG_TEXTURE_FORMAT,
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
                    ty: BufferBindingType::Storage { read_only: false },
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

    commands.insert_resource(LightPipeline {
        view_layout,
        deferred_layout,
        mesh_material_layout,
        texture_count,
        texture_layout,
        noise_layout,
        render_layout,
        reservoir_layout,
    });
}

#[derive(Component)]
pub struct LightTextures {
    /// Index of the current frame's output denoised texture.
    pub head: usize,
    pub albedo: TextureView,
    pub debug: TextureView,
    pub variance: [TextureView; 3],
    pub render: [TextureView; 3],
}

#[allow(clippy::too_many_arguments)]
fn prepare_light_textures(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut texture_cache: ResMut<TextureCache>,
    mut reservoir_cache: ResMut<ReservoirCache>,
    cameras: Query<(Entity, &ExtractedCamera, &FrameCounter, &HikariSettings)>,
) {
    for (entity, camera, counter, settings) in &cameras {
        if let Some(size) = camera.physical_target_size {
            let texture_usage = TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
            let scale = settings.upscale.ratio().recip();
            let size = (scale * size.as_vec2()).ceil().as_uvec2();
            let mut create_texture = |texture_format| {
                let extent = Extent3d {
                    width: size.x,
                    height: size.y,
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
                            usage: texture_usage,
                        },
                    )
                    .default_view
            };

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

            macro_rules! create_texture_array {
                [$texture_format:ident; $count:literal] => {
                    [(); $count].map(|_| create_texture($texture_format))
                };
            }

            let variance = create_texture_array![VARIANCE_TEXTURE_FORMAT; 3];
            let render = create_texture_array![RENDER_TEXTURE_FORMAT; 3];

            commands.entity(entity).insert(LightTextures {
                head: counter.0 % 2,
                albedo: create_texture(ALBEDO_TEXTURE_FORMAT),
                debug: create_texture(DEBUG_TEXTURE_FORMAT),
                variance,
                render,
            });
        }
    }
}

#[derive(Resource)]
pub struct CachedLightPipelines {
    direct_lit: CachedComputePipelineId,
    direct_emissive: CachedComputePipelineId,
    indirect: CachedComputePipelineId,
    indirect_multiple_bounces: CachedComputePipelineId,
    emissive_spatial_reuse: CachedComputePipelineId,
    indirect_spatial_reuse: CachedComputePipelineId,
}

fn queue_light_pipelines(
    mut commands: Commands,
    pipeline: Res<LightPipeline>,
    mut pipelines: ResMut<SpecializedComputePipelines<LightPipeline>>,
    mut pipeline_cache: ResMut<PipelineCache>,
) {
    let key = LightPipelineKey::from_texture_count(pipeline.texture_count);

    let (direct_lit, direct_emissive) = {
        let key = key | LightPipelineKey::from_entry_point(LightEntryPoint::DirectLit);
        let direct_lit = pipelines.specialize(&mut pipeline_cache, &pipeline, key);
        let direct_emissive = pipelines.specialize(
            &mut pipeline_cache,
            &pipeline,
            key | LightPipelineKey::INCLUDE_EMISSIVE_BIT,
        );
        (direct_lit, direct_emissive)
    };

    let (indirect, indirect_multiple_bounces) = {
        let key = key | LightPipelineKey::from_entry_point(LightEntryPoint::IndirectLitAmbient);
        let indirect = pipelines.specialize(&mut pipeline_cache, &pipeline, key);
        let indirect_multiple_bounces = pipelines.specialize(
            &mut pipeline_cache,
            &pipeline,
            key | LightPipelineKey::MULTIPLE_BOUNCES_BIT,
        );
        (indirect, indirect_multiple_bounces)
    };

    let (emissive_spatial_reuse, indirect_spatial_reuse) = {
        let key = key | LightPipelineKey::from_entry_point(LightEntryPoint::SpatialReuse);
        let emissive_spatial_reuse = pipelines.specialize(
            &mut pipeline_cache,
            &pipeline,
            key | LightPipelineKey::INCLUDE_EMISSIVE_BIT,
        );
        let indirect_spatial_reuse = pipelines.specialize(&mut pipeline_cache, &pipeline, key);
        (emissive_spatial_reuse, indirect_spatial_reuse)
    };
    commands.insert_resource(CachedLightPipelines {
        direct_lit,
        direct_emissive,
        indirect,
        indirect_multiple_bounces,
        emissive_spatial_reuse,
        indirect_spatial_reuse,
    })
}

#[derive(Component, Clone)]
pub struct LightBindGroup {
    pub deferred: BindGroup,
    pub noise: BindGroup,
    pub render: [BindGroup; 3],
    pub reservoir: [BindGroup; 3],
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
    query: Query<(Entity, &PrepassTextures, &LightTextures), With<ExtractedCamera>>,
) {
    for (entity, prepass, light) in &query {
        let reservoirs = reservoir_cache.get(&entity).unwrap();
        if let Some(reservoir_bindings) = reservoirs
            .iter()
            .map(|buffer| buffer.binding())
            .collect::<Option<Vec<_>>>()
        {
            let current = light.head;
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

            let render = [0, 1, 2].map(|id| {
                let variance = &light.variance[id];
                let render = &light.render[id];

                render_device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.render_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::TextureView(&light.albedo),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::TextureView(variance),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: BindingResource::TextureView(render),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: BindingResource::TextureView(&light.debug),
                        },
                    ],
                })
            });

            let reservoir = [(0, 4), (2, 4), (6, 8)].map(|(temporal, spatial)| {
                let current_temporal = reservoir_bindings[current + temporal].clone();
                let previous_temporal = reservoir_bindings[previous + temporal].clone();
                let current_spatial = reservoir_bindings[current + spatial].clone();
                let previous_spatial = reservoir_bindings[previous + spatial].clone();

                render_device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.reservoir_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: current_temporal,
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: previous_temporal,
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: current_spatial,
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: previous_spatial,
                        },
                    ],
                })
            });

            commands.entity(entity).insert(LightBindGroup {
                deferred,
                noise,
                render,
                reservoir,
            });
        }
    }
}

#[allow(clippy::type_complexity)]
pub struct LightNode {
    query: QueryState<(
        &'static ExtractedCamera,
        &'static DynamicUniformIndex<FrameUniform>,
        &'static ViewUniformOffset,
        &'static PreviousViewUniformOffset,
        &'static ViewLightsUniformOffset,
        &'static LightBindGroup,
        &'static HikariSettings,
    )>,
}

impl LightNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            query: world.query_filtered(),
        }
    }
}

impl Node for LightNode {
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
        let (
            camera,
            frame_uniform,
            view_uniform,
            previous_view_uniform,
            view_lights,
            light_bind_group,
            settings,
        ) = match self.query.get_manual(world, entity) {
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

        let size = camera.physical_target_size.unwrap();
        let scale = settings.upscale.ratio().recip();
        let scaled_size = (scale * size.as_vec2()).ceil().as_uvec2();

        let mut pass = render_context
            .command_encoder
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(
            0,
            view_bind_group,
            &[
                frame_uniform.index(),
                view_uniform.offset,
                previous_view_uniform.offset,
                view_lights.offset,
            ],
        );
        pass.set_bind_group(1, &light_bind_group.deferred, &[]);
        pass.set_bind_group(2, &mesh_material_bind_group.mesh_material, &[]);
        pass.set_bind_group(3, &mesh_material_bind_group.texture, &[]);
        pass.set_bind_group(4, &light_bind_group.noise, &[]);

        let indirect_pipeline = match settings.indirect_bounces {
            0 | 1 => &pipelines.indirect,
            _ => &pipelines.indirect_multiple_bounces,
        };

        for (render, reservoir, temporal_pipeline, spatial_pipeline) in multizip((
            light_bind_group.render.iter(),
            light_bind_group.reservoir.iter(),
            [
                &pipelines.direct_lit,
                &pipelines.direct_emissive,
                indirect_pipeline,
            ],
            [
                None,
                Some(&pipelines.emissive_spatial_reuse),
                Some(&pipelines.indirect_spatial_reuse),
            ],
        )) {
            pass.set_bind_group(5, render, &[]);
            pass.set_bind_group(6, reservoir, &[]);

            if let Some(pipeline) = pipeline_cache.get_compute_pipeline(*temporal_pipeline) {
                pass.set_pipeline(pipeline);

                let count = (scaled_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                pass.dispatch_workgroups(count.x, count.y, 1);

                if let Some(pipeline) = spatial_pipeline
                    .filter(|_| settings.spatial_reuse)
                    .and_then(|pipeline| pipeline_cache.get_compute_pipeline(*pipeline))
                {
                    pass.set_pipeline(pipeline);

                    let count = (scaled_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                    pass.dispatch_workgroups(count.x, count.y, 1);
                }
            }
        }

        Ok(())
    }
}
