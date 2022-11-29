use crate::{
    light::{LightTextures, VARIANCE_TEXTURE_FORMAT},
    prepass::{PrepassBindGroup, PrepassPipeline, PrepassTextures},
    view::{FrameCounter, FrameUniform, PreviousViewUniformOffset},
    HikariSettings, Taa, Upscale, DENOISE_SHADER_HANDLE, FSR1_EASU_SHADER_HANDLE,
    FSR1_RCAS_SHADER_HANDLE, SMAA_SHADER_HANDLE, TAA_SHADER_HANDLE, TONE_MAPPING_SHADER_HANDLE,
    WORKGROUP_SIZE,
};
use bevy::{
    ecs::query::QueryItem,
    pbr::{ViewLightsUniformOffset, MAX_DIRECTIONAL_LIGHTS},
    prelude::*,
    render::{
        camera::ExtractedCamera,
        extract_component::{
            ComponentUniforms, DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin,
            UniformComponentPlugin,
        },
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
        texture::{FallbackImage, TextureCache},
        view::ViewUniformOffset,
        RenderApp, RenderStage,
    },
};
use serde::Serialize;

pub const HDR_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

pub struct PostProcessPlugin;
impl Plugin for PostProcessPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(ExtractComponentPlugin::<FsrConstantsUniform>::default())
            .add_plugin(UniformComponentPlugin::<FsrConstantsUniform>::default());

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

#[derive(Resource)]
pub struct PostProcessPipeline {
    pub view_layout: BindGroupLayout,
    pub deferred_layout: BindGroupLayout,
    pub sampler_layout: BindGroupLayout,
    pub denoise_internal_layout: BindGroupLayout,
    pub denoise_render_layout: BindGroupLayout,
    pub tone_mapping_layout: BindGroupLayout,
    pub smaa_layout: BindGroupLayout,
    pub taa_layout: BindGroupLayout,
    pub upscale_layout: BindGroupLayout,
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
                    visibility: ShaderStages::all(),
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let denoise_internal_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    // Internal 0
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::ReadWrite,
                            format: HDR_TEXTURE_FORMAT,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Internal 1
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::ReadWrite,
                            format: HDR_TEXTURE_FORMAT,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Internal 2
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::ReadWrite,
                            format: HDR_TEXTURE_FORMAT,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Internal 3
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::ReadWrite,
                            format: HDR_TEXTURE_FORMAT,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Internal Variance
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::ReadWrite,
                            format: VARIANCE_TEXTURE_FORMAT,
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
                    // Variance
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
                    // Previous Radiance
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
                    // Render
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Radiance Output
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::ReadWrite,
                            format: HDR_TEXTURE_FORMAT,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Output
                    BindGroupLayoutEntry {
                        binding: 5,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::ReadWrite,
                            format: HDR_TEXTURE_FORMAT,
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
            ],
        });

        let upscale_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(FsrConstantsUniform::min_size()),
                    },
                    count: None,
                },
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
            ],
        });

        let output_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::all(),
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: HDR_TEXTURE_FORMAT,
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
            smaa_layout,
            taa_layout,
            upscale_layout,
            output_layout,
        }
    }
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, FromPrimitive)]
#[serde(rename_all = "snake_case")]
pub enum PostProcessEntryPoint {
    #[default]
    Demodulation = 0,
    Denoise = 1,
    ToneMapping = 2,
    JasmineTaa = 3,
    SmaaTu4x = 4,
    SmaaTu4xExtrapolate = 5,
    Upscale = 6,
    UpscaleSharpen = 7,
}

bitflags::bitflags! {
    #[repr(transparent)]
    pub struct PostProcessPipelineKey: u32 {
        const ENTRY_POINT_BITS          = PostProcessPipelineKey::ENTRY_POINT_MASK_BITS;
        const FIREFLY_FILTERING_BITS    = 1 << PostProcessPipelineKey::FIREFLY_FILTERING_SHIFT_BITS;
        const DENOISE_LEVEL_BITS        = PostProcessPipelineKey::DENOISE_LEVEL_MASK_BITS << PostProcessPipelineKey::DENOISE_LEVEL_SHIFT_BITS;
    }
}

impl PostProcessPipelineKey {
    const ENTRY_POINT_MASK_BITS: u32 = 0xF;
    const DENOISE_LEVEL_MASK_BITS: u32 = 0b11;
    const DENOISE_LEVEL_SHIFT_BITS: u32 = 32 - 2;
    const FIREFLY_FILTERING_SHIFT_BITS: u32 = 8;

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
        let mut entry_point = serde_variant::to_variant_name(&key.entry_point())
            .unwrap()
            .into();

        let mut shader_defs: Vec<_> = vec![];
        shader_defs.push(ShaderDefVal::Int(
            "MAX_DIRECTIONAL_LIGHTS".into(),
            MAX_DIRECTIONAL_LIGHTS as i32,
        ));

        if key.contains(PostProcessPipelineKey::FIREFLY_FILTERING_BITS) {
            shader_defs.push("FIREFLY_FILTERING".into());
        }

        let (layout, shader) = match key.entry_point() {
            PostProcessEntryPoint::Demodulation | PostProcessEntryPoint::Denoise => {
                let layout = vec![
                    self.view_layout.clone(),
                    self.deferred_layout.clone(),
                    self.sampler_layout.clone(),
                    self.denoise_internal_layout.clone(),
                    self.denoise_render_layout.clone(),
                ];
                shader_defs.push(format!("DENOISE_LEVEL_{}", key.denoise_level()).into());
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
            PostProcessEntryPoint::JasmineTaa => {
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
            PostProcessEntryPoint::SmaaTu4x => {
                let layout = vec![
                    self.view_layout.clone(),
                    self.deferred_layout.clone(),
                    self.sampler_layout.clone(),
                    self.smaa_layout.clone(),
                    self.output_layout.clone(),
                ];
                let shader = SMAA_SHADER_HANDLE.typed();
                (layout, shader)
            }
            PostProcessEntryPoint::SmaaTu4xExtrapolate => {
                let layout = vec![
                    self.view_layout.clone(),
                    self.deferred_layout.clone(),
                    self.sampler_layout.clone(),
                    self.smaa_layout.clone(),
                    self.output_layout.clone(),
                ];
                let shader = SMAA_SHADER_HANDLE.typed();
                (layout, shader)
            }
            PostProcessEntryPoint::Upscale => {
                let layout = vec![
                    self.sampler_layout.clone(),
                    self.upscale_layout.clone(),
                    self.output_layout.clone(),
                ];
                entry_point = "main".into();
                let shader = FSR1_EASU_SHADER_HANDLE.typed();
                (layout, shader)
            }
            PostProcessEntryPoint::UpscaleSharpen => {
                let layout = vec![
                    self.sampler_layout.clone(),
                    self.upscale_layout.clone(),
                    self.output_layout.clone(),
                ];
                entry_point = "main".into();
                let shader = FSR1_RCAS_SHADER_HANDLE.typed();
                (layout, shader)
            }
        };

        ComputePipelineDescriptor {
            label: None,
            layout: Some(layout),
            shader,
            shader_defs,
            entry_point,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Component, ShaderType)]
pub struct FsrConstantsUniform {
    // NOTE: do not remove this yet
    //pub const_0: UVec4,
    //pub const_1: UVec4,
    //pub const_2: UVec4,
    //pub const_3: UVec4,
    //pub sample: UVec4,
    pub input_viewport_in_pixels: Vec2,
    pub input_size_in_pixels: Vec2,
    pub output_size_in_pixels: Vec2,
    pub sharpness: f32,
    pub hdr: u32,
}

impl ExtractComponent for FsrConstantsUniform {
    type Query = (&'static Camera, &'static HikariSettings);
    type Filter = ();
    type Out = Self;

    fn extract_component((camera, settings): QueryItem<Self::Query>) -> Option<Self::Out> {
        let size = camera.physical_target_size().unwrap_or_default();
        let scale = settings.upscale.ratio().recip();
        let scaled_size = (scale * size.as_vec2()).ceil();
        Some(Self {
            input_viewport_in_pixels: scaled_size,
            input_size_in_pixels: scaled_size,
            output_size_in_pixels: size.as_vec2(),
            sharpness: settings.upscale.sharpness(),
            hdr: 0,
        })
    }
}

// NOTE! Don't delete, might be used soon, instead of calulating this on GPU
/*fn get_fsr_constants(ratio: f32, hdr_rcas: bool, camera: &ExtractedCamera) -> FSRConstantsUniform {
    let mut fsr_constant = FSRConstantsUniform::default();
    let size = camera.physical_target_size.unwrap();
    let size_x = size.x as f32;
    let size_y = size.x as f32;
    compute_fsr_constants(
        &mut fsr_constant.const_0,
        &mut fsr_constant.const_1,
        &mut fsr_constant.const_2,
        &mut fsr_constant.const_3,
        size_x,
        size_y,
        size_x,
        size_y,
        size_x * ratio,
        size_y * ratio,
    );

    fsr_constant.sample.x = if hdr_rcas { 1 } else { 0 };
    fsr_constant
}

pub union U32F32Union {
    pub u: u32,
    pub f: f32,
}

fn f32_u32(a: f32) -> u32 {
    let uf = U32F32Union { f: a };
    unsafe { uf.u }
}

fn compute_fsr_constants(
    con0: &mut UVec4,
    con1: &mut UVec4,
    con2: &mut UVec4,
    con3: &mut UVec4,
    // This the rendered image resolution being upscaled
    input_viewport_in_pixels_x: f32,
    input_viewport_in_pixels_y: f32,
    // This is the resolution of the resource containing the input image (useful for dynamic resolution)
    input_size_in_pixels_x: f32,
    input_size_in_pixels_y: f32,
    // This is the display resolution which the input image gets upscaled to
    output_size_in_pixels_x: f32,
    output_size_in_pixels_y: f32,
) {
    // Output integer position to a pixel position in viewport.
    con0[0] = f32_u32(input_viewport_in_pixels_x * (1.0 / output_size_in_pixels_x));
    con0[1] = f32_u32(input_viewport_in_pixels_y * (1.0 / output_size_in_pixels_y));
    con0[2] = f32_u32(0.5 * input_viewport_in_pixels_x * (1.0 / output_size_in_pixels_x) - 0.5);
    con0[3] = f32_u32(0.5 * input_viewport_in_pixels_y * (1.0 / output_size_in_pixels_y) - 0.5);

    // Viewport pixel position to normalized image space.
    // This is used to get upper-left of 'F' tap.
    con1[0] = (1.0 / input_size_in_pixels_x) as u32;
    con1[1] = (1.0 / input_size_in_pixels_y) as u32;
    // Centers of gather4, first offset from upper-left of 'F'.
    //      +---+---+
    //      |   |   |
    //      +--(0)--+
    //      | b | c |
    //  +---F---+---+---+
    //  | e | f | g | h |
    //  +--(1)--+--(2)--+
    //  | i | j | k | l |
    //  +---+---+---+---+
    //      | n | o |
    //      +--(3)--+
    //      |   |   |
    //      +---+---+
    con1[2] = f32_u32(1.0 * (1.0 / input_size_in_pixels_x));
    con1[3] = f32_u32(-1.0 * (1.0 / input_size_in_pixels_y));
    // These are from (0) instead of 'F'.
    con2[0] = f32_u32(-1.0 * (1.0 / input_size_in_pixels_x));
    con2[1] = f32_u32(2.0 * (1.0 / input_size_in_pixels_y));
    con2[2] = f32_u32(1.0 * (1.0 / input_size_in_pixels_x));
    con2[3] = f32_u32(2.0 * (1.0 / input_size_in_pixels_y));
    con3[0] = f32_u32(0.0 * (1.0 / input_size_in_pixels_x));
    con3[1] = f32_u32(4.0 * (1.0 / input_size_in_pixels_y));
    con3[2] = 0;
    con3[3] = 0;
}*/

#[derive(Component)]
pub struct PostProcessTextures {
    pub head: usize,
    pub nearest_sampler: Sampler,
    pub linear_sampler: Sampler,
    pub fallback: TextureView,
    pub denoise_internal: [TextureView; 4],
    pub denoise_internal_variance: TextureView,
    pub denoise_radiance: [TextureView; 6],
    pub denoise_render: [TextureView; 3],
    pub tone_mapping_output: [TextureView; 2],
    pub taa_output: [TextureView; 2],
    pub upscale_output: [TextureView; 2],
}

fn prepare_post_process_textures(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    cameras: Query<(Entity, &ExtractedCamera, &FrameCounter, &HikariSettings)>,
) {
    let texture_usage = TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
    let fallback = texture_cache
        .get(
            &render_device,
            TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: HDR_TEXTURE_FORMAT,
                usage: texture_usage,
            },
        )
        .default_view;

    for (entity, camera, counter, settings) in &cameras {
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
                            usage: texture_usage,
                        },
                    )
                    .default_view
            };

            macro_rules! create_texture_array {
                [$texture_format:ident, $scale:ident; $count:literal] => {
                    [(); $count].map(|_| create_texture($texture_format, $scale))
                };
                [$texture_format:ident, $scale:literal; $count:literal] => {
                    [(); $count].map(|_| create_texture($texture_format, $scale))
                };
                [$texture:ident; $count:literal] => {
                    [(); $count].map(|_| $texture.clone())
                };
            }

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

            let mut scale = settings.upscale.ratio().recip();

            let denoise_internal_variance = create_texture(VARIANCE_TEXTURE_FORMAT, scale);
            let denoise_internal = create_texture_array![HDR_TEXTURE_FORMAT, scale; 4];
            let denoise_radiance = create_texture_array![HDR_TEXTURE_FORMAT, scale; 6];
            let denoise_render = create_texture_array![HDR_TEXTURE_FORMAT, scale; 3];

            let tone_mapping_output = create_texture_array![HDR_TEXTURE_FORMAT, scale; 2];

            let upscale_output = match settings.upscale {
                Upscale::SmaaTu4x { .. } => {
                    scale *= 2.0;
                    create_texture_array![HDR_TEXTURE_FORMAT, scale; 2]
                }
                Upscale::Fsr1 { .. } => create_texture_array![HDR_TEXTURE_FORMAT, 1.0; 2],
                Upscale::None => create_texture_array![fallback; 2],
            };

            let taa_output = match settings.taa {
                Taa::Jasmine => {
                    create_texture_array![HDR_TEXTURE_FORMAT, scale; 2]
                }
                Taa::None => create_texture_array![fallback; 2],
            };

            commands.entity(entity).insert(PostProcessTextures {
                head: counter.0 % 2,
                nearest_sampler,
                linear_sampler,
                fallback: fallback.clone(),
                denoise_internal,
                denoise_internal_variance,
                denoise_radiance,
                denoise_render,
                tone_mapping_output,
                taa_output,
                upscale_output,
            });
        }
    }
}

#[derive(Resource)]
pub struct CachedPostProcessPipelines {
    demodulation: CachedComputePipelineId,
    denoise: [CachedComputePipelineId; 4],
    denoise_indirect: [CachedComputePipelineId; 4],
    tone_mapping: CachedComputePipelineId,
    taa_jasmine: CachedComputePipelineId,
    smaa_tu4x: CachedComputePipelineId,
    smaa_tu4x_extrapolate: CachedComputePipelineId,
    upscale: CachedComputePipelineId,
    upscale_sharpen: CachedComputePipelineId,
}

fn queue_post_process_pipelines(
    mut commands: Commands,
    pipeline: Res<PostProcessPipeline>,
    mut pipelines: ResMut<SpecializedComputePipelines<PostProcessPipeline>>,
    mut pipeline_cache: ResMut<PipelineCache>,
) {
    let key = PostProcessPipelineKey::from_entry_point(PostProcessEntryPoint::Demodulation);
    let demodulation = pipelines.specialize(&mut pipeline_cache, &pipeline, key);

    let denoise = [0, 1, 2, 3].map(|level| {
        let mut key = PostProcessPipelineKey::from_entry_point(PostProcessEntryPoint::Denoise);
        key |= PostProcessPipelineKey::from_denoise_level(level);
        pipelines.specialize(&mut pipeline_cache, &pipeline, key)
    });

    let denoise_indirect = [0, 1, 2, 3].map(|level| {
        let mut key = PostProcessPipelineKey::from_entry_point(PostProcessEntryPoint::Denoise);
        key |= PostProcessPipelineKey::from_denoise_level(level);
        match level {
            0 | 1 | 2 => key |= PostProcessPipelineKey::FIREFLY_FILTERING_BITS,
            _ => {}
        }
        pipelines.specialize(&mut pipeline_cache, &pipeline, key)
    });

    let key = PostProcessPipelineKey::from_entry_point(PostProcessEntryPoint::ToneMapping);
    let tone_mapping = pipelines.specialize(&mut pipeline_cache, &pipeline, key);

    let key = PostProcessPipelineKey::from_entry_point(PostProcessEntryPoint::JasmineTaa);
    let taa_jasmine = pipelines.specialize(&mut pipeline_cache, &pipeline, key);

    let key = PostProcessPipelineKey::from_entry_point(PostProcessEntryPoint::SmaaTu4x);
    let smaa_tu4x = pipelines.specialize(&mut pipeline_cache, &pipeline, key);

    let key = PostProcessPipelineKey::from_entry_point(PostProcessEntryPoint::SmaaTu4xExtrapolate);
    let smaa_tu4x_extrapolate = pipelines.specialize(&mut pipeline_cache, &pipeline, key);

    let key = PostProcessPipelineKey::from_entry_point(PostProcessEntryPoint::Upscale);
    let upscale = pipelines.specialize(&mut pipeline_cache, &pipeline, key);

    let key = PostProcessPipelineKey::from_entry_point(PostProcessEntryPoint::UpscaleSharpen);
    let upscale_sharpen = pipelines.specialize(&mut pipeline_cache, &pipeline, key);

    commands.insert_resource(CachedPostProcessPipelines {
        demodulation,
        denoise,
        denoise_indirect,
        tone_mapping,
        taa_jasmine,
        smaa_tu4x,
        smaa_tu4x_extrapolate,
        upscale,
        upscale_sharpen,
    })
}

#[derive(Component, Clone)]
pub struct PostProcessBindGroup {
    pub deferred: BindGroup,
    pub sampler: BindGroup,
    pub denoise_internal: BindGroup,
    pub denoise_render: Vec<BindGroup>,
    pub tone_mapping: BindGroup,
    pub tone_mapping_output: BindGroup,
    pub smaa: BindGroup,
    pub smaa_output: BindGroup,
    pub taa: BindGroup,
    pub taa_output: BindGroup,
    pub upscale: BindGroup,
    pub upscale_output: BindGroup,
    pub upscale_sharpen: BindGroup,
    pub upscale_sharpen_output: BindGroup,
}

#[allow(clippy::type_complexity)]
fn queue_post_process_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline: Res<PostProcessPipeline>,
    images: Res<RenderAssets<Image>>,
    fallback: Res<FallbackImage>,
    fsr_constants_uniforms: Res<ComponentUniforms<FsrConstantsUniform>>,
    query: Query<
        (
            Entity,
            &PrepassTextures,
            &LightTextures,
            &PostProcessTextures,
            &HikariSettings,
        ),
        With<ExtractedCamera>,
    >,
) {
    let fsr_constants_binding = match fsr_constants_uniforms.binding() {
        Some(binding) => binding,
        None => return,
    };

    for (entity, prepass, light, post_process, settings) in &query {
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
                    resource: BindingResource::TextureView(&post_process.denoise_internal[0]),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&post_process.denoise_internal[1]),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&post_process.denoise_internal[2]),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&post_process.denoise_internal[3]),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&post_process.denoise_internal_variance),
                },
            ],
        });

        let mut denoise_render = [0, 1, 2]
            .map(|id| {
                render_device.create_bind_group(&BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.denoise_render_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::TextureView(&light.albedo),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::TextureView(&light.variance[id]),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: BindingResource::TextureView(
                                &post_process.denoise_radiance[previous + 2 * id],
                            ),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: BindingResource::TextureView(&light.render[id]),
                        },
                        BindGroupEntry {
                            binding: 4,
                            resource: BindingResource::TextureView(
                                &post_process.denoise_radiance[current + 2 * id],
                            ),
                        },
                        BindGroupEntry {
                            binding: 5,
                            resource: BindingResource::TextureView(
                                &post_process.denoise_render[id],
                            ),
                        },
                    ],
                })
            })
            .to_vec();

        let (direct_render, emissive_render, mut indirect_render) = match settings.denoise {
            false => (&light.render[0], &light.render[1], &light.render[2]),
            true => (
                &post_process.denoise_render[0],
                &post_process.denoise_render[1],
                &post_process.denoise_render[2],
            ),
        };

        if settings.indirect_bounces == 0 {
            // Do not denoise when there is no indirect rendering pass.
            denoise_render.pop();
            // Use fallback texture when there is no indirect denoise pass.
            indirect_render = &post_process.fallback;
        }

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
                resource: BindingResource::TextureView(&post_process.tone_mapping_output[current]),
            }],
        });

        let smaa = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.smaa_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &post_process.tone_mapping_output[previous],
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &post_process.tone_mapping_output[current],
                    ),
                },
            ],
        });
        let smaa_output = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.output_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&post_process.upscale_output[0]),
            }],
        });

        let taa_input_texture = match settings.upscale {
            Upscale::SmaaTu4x { .. } => &post_process.upscale_output[0],
            _ => &post_process.tone_mapping_output[current],
        };
        let taa = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.taa_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&post_process.taa_output[previous]),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(taa_input_texture),
                },
            ],
        });
        let taa_output = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.output_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&post_process.taa_output[current]),
            }],
        });

        let upscale_input_texture = match settings.taa {
            Taa::Jasmine => &post_process.taa_output[current],
            Taa::None => &post_process.tone_mapping_output[current],
        };

        let upscale = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.upscale_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: fsr_constants_binding.clone(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(upscale_input_texture),
                },
            ],
        });
        let upscale_output = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.output_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&post_process.upscale_output[0]),
            }],
        });

        let upscale_sharpen = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.upscale_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: fsr_constants_binding.clone(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&post_process.upscale_output[0]),
                },
            ],
        });
        let upscale_sharpen_output = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.output_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&post_process.upscale_output[1]),
            }],
        });

        commands.entity(entity).insert(PostProcessBindGroup {
            deferred,
            sampler,
            denoise_internal,
            denoise_render,
            tone_mapping,
            tone_mapping_output,
            smaa,
            smaa_output,
            taa,
            taa_output,
            upscale,
            upscale_output,
            upscale_sharpen,
            upscale_sharpen_output,
        });
    }
}

#[allow(clippy::type_complexity)]
pub struct PostProcessNode {
    query: QueryState<(
        &'static ExtractedCamera,
        &'static DynamicUniformIndex<FrameUniform>,
        &'static ViewUniformOffset,
        &'static PreviousViewUniformOffset,
        &'static ViewLightsUniformOffset,
        &'static PostProcessBindGroup,
        &'static DynamicUniformIndex<FsrConstantsUniform>,
        &'static HikariSettings,
    )>,
}

impl PostProcessNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            query: world.query_filtered(),
        }
    }
}

impl Node for PostProcessNode {
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
            post_process_bind_group,
            fsr_constants_uniform,
            settings,
        ) = match self.query.get_manual(world, entity) {
            Ok(query) => query,
            Err(_) => return Ok(()),
        };
        let view_bind_group = match world.get_resource::<PrepassBindGroup>() {
            Some(bind_group) => &bind_group.view,
            None => return Ok(()),
        };

        let pipelines = world.resource::<CachedPostProcessPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let size = camera.physical_target_size.unwrap();
        let scale = settings.upscale.ratio().recip();
        let mut scaled_size = (scale * size.as_vec2()).ceil().as_uvec2();

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
        pass.set_bind_group(1, &post_process_bind_group.deferred, &[]);
        pass.set_bind_group(2, &post_process_bind_group.sampler, &[]);

        if settings.denoise {
            pass.set_bind_group(3, &post_process_bind_group.denoise_internal, &[]);

            let denoise_pipelines = [
                pipelines.denoise,
                pipelines.denoise,
                pipelines.denoise_indirect,
            ];

            for (render_bind_group, denoise) in post_process_bind_group
                .denoise_render
                .iter()
                .zip(denoise_pipelines.iter())
            {
                pass.set_bind_group(4, render_bind_group, &[]);

                if let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.demodulation)
                {
                    pass.set_pipeline(pipeline);

                    let count = (scaled_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                    pass.dispatch_workgroups(count.x, count.y, 1);
                }

                for pipeline in denoise
                    .iter()
                    .filter_map(|pipeline| pipeline_cache.get_compute_pipeline(*pipeline))
                {
                    pass.set_pipeline(pipeline);

                    let count = (scaled_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                    pass.dispatch_workgroups(count.x, count.y, 1);
                }
            }
        }

        pass.set_bind_group(3, &post_process_bind_group.tone_mapping, &[]);
        pass.set_bind_group(4, &post_process_bind_group.tone_mapping_output, &[]);

        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.tone_mapping) {
            pass.set_pipeline(pipeline);

            let count = (scaled_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(count.x, count.y, 1);
        }

        if matches!(settings.upscale, Upscale::SmaaTu4x { .. }) {
            pass.set_bind_group(3, &post_process_bind_group.smaa, &[]);
            pass.set_bind_group(4, &post_process_bind_group.smaa_output, &[]);

            if let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.smaa_tu4x) {
                pass.set_pipeline(pipeline);

                let count = (scaled_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                pass.dispatch_workgroups(count.x, count.y, 1);
            }

            if let Some(pipeline) =
                pipeline_cache.get_compute_pipeline(pipelines.smaa_tu4x_extrapolate)
            {
                pass.set_pipeline(pipeline);

                let count = (scaled_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                pass.dispatch_workgroups(count.x, count.y, 1);
            }

            // The in-out size after the upscale pass is 4x larger.
            scaled_size *= 2;
        }

        if matches!(settings.taa, Taa::Jasmine) {
            let pipeline = pipelines.taa_jasmine;

            pass.set_bind_group(3, &post_process_bind_group.taa, &[]);
            pass.set_bind_group(4, &post_process_bind_group.taa_output, &[]);

            if let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline) {
                pass.set_pipeline(pipeline);

                let count = (scaled_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
                pass.dispatch_workgroups(count.x, count.y, 1);
            }
        }

        if matches!(settings.upscale, Upscale::Fsr1 { .. }) {
            pass.set_bind_group(0, &post_process_bind_group.sampler, &[]);
            pass.set_bind_group(
                1,
                &post_process_bind_group.upscale,
                &[fsr_constants_uniform.index()],
            );
            pass.set_bind_group(2, &post_process_bind_group.upscale_output, &[]);

            if let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.upscale) {
                pass.set_pipeline(pipeline);

                let count = (size * 2 + 15) / 16;
                pass.dispatch_workgroups(count.x, count.y, 1);
            }

            pass.set_bind_group(
                1,
                &post_process_bind_group.upscale_sharpen,
                &[fsr_constants_uniform.index()],
            );
            pass.set_bind_group(2, &post_process_bind_group.upscale_sharpen_output, &[]);

            if let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.upscale_sharpen) {
                pass.set_pipeline(pipeline);

                let count = (size * 2 + 15) / 16;
                pass.dispatch_workgroups(count.x, count.y, 1);
            }
        }

        Ok(())
    }
}
