use crate::{MIN_VOXEL_COUNT, MIPMAP_SHADER_HANDLE};
use bevy::{
    ecs::system::{
        lifetimeless::{SRes, SResMut},
        SystemParamItem,
    },
    pbr::{MeshPipeline, SpecializedMaterial},
    prelude::*,
    reflect::TypeUuid,
    render::{
        primitives::Aabb,
        render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin},
        render_resource::{std140::AsStd140, std430::AsStd430, *},
        renderer::{RenderDevice, RenderQueue},
        texture::TextureCache,
        RenderApp,
    },
};
use itertools::Itertools;
use std::num::NonZeroU32;

pub struct VolumePlugin;
impl Plugin for VolumePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(RenderAssetPlugin::<VolumeAsset>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<VolumePipeline>();
    }
}

pub struct VolumePipeline {
    pub mesh_pipeline: MeshPipeline,
    pub material_layout: BindGroupLayout,
    pub volume_layout: BindGroupLayout,

    pub voxel_buffer_layout: BindGroupLayout,
    pub clear_pipeline: CachedComputePipelineId,
    pub transfer_pipeline: CachedComputePipelineId,

    pub mipmap_layout: BindGroupLayout,
    pub mipmap_pipeline: CachedComputePipelineId,

    pub anisotropic_layout: BindGroupLayout,
    pub anisotropic_pipeline: CachedComputePipelineId,
}

impl FromWorld for VolumePipeline {
    fn from_world(world: &mut World) -> Self {
        let mesh_pipeline = world.resource::<MeshPipeline>().clone();

        let render_device = world.resource::<RenderDevice>();
        let material_layout = StandardMaterial::bind_group_layout(render_device);

        let volume_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // volume bound
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(GpuVolume::std140_size_static() as u64),
                    },
                    count: None,
                },
                // volume clusters
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(u32::std140_size_static() as u64),
                    },
                    count: None,
                },
            ],
        });

        let voxel_buffer_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    // voxel buffer
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: BufferSize::new(
                                <[u32; MIN_VOXEL_COUNT]>::std430_size_static() as u64,
                            ),
                        },
                        count: None,
                    },
                    // voxel texture
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba16Float,
                            view_dimension: TextureViewDimension::D3,
                        },
                        count: None,
                    },
                ],
            });

        let mipmap_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: (0u32..6)
                .map(
                    // output textures
                    |direction| BindGroupLayoutEntry {
                        binding: direction,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba16Float,
                            view_dimension: TextureViewDimension::D3,
                        },
                        count: None,
                    },
                )
                .chain([
                    // input texture
                    BindGroupLayoutEntry {
                        binding: 6,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    },
                ])
                .collect_vec()
                .as_slice(),
        });

        let anisotropic_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    // output texture
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba16Float,
                            view_dimension: TextureViewDimension::D3,
                        },
                        count: None,
                    },
                    // input texture
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // direction uniform
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: BufferSize::new(
                                <[u32; 6]>::std140_size_static() as u64
                            ),
                        },
                        count: None,
                    },
                ],
            });

        let mut pipeline_cache = world.resource_mut::<PipelineCache>();

        let clear_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![voxel_buffer_layout.clone()]),
            shader: MIPMAP_SHADER_HANDLE.typed(),
            shader_defs: vec!["VOXEL_BUFFER".into()],
            entry_point: "clear".into(),
        });

        let transfer_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![voxel_buffer_layout.clone()]),
            shader: MIPMAP_SHADER_HANDLE.typed(),
            shader_defs: vec!["VOXEL_BUFFER".into()],
            entry_point: "transfer".into(),
        });

        let mipmap_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![mipmap_layout.clone()]),
            shader: MIPMAP_SHADER_HANDLE.typed(),
            shader_defs: vec![],
            entry_point: "mipmap".into(),
        });

        let anisotropic_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: None,
                layout: Some(vec![anisotropic_layout.clone()]),
                shader: MIPMAP_SHADER_HANDLE.typed(),
                shader_defs: vec!["MIPMAP_ANISOTROPIC".into()],
                entry_point: "mipmap".into(),
            });

        Self {
            mesh_pipeline,
            material_layout,
            volume_layout,
            voxel_buffer_layout,
            clear_pipeline,
            transfer_pipeline,
            mipmap_layout,
            mipmap_pipeline,
            anisotropic_layout,
            anisotropic_pipeline,
        }
    }
}

#[derive(Debug, Clone, TypeUuid)]
#[uuid = "6c15c8a0-cb01-4913-b1ee-86b11a60cf58"]
pub struct VolumeAsset {
    pub resolution: u32,
}

impl RenderAsset for VolumeAsset {
    type ExtractedAsset = VolumeAsset;
    type PreparedAsset = GpuVolumeAsset;
    type Param = (
        SRes<RenderDevice>,
        SRes<RenderQueue>,
        SResMut<TextureCache>,
        SRes<VolumePipeline>,
    );

    fn extract_asset(&self) -> Self::ExtractedAsset {
        self.clone()
    }

    fn prepare_asset(
        volume: Self::ExtractedAsset,
        (render_device, render_queue, texture_cache, volume_pipeline): &mut SystemParamItem<
            Self::Param,
        >,
    ) -> Result<Self::PreparedAsset, PrepareAssetError<Self::ExtractedAsset>> {
        let mut mipmap_uniforms = DynamicUniformVec::default();
        let mut mipmap_uniform_offsets = vec![];
        for direction in 0u32..6 {
            mipmap_uniform_offsets.push(mipmap_uniforms.push(direction));
        }
        mipmap_uniforms.write_buffer(render_device, render_queue);
        let mipmap_uniform_offsets = mipmap_uniform_offsets.try_into().unwrap();

        let mut voxel_buffer = StorageBuffer::default();
        let mut voxel_data = vec![0; volume.resolution.pow(3) as usize];
        voxel_buffer.append(&mut voxel_data);
        voxel_buffer.write_buffer(render_device, render_queue);

        let voxel_texture = texture_cache.get(
            render_device,
            TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: volume.resolution,
                    height: volume.resolution,
                    depth_or_array_layers: volume.resolution,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            },
        );

        let size = Extent3d {
            width: volume.resolution / 2,
            height: volume.resolution / 2,
            depth_or_array_layers: volume.resolution / 2,
        };
        let anisotropic_textures = [(); 6].map(|_| {
            texture_cache.get(
                render_device,
                TextureDescriptor {
                    label: None,
                    size,
                    mip_level_count: size.max_mips(),
                    sample_count: 1,
                    dimension: TextureDimension::D3,
                    format: TextureFormat::Rgba16Float,
                    usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                },
            )
        });
        let voxel_sampler = render_device.create_sampler(&SamplerDescriptor {
            label: None,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });

        let voxel_buffer_bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &volume_pipeline.voxel_buffer_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: voxel_buffer.binding().unwrap(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&voxel_texture.default_view),
                },
            ],
        });

        let texture_views = anisotropic_textures
            .clone()
            .map(|texture| texture.default_view);
        let mipmap_bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &volume_pipeline.mipmap_layout,
            entries: texture_views
                .iter()
                .enumerate()
                .map(|(direction, texture_view)| BindGroupEntry {
                    binding: direction as u32,
                    resource: BindingResource::TextureView(texture_view),
                })
                .chain([BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::TextureView(&voxel_texture.default_view),
                }])
                .collect_vec()
                .as_slice(),
        });

        let texture_views = (0..size.max_mips()).map(|level| {
            anisotropic_textures.clone().map(|texture| {
                texture.texture.create_view(&TextureViewDescriptor {
                    base_mip_level: level as u32,
                    mip_level_count: NonZeroU32::new(1),
                    ..Default::default()
                })
            })
        });
        let anisotropic_bind_groups = texture_views
            .tuple_windows()
            .map(|(input_textures, output_textures)| {
                itertools::zip_eq(input_textures, output_textures)
                    .map(|(texture_in, texture_out)| {
                        render_device.create_bind_group(&BindGroupDescriptor {
                            label: None,
                            layout: &volume_pipeline.anisotropic_layout,
                            entries: &[
                                BindGroupEntry {
                                    binding: 0,
                                    resource: BindingResource::TextureView(&texture_out),
                                },
                                BindGroupEntry {
                                    binding: 1,
                                    resource: BindingResource::TextureView(&texture_in),
                                },
                                BindGroupEntry {
                                    binding: 2,
                                    resource: mipmap_uniforms.binding().unwrap(),
                                },
                            ],
                        })
                    })
                    .collect_vec()
                    .try_into()
                    .unwrap()
            })
            .collect_vec();

        let voxel_texture = voxel_texture.texture;
        let anisotropic_textures = anisotropic_textures.map(|texture| texture.texture);

        Ok(GpuVolumeAsset {
            mipmap_uniforms,
            mipmap_uniform_offsets,
            voxel_buffer,
            voxel_texture,
            anisotropic_textures,
            voxel_sampler,
            voxel_buffer_bind_group,
            mipmap_bind_group,
            anisotropic_bind_groups,
        })
    }
}

pub struct GpuVolumeAsset {
    pub mipmap_uniforms: DynamicUniformVec<u32>,
    pub mipmap_uniform_offsets: [u32; 6],

    pub voxel_buffer: StorageBuffer<u32>,
    pub voxel_texture: Texture,
    pub anisotropic_textures: [Texture; 6],
    pub voxel_sampler: Sampler,

    pub voxel_buffer_bind_group: BindGroup,
    pub mipmap_bind_group: BindGroup,
    pub anisotropic_bind_groups: Vec<[BindGroup; 6]>,
}

#[derive(Component, Debug, Clone)]
pub struct Volume {
    pub asset: Handle<VolumeAsset>,
    pub bound: Aabb,
    views: Option<[Entity; 3]>,
    clusters: Vec<u32>,
}

impl Volume {
    pub fn new(asset: Handle<VolumeAsset>, min: Vec3, max: Vec3) -> Self {
        Self {
            asset,
            bound: Aabb::from_min_max(min, max),
            views: None,
            clusters: vec![],
        }
    }
}

#[derive(AsStd140)]
pub struct GpuVolume {
    pub min: Vec3,
    pub max: Vec3,
    pub resolution: u32,
}

impl GpuVolume {
    pub fn new(aabb: Aabb, resolution: u32) -> Self {
        Self {
            min: aabb.min().into(),
            max: aabb.max().into(),
            resolution,
        }
    }
}

#[derive(Component)]
pub struct VolumeMeta {
    pub volume_uniform: UniformVec<GpuVolume>,
    pub clusters_uniform: UniformVec<u32>,
}
