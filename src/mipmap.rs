use crate::{volume::GpuVoxelBuffer, MIPMAP_SHADER_HANDLE};
use bevy::{
    prelude::*,
    render::{
        render_resource::{std140::AsStd140, std430::AsStd430, *},
        renderer::{RenderDevice, RenderQueue},
        RenderApp,
    },
};
use itertools::Itertools;

pub struct MipmapPlugin;
impl Plugin for MipmapPlugin {
    fn build(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<MipmapPipeline>()
            .init_resource::<MipmapMeta>();
    }
}

pub struct MipmapPipeline {
    pub voxel_buffer_bind_group_layout: BindGroupLayout,
    pub clear_pipeline: CachedComputePipelineId,
    pub fill_pipeline: CachedComputePipelineId,

    pub input_texture_bind_group_layout: BindGroupLayout,
    pub output_texture_base_bind_group_layout: BindGroupLayout,
    pub output_texture_bind_group_layout: BindGroupLayout,
    pub mipmap_bind_group_layout: BindGroupLayout,

    pub mipmap_base_pipeline: CachedComputePipelineId,
    pub mipmap_pipeline: CachedComputePipelineId,
}
impl FromWorld for MipmapPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let voxel_buffer_bind_group_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: BufferSize::new(
                                GpuVoxelBuffer::std430_size_static() as u64
                            ),
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba16Float,
                            view_dimension: TextureViewDimension::D3,
                        },
                        count: None,
                    },
                ],
            });

        let input_texture_bind_group_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        let output_texture_base_bind_group_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: (0u32..6)
                    .map(|direction| BindGroupLayoutEntry {
                        binding: direction,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba16Float,
                            view_dimension: TextureViewDimension::D3,
                        },
                        count: None,
                    })
                    .collect_vec()
                    .as_slice(),
            });

        let output_texture_bind_group_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D3,
                    },
                    count: None,
                }],
            });

        let mipmap_bind_group_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: BufferSize::new(GpuMipmap::std140_size_static() as u64),
                    },
                    count: None,
                }],
            });

        let mut pipeline_cache = world.resource_mut::<PipelineCache>();

        let clear_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![voxel_buffer_bind_group_layout.clone()]),
            shader: MIPMAP_SHADER_HANDLE.typed(),
            shader_defs: vec!["VOXEL_BUFFER".into()],
            entry_point: "clear".into(),
        });

        let fill_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![voxel_buffer_bind_group_layout.clone()]),
            shader: MIPMAP_SHADER_HANDLE.typed(),
            shader_defs: vec!["VOXEL_BUFFER".into()],
            entry_point: "fill".into(),
        });

        let mipmap_base_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: None,
                layout: Some(vec![
                    input_texture_bind_group_layout.clone(),
                    output_texture_base_bind_group_layout.clone(),
                ]),
                shader: MIPMAP_SHADER_HANDLE.typed(),
                shader_defs: vec!["BASE_LEVEL".into()],
                entry_point: "mipmap".into(),
            });

        let mipmap_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![
                input_texture_bind_group_layout.clone(),
                output_texture_bind_group_layout.clone(),
                mipmap_bind_group_layout.clone(),
            ]),
            shader: MIPMAP_SHADER_HANDLE.typed(),
            shader_defs: vec![],
            entry_point: "mipmap".into(),
        });

        Self {
            voxel_buffer_bind_group_layout,
            clear_pipeline,
            fill_pipeline,
            input_texture_bind_group_layout,
            output_texture_base_bind_group_layout,
            output_texture_bind_group_layout,
            mipmap_bind_group_layout,
            mipmap_base_pipeline,
            mipmap_pipeline,
        }
    }
}

pub struct MipmapMeta {
    pub data: UniformVec<GpuMipmap>,
}
impl FromWorld for MipmapMeta {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let render_queue = world.resource::<RenderQueue>();

        let mut data = UniformVec::default();
        for direction in 0u32..6 {
            data.push(GpuMipmap { direction });
        }
        data.write_buffer(render_device, render_queue);

        Self { data }
    }
}

#[derive(Clone, AsStd140)]
pub struct GpuMipmap {
    direction: u32,
}
