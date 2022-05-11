use crate::{
    volume::{GpuVoxelBuffer, VolumeMeta},
    MIPMAP_SHADER_HANDLE, VOXEL_MIPMAP_LEVEL_COUNT, VOXEL_SIZE,
};
use bevy::{
    core_pipeline::node,
    prelude::*,
    render::{
        render_graph::{Node, RenderGraph},
        render_resource::{std140::AsStd140, std430::AsStd430, *},
        renderer::{RenderDevice, RenderQueue},
        RenderApp,
    },
};
use itertools::Itertools;
use std::num::NonZeroU32;

pub struct MipmapPlugin;
impl Plugin for MipmapPlugin {
    fn build(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<MipmapPipeline>()
            .init_resource::<MipmapMeta>();

        render_app
            .world
            .resource_scope(|_world, mut graph: Mut<RenderGraph>| {
                use crate::node::{MIPMAP_PASS, VOXEL_CLEAR_PASS, VOXEL_PASS_DRIVER};
                use node::CLEAR_PASS_DRIVER;

                graph.add_node(VOXEL_CLEAR_PASS, VoxelClearPassNode);
                graph.add_node(MIPMAP_PASS, MipmapPassNode);

                graph
                    .add_node_edge(CLEAR_PASS_DRIVER, VOXEL_CLEAR_PASS)
                    .unwrap();
                graph
                    .add_node_edge(VOXEL_CLEAR_PASS, VOXEL_PASS_DRIVER)
                    .unwrap();

                graph.add_node_edge(VOXEL_PASS_DRIVER, MIPMAP_PASS).unwrap();
            });
    }
}

pub struct MipmapPipeline {
    pub voxel_buffer_bind_group_layout: BindGroupLayout,
    pub clear_pipeline: CachedComputePipelineId,
    pub fill_pipeline: CachedComputePipelineId,

    pub mipmap_bind_group_layout: BindGroupLayout,
    pub mipmap_pipeline: CachedComputePipelineId,

    pub mipmap_anisotropic_bind_group_layout: BindGroupLayout,
    pub mipmap_anisotropic_pipeline: CachedComputePipelineId,
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

        let mipmap_bind_group_layout =
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
                    .chain([BindGroupLayoutEntry {
                        binding: 6,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    }])
                    .collect_vec()
                    .as_slice(),
            });

        let mipmap_anisotropic_bind_group_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[
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
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: BufferSize::new(
                                GpuMipmap::std140_size_static() as u64
                            ),
                        },
                        count: None,
                    },
                ],
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

        let mipmap_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![mipmap_bind_group_layout.clone()]),
            shader: MIPMAP_SHADER_HANDLE.typed(),
            shader_defs: vec![],
            entry_point: "mipmap".into(),
        });

        let mipmap_anisotropic_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: None,
                layout: Some(vec![mipmap_anisotropic_bind_group_layout.clone()]),
                shader: MIPMAP_SHADER_HANDLE.typed(),
                shader_defs: vec!["MIPMAP_ANISOTROPIC".into()],
                entry_point: "mipmap".into(),
            });

        Self {
            voxel_buffer_bind_group_layout,
            clear_pipeline,
            fill_pipeline,
            mipmap_bind_group_layout,
            mipmap_pipeline,
            mipmap_anisotropic_bind_group_layout,
            mipmap_anisotropic_pipeline,
        }
    }
}

pub struct MipmapMeta {
    pub mipmap_uniforms: DynamicUniformVec<GpuMipmap>,
    pub mipmap_uniform_offsets: [u32; 6],

    pub voxel_buffer_bind_group: BindGroup,
    pub mipmap_bind_group: BindGroup,
    pub mipmap_anisotropic_bind_groups: Vec<[BindGroup; 6]>,
}

impl FromWorld for MipmapMeta {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let render_queue = world.resource::<RenderQueue>();

        let mut mipmap_uniforms = DynamicUniformVec::default();
        let mut mipmap_uniform_offsets = vec![];
        for direction in 0u32..6 {
            mipmap_uniform_offsets.push(mipmap_uniforms.push(GpuMipmap { direction }));
        }
        mipmap_uniforms.write_buffer(render_device, render_queue);
        let mipmap_uniform_offsets = mipmap_uniform_offsets.try_into().unwrap();

        let mipmap_pipeline = world.resource::<MipmapPipeline>();
        let volume_meta = world.resource::<VolumeMeta>();

        let voxel_buffer_bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &mipmap_pipeline.voxel_buffer_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: volume_meta.voxel_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &volume_meta
                            .voxel_texture
                            .texture
                            .create_view(&TextureViewDescriptor {
                                base_mip_level: 0,
                                mip_level_count: NonZeroU32::new(1),
                                ..default()
                            }),
                    ),
                },
            ],
        });

        let mipmap_bind_group = volume_meta
            .anisotropic_textures
            .iter()
            .map(|cached_texture| {
                cached_texture.texture.create_view(&TextureViewDescriptor {
                    base_mip_level: 0,
                    mip_level_count: NonZeroU32::new(1),
                    ..default()
                })
            })
            .collect_vec();
        let mipmap_bind_group = mipmap_bind_group
            .iter()
            .enumerate()
            .map(|(direction, texture_view)| BindGroupEntry {
                binding: direction as u32,
                resource: BindingResource::TextureView(&texture_view),
            })
            .chain([BindGroupEntry {
                binding: 6,
                resource: BindingResource::TextureView(&volume_meta.voxel_texture.default_view),
            }])
            .collect_vec();
        let mipmap_bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &mipmap_pipeline.mipmap_bind_group_layout,
            entries: &mipmap_bind_group,
        });

        let mipmap_texture_views = (0..VOXEL_MIPMAP_LEVEL_COUNT).map(|level| {
            volume_meta
                .anisotropic_textures
                .iter()
                .map(move |cached_texture| {
                    cached_texture.texture.create_view(&TextureViewDescriptor {
                        base_mip_level: level as u32,
                        mip_level_count: NonZeroU32::new(1),
                        ..default()
                    })
                })
        });
        let mipmap_anisotropic_bind_groups = mipmap_texture_views
            .tuple_windows::<(_, _)>()
            .map(|(texture_in, texture_out)| {
                texture_in
                    .zip_eq(texture_out)
                    .map(|(texture_in, texture_out)| {
                        render_device.create_bind_group(&BindGroupDescriptor {
                            label: None,
                            layout: &mipmap_pipeline.mipmap_anisotropic_bind_group_layout,
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

        Self {
            mipmap_uniforms,
            mipmap_uniform_offsets,
            voxel_buffer_bind_group,
            mipmap_bind_group,
            mipmap_anisotropic_bind_groups,
        }
    }
}

#[derive(Clone, AsStd140)]
pub struct GpuMipmap {
    direction: u32,
}

pub struct VoxelClearPassNode;
impl Node for VoxelClearPassNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let mipmap_pipeline = world.resource::<MipmapPipeline>();
        let mipmap_meta = world.resource::<MipmapMeta>();
        let pipeline_cache = world.resource::<PipelineCache>();

        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(mipmap_pipeline.clear_pipeline)
        {
            let mut pass = render_context
                .command_encoder
                .begin_compute_pass(&default());
            let bind_group = &mipmap_meta.voxel_buffer_bind_group;

            let count = (VOXEL_SIZE / 8) as u32;
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch(count, count, count);
        }

        Ok(())
    }
}

pub struct MipmapPassNode;
impl Node for MipmapPassNode {
    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let mipmap_pipeline = world.resource::<MipmapPipeline>();
        let mipmap_meta = world.resource::<MipmapMeta>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let mut pass = render_context
            .command_encoder
            .begin_compute_pass(&default());

        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(mipmap_pipeline.fill_pipeline) {
            let count = (VOXEL_SIZE / 8) as u32;
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &mipmap_meta.voxel_buffer_bind_group, &[]);
            pass.dispatch(count, count, count);
        }

        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(mipmap_pipeline.mipmap_pipeline)
        {
            let size = (VOXEL_SIZE / 2) as u32;
            let count = (size / 8).max(1);
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &mipmap_meta.mipmap_bind_group, &[]);
            pass.dispatch(count, count, size);
        }

        if let Some(pipeline) =
            pipeline_cache.get_compute_pipeline(mipmap_pipeline.mipmap_anisotropic_pipeline)
        {
            pass.set_pipeline(&pipeline);

            for (level, bind_groups) in mipmap_meta
                .mipmap_anisotropic_bind_groups
                .iter()
                .enumerate()
            {
                let level = level + 1;
                for (bind_group, offset) in bind_groups
                    .iter()
                    .zip_eq(mipmap_meta.mipmap_uniform_offsets.iter())
                {
                    let size = (VOXEL_SIZE / (2 << level)) as u32;
                    let count = (size / 8).max(1);
                    pass.set_bind_group(0, &bind_group, &[*offset]);
                    pass.dispatch(count, count, count);
                }
            }
        }

        Ok(())
    }
}
