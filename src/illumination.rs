use crate::{
    mesh::{
        GpuInstanceBuffer, GpuNodeBuffer, GpuPrimitiveBuffer, GpuVertexBuffer, MeshRenderAssets,
    },
    prepass::PrepassTarget,
    ILLUMINATION_SHADER_HANDLE,
};
use bevy::{
    pbr::MeshPipeline,
    prelude::*,
    render::{
        camera::ExtractedCamera,
        render_resource::{ComputePipelineDescriptor, *},
        renderer::RenderDevice,
        texture::TextureCache,
        RenderApp, RenderStage,
    },
};

pub struct IlluminationPlugin;
impl Plugin for IlluminationPlugin {
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<IlluminationPipeline>()
                .init_resource::<SpecializedComputePipelines<IlluminationPipeline>>()
                .add_system_to_stage(RenderStage::Prepare, prepare_illumination_targets)
                .add_system_to_stage(RenderStage::Queue, queue_mesh_bind_group)
                .add_system_to_stage(RenderStage::Queue, queue_deferred_bind_group);
        }
    }
}

pub struct IlluminationPipeline {
    view_layout: BindGroupLayout,
    mesh_layout: BindGroupLayout,
    deferred_layout: BindGroupLayout,
    render_layout: BindGroupLayout,
}

impl FromWorld for IlluminationPipeline {
    fn from_world(world: &mut World) -> Self {
        let mesh_pipeline = world.resource::<MeshPipeline>();
        let view_layout = mesh_pipeline.view_layout.clone();

        let render_device = world.resource::<RenderDevice>();

        let mesh_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Vertices
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuVertexBuffer::min_size()),
                    },
                    count: None,
                },
                // Primitives
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuPrimitiveBuffer::min_size()),
                    },
                    count: None,
                },
                // Asset nodes
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuNodeBuffer::min_size()),
                    },
                    count: None,
                },
                // Instances
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuInstanceBuffer::min_size()),
                    },
                    count: None,
                },
                // Instance nodes
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuNodeBuffer::min_size()),
                    },
                    count: None,
                },
            ],
        });

        let deferred_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Depth buffer.
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                    count: None,
                },
                // Normal-velocity buffer.
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        let render_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        Self {
            view_layout,
            mesh_layout,
            deferred_layout,
            render_layout,
        }
    }
}

impl SpecializedComputePipeline for IlluminationPipeline {
    type Key = ();

    fn specialize(&self, _key: Self::Key) -> ComputePipelineDescriptor {
        let layout = vec![
            self.view_layout.clone(),
            self.mesh_layout.clone(),
            self.deferred_layout.clone(),
            self.render_layout.clone(),
        ];

        ComputePipelineDescriptor {
            label: None,
            layout: Some(layout),
            shader: ILLUMINATION_SHADER_HANDLE.typed::<Shader>(),
            shader_defs: vec![],
            entry_point: "direct".into(),
        }
    }
}

#[derive(Component)]
pub struct IlluminationTarget {
    pub direct_view: TextureView,
}

fn prepare_illumination_targets(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    cameras: Query<(Entity, &ExtractedCamera)>,
) {
    for (entity, camera) in &cameras {
        if let Some(target_size) = camera.physical_target_size {
            let size = Extent3d {
                width: target_size.x,
                height: target_size.y,
                depth_or_array_layers: 1,
            };

            let direct_view = texture_cache
                .get(
                    &render_device,
                    TextureDescriptor {
                        label: Some("illumination_direct_texture"),
                        size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: TextureDimension::D2,
                        format: TextureFormat::Rgba32Float,
                        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                    },
                )
                .default_view;

            commands
                .entity(entity)
                .insert(IlluminationTarget { direct_view });
        }
    }
}

pub struct MeshBindGroup(BindGroup);

fn queue_mesh_bind_group(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline: Res<IlluminationPipeline>,
    render_assets: Res<MeshRenderAssets>,
) {
    if let (
        Some(vertex_binding),
        Some(primitive_binding),
        Some(asset_node_binding),
        Some(instance_binding),
        Some(instance_node_binding),
    ) = (
        render_assets.vertex_buffer.binding(),
        render_assets.primitive_buffer.binding(),
        render_assets.asset_node_buffer.binding(),
        render_assets.instance_buffer.binding(),
        render_assets.instance_node_buffer.binding(),
    ) {
        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.mesh_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: vertex_binding,
                },
                BindGroupEntry {
                    binding: 1,
                    resource: primitive_binding,
                },
                BindGroupEntry {
                    binding: 2,
                    resource: asset_node_binding,
                },
                BindGroupEntry {
                    binding: 3,
                    resource: instance_binding,
                },
                BindGroupEntry {
                    binding: 4,
                    resource: instance_node_binding,
                },
            ],
        });
        commands.insert_resource(MeshBindGroup(bind_group));
    }
}

#[derive(Component)]
pub struct DeferredBindGroup(BindGroup);

fn queue_deferred_bind_group(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline: Res<IlluminationPipeline>,
    query: Query<(Entity, &PrepassTarget)>,
) {
    for (entity, prepass_target) in &query {
        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.deferred_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&prepass_target.depth_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&prepass_target.depth_sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&prepass_target.normal_velocity_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(&prepass_target.normal_velocity_sampler),
                },
            ],
        });
        commands
            .entity(entity)
            .insert(DeferredBindGroup(bind_group));
    }
}
