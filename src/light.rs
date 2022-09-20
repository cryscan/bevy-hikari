use crate::{
    mesh::{MeshMaterialBindGroup, MeshMaterialBindGroupLayout, TextureBindGroupLayout},
    prepass::{DeferredBindGroup, PrepassPipeline},
    LIGHT_SHADER_HANDLE, WORKGROUP_SIZE,
};
use bevy::{
    pbr::{
        GlobalLightMeta, GpuLights, GpuPointLights, LightMeta, ShadowPipeline, ViewClusterBindings,
        ViewLightsUniformOffset, ViewShadowBindings,
    },
    prelude::*,
    render::{
        camera::ExtractedCamera,
        render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
        texture::TextureCache,
        view::{ViewUniform, ViewUniformOffset, ViewUniforms},
        RenderApp, RenderStage,
    },
};

pub struct LightPlugin;
impl Plugin for LightPlugin {
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<LightPipeline>()
                .init_resource::<SpecializedComputePipelines<LightPipeline>>()
                .add_system_to_stage(RenderStage::Prepare, prepare_light_pass_targets)
                .add_system_to_stage(RenderStage::Queue, queue_view_bind_groups)
                .add_system_to_stage(RenderStage::Queue, queue_render_bind_groups)
                .add_system_to_stage(RenderStage::Queue, queue_light_pipelines);
        }
    }
}

pub struct LightPipeline {
    pub view_layout: BindGroupLayout,
    pub deferred_layout: BindGroupLayout,
    pub mesh_material_layout: BindGroupLayout,
    pub texture_layout: Option<BindGroupLayout>,
    pub render_layout: BindGroupLayout,
}

impl FromWorld for LightPipeline {
    fn from_world(world: &mut World) -> Self {
        let buffer_binding_type = BufferBindingType::Storage { read_only: true };

        let render_device = world.resource::<RenderDevice>();
        let prepass_pipeline = world.resource::<PrepassPipeline>();
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
                // Point Shadow Texture Cube Array
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Depth,
                        #[cfg(not(feature = "webgl"))]
                        view_dimension: TextureViewDimension::CubeArray,
                        #[cfg(feature = "webgl")]
                        view_dimension: TextureViewDimension::Cube,
                    },
                    count: None,
                },
                // Point Shadow Texture Array Sampler
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Comparison),
                    count: None,
                },
                // Directional Shadow Texture Array
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: TextureSampleType::Depth,
                        #[cfg(not(feature = "webgl"))]
                        view_dimension: TextureViewDimension::D2Array,
                        #[cfg(feature = "webgl")]
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Directional Shadow Texture Array Sampler
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Comparison),
                    count: None,
                },
                // PointLights
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: buffer_binding_type,
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuPointLights::min_size(buffer_binding_type)),
                    },
                    count: None,
                },
                // ClusteredLightIndexLists
                BindGroupLayoutEntry {
                    binding: 7,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: buffer_binding_type,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            ViewClusterBindings::min_size_cluster_light_index_lists(
                                buffer_binding_type,
                            ),
                        ),
                    },
                    count: None,
                },
                // ClusterOffsetsAndCounts
                BindGroupLayoutEntry {
                    binding: 8,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: buffer_binding_type,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            ViewClusterBindings::min_size_cluster_offsets_and_counts(
                                buffer_binding_type,
                            ),
                        ),
                    },
                    count: None,
                },
            ],
            label: None,
        });

        let deferred_layout = prepass_pipeline.deferred_layout.clone();

        let render_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba16Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        Self {
            view_layout,
            deferred_layout,
            mesh_material_layout,
            texture_layout: None,
            render_layout,
        }
    }
}

impl SpecializedComputePipeline for LightPipeline {
    type Key = usize;

    fn specialize(&self, _key: Self::Key) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![
                self.view_layout.clone(),
                self.deferred_layout.clone(),
                self.mesh_material_layout.clone(),
                self.texture_layout.clone().unwrap(),
                self.render_layout.clone(),
            ]),
            shader: LIGHT_SHADER_HANDLE.typed::<Shader>(),
            shader_defs: vec![],
            entry_point: "direct_cast".into(),
        }
    }
}

#[derive(Component)]
pub struct LightPassTarget {
    pub direct_view: TextureView,
}

fn prepare_light_pass_targets(
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
                        label: Some("light_direct_texture"),
                        size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: TextureDimension::D2,
                        format: TextureFormat::Rgba16Float,
                        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                    },
                )
                .default_view;

            commands
                .entity(entity)
                .insert(LightPassTarget { direct_view });
        }
    }
}

pub struct CachedLightPipelines {
    direct_cast: CachedComputePipelineId,
}

fn queue_light_pipelines(
    mut commands: Commands,
    layout: Res<TextureBindGroupLayout>,
    mut pipeline: ResMut<LightPipeline>,
    mut pipelines: ResMut<SpecializedComputePipelines<LightPipeline>>,
    mut pipeline_cache: ResMut<PipelineCache>,
) {
    pipeline.texture_layout = Some(layout.layout.clone());
    let direct_cast = pipelines.specialize(&mut pipeline_cache, &pipeline, layout.count);
    commands.insert_resource(CachedLightPipelines { direct_cast })
}

#[derive(Component)]
pub struct ViewBindGroup(pub BindGroup);

#[allow(clippy::too_many_arguments)]
pub fn queue_view_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline: Res<LightPipeline>,
    shadow_pipeline: Res<ShadowPipeline>,
    light_meta: Res<LightMeta>,
    global_light_meta: Res<GlobalLightMeta>,
    view_uniforms: Res<ViewUniforms>,
    views: Query<(Entity, &ViewShadowBindings, &ViewClusterBindings)>,
) {
    if let (Some(view_binding), Some(light_binding), Some(point_light_binding)) = (
        view_uniforms.uniforms.binding(),
        light_meta.view_gpu_lights.binding(),
        global_light_meta.gpu_point_lights.binding(),
    ) {
        for (entity, view_shadow_bindings, view_cluster_bindings) in &views {
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
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::TextureView(
                            &view_shadow_bindings.point_light_depth_texture_view,
                        ),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::Sampler(&shadow_pipeline.point_light_sampler),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: BindingResource::TextureView(
                            &view_shadow_bindings.directional_light_depth_texture_view,
                        ),
                    },
                    BindGroupEntry {
                        binding: 5,
                        resource: BindingResource::Sampler(
                            &shadow_pipeline.directional_light_sampler,
                        ),
                    },
                    BindGroupEntry {
                        binding: 6,
                        resource: point_light_binding.clone(),
                    },
                    BindGroupEntry {
                        binding: 7,
                        resource: view_cluster_bindings.light_index_lists_binding().unwrap(),
                    },
                    BindGroupEntry {
                        binding: 8,
                        resource: view_cluster_bindings.offsets_and_counts_binding().unwrap(),
                    },
                ],
                label: None,
                layout: &pipeline.view_layout,
            });

            commands.entity(entity).insert(ViewBindGroup(bind_group));
        }
    }
}

#[derive(Component)]
pub struct RenderBindGroup(pub BindGroup);

fn queue_render_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline: Res<LightPipeline>,
    query: Query<(Entity, &LightPassTarget)>,
) {
    for (entity, light_pass_target) in &query {
        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.render_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&light_pass_target.direct_view),
            }],
        });
        commands.entity(entity).insert(RenderBindGroup(bind_group));
    }
}

pub struct LightPassNode {
    query: QueryState<(
        &'static ExtractedCamera,
        &'static ViewUniformOffset,
        &'static ViewLightsUniformOffset,
        &'static ViewBindGroup,
        &'static DeferredBindGroup,
        &'static RenderBindGroup,
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
        let (
            camera,
            view_uniform,
            view_lights,
            view_bind_group,
            deferred_bind_group,
            render_bind_group,
        ) = match self.query.get_manual(world, entity) {
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
        pass.set_bind_group(1, &deferred_bind_group.0, &[]);
        pass.set_bind_group(2, &mesh_material_bind_group.mesh_material, &[0, 0]);
        pass.set_bind_group(3, &mesh_material_bind_group.texture, &[]);
        pass.set_bind_group(4, &render_bind_group.0, &[]);

        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.direct_cast) {
            pass.set_pipeline(pipeline);

            let size = camera.physical_target_size.unwrap();
            let count = (size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(count.x, count.y, 1);
        }

        Ok(())
    }
}