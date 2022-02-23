use bevy::{
    core::FloatOrd,
    core_pipeline,
    ecs::system::{
        lifetimeless::{Read, SQuery},
        SystemParamItem,
    },
    pbr::{
        DrawMesh, GpuLights, LightMeta, MeshPipeline, MeshPipelineKey, SetMaterialBindGroup,
        SetMeshBindGroup, ShadowPipeline, SpecializedMaterial, ViewLightsUniformOffset,
        ViewShadowBindings,
    },
    prelude::*,
    reflect::TypeUuid,
    render::{
        camera::CameraProjection,
        primitives::{Aabb, Frustum},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph},
        render_phase::{
            sort_phase_system, AddRenderCommand, CachedPipelinePhaseItem, DrawFunctionId,
            DrawFunctions, EntityPhaseItem, EntityRenderCommand, PhaseItem, RenderCommandResult,
            RenderPhase, SetItemPipeline, TrackedRenderPass,
        },
        render_resource::{std140::AsStd140, *},
        renderer::{RenderDevice, RenderQueue},
        texture::TextureCache,
        view::{
            ExtractedView, RenderLayers, ViewUniform, ViewUniformOffset, ViewUniforms,
            VisibleEntities,
        },
        RenderApp, RenderStage,
    },
};
use std::{borrow::Cow, f32::consts::FRAC_PI_2, num::NonZeroU32};

pub const VOXEL_SIZE: usize = 256;
pub const VOXEL_MIPMAP_LEVEL_COUNT: usize = 9;

pub const VOXEL_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 14750151725749984738);

pub mod draw_3d_graph {
    pub mod node {
        pub const VOXEL_PASS: &str = "voxel_pass";
        pub const MIPMAP_PASS: &str = "mipmap_pass";
    }
}

#[derive(Default)]
pub struct VoxelConeTracingPlugin;

impl Plugin for VoxelConeTracingPlugin {
    fn build(&self, app: &mut App) {
        app.add_system_to_stage(CoreStage::PostUpdate, add_volume_views.exclusive_system())
            .add_system_to_stage(CoreStage::PostUpdate, check_visibility);

        let mut shaders = app.world.get_resource_mut::<Assets<Shader>>().unwrap();
        shaders.set_untracked(
            VOXEL_SHADER_HANDLE,
            Shader::from_wgsl(include_str!("shaders/voxel.wgsl").replace("\r\n", "\n")),
        );

        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => return,
        };

        let voxel_pass_node = VoxelPassNode::new(&mut render_app.world);
        let mipmap_pass_node = MipmapPassNode::new(&mut render_app.world);

        render_app
            .init_resource::<VoxelPipeline>()
            .init_resource::<SpecializedPipelines<VoxelPipeline>>()
            .init_resource::<VoxelMeta>()
            .init_resource::<DrawFunctions<Voxel>>()
            .add_render_command::<Voxel, DrawVoxelMesh>()
            .add_system_to_stage(
                RenderStage::Extract,
                extract_volumes.label(VoxelConeTracingSystems::ExtractVolumes),
            )
            .add_system_to_stage(
                RenderStage::Extract,
                extract_views.label(VoxelConeTracingSystems::ExtractViews),
            )
            .add_system_to_stage(
                RenderStage::Prepare,
                prepare_volumes.label(VoxelConeTracingSystems::PrepareVolumes),
            )
            .add_system_to_stage(
                RenderStage::Queue,
                queue_volume_view_bind_groups
                    .label(VoxelConeTracingSystems::QueueVolumeViewBindGroups),
            )
            .add_system_to_stage(
                RenderStage::Queue,
                queue_voxel_bind_groups.label(VoxelConeTracingSystems::QueueVoxelBindGroups),
            )
            .add_system_to_stage(
                RenderStage::Queue,
                queue_voxel.label(VoxelConeTracingSystems::QueueVoxel),
            )
            .add_system_to_stage(
                RenderStage::Queue,
                queue_mipmap_bind_groups.label(VoxelConeTracingSystems::QueueMipmapBindGroups),
            )
            .add_system_to_stage(RenderStage::PhaseSort, sort_phase_system::<Voxel>);

        let mut render_graph = render_app.world.get_resource_mut::<RenderGraph>().unwrap();

        let draw_3d_graph = render_graph
            .get_sub_graph_mut(core_pipeline::draw_3d_graph::NAME)
            .unwrap();

        draw_3d_graph.add_node(draw_3d_graph::node::VOXEL_PASS, voxel_pass_node);
        draw_3d_graph.add_node(draw_3d_graph::node::MIPMAP_PASS, mipmap_pass_node);

        draw_3d_graph
            .add_node_edge(
                bevy::pbr::draw_3d_graph::node::SHADOW_PASS,
                draw_3d_graph::node::VOXEL_PASS,
            )
            .unwrap();
        draw_3d_graph
            .add_node_edge(
                draw_3d_graph::node::VOXEL_PASS,
                draw_3d_graph::node::MIPMAP_PASS,
            )
            .unwrap();
        draw_3d_graph
            .add_node_edge(
                draw_3d_graph::node::MIPMAP_PASS,
                core_pipeline::draw_3d_graph::node::MAIN_PASS,
            )
            .unwrap();
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
pub enum VoxelConeTracingSystems {
    ExtractVolumes,
    ExtractViews,
    PrepareVolumes,
    QueueVolumeViewBindGroups,
    QueueVoxelBindGroups,
    QueueVoxel,
    QueueMipmapBindGroups,
}

#[derive(Component, Clone)]
pub struct Volume {
    pub min: Vec3,
    pub max: Vec3,
    views: Vec<Entity>,
}

impl Volume {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self {
            min,
            max,
            views: vec![],
        }
    }
}

impl Default for Volume {
    fn default() -> Self {
        Self::new(Vec3::new(-5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0))
    }
}

#[derive(Component)]
pub struct VolumeView;

#[derive(Component, Clone)]
pub struct VolumeUniformOffset {
    pub offset: u32,
}

#[derive(Component)]
pub struct VolumeColorAttachment {
    pub texture_view: TextureView,
}

#[derive(Component, Clone)]
pub struct VolumeBindings {
    pub voxel_texture: Texture,
    pub voxel_texture_views: Vec<TextureView>,
}

#[derive(Clone, AsStd140)]
struct GpuVolume {
    min: Vec3,
    max: Vec3,
}

#[derive(Default)]
struct VoxelMeta {
    volume_uniforms: DynamicUniformVec<GpuVolume>,
}

#[derive(Component)]
pub struct VolumeViewBindGroup {
    pub value: BindGroup,
}

#[derive(Component)]
pub struct VoxelBindGroup {
    pub value: BindGroup,
}

#[derive(Component, Default, Clone)]
pub struct MipmapBindGroup {
    pub values: Vec<(usize, BindGroup)>,
}

pub struct VoxelPipeline {
    view_layout: BindGroupLayout,
    material_layout: BindGroupLayout,
    voxel_layout: BindGroupLayout,
    mesh_pipeline: MeshPipeline,

    mipmap_layout: BindGroupLayout,
    mipmap_pipeline: ComputePipeline,
}

impl FromWorld for VoxelPipeline {
    fn from_world(world: &mut World) -> Self {
        let mesh_pipeline = world.get_resource::<MeshPipeline>().unwrap().clone();

        let render_device = world.get_resource::<RenderDevice>().unwrap();

        let material_layout = StandardMaterial::bind_group_layout(render_device);

        let view_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("voxel_view_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: BufferSize::new(ViewUniform::std140_size_static() as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: BufferSize::new(GpuLights::std140_size_static() as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        #[cfg(not(feature = "webgl"))]
                        view_dimension: TextureViewDimension::D2Array,
                        #[cfg(feature = "webgl")]
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        let voxel_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("voxel_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: BufferSize::new(GpuVolume::std140_size_static() as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });

        let mipmap_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("mipmap_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D3,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });

        let mipmap_pipeline_layout =
            render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("mipmap_pipeline_layout"),
                bind_group_layouts: &[&mipmap_layout],
                push_constant_ranges: &[],
            });

        let shader = render_device.create_shader_module(&ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(
                &include_str!("shaders/mipmap.wgsl").replace("\r\n", "\n"),
            )),
        });

        let mipmap_pipeline = render_device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("mipmap_pipeline"),
            layout: Some(&mipmap_pipeline_layout),
            module: &shader,
            entry_point: "mipmap",
        });

        Self {
            view_layout,
            material_layout,
            voxel_layout,
            mesh_pipeline,
            mipmap_layout,
            mipmap_pipeline,
        }
    }
}

impl SpecializedPipeline for VoxelPipeline {
    type Key = MeshPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let shader = VOXEL_SHADER_HANDLE.typed::<Shader>();

        let mut descriptor = self.mesh_pipeline.specialize(key);
        descriptor.fragment.as_mut().unwrap().shader = shader.clone();
        descriptor.layout = Some(vec![
            self.view_layout.clone(),
            self.material_layout.clone(),
            self.mesh_pipeline.mesh_layout.clone(),
            self.voxel_layout.clone(),
        ]);
        descriptor.primitive.cull_mode = None;
        descriptor.depth_stencil = None;

        descriptor
    }
}

fn add_volume_views(mut commands: Commands, mut volumes: Query<&mut Volume>) {
    for mut volume in volumes.iter_mut() {
        if !volume.views.is_empty() {
            continue;
        }

        let center = (volume.max + volume.min) / 2.0;
        let extend = (volume.max - volume.min) / 2.0;

        for rotation in [
            Quat::IDENTITY,
            Quat::from_rotation_y(FRAC_PI_2),
            Quat::from_rotation_x(FRAC_PI_2),
        ] {
            let transform = GlobalTransform::from_translation(center)
                * GlobalTransform::from_rotation(rotation);

            let projection = OrthographicProjection {
                left: -extend.x,
                right: extend.x,
                bottom: -extend.y,
                top: extend.y,
                near: -extend.z,
                far: extend.z,
                ..Default::default()
            };

            let entity = commands
                .spawn_bundle((
                    VolumeView,
                    transform,
                    projection,
                    Frustum::default(),
                    VisibleEntities::default(),
                ))
                .id();
            volume.views.push(entity);
        }
    }
}

pub fn check_visibility(
    mut view_query: Query<
        (&mut VisibleEntities, &Frustum, Option<&RenderLayers>),
        With<VolumeView>,
    >,
    mut visible_entity_query: QuerySet<(
        QueryState<&mut ComputedVisibility>,
        QueryState<(
            Entity,
            &Visibility,
            &mut ComputedVisibility,
            Option<&RenderLayers>,
            Option<&Aabb>,
            Option<&GlobalTransform>,
        )>,
    )>,
) {
    // Reset the computed visibility to false
    for mut computed_visibility in visible_entity_query.q0().iter_mut() {
        computed_visibility.is_visible = false;
    }

    for (mut visible_entities, frustum, maybe_view_mask) in view_query.iter_mut() {
        visible_entities.entities.clear();
        let view_mask = maybe_view_mask.copied().unwrap_or_default();

        for (
            entity,
            visibility,
            mut computed_visibility,
            maybe_entity_mask,
            maybe_aabb,
            maybe_transform,
        ) in visible_entity_query.q1().iter_mut()
        {
            if !visibility.is_visible {
                continue;
            }

            let entity_mask = maybe_entity_mask.copied().unwrap_or_default();
            if !view_mask.intersects(&entity_mask) {
                continue;
            }

            // If we have an aabb and transform, do frustum culling
            if let (Some(aabb), Some(transform)) = (maybe_aabb, maybe_transform) {
                if !frustum.intersects_obb(aabb, &transform.compute_matrix()) {
                    continue;
                }
            }

            computed_visibility.is_visible = true;
            visible_entities.entities.push(entity);
        }

        // TODO: check for big changes in visible entities len() vs capacity() (ex: 2x) and resize
        // to prevent holding unneeded memory
    }
}

fn extract_views(
    mut commands: Commands,
    query: Query<
        (
            Entity,
            &GlobalTransform,
            &OrthographicProjection,
            &VisibleEntities,
        ),
        With<VolumeView>,
    >,
) {
    for (entity, transform, projection, visible_entities) in query.iter() {
        commands.get_or_spawn(entity).insert_bundle((
            ExtractedView {
                projection: projection.get_projection_matrix(),
                transform: *transform,
                width: VOXEL_SIZE as u32,
                height: VOXEL_SIZE as u32,
                near: projection.near,
                far: projection.far,
            },
            visible_entities.clone(),
            VolumeView,
        ));
    }
}

fn extract_volumes(mut commands: Commands, volumes: Query<(Entity, &Volume)>) {
    for (entity, volume) in volumes.iter() {
        commands.get_or_spawn(entity).insert(volume.clone());
    }
}

fn prepare_volumes(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut texture_cache: ResMut<TextureCache>,
    mut volumes: Query<(Entity, &Volume)>,
    mut voxel_meta: ResMut<VoxelMeta>,
) {
    voxel_meta.volume_uniforms.clear();

    for (entity, volume) in volumes.iter_mut() {
        let texture_view = texture_cache
            .get(
                &render_device,
                TextureDescriptor {
                    label: Some("voxel_volume_texture"),
                    size: Extent3d {
                        width: VOXEL_SIZE as u32,
                        height: VOXEL_SIZE as u32,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Bgra8UnormSrgb,
                    usage: TextureUsages::RENDER_ATTACHMENT,
                },
            )
            .texture
            .create_view(&TextureViewDescriptor {
                label: Some("voxel_volume_texture_view"),
                format: None,
                dimension: Some(TextureViewDimension::D2),
                aspect: TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: 0,
                array_layer_count: None,
            });

        for view in &volume.views {
            let texture_view = texture_view.clone();
            commands.entity(*view).insert_bundle((
                VolumeColorAttachment { texture_view },
                RenderPhase::<Voxel>::default(),
            ));
        }

        let volume_uniform_offset = VolumeUniformOffset {
            offset: voxel_meta.volume_uniforms.push(GpuVolume {
                min: volume.min,
                max: volume.max,
            }),
        };

        let voxel_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: VOXEL_SIZE as u32,
                    height: VOXEL_SIZE as u32,
                    depth_or_array_layers: VOXEL_SIZE as u32,
                },
                mip_level_count: VOXEL_MIPMAP_LEVEL_COUNT as u32,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            },
        );

        let voxel_texture_views = (0..VOXEL_MIPMAP_LEVEL_COUNT)
            .map(|i| {
                voxel_texture.texture.create_view(&TextureViewDescriptor {
                    label: Some(&format!("voxel_texture_view_{}_{}", entity.id(), i)),
                    format: None,
                    dimension: Some(TextureViewDimension::D3),
                    aspect: TextureAspect::All,
                    base_mip_level: i as u32,
                    mip_level_count: NonZeroU32::new(1),
                    base_array_layer: 0,
                    array_layer_count: None,
                })
            })
            .collect();

        let volume_bindings = VolumeBindings {
            voxel_texture: voxel_texture.texture,
            voxel_texture_views,
        };

        for view in volume.views.iter().cloned() {
            commands.entity(view).insert(volume_uniform_offset.clone());
        }

        commands.entity(entity).insert(volume_bindings);
    }

    voxel_meta
        .volume_uniforms
        .write_buffer(&render_device, &render_queue);
}

fn queue_volume_view_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    voxel_pipeline: Res<VoxelPipeline>,
    shadow_pipeline: Res<ShadowPipeline>,
    light_meta: Res<LightMeta>,
    view_uniforms: Res<ViewUniforms>,
    volume_query: Query<(&Volume, &ViewShadowBindings, &ViewLightsUniformOffset)>,
) {
    if let (Some(view_binding), Some(light_binding)) = (
        view_uniforms.uniforms.binding(),
        light_meta.view_gpu_lights.binding(),
    ) {
        for (volume, shadow_bindings, lights_uniform_offset) in volume_query.iter() {
            for view in volume.views.iter().cloned() {
                let view_bind_group = render_device.create_bind_group(&BindGroupDescriptor {
                    label: Some("volume_view_bind_group"),
                    layout: &voxel_pipeline.view_layout,
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
                                &shadow_bindings.directional_light_depth_texture_view,
                            ),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: BindingResource::Sampler(
                                &shadow_pipeline.directional_light_sampler,
                            ),
                        },
                    ],
                });

                commands
                    .entity(view)
                    .insert(VolumeViewBindGroup {
                        value: view_bind_group,
                    })
                    .insert(ViewLightsUniformOffset {
                        offset: lights_uniform_offset.offset,
                    });
            }
        }
    }
}

fn queue_voxel_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    voxel_pipeline: Res<VoxelPipeline>,
    voxel_meta: Res<VoxelMeta>,
    volumes: Query<(&Volume, &VolumeBindings)>,
) {
    for (volume, bindings) in volumes.iter() {
        for view in volume.views.iter().cloned() {
            let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
                label: Some("voxel_bind_group"),
                layout: &voxel_pipeline.voxel_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: voxel_meta.volume_uniforms.binding().unwrap(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&bindings.voxel_texture_views[0]),
                    },
                ],
            });

            commands
                .entity(view)
                .insert(VoxelBindGroup { value: bind_group });
        }
    }
}

fn queue_voxel(
    voxel_draw_functions: Res<DrawFunctions<Voxel>>,
    voxel_pipeline: Res<VoxelPipeline>,
    meshes: Query<&Handle<Mesh>>,
    render_meshes: Res<RenderAssets<Mesh>>,
    mut pipelines: ResMut<SpecializedPipelines<VoxelPipeline>>,
    mut pipeline_cache: ResMut<RenderPipelineCache>,
    volumes: Query<&Volume, Without<VolumeView>>,
    mut view_query: Query<(&VisibleEntities, &mut RenderPhase<Voxel>), With<VolumeView>>,
) {
    let draw_mesh = voxel_draw_functions
        .read()
        .get_id::<DrawVoxelMesh>()
        .unwrap();

    for volume in volumes.iter() {
        for view in volume.views.iter().cloned() {
            let (visible_entities, mut phase) = view_query.get_mut(view).unwrap();
            for entity in visible_entities.entities.iter().cloned() {
                if let Ok(mesh_handle) = meshes.get(entity) {
                    let mut key = MeshPipelineKey::empty();
                    if let Some(mesh) = render_meshes.get(mesh_handle) {
                        if mesh.has_tangents {
                            key |= MeshPipelineKey::VERTEX_TANGENTS;
                        }
                        key |= MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);
                        key |= MeshPipelineKey::from_msaa_samples(1);
                    }

                    let pipeline_id =
                        pipelines.specialize(&mut pipeline_cache, &voxel_pipeline, key);
                    phase.add(Voxel {
                        draw_function: draw_mesh,
                        pipeline: pipeline_id,
                        entity,
                        distance: 0.0,
                    });
                }
            }
        }
    }
}

fn queue_mipmap_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    voxel_pipeline: Res<VoxelPipeline>,
    volumes: Query<(Entity, &VolumeBindings), With<Volume>>,
) {
    for (entity, volume_bindings) in volumes.iter() {
        let mut mipmap_bind_group = MipmapBindGroup::default();
        for i in 0..VOXEL_MIPMAP_LEVEL_COUNT - 1 {
            let ref texture_in = volume_bindings.voxel_texture_views[i];
            let ref texture_out = volume_bindings.voxel_texture_views[i + 1];
            let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &voxel_pipeline.mipmap_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(texture_in),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(texture_out),
                    },
                ],
            });
            let size = (VOXEL_SIZE / 2) / (1usize << i);
            mipmap_bind_group.values.push((size, bind_group));
        }

        commands.entity(entity).insert(mipmap_bind_group);
    }
}

struct Voxel {
    distance: f32,
    entity: Entity,
    pipeline: CachedPipelineId,
    draw_function: DrawFunctionId,
}

impl PhaseItem for Voxel {
    type SortKey = FloatOrd;

    fn sort_key(&self) -> Self::SortKey {
        FloatOrd(self.distance)
    }

    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }
}

impl EntityPhaseItem for Voxel {
    fn entity(&self) -> Entity {
        self.entity
    }
}

impl CachedPipelinePhaseItem for Voxel {
    fn cached_pipeline(&self) -> CachedPipelineId {
        self.pipeline
    }
}

pub type DrawVoxelMesh = (
    SetItemPipeline,
    SetVolumeViewBindGroup<0>,
    SetMaterialBindGroup<StandardMaterial, 1>,
    SetMeshBindGroup<2>,
    SetVoxelBindGroup<3>,
    DrawMesh,
);

pub struct SetVolumeViewBindGroup<const I: usize>;
impl<const I: usize> EntityRenderCommand for SetVolumeViewBindGroup<I> {
    type Param = SQuery<(
        Read<ViewUniformOffset>,
        Read<ViewLightsUniformOffset>,
        Read<VolumeViewBindGroup>,
    )>;

    fn render<'w>(
        view: Entity,
        _item: Entity,
        query: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let (view_uniform_offset, lights_uniform_offset, bind_group) = query.get(view).unwrap();
        pass.set_bind_group(
            I,
            &bind_group.value,
            &[view_uniform_offset.offset, lights_uniform_offset.offset],
        );
        RenderCommandResult::Success
    }
}

pub struct SetVoxelBindGroup<const I: usize>;
impl<const I: usize> EntityRenderCommand for SetVoxelBindGroup<I> {
    type Param = SQuery<(Read<VolumeUniformOffset>, Read<VoxelBindGroup>)>;

    fn render<'w>(
        view: Entity,
        _item: Entity,
        query: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let (volume_uniform_offset, bind_group) = query.get(view).unwrap();
        pass.set_bind_group(I, &bind_group.value, &[volume_uniform_offset.offset]);
        RenderCommandResult::Success
    }
}

pub struct VoxelPassNode {
    volume_view_query: QueryState<(
        Entity,
        &'static VolumeColorAttachment,
        &'static RenderPhase<Voxel>,
    )>,
}

impl VoxelPassNode {
    pub fn new(world: &mut World) -> Self {
        let volume_view_query = QueryState::new(world);
        Self { volume_view_query }
    }
}

impl render_graph::Node for VoxelPassNode {
    fn update(&mut self, world: &mut World) {
        self.volume_view_query.update_archetypes(world);
    }

    fn run(
        &self,
        _graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        for (entity, volume_view, phase) in self.volume_view_query.iter_manual(world) {
            let descriptor = RenderPassDescriptor {
                label: None,
                color_attachments: &[RenderPassColorAttachment {
                    view: &volume_view.texture_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK.into()),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            };

            let draw_functions = world.get_resource::<DrawFunctions<Voxel>>().unwrap();
            let render_pass = render_context
                .command_encoder
                .begin_render_pass(&descriptor);
            let mut draw_functions = draw_functions.write();
            let mut tracked_pass = TrackedRenderPass::new(render_pass);
            for item in &phase.items {
                let draw_function = draw_functions.get_mut(item.draw_function).unwrap();
                draw_function.draw(world, &mut tracked_pass, entity, item);
            }
        }

        Ok(())
    }
}

pub struct MipmapPassNode {
    volume_query: QueryState<&'static MipmapBindGroup, With<Volume>>,
}

impl MipmapPassNode {
    pub fn new(world: &mut World) -> Self {
        let volume_query = QueryState::new(world);
        Self { volume_query }
    }
}

impl render_graph::Node for MipmapPassNode {
    fn update(&mut self, world: &mut World) {
        self.volume_query.update_archetypes(world);
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let pipeline = world.get_resource::<VoxelPipeline>().unwrap();
        let mut pass = render_context
            .command_encoder
            .begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline.mipmap_pipeline);

        for mipmap_bind_group in self.volume_query.iter_manual(world) {
            for (size, bind_group) in &mipmap_bind_group.values {
                pass.set_bind_group(0, bind_group, &[]);
                let size = (size / 4).max(1usize) as u32;
                pass.dispatch(size, size, size);
            }
        }

        Ok(())
    }
}
