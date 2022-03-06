use super::{
    GpuVolume, Volume, VolumeBindings, VolumeColorAttachment, VolumeMeta, VolumeUniformOffset,
    VolumeView, VoxelConeTracingSystems, VOXEL_ANISOTROPIC_MIPMAP_LEVEL_COUNT, VOXEL_SHADER_HANDLE,
    VOXEL_SIZE,
};
use bevy::{
    core::FloatOrd,
    ecs::system::{
        lifetimeless::{Read, SQuery},
        SystemParamItem,
    },
    pbr::{
        DrawMesh, GlobalLightMeta, LightMeta, MeshPipeline, MeshPipelineKey, MeshViewBindGroup,
        SetMaterialBindGroup, SetMeshBindGroup, SetMeshViewBindGroup, ShadowPipeline,
        SpecializedMaterial, ViewClusterBindings, ViewLightsUniformOffset, ViewShadowBindings,
    },
    prelude::*,
    render::{
        camera::CameraProjection,
        primitives::{Aabb, Frustum},
        render_asset::RenderAssets,
        render_graph::{self, SlotInfo, SlotType},
        render_phase::{
            AddRenderCommand, CachedPipelinePhaseItem, DrawFunctionId, DrawFunctions,
            EntityPhaseItem, EntityRenderCommand, PhaseItem, RenderCommandResult, RenderPhase,
            SetItemPipeline, TrackedRenderPass,
        },
        render_resource::{std140::AsStd140, *},
        renderer::RenderDevice,
        view::{ExtractedView, RenderLayers, ViewUniforms, VisibleEntities},
        RenderApp, RenderStage,
    },
};
use std::{borrow::Cow, f32::consts::FRAC_PI_2, marker::PhantomData, num::NonZeroU32};

pub struct VoxelPlugin;
impl Plugin for VoxelPlugin {
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<VoxelPipeline>()
                .init_resource::<SpecializedPipelines<VoxelPipeline>>()
                .init_resource::<DrawFunctions<Voxel>>()
                .add_system_to_stage(
                    RenderStage::Queue,
                    queue_voxel_bind_groups.label(VoxelConeTracingSystems::QueueVoxelBindGroups),
                )
                .add_system_to_stage(
                    RenderStage::Queue,
                    queue_mipmap_bind_groups.label(VoxelConeTracingSystems::QueueMipmapBindGroups),
                );
        }
    }
}

/// The plugin registers the voxel draw functions/systems for a [`SpecializedMaterial`].
#[derive(Default)]
pub struct VoxelMaterialPlugin<M: SpecializedMaterial>(PhantomData<M>);
impl<M: SpecializedMaterial> Plugin for VoxelMaterialPlugin<M> {
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<Voxel, DrawVoxelMesh<M>>()
                .add_system_to_stage(
                    RenderStage::Queue,
                    queue_voxel_meshes::<M>.label(VoxelConeTracingSystems::QueueVoxel),
                );
        }
    }
}

#[derive(Component)]
pub struct VoxelBindGroup {
    pub value: BindGroup,
}

#[derive(Component)]
pub struct MipmapBindGroup {
    pub mipmaps: Vec<Vec<BindGroup>>,
    pub clear: BindGroup,
}

pub struct VoxelPipeline {
    pub material_layout: BindGroupLayout,
    pub voxel_layout: BindGroupLayout,
    pub mesh_pipeline: MeshPipeline,

    pub mipmap_layout: BindGroupLayout,
    pub mipmap_pipelines: Vec<ComputePipeline>,
    pub clear_pipeline: ComputePipeline,
}

impl FromWorld for VoxelPipeline {
    fn from_world(world: &mut World) -> Self {
        let mesh_pipeline = world.get_resource::<MeshPipeline>().unwrap().clone();

        let render_device = world.get_resource::<RenderDevice>().unwrap();

        let material_layout = StandardMaterial::bind_group_layout(render_device);

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
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D3,
                        multisampled: false,
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

        let mipmap_pipelines = (0..6)
            .map(|direction| {
                render_device.create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some(&format!("mipmap_pipeline_{direction}")),
                    layout: Some(&mipmap_pipeline_layout),
                    module: &shader,
                    entry_point: &format!("mipmap_{direction}"),
                })
            })
            .collect();
        let clear_pipeline = render_device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("clear_pipeline"),
            layout: Some(&mipmap_pipeline_layout),
            module: &shader,
            entry_point: "clear",
        });

        Self {
            material_layout,
            voxel_layout,
            mesh_pipeline,
            mipmap_layout,
            mipmap_pipelines,
            clear_pipeline,
        }
    }
}

impl SpecializedPipeline for VoxelPipeline {
    type Key = MeshPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let shader = VOXEL_SHADER_HANDLE.typed::<Shader>();

        let mut descriptor = self.mesh_pipeline.specialize(key);
        descriptor.fragment.as_mut().unwrap().shader = shader;
        descriptor.layout = Some(vec![
            self.mesh_pipeline.view_layout.clone(),
            self.material_layout.clone(),
            self.mesh_pipeline.mesh_layout.clone(),
            self.voxel_layout.clone(),
        ]);
        descriptor.primitive.cull_mode = None;
        descriptor.depth_stencil = None;
        descriptor.multisample = MultisampleState {
            count: 1,
            ..Default::default()
        };

        descriptor
    }
}

pub fn add_volume_views(mut commands: Commands, mut volumes: Query<&mut Volume>) {
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

#[allow(clippy::type_complexity)]
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

pub fn extract_views(
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

#[allow(clippy::too_many_arguments)]
pub fn queue_volume_view_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mesh_pipeline: Res<MeshPipeline>,
    shadow_pipeline: Res<ShadowPipeline>,
    light_meta: Res<LightMeta>,
    global_light_meta: Res<GlobalLightMeta>,
    view_uniforms: Res<ViewUniforms>,
    volume_query: Query<(
        &Volume,
        &ViewLightsUniformOffset,
        &ViewShadowBindings,
        &ViewClusterBindings,
    )>,
) {
    if let (Some(view_binding), Some(light_binding), Some(point_light_binding)) = (
        view_uniforms.uniforms.binding(),
        light_meta.view_gpu_lights.binding(),
        global_light_meta.gpu_point_lights.binding(),
    ) {
        for (volume, view_lights, view_shadow_bindings, view_cluster_bindings) in
            volume_query.iter()
        {
            let view_bind_group = render_device.create_bind_group(&BindGroupDescriptor {
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
                        resource: view_cluster_bindings
                            .cluster_light_index_lists
                            .binding()
                            .unwrap(),
                    },
                    BindGroupEntry {
                        binding: 8,
                        resource: view_cluster_bindings
                            .cluster_offsets_and_counts
                            .binding()
                            .unwrap(),
                    },
                ],
                label: Some("mesh_view_bind_group"),
                layout: &mesh_pipeline.view_layout,
            });

            for view in volume.views.iter().cloned() {
                commands
                    .entity(view)
                    .insert(ViewLightsUniformOffset {
                        offset: view_lights.offset,
                    })
                    .insert(MeshViewBindGroup {
                        value: view_bind_group.clone(),
                    });
            }
        }
    }
}

pub fn queue_voxel_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    voxel_pipeline: Res<VoxelPipeline>,
    volume_meta: Res<VolumeMeta>,
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
                        resource: volume_meta.volume_uniforms.binding().unwrap(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(
                            &bindings.voxel_texture.default_view,
                        ),
                    },
                ],
            });

            commands
                .entity(view)
                .insert(VoxelBindGroup { value: bind_group });
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn queue_voxel_meshes<M: SpecializedMaterial>(
    voxel_draw_functions: Res<DrawFunctions<Voxel>>,
    voxel_pipeline: Res<VoxelPipeline>,
    material_meshes: Query<(&Handle<M>, &Handle<Mesh>)>,
    render_meshes: Res<RenderAssets<Mesh>>,
    render_materials: Res<RenderAssets<M>>,
    mut pipelines: ResMut<SpecializedPipelines<VoxelPipeline>>,
    mut pipeline_cache: ResMut<RenderPipelineCache>,
    volumes: Query<&Volume, Without<VolumeView>>,
    mut view_query: Query<(&VisibleEntities, &mut RenderPhase<Voxel>), With<VolumeView>>,
) {
    let draw_mesh = voxel_draw_functions
        .read()
        .get_id::<DrawVoxelMesh<M>>()
        .unwrap();

    for volume in volumes.iter() {
        for view in volume.views.iter().cloned() {
            let (visible_entities, mut phase) = view_query.get_mut(view).unwrap();
            for entity in visible_entities.entities.iter().cloned() {
                if let Ok((material_handle, mesh_handle)) = material_meshes.get(entity) {
                    if !render_materials.contains_key(material_handle) {
                        continue;
                    }

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

pub fn queue_mipmap_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    voxel_pipeline: Res<VoxelPipeline>,
    volumes: Query<(Entity, &VolumeBindings), With<Volume>>,
) {
    for (entity, volume_bindings) in volumes.iter() {
        let anisotropic_mipmaps = (0..VOXEL_ANISOTROPIC_MIPMAP_LEVEL_COUNT)
            .map(|level| {
                volume_bindings
                    .anisotropic_textures
                    .iter()
                    .map(|cached_texture| {
                        cached_texture.texture.create_view(&TextureViewDescriptor {
                            base_mip_level: level as u32,
                            mip_level_count: NonZeroU32::new(1),
                            ..Default::default()
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mipmaps = (0..VOXEL_ANISOTROPIC_MIPMAP_LEVEL_COUNT)
            .map(|level| {
                let mut bind_groups = vec![];

                for direction in 0..6 {
                    let texture_in = match level {
                        0 => &volume_bindings.voxel_texture.default_view,
                        level => &anisotropic_mipmaps[level - 1][direction],
                    };
                    let texture_out = &anisotropic_mipmaps[level][direction];

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
                    bind_groups.push(bind_group);
                }

                bind_groups
            })
            .collect();

        let clear = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &voxel_pipeline.mipmap_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &volume_bindings.anisotropic_textures[0].default_view,
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &volume_bindings.voxel_texture.default_view,
                    ),
                },
            ],
        });

        commands
            .entity(entity)
            .insert(MipmapBindGroup { mipmaps, clear });
    }
}

pub struct Voxel {
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

pub type DrawVoxelMesh<M> = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMaterialBindGroup<M, 1>,
    SetMeshBindGroup<2>,
    SetVoxelBindGroup<3>,
    DrawMesh,
);

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
    volume_query: QueryState<&'static Volume>,
    volume_view_query: QueryState<(&'static VolumeColorAttachment, &'static RenderPhase<Voxel>)>,
}

impl VoxelPassNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        let volume_query = QueryState::new(world);
        let volume_view_query = QueryState::new(world);
        Self {
            volume_query,
            volume_view_query,
        }
    }
}

impl render_graph::Node for VoxelPassNode {
    fn input(&self) -> Vec<render_graph::SlotInfo> {
        vec![SlotInfo::new(Self::IN_VIEW, SlotType::Entity)]
    }

    fn update(&mut self, world: &mut World) {
        self.volume_query.update_archetypes(world);
        self.volume_view_query.update_archetypes(world);
    }

    fn run(
        &self,
        graph: &mut bevy::render::render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext,
        world: &World,
    ) -> Result<(), bevy::render::render_graph::NodeRunError> {
        let entity = graph.get_input_entity(Self::IN_VIEW)?;
        if let Ok(volume) = self.volume_query.get_manual(world, entity) {
            for view in volume.views.iter().cloned() {
                let (volume_color_attachment, phase) =
                    self.volume_view_query.get_manual(world, view).unwrap();
                let descriptor = RenderPassDescriptor {
                    label: None,
                    color_attachments: &[RenderPassColorAttachment {
                        view: &volume_color_attachment.texture.default_view,
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
                    draw_function.draw(world, &mut tracked_pass, view, item);
                }
            }
        }

        Ok(())
    }
}

pub struct MipmapPassNode {
    query: QueryState<&'static MipmapBindGroup, With<Volume>>,
}

impl MipmapPassNode {
    pub fn new(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl render_graph::Node for MipmapPassNode {
    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
    }

    #[allow(clippy::needless_range_loop)]
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

        for mipmap_bind_group in self.query.iter_manual(world) {
            for (level, bind_groups) in mipmap_bind_group.mipmaps.iter().enumerate() {
                for direction in 0..6 {
                    let size = (VOXEL_SIZE / (2 << level)) as u32;
                    let count = (size / 4).max(1);
                    pass.set_pipeline(&pipeline.mipmap_pipelines[direction]);
                    pass.set_bind_group(0, &bind_groups[direction], &[]);
                    pass.dispatch(count, count, count);
                }
            }
        }

        Ok(())
    }
}

pub struct VoxelClearPassNode {
    query: QueryState<&'static MipmapBindGroup, With<Volume>>,
}

impl VoxelClearPassNode {
    pub fn new(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl render_graph::Node for VoxelClearPassNode {
    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
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

        pass.set_pipeline(&pipeline.clear_pipeline);

        for mipmap_bind_group in self.query.iter_manual(world) {
            let count = (VOXEL_SIZE / 4) as u32;
            pass.set_bind_group(0, &mipmap_bind_group.clear, &[]);
            pass.dispatch(count, count, count);
        }

        Ok(())
    }
}
