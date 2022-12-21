use crate::{
    mesh_material::{
        DynamicInstanceIndex, InstanceIndex, InstanceRenderAssets, PreviousMeshUniform,
    },
    view::{FrameUniform, PreviousViewUniform, PreviousViewUniformOffset, PreviousViewUniforms},
    HikariSettings, Taa, Upscale, PREPASS_SHADER_HANDLE,
};
use bevy::{
    ecs::{
        query::QueryItem,
        system::{
            lifetimeless::{Read, SQuery, SRes},
            SystemParamItem,
        },
    },
    pbr::{
        DrawMesh, GpuLights, LightMeta, MeshPipelineKey, MeshUniform, ViewLightsUniformOffset,
        SHADOW_FORMAT,
    },
    prelude::*,
    render::{
        camera::ExtractedCamera,
        extract_component::{
            ComponentUniforms, DynamicUniformIndex, ExtractComponent, ExtractComponentPlugin,
        },
        mesh::MeshVertexBufferLayout,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphContext, SlotInfo, SlotType},
        render_phase::{
            sort_phase_system, AddRenderCommand, CachedRenderPipelinePhaseItem, DrawFunctionId,
            DrawFunctions, EntityPhaseItem, EntityRenderCommand, PhaseItem, RenderCommandResult,
            RenderPhase, SetItemPipeline, TrackedRenderPass,
        },
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
        texture::{FallbackImage, GpuImage, ImageSampler, TextureCache},
        view::{ExtractedView, ViewUniform, ViewUniformOffset, ViewUniforms, VisibleEntities},
        Extract, RenderApp, RenderStage,
    },
    utils::FloatOrd,
};

pub const POSITION_FORMAT: TextureFormat = TextureFormat::Rgba32Float;
pub const NORMAL_FORMAT: TextureFormat = TextureFormat::Rgba8Snorm;
pub const DEPTH_GRADIENT_FORMAT: TextureFormat = TextureFormat::Rg32Float;
pub const INSTANCE_MATERIAL_FORMAT: TextureFormat = TextureFormat::Rg16Float;
pub const VELOCITY_UV_FORMAT: TextureFormat = TextureFormat::Rgba32Float;

pub struct PrepassPlugin;
impl Plugin for PrepassPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(ExtractComponentPlugin::<PrepassTextures>::default())
            .add_system(prepass_textures_system);

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<DrawFunctions<Prepass>>()
                .init_resource::<PrepassPipeline>()
                .init_resource::<SpecializedMeshPipelines<PrepassPipeline>>()
                .add_render_command::<Prepass, DrawPrepass>()
                .add_system_to_stage(RenderStage::Extract, extract_prepass_camera_phases)
                .add_system_to_stage(RenderStage::Queue, queue_prepass_depth_texture)
                .add_system_to_stage(RenderStage::Queue, queue_prepass_meshes)
                .add_system_to_stage(RenderStage::Queue, queue_prepass_bind_groups)
                .add_system_to_stage(RenderStage::Queue, queue_deferred_bind_group)
                .add_system_to_stage(RenderStage::PhaseSort, sort_phase_system::<Prepass>);
        }
    }
}

#[derive(Resource)]
pub struct PrepassPipeline {
    pub view_layout: BindGroupLayout,
    pub mesh_layout: BindGroupLayout,
}

impl FromWorld for PrepassPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let view_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(FrameUniform::min_size()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(ViewUniform::min_size()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(PreviousViewUniform::min_size()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(GpuLights::min_size()),
                    },
                    count: None,
                },
            ],
        });

        let mesh_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(MeshUniform::min_size()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(PreviousMeshUniform::min_size()),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(InstanceIndex::min_size()),
                    },
                    count: None,
                },
            ],
        });

        Self {
            view_layout,
            mesh_layout,
        }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct PrepassPipelineKey {
    pub mesh_pipeline_key: MeshPipelineKey,
    pub temporal_anti_aliasing: bool,
    pub smaa_tu4x: bool,
}

impl SpecializedMeshPipeline for PrepassPipeline {
    type Key = PrepassPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayout,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let vertex_attributes = vec![
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            Mesh::ATTRIBUTE_UV_0.at_shader_location(2),
        ];
        let vertex_buffer_layout = layout.get_layout(&vertex_attributes)?;
        let bind_group_layout = vec![self.view_layout.clone(), self.mesh_layout.clone()];

        let mut shader_defs = vec![];
        if key.temporal_anti_aliasing {
            shader_defs.push("TEMPORAL_ANTI_ALIASING".into());
        }
        if key.smaa_tu4x {
            shader_defs.push("SMAA_TU4X".into());
        }

        Ok(RenderPipelineDescriptor {
            label: None,
            layout: Some(bind_group_layout),
            vertex: VertexState {
                shader: PREPASS_SHADER_HANDLE.typed::<Shader>(),
                shader_defs: shader_defs.clone(),
                entry_point: "vertex".into(),
                buffers: vec![vertex_buffer_layout],
            },
            fragment: Some(FragmentState {
                shader: PREPASS_SHADER_HANDLE.typed::<Shader>(),
                shader_defs,
                entry_point: "fragment".into(),
                targets: vec![
                    Some(ColorTargetState {
                        format: POSITION_FORMAT,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: NORMAL_FORMAT,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: DEPTH_GRADIENT_FORMAT,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: INSTANCE_MATERIAL_FORMAT,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: VELOCITY_UV_FORMAT,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                ],
            }),
            primitive: PrimitiveState {
                topology: key.mesh_pipeline_key.primitive_topology(),
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: SHADOW_FORMAT,
                depth_write_enabled: true,
                depth_compare: CompareFunction::GreaterEqual,
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: MultisampleState::default(),
        })
    }
}

fn extract_prepass_camera_phases(
    mut commands: Commands,
    cameras_3d: Extract<Query<(Entity, &Camera), With<Camera3d>>>,
) {
    for (entity, camera) in cameras_3d.iter() {
        if camera.is_active {
            commands
                .get_or_spawn(entity)
                .insert(RenderPhase::<Prepass>::default());
        }
    }
}

#[derive(Clone, Component, AsBindGroup)]
pub struct PrepassTextures {
    pub size: Extent3d,
    #[texture(0, visibility(all))]
    pub position: Handle<Image>,
    #[texture(1, visibility(all))]
    pub normal: Handle<Image>,
    #[texture(2, visibility(all))]
    pub depth_gradient: Handle<Image>,
    #[texture(3, visibility(all))]
    pub instance_material: Handle<Image>,
    #[texture(4, visibility(all))]
    pub velocity_uv: Handle<Image>,
    #[texture(5, visibility(all))]
    pub previous_position: Handle<Image>,
    #[texture(6, visibility(all))]
    pub previous_normal: Handle<Image>,
    #[texture(7, visibility(all))]
    pub previous_instance_material: Handle<Image>,
    #[texture(8, visibility(all))]
    pub previous_velocity_uv: Handle<Image>,
}

impl PrepassTextures {
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.position, &mut self.previous_position);
        std::mem::swap(&mut self.normal, &mut self.previous_normal);
        std::mem::swap(
            &mut self.instance_material,
            &mut self.previous_instance_material,
        );
        std::mem::swap(&mut self.velocity_uv, &mut self.previous_velocity_uv);
    }
}

impl ExtractComponent for PrepassTextures {
    type Query = &'static Self;
    type Filter = ();

    fn extract_component(item: QueryItem<Self::Query>) -> Self {
        item.clone()
    }
}

pub struct PreparedPrepassTextures<'a> {
    pub position: &'a GpuImage,
    pub normal: &'a GpuImage,
    pub depth_gradient: &'a GpuImage,
    pub instance_material: &'a GpuImage,
    pub velocity_uv: &'a GpuImage,
}

impl PrepassTextures {
    pub fn prepared<'a>(
        &self,
        assets: &'a RenderAssets<Image>,
    ) -> Option<PreparedPrepassTextures<'a>> {
        let prepared = PreparedPrepassTextures {
            position: assets.get(&self.position)?,
            normal: assets.get(&self.normal)?,
            depth_gradient: assets.get(&self.depth_gradient)?,
            instance_material: assets.get(&self.instance_material)?,
            velocity_uv: assets.get(&self.velocity_uv)?,
        };
        Some(prepared)
    }
}

#[allow(clippy::type_complexity)]
fn prepass_textures_system(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut queries: ParamSet<(
        Query<(Entity, &Camera, &HikariSettings), Changed<Camera>>,
        Query<&mut PrepassTextures>,
    )>,
) {
    for (entity, camera, _settings) in &queries.p0() {
        if let Some(size) = camera.physical_target_size() {
            // let scale = settings.upscale.ratio().recip();
            let size = size.as_vec2().ceil().as_uvec2();
            let size = Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            };
            let texture_usage = TextureUsages::COPY_DST
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::RENDER_ATTACHMENT;

            let create_texture = |texture_format| -> Image {
                let texture_descriptor = TextureDescriptor {
                    label: None,
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: texture_format,
                    usage: texture_usage,
                };
                let sampler_descriptor = ImageSampler::Descriptor(SamplerDescriptor {
                    mag_filter: FilterMode::Nearest,
                    min_filter: FilterMode::Nearest,
                    mipmap_filter: FilterMode::Nearest,
                    ..Default::default()
                });

                let mut image = Image {
                    texture_descriptor,
                    sampler_descriptor,
                    ..Default::default()
                };
                image.resize(size);
                image
            };

            let position = images.add(create_texture(POSITION_FORMAT));
            let normal = images.add(create_texture(NORMAL_FORMAT));
            let depth_gradient = images.add(create_texture(DEPTH_GRADIENT_FORMAT));
            let instance_material = images.add(create_texture(INSTANCE_MATERIAL_FORMAT));
            let velocity_uv = images.add(create_texture(VELOCITY_UV_FORMAT));

            let previous_position = images.add(create_texture(POSITION_FORMAT));
            let previous_normal = images.add(create_texture(NORMAL_FORMAT));
            let previous_instance_material = images.add(create_texture(INSTANCE_MATERIAL_FORMAT));
            let previous_velocity_uv = images.add(create_texture(VELOCITY_UV_FORMAT));

            commands.entity(entity).insert(PrepassTextures {
                size,
                position,
                normal,
                depth_gradient,
                instance_material,
                velocity_uv,
                previous_position,
                previous_normal,
                previous_instance_material,
                previous_velocity_uv,
            });
        }
    }

    queries.p1().for_each_mut(|mut textures| textures.swap());
}

#[derive(Component, Deref, DerefMut)]
pub struct PrepassDepthTexture(pub TextureView);

fn queue_prepass_depth_texture(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    query: Query<(Entity, &PrepassTextures)>,
) {
    for (entity, textures) in &query {
        let size = textures.size;
        let texture_usage = TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT;
        let texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: None,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Depth32Float,
                usage: texture_usage,
            },
        );
        commands
            .entity(entity)
            .insert(PrepassDepthTexture(texture.default_view));
    }
}

#[allow(clippy::too_many_arguments)]
fn queue_prepass_meshes(
    draw_functions: Res<DrawFunctions<Prepass>>,
    render_meshes: Res<RenderAssets<Mesh>>,
    prepass_pipeline: Res<PrepassPipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<PrepassPipeline>>,
    mut pipeline_cache: ResMut<PipelineCache>,
    meshes: Query<(Entity, &Handle<Mesh>, &MeshUniform, &DynamicInstanceIndex)>,
    mut views: Query<(
        &ExtractedView,
        &VisibleEntities,
        &mut RenderPhase<Prepass>,
        &HikariSettings,
    )>,
) {
    let draw_function = draw_functions.read().get_id::<DrawPrepass>().unwrap();
    for (view, visible_entities, mut prepass_phase, settings) in &mut views {
        let rangefinder = view.rangefinder3d();

        let add_render_phase = |(entity, mesh_handle, mesh_uniform, _): (
            Entity,
            &Handle<Mesh>,
            &MeshUniform,
            &DynamicInstanceIndex,
        )| {
            if let Some(mesh) = render_meshes.get(mesh_handle) {
                let key = MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);
                let key = PrepassPipelineKey {
                    mesh_pipeline_key: key,
                    temporal_anti_aliasing: matches!(settings.taa, Taa::Jasmine),
                    smaa_tu4x: matches!(settings.upscale, Upscale::SmaaTu4x { .. }),
                };
                let pipeline_id =
                    pipelines.specialize(&mut pipeline_cache, &prepass_pipeline, key, &mesh.layout);
                let pipeline_id = match pipeline_id {
                    Ok(id) => id,
                    Err(err) => {
                        error!("{}", err);
                        return;
                    }
                };
                prepass_phase.add(Prepass {
                    distance: rangefinder.distance(&mesh_uniform.transform),
                    entity,
                    pipeline: pipeline_id,
                    draw_function,
                });
            }
        };

        visible_entities
            .entities
            .iter()
            .filter_map(|visible_entity| meshes.get(*visible_entity).ok())
            .for_each(add_render_phase);
    }
}

#[derive(Resource)]
pub struct PrepassBindGroups {
    pub view: BindGroup,
    pub mesh: BindGroup,
}

#[allow(clippy::too_many_arguments)]
fn queue_prepass_bind_groups(
    mut commands: Commands,
    prepass_pipeline: Res<PrepassPipeline>,
    render_device: Res<RenderDevice>,
    mesh_uniforms: Res<ComponentUniforms<MeshUniform>>,
    previous_mesh_uniforms: Res<ComponentUniforms<PreviousMeshUniform>>,
    instance_render_assets: Res<InstanceRenderAssets>,
    view_uniforms: Res<ViewUniforms>,
    frame_uniforms: Res<ComponentUniforms<FrameUniform>>,
    light_meta: Res<LightMeta>,
    previous_view_uniforms: Res<PreviousViewUniforms>,
) {
    if let (
        Some(view_binding),
        Some(previous_view_binding),
        Some(frame_binding),
        Some(light_binding),
        Some(mesh_binding),
        Some(previous_mesh_binding),
        Some(instance_indices_binding),
    ) = (
        view_uniforms.uniforms.binding(),
        previous_view_uniforms.uniforms.binding(),
        frame_uniforms.uniforms().binding(),
        light_meta.view_gpu_lights.binding(),
        mesh_uniforms.binding(),
        previous_mesh_uniforms.binding(),
        instance_render_assets.instance_indices.binding(),
    ) {
        let view = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &prepass_pipeline.view_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: frame_binding,
                },
                BindGroupEntry {
                    binding: 1,
                    resource: view_binding,
                },
                BindGroupEntry {
                    binding: 2,
                    resource: previous_view_binding,
                },
                BindGroupEntry {
                    binding: 3,
                    resource: light_binding,
                },
            ],
        });
        let mesh = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &prepass_pipeline.mesh_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: mesh_binding,
                },
                BindGroupEntry {
                    binding: 1,
                    resource: previous_mesh_binding,
                },
                BindGroupEntry {
                    binding: 2,
                    resource: instance_indices_binding,
                },
            ],
        });
        commands.insert_resource(PrepassBindGroups { view, mesh });
    }
}

#[derive(Component)]
pub struct DeferredBindGroup(pub BindGroup);

fn queue_deferred_bind_group(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    images: Res<RenderAssets<Image>>,
    fallback_image: Res<FallbackImage>,
    cameras: Query<(Entity, &PrepassTextures), With<ExtractedCamera>>,
) {
    let layout = PrepassTextures::bind_group_layout(&render_device);

    for (entity, prepass) in &cameras {
        if let Ok(prepared_bind_group) =
            prepass.as_bind_group(&layout, &render_device, &images, &fallback_image)
        {
            commands
                .entity(entity)
                .insert(DeferredBindGroup(prepared_bind_group.bind_group));
        }
    }
}

pub struct Prepass {
    pub distance: f32,
    pub entity: Entity,
    pub pipeline: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
}

impl PhaseItem for Prepass {
    type SortKey = FloatOrd;

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        FloatOrd(self.distance)
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }
}

impl EntityPhaseItem for Prepass {
    #[inline]
    fn entity(&self) -> Entity {
        self.entity
    }
}

impl CachedRenderPipelinePhaseItem for Prepass {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline
    }
}

type DrawPrepass = (
    SetItemPipeline,
    SetViewBindGroup<0>,
    SetMeshBindGroup<1>,
    DrawMesh,
);

pub struct SetViewBindGroup<const I: usize>;
impl<const I: usize> EntityRenderCommand for SetViewBindGroup<I> {
    type Param = (
        SRes<PrepassBindGroups>,
        SQuery<(
            Read<DynamicUniformIndex<FrameUniform>>,
            Read<ViewUniformOffset>,
            Read<PreviousViewUniformOffset>,
            Read<ViewLightsUniformOffset>,
        )>,
    );

    fn render<'w>(
        view: Entity,
        _item: Entity,
        (bind_group, view_query): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        if let Ok((frame_uniform, view_uniform, previous_view_uniform, view_lights)) =
            view_query.get_inner(view)
        {
            pass.set_bind_group(
                I,
                &bind_group.into_inner().view,
                &[
                    frame_uniform.index(),
                    view_uniform.offset,
                    previous_view_uniform.offset,
                    view_lights.offset,
                ],
            );
            RenderCommandResult::Success
        } else {
            RenderCommandResult::Failure
        }
    }
}

pub struct SetMeshBindGroup<const I: usize>;
impl<const I: usize> EntityRenderCommand for SetMeshBindGroup<I> {
    type Param = (
        Option<SRes<PrepassBindGroups>>,
        SQuery<(
            Read<DynamicUniformIndex<MeshUniform>>,
            Read<DynamicUniformIndex<PreviousMeshUniform>>,
            Read<DynamicInstanceIndex>,
        )>,
    );

    fn render<'w>(
        _view: Entity,
        item: Entity,
        (bind_group, mesh_query): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        if let (Some(bind_group), Ok((mesh_uniform, previous_mesh_uniform, instance_index))) =
            (bind_group, mesh_query.get_inner(item))
        {
            pass.set_bind_group(
                I,
                &bind_group.into_inner().mesh,
                &[
                    mesh_uniform.index(),
                    previous_mesh_uniform.index(),
                    instance_index.0,
                ],
            );
            RenderCommandResult::Success
        } else {
            RenderCommandResult::Failure
        }
    }
}

pub struct PrepassNode {
    #[allow(clippy::type_complexity)]
    query: QueryState<
        (
            &'static ExtractedCamera,
            &'static RenderPhase<Prepass>,
            &'static Camera3d,
            &'static PrepassDepthTexture,
            &'static PrepassTextures,
        ),
        With<ExtractedView>,
    >,
}

impl PrepassNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            query: world.query_filtered(),
        }
    }
}

impl Node for PrepassNode {
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
        let (camera, prepass_phase, camera_3d, depth, textures) =
            match self.query.get_manual(world, entity) {
                Ok(query) => query,
                Err(_) => return Ok(()),
            };

        let images = world.resource::<RenderAssets<Image>>();
        let textures = match textures.prepared(images) {
            Some(textures) => textures,
            None => return Ok(()),
        };

        {
            #[cfg(feature = "trace")]
            let _main_prepass_span = info_span!("main_prepass").entered();
            let ops = Operations {
                load: LoadOp::Clear(Color::NONE.into()),
                store: true,
            };
            let pass_descriptor = RenderPassDescriptor {
                label: Some("main_prepass"),
                color_attachments: &[
                    Some(RenderPassColorAttachment {
                        view: &textures.position.texture_view,
                        resolve_target: None,
                        ops,
                    }),
                    Some(RenderPassColorAttachment {
                        view: &textures.normal.texture_view,
                        resolve_target: None,
                        ops,
                    }),
                    Some(RenderPassColorAttachment {
                        view: &textures.depth_gradient.texture_view,
                        resolve_target: None,
                        ops,
                    }),
                    Some(RenderPassColorAttachment {
                        view: &textures.instance_material.texture_view,
                        resolve_target: None,
                        ops,
                    }),
                    Some(RenderPassColorAttachment {
                        view: &textures.velocity_uv.texture_view,
                        resolve_target: None,
                        ops,
                    }),
                ],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(Operations {
                        load: camera_3d.depth_load_op.clone().into(),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            };

            let draw_functions = world.resource::<DrawFunctions<Prepass>>();

            let render_pass = render_context
                .command_encoder
                .begin_render_pass(&pass_descriptor);
            let mut draw_functions = draw_functions.write();
            let mut tracked_pass = TrackedRenderPass::new(render_pass);
            if let Some(viewport) = camera.viewport.as_ref() {
                tracked_pass.set_camera_viewport(viewport);
            }
            for item in &prepass_phase.items {
                let draw_function = draw_functions.get_mut(item.draw_function).unwrap();
                draw_function.draw(world, &mut tracked_pass, entity, item);
            }
        }

        Ok(())
    }
}
