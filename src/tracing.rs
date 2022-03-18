use crate::{
    overlay::ScreenOverlay, GpuVolume, NotGiReceiver, Volume, VolumeBindings, VolumeMeta,
    VolumeUniformOffset, TRACING_SHADER_HANDLE,
};
use bevy::{
    core::FloatOrd,
    core_pipeline::{AlphaMask3d, Opaque3d, Transparent3d},
    ecs::system::lifetimeless::{Read, SQuery},
    pbr::{
        DrawMesh, MeshPipeline, MeshPipelineKey, MeshUniform, SetMaterialBindGroup,
        SetMeshBindGroup, SetMeshViewBindGroup, SpecializedMaterial,
    },
    prelude::*,
    render::{
        render_asset::RenderAssets,
        render_graph::{self, SlotInfo, SlotType},
        render_phase::{
            sort_phase_system, AddRenderCommand, CachedPipelinePhaseItem, DrawFunctionId,
            DrawFunctions, EntityPhaseItem, EntityRenderCommand, PhaseItem, RenderCommandResult,
            RenderPhase, SetItemPipeline, TrackedRenderPass,
        },
        render_resource::{
            std140::{AsStd140, Std140},
            *,
        },
        renderer::RenderDevice,
        view::{ExtractedView, VisibleEntities},
        RenderApp, RenderStage,
    },
};
use std::marker::PhantomData;

pub struct TracingPlugin;
impl Plugin for TracingPlugin {
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<DirectionsUniform>()
                .init_resource::<TracingPipeline>()
                .init_resource::<SpecializedPipelines<TracingPipeline>>()
                .init_resource::<DrawFunctions<Tracing<Opaque3d>>>()
                .init_resource::<DrawFunctions<Tracing<AlphaMask3d>>>()
                .init_resource::<DrawFunctions<Tracing<Transparent3d>>>()
                .init_resource::<DrawFunctions<AmbientOcclusion>>()
                .add_system_to_stage(RenderStage::Extract, extract_receiver)
                .add_system_to_stage(RenderStage::Queue, queue_tracing_bind_groups)
                .add_system_to_stage(
                    RenderStage::PhaseSort,
                    sort_phase_system::<Tracing<Opaque3d>>,
                )
                .add_system_to_stage(
                    RenderStage::PhaseSort,
                    sort_phase_system::<Tracing<AlphaMask3d>>,
                )
                .add_system_to_stage(
                    RenderStage::PhaseSort,
                    sort_phase_system::<Tracing<Transparent3d>>,
                )
                .add_system_to_stage(
                    RenderStage::PhaseSort,
                    sort_phase_system::<AmbientOcclusion>,
                );
        }
    }
}

/// The plugin registers the GI draw functions/systems for a [`SpecializedMaterial`].
#[derive(Default)]
pub struct TracingMaterialPlugin<M: SpecializedMaterial>(PhantomData<M>);
impl<M: SpecializedMaterial> Plugin for TracingMaterialPlugin<M> {
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<Tracing<Opaque3d>, DrawTracingMesh<M>>()
                .add_render_command::<Tracing<AlphaMask3d>, DrawTracingMesh<M>>()
                .add_render_command::<Tracing<Transparent3d>, DrawTracingMesh<M>>()
                .add_render_command::<AmbientOcclusion, DrawTracingMesh<M>>()
                .add_system_to_stage(RenderStage::Queue, queue_tracing_meshes::<M>);
        }
    }
}

pub struct TracingPipeline {
    pub material_layout: BindGroupLayout,
    pub tracing_layout: BindGroupLayout,
    pub mesh_pipeline: MeshPipeline,
}

impl FromWorld for TracingPipeline {
    fn from_world(world: &mut World) -> Self {
        let mesh_pipeline = world.get_resource::<MeshPipeline>().unwrap().clone();

        let render_device = world.get_resource::<RenderDevice>().unwrap();

        let material_layout = StandardMaterial::bind_group_layout(render_device);

        let anisotropic_layout_entries = (0..6).map(|direction| BindGroupLayoutEntry {
            binding: direction,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        });
        let tracing_layout_entries = vec![
            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(GpuDirections::std140_size_static() as u64),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 7,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: BufferSize::new(GpuVolume::std140_size_static() as u64),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 8,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 9,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ];

        let tracing_layout_entries = anisotropic_layout_entries
            .chain(tracing_layout_entries)
            .collect::<Vec<_>>();

        let tracing_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("tracing_layout"),
            entries: tracing_layout_entries.as_slice(),
        });

        Self {
            material_layout,
            tracing_layout,
            mesh_pipeline,
        }
    }
}

#[derive(Default, Hash, PartialEq, Eq, Clone, Copy)]
pub struct TracingPipelineKey {
    pub not_receiver: bool,
    pub ambient_occlusion: bool,
}

impl SpecializedPipeline for TracingPipeline {
    type Key = (MeshPipelineKey, TracingPipelineKey);

    fn specialize(&self, (mesh_key, tracing_key): Self::Key) -> RenderPipelineDescriptor {
        let shader = TRACING_SHADER_HANDLE.typed::<Shader>();

        let mut descriptor = self.mesh_pipeline.specialize(mesh_key);
        let mut fragment = descriptor.fragment.as_mut().unwrap();
        fragment.shader = shader;
        if tracing_key.not_receiver {
            fragment.shader_defs.push("NOT_GI_RECEIVER".to_string());
        }
        if tracing_key.ambient_occlusion {
            fragment.shader_defs.push("AMBIENT_OCCLUSION".to_string());
            fragment.targets[0].write_mask = ColorWrites::ALPHA;
        }

        descriptor.depth_stencil = Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: CompareFunction::Greater,
            stencil: StencilState::default(),
            bias: DepthBiasState {
                constant: 0,
                slope_scale: 0.0,
                clamp: 0.0,
            },
        });
        descriptor.layout = Some(vec![
            self.mesh_pipeline.view_layout.clone(),
            self.material_layout.clone(),
            self.mesh_pipeline.mesh_layout.clone(),
            self.tracing_layout.clone(),
        ]);

        descriptor
    }
}

pub struct DirectionsUniform(pub Buffer);

impl FromWorld for DirectionsUniform {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: None,
            contents: GpuDirections::default().as_std140().as_bytes(),
            usage: BufferUsages::UNIFORM,
        });
        Self(buffer)
    }
}

#[derive(AsStd140)]
struct GpuDirections {
    data: [Vec3; 14],
}

impl Default for GpuDirections {
    fn default() -> Self {
        Self {
            data: [
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(-1.0, 1.0, 1.0),
                Vec3::new(1.0, -1.0, 1.0),
                Vec3::new(-1.0, -1.0, 1.0),
                Vec3::new(1.0, 1.0, -1.0),
                Vec3::new(-1.0, 1.0, -1.0),
                Vec3::new(1.0, -1.0, -1.0),
                Vec3::new(-1.0, -1.0, -1.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::new(0.0, 0.0, -1.0),
            ],
        }
    }
}

#[derive(Component)]
pub struct TracingBindGroup {
    value: BindGroup,
}

fn extract_receiver(
    mut commands: Commands,
    query: Query<(Entity, &Handle<Mesh>), With<NotGiReceiver>>,
) {
    for (entity, _) in query.iter() {
        commands.get_or_spawn(entity).insert(NotGiReceiver);
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn queue_tracing_meshes<M: SpecializedMaterial>(
    opaque_draw_functions: Res<DrawFunctions<Tracing<Opaque3d>>>,
    alpha_mask_draw_functions: Res<DrawFunctions<Tracing<AlphaMask3d>>>,
    transparent_draw_functions: Res<DrawFunctions<Tracing<Transparent3d>>>,
    ambient_occlusion_draw_functions: Res<DrawFunctions<AmbientOcclusion>>,
    tracing_pipeline: Res<TracingPipeline>,
    material_meshes: Query<(
        &Handle<M>,
        &Handle<Mesh>,
        &MeshUniform,
        Option<&NotGiReceiver>,
    )>,
    render_meshes: Res<RenderAssets<Mesh>>,
    render_materials: Res<RenderAssets<M>>,
    mut pipelines: ResMut<SpecializedPipelines<TracingPipeline>>,
    mut pipeline_cache: ResMut<RenderPipelineCache>,
    msaa: Res<Msaa>,
    mut view_query: Query<(
        &ExtractedView,
        &VisibleEntities,
        &mut RenderPhase<Tracing<Opaque3d>>,
        &mut RenderPhase<Tracing<AlphaMask3d>>,
        &mut RenderPhase<Tracing<Transparent3d>>,
        &mut RenderPhase<AmbientOcclusion>,
    )>,
) {
    let draw_opaque = opaque_draw_functions
        .read()
        .get_id::<DrawTracingMesh<M>>()
        .unwrap();
    let draw_alpha_mask = alpha_mask_draw_functions
        .read()
        .get_id::<DrawTracingMesh<M>>()
        .unwrap();
    let draw_transparent = transparent_draw_functions
        .read()
        .get_id::<DrawTracingMesh<M>>()
        .unwrap();
    let draw_ambient_occlusion = ambient_occlusion_draw_functions
        .read()
        .get_id::<DrawTracingMesh<M>>()
        .unwrap();

    for (
        view,
        visible_entities,
        mut opaque_phase,
        mut alpha_mask_phase,
        mut transparent_phase,
        mut ambient_occlusion_phase,
    ) in view_query.iter_mut()
    {
        let inverse_view_matrix = view.transform.compute_matrix().inverse();
        let inverse_view_row_2 = inverse_view_matrix.row(2);

        for entity in visible_entities.entities.iter().cloned() {
            if let Ok((material_handle, mesh_handle, mesh_uniform, not_receiver)) =
                material_meshes.get(entity)
            {
                if let Some(material) = render_materials.get(material_handle) {
                    let mut mesh_key = MeshPipelineKey::from_msaa_samples(msaa.samples);
                    if let Some(mesh) = render_meshes.get(mesh_handle) {
                        let mesh_z = inverse_view_row_2.dot(mesh_uniform.transform.col(3));
                        let not_receiver = not_receiver.is_some();

                        if mesh.has_tangents {
                            mesh_key |= MeshPipelineKey::VERTEX_TANGENTS;
                        }
                        mesh_key |=
                            MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);

                        let tracing_key = TracingPipelineKey {
                            not_receiver,
                            ambient_occlusion: true,
                        };
                        let pipeline_id = pipelines.specialize(
                            &mut pipeline_cache,
                            &tracing_pipeline,
                            (mesh_key, tracing_key),
                        );

                        ambient_occlusion_phase.add(AmbientOcclusion {
                            distance: -mesh_z,
                            entity,
                            pipeline: pipeline_id,
                            draw_function: draw_ambient_occlusion,
                        });

                        let alpha_mode = M::alpha_mode(material);
                        if let AlphaMode::Blend = alpha_mode {
                            mesh_key |= MeshPipelineKey::TRANSPARENT_MAIN_PASS;
                        }

                        let tracing_key = TracingPipelineKey {
                            not_receiver,
                            ambient_occlusion: false,
                        };
                        let pipeline_id = pipelines.specialize(
                            &mut pipeline_cache,
                            &tracing_pipeline,
                            (mesh_key, tracing_key),
                        );

                        match alpha_mode {
                            AlphaMode::Opaque => opaque_phase.add(Tracing(Opaque3d {
                                distance: -mesh_z,
                                pipeline: pipeline_id,
                                entity,
                                draw_function: draw_opaque,
                            })),
                            AlphaMode::Mask(_) => alpha_mask_phase.add(Tracing(AlphaMask3d {
                                distance: -mesh_z,
                                pipeline: pipeline_id,
                                entity,
                                draw_function: draw_alpha_mask,
                            })),
                            AlphaMode::Blend => transparent_phase.add(Tracing(Transparent3d {
                                distance: mesh_z,
                                pipeline: pipeline_id,
                                entity,
                                draw_function: draw_transparent,
                            })),
                        }
                    }
                }
            }
        }
    }
}

pub fn queue_tracing_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    tracing_pipeline: Res<TracingPipeline>,
    directions: Res<DirectionsUniform>,
    volume_meta: Res<VolumeMeta>,
    volume_query: Query<(Entity, &VolumeBindings), With<Volume>>,
) {
    for (view, volume_bindings) in volume_query.iter() {
        let anisotropic_bindings = (0..6).map(|direction| {
            let view = &volume_bindings.anisotropic_textures[direction].default_view;
            BindGroupEntry {
                binding: direction as u32,
                resource: BindingResource::TextureView(view),
            }
        });
        let tracing_bindings = vec![
            BindGroupEntry {
                binding: 6,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &directions.0,
                    offset: 0,
                    size: None,
                }),
            },
            BindGroupEntry {
                binding: 7,
                resource: volume_meta.volume_uniforms.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 8,
                resource: BindingResource::TextureView(&volume_bindings.voxel_texture.default_view),
            },
            BindGroupEntry {
                binding: 9,
                resource: BindingResource::Sampler(&volume_bindings.texture_sampler),
            },
        ];

        let tracing_bindings = anisotropic_bindings
            .chain(tracing_bindings)
            .collect::<Vec<_>>();

        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: Some("tracing_bind_group"),
            layout: &tracing_pipeline.tracing_layout,
            entries: tracing_bindings.as_slice(),
        });

        commands
            .entity(view)
            .insert(TracingBindGroup { value: bind_group });
    }
}

pub struct Tracing<T: PhaseItem + EntityPhaseItem + CachedPipelinePhaseItem>(T);

impl<T> PhaseItem for Tracing<T>
where
    T: PhaseItem + EntityPhaseItem + CachedPipelinePhaseItem,
{
    type SortKey = T::SortKey;

    fn sort_key(&self) -> Self::SortKey {
        self.0.sort_key()
    }

    fn draw_function(&self) -> DrawFunctionId {
        self.0.draw_function()
    }
}

impl<T> EntityPhaseItem for Tracing<T>
where
    T: PhaseItem + EntityPhaseItem + CachedPipelinePhaseItem,
{
    fn entity(&self) -> Entity {
        self.0.entity()
    }
}

impl<T> CachedPipelinePhaseItem for Tracing<T>
where
    T: PhaseItem + EntityPhaseItem + CachedPipelinePhaseItem,
{
    fn cached_pipeline(&self) -> CachedPipelineId {
        self.0.cached_pipeline()
    }
}

pub struct AmbientOcclusion {
    distance: f32,
    entity: Entity,
    pipeline: CachedPipelineId,
    draw_function: DrawFunctionId,
}

impl PhaseItem for AmbientOcclusion {
    type SortKey = FloatOrd;

    fn sort_key(&self) -> Self::SortKey {
        FloatOrd(self.distance)
    }

    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }
}

impl EntityPhaseItem for AmbientOcclusion {
    fn entity(&self) -> Entity {
        self.entity
    }
}

impl CachedPipelinePhaseItem for AmbientOcclusion {
    fn cached_pipeline(&self) -> CachedPipelineId {
        self.pipeline
    }
}

pub type DrawTracingMesh<M> = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMaterialBindGroup<M, 1>,
    SetMeshBindGroup<2>,
    SetTracingBindGroup<3>,
    DrawMesh,
);

pub struct SetTracingBindGroup<const I: usize>;
impl<const I: usize> EntityRenderCommand for SetTracingBindGroup<I> {
    type Param = SQuery<(Read<VolumeUniformOffset>, Read<TracingBindGroup>)>;

    fn render<'w>(
        view: Entity,
        _item: Entity,
        query: bevy::ecs::system::SystemParamItem<'w, '_, Self::Param>,
        pass: &mut bevy::render::render_phase::TrackedRenderPass<'w>,
    ) -> bevy::render::render_phase::RenderCommandResult {
        let (volume_uniform_offset, bind_group) = query.get(view).unwrap();
        pass.set_bind_group(I, &bind_group.value, &[volume_uniform_offset.offset]);
        RenderCommandResult::Success
    }
}

#[allow(clippy::type_complexity)]
pub struct TracingPassNode {
    query: QueryState<
        (
            &'static RenderPhase<Tracing<Opaque3d>>,
            &'static RenderPhase<Tracing<AlphaMask3d>>,
            &'static RenderPhase<Tracing<Transparent3d>>,
            &'static RenderPhase<AmbientOcclusion>,
            &'static VolumeBindings,
        ),
        With<ExtractedView>,
    >,
}

impl TracingPassNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl render_graph::Node for TracingPassNode {
    fn input(&self) -> Vec<SlotInfo> {
        vec![SlotInfo::new(Self::IN_VIEW, SlotType::Entity)]
    }

    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
    }

    fn run(
        &self,
        graph: &mut render_graph::RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let view_entity = graph.get_input_entity(Self::IN_VIEW)?;
        let (opaque_phase, alpha_mask_phase, transparent_phase, ambient_occlusion_phase, bindings) =
            match self.query.get_manual(world, view_entity) {
                Ok(query) => query,
                Err(_) => return Ok(()),
            };

        let overlay = world.get_resource::<ScreenOverlay>().unwrap();
        let images = world.get_resource::<RenderAssets<Image>>().unwrap();
        let view = &images[&overlay.irradiance].texture_view;
        let resolve_target = &images[&overlay.irradiance_resolve].texture_view;

        {
            let pass_descriptor = RenderPassDescriptor {
                label: Some("tracing_opaque_pass"),
                color_attachments: &[RenderPassColorAttachment {
                    view,
                    resolve_target: Some(resolve_target),
                    ops: Operations {
                        load: LoadOp::Clear(Color::NONE.into()),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &bindings.irradiance_depth_texture.default_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(0.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            };

            let render_pass = render_context
                .command_encoder
                .begin_render_pass(&pass_descriptor);

            let mut draw_functions = world
                .get_resource::<DrawFunctions<Tracing<Opaque3d>>>()
                .unwrap()
                .write();
            let mut tracked_pass = TrackedRenderPass::new(render_pass);

            for item in &opaque_phase.items {
                let draw_function = draw_functions.get_mut(item.0.draw_function).unwrap();
                draw_function.draw(world, &mut tracked_pass, view_entity, item);
            }
        }

        {
            let pass_descriptor = RenderPassDescriptor {
                label: Some("tracing_alpha_mask_pass"),
                color_attachments: &[RenderPassColorAttachment {
                    view,
                    resolve_target: Some(resolve_target),
                    ops: Operations {
                        load: LoadOp::Load,
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &bindings.irradiance_depth_texture.default_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Load,
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            };

            let render_pass = render_context
                .command_encoder
                .begin_render_pass(&pass_descriptor);

            let mut draw_functions = world
                .get_resource::<DrawFunctions<Tracing<AlphaMask3d>>>()
                .unwrap()
                .write();
            let mut tracked_pass = TrackedRenderPass::new(render_pass);

            for item in &alpha_mask_phase.items {
                let draw_function = draw_functions.get_mut(item.0.draw_function).unwrap();
                draw_function.draw(world, &mut tracked_pass, view_entity, item);
            }
        }

        {
            let pass_descriptor = RenderPassDescriptor {
                label: Some("tracing_transparent_pass"),
                color_attachments: &[RenderPassColorAttachment {
                    view,
                    resolve_target: Some(resolve_target),
                    ops: Operations {
                        load: LoadOp::Load,
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &bindings.irradiance_depth_texture.default_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Load,
                        store: false,
                    }),
                    stencil_ops: None,
                }),
            };

            let render_pass = render_context
                .command_encoder
                .begin_render_pass(&pass_descriptor);

            let mut draw_functions = world
                .get_resource::<DrawFunctions<Tracing<Transparent3d>>>()
                .unwrap()
                .write();
            let mut tracked_pass = TrackedRenderPass::new(render_pass);

            for item in &transparent_phase.items {
                let draw_function = draw_functions.get_mut(item.0.draw_function).unwrap();
                draw_function.draw(world, &mut tracked_pass, view_entity, item);
            }
        }

        {
            let pass_descriptor = RenderPassDescriptor {
                label: Some("tracing_ambient_occlusion_pass"),
                color_attachments: &[RenderPassColorAttachment {
                    view,
                    resolve_target: Some(resolve_target),
                    ops: Operations {
                        load: LoadOp::Load,
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &bindings.irradiance_depth_texture.default_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(0.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            };

            let render_pass = render_context
                .command_encoder
                .begin_render_pass(&pass_descriptor);

            let mut draw_functions = world
                .get_resource::<DrawFunctions<AmbientOcclusion>>()
                .unwrap()
                .write();
            let mut tracked_pass = TrackedRenderPass::new(render_pass);

            for item in &ambient_occlusion_phase.items {
                let draw_function = draw_functions.get_mut(item.draw_function).unwrap();
                draw_function.draw(world, &mut tracked_pass, view_entity, item);
            }
        }

        Ok(())
    }
}
