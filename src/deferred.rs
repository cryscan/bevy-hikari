use crate::{overlay::GpuScreenOverlay, GiConfig, ALBEDO_SHADER_HANDLE};
use bevy::{
    core_pipeline::{AlphaMask3d, Opaque3d, Transparent3d},
    pbr::{
        DrawMesh, MeshPipeline, MeshPipelineKey, MeshUniform, SetMaterialBindGroup,
        SetMeshBindGroup, SetMeshViewBindGroup, SpecializedMaterial,
    },
    prelude::FromWorld,
    prelude::*,
    render::{
        render_asset::RenderAssets,
        render_graph::{self, SlotInfo, SlotType},
        render_phase::{
            sort_phase_system, AddRenderCommand, CachedPipelinePhaseItem, DrawFunctionId,
            DrawFunctions, EntityPhaseItem, PhaseItem, RenderPhase, SetItemPipeline,
            TrackedRenderPass,
        },
        render_resource::*,
        renderer::RenderDevice,
        view::{ExtractedView, VisibleEntities},
        RenderApp, RenderStage,
    },
};
use std::marker::PhantomData;

pub struct DeferredPlugin;
impl Plugin for DeferredPlugin {
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<DeferredPipeline>()
                .init_resource::<SpecializedPipelines<DeferredPipeline>>()
                .init_resource::<DrawFunctions<Deferred<Opaque3d>>>()
                .init_resource::<DrawFunctions<Deferred<AlphaMask3d>>>()
                .init_resource::<DrawFunctions<Deferred<Transparent3d>>>()
                .add_system_to_stage(
                    RenderStage::PhaseSort,
                    sort_phase_system::<Deferred<Opaque3d>>,
                )
                .add_system_to_stage(
                    RenderStage::PhaseSort,
                    sort_phase_system::<Deferred<AlphaMask3d>>,
                )
                .add_system_to_stage(
                    RenderStage::PhaseSort,
                    sort_phase_system::<Deferred<Transparent3d>>,
                );
        }
    }
}

#[derive(Default)]
pub struct DeferredMaterialPlugin<M: SpecializedMaterial>(PhantomData<M>);
impl<M: SpecializedMaterial> Plugin for DeferredMaterialPlugin<M> {
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<Deferred<Opaque3d>, DrawDeferredMesh<M>>()
                .add_render_command::<Deferred<AlphaMask3d>, DrawDeferredMesh<M>>()
                .add_render_command::<Deferred<Transparent3d>, DrawDeferredMesh<M>>()
                .add_system_to_stage(RenderStage::Queue, queue_deferred_meshes::<M>);
        }
    }
}

pub struct DeferredPipeline {
    pub material_layout: BindGroupLayout,
    pub mesh_pipeline: MeshPipeline,
}

impl FromWorld for DeferredPipeline {
    fn from_world(world: &mut World) -> Self {
        let mesh_pipeline = world.get_resource::<MeshPipeline>().unwrap().clone();

        let render_device = world.get_resource::<RenderDevice>().unwrap();
        let material_layout = StandardMaterial::bind_group_layout(render_device);

        Self {
            material_layout,
            mesh_pipeline,
        }
    }
}

impl SpecializedPipeline for DeferredPipeline {
    type Key = MeshPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let shader = ALBEDO_SHADER_HANDLE.typed::<Shader>();

        let mut descriptor = self.mesh_pipeline.specialize(key);
        descriptor.fragment.as_mut().unwrap().shader = shader;
        descriptor.layout = Some(vec![
            self.mesh_pipeline.view_layout.clone(),
            self.material_layout.clone(),
            self.mesh_pipeline.mesh_layout.clone(),
        ]);

        descriptor
    }
}

pub type DrawDeferredMesh<M> = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMaterialBindGroup<M, 1>,
    SetMeshBindGroup<2>,
    DrawMesh,
);

pub struct Deferred<T: PhaseItem + EntityPhaseItem + CachedPipelinePhaseItem>(T);

impl<T> PhaseItem for Deferred<T>
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

impl<T> EntityPhaseItem for Deferred<T>
where
    T: PhaseItem + EntityPhaseItem + CachedPipelinePhaseItem,
{
    fn entity(&self) -> Entity {
        self.0.entity()
    }
}

impl<T> CachedPipelinePhaseItem for Deferred<T>
where
    T: PhaseItem + EntityPhaseItem + CachedPipelinePhaseItem,
{
    fn cached_pipeline(&self) -> CachedPipelineId {
        self.0.cached_pipeline()
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn queue_deferred_meshes<M: SpecializedMaterial>(
    opaque_draw_functions: Res<DrawFunctions<Deferred<Opaque3d>>>,
    alpha_mask_draw_functions: Res<DrawFunctions<Deferred<AlphaMask3d>>>,
    transparent_draw_functions: Res<DrawFunctions<Deferred<Transparent3d>>>,
    deferred_pipeline: Res<DeferredPipeline>,
    material_meshes: Query<(&Handle<M>, &Handle<Mesh>, &MeshUniform)>,
    render_meshes: Res<RenderAssets<Mesh>>,
    render_materials: Res<RenderAssets<M>>,
    mut pipelines: ResMut<SpecializedPipelines<DeferredPipeline>>,
    mut pipeline_cache: ResMut<RenderPipelineCache>,
    msaa: Res<Msaa>,
    mut view_query: Query<(
        &ExtractedView,
        &VisibleEntities,
        &mut RenderPhase<Deferred<Opaque3d>>,
        &mut RenderPhase<Deferred<AlphaMask3d>>,
        &mut RenderPhase<Deferred<Transparent3d>>,
    )>,
    config: Res<GiConfig>,
) {
    if !config.enabled {
        return;
    }

    let draw_opaque = opaque_draw_functions
        .read()
        .get_id::<DrawDeferredMesh<M>>()
        .unwrap();
    let draw_alpha_mask = alpha_mask_draw_functions
        .read()
        .get_id::<DrawDeferredMesh<M>>()
        .unwrap();
    let draw_transparent = transparent_draw_functions
        .read()
        .get_id::<DrawDeferredMesh<M>>()
        .unwrap();

    for (view, visible_entities, mut opaque_phase, mut alpha_mask_phase, mut transparent_phase) in
        view_query.iter_mut()
    {
        let inverse_view_matrix = view.transform.compute_matrix().inverse();
        let inverse_view_row_2 = inverse_view_matrix.row(2);

        for entity in visible_entities.entities.iter().cloned() {
            if let Ok((material_handle, mesh_handle, mesh_uniform)) = material_meshes.get(entity) {
                if let Some(material) = render_materials.get(material_handle) {
                    let mut mesh_key = MeshPipelineKey::from_msaa_samples(msaa.samples);
                    if let Some(mesh) = render_meshes.get(mesh_handle) {
                        let mesh_z = inverse_view_row_2.dot(mesh_uniform.transform.col(3));

                        if mesh.has_tangents {
                            mesh_key |= MeshPipelineKey::VERTEX_TANGENTS;
                        }
                        mesh_key |=
                            MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);

                        let alpha_mode = M::alpha_mode(material);
                        if let AlphaMode::Blend = alpha_mode {
                            mesh_key |= MeshPipelineKey::TRANSPARENT_MAIN_PASS;
                        }

                        let pipeline_id =
                            pipelines.specialize(&mut pipeline_cache, &deferred_pipeline, mesh_key);

                        match alpha_mode {
                            AlphaMode::Opaque => opaque_phase.add(Deferred(Opaque3d {
                                distance: -mesh_z,
                                pipeline: pipeline_id,
                                entity,
                                draw_function: draw_opaque,
                            })),
                            AlphaMode::Mask(_) => alpha_mask_phase.add(Deferred(AlphaMask3d {
                                distance: -mesh_z,
                                pipeline: pipeline_id,
                                entity,
                                draw_function: draw_alpha_mask,
                            })),
                            AlphaMode::Blend => transparent_phase.add(Deferred(Transparent3d {
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

#[allow(clippy::type_complexity)]
pub struct DeferredPassNode {
    query: QueryState<
        (
            &'static RenderPhase<Deferred<Opaque3d>>,
            &'static RenderPhase<Deferred<AlphaMask3d>>,
            &'static RenderPhase<Deferred<Transparent3d>>,
        ),
        With<ExtractedView>,
    >,
}

impl DeferredPassNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl render_graph::Node for DeferredPassNode {
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
        let (opaque_phase, alpha_mask_phase, transparent_phase) =
            match self.query.get_manual(world, view_entity) {
                Ok(query) => query,
                Err(_) => return Ok(()),
            };

        let overlay = match world.get_resource::<GpuScreenOverlay>() {
            Some(overlay) => overlay,
            None => return Ok(()),
        };

        {
            let pass_descriptor = RenderPassDescriptor {
                label: Some("deferred_opaque_pass"),
                color_attachments: &[overlay.albedo_color_attachment(Operations {
                    load: LoadOp::Clear(Color::NONE.into()),
                    store: true,
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &overlay.albedo_depth.default_view,
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
                .get_resource::<DrawFunctions<Deferred<Opaque3d>>>()
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
                label: Some("deferred_alpha_mask_pass"),
                color_attachments: &[overlay.albedo_color_attachment(Operations {
                    load: LoadOp::Load,
                    store: true,
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &overlay.albedo_depth.default_view,
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
                .get_resource::<DrawFunctions<Deferred<AlphaMask3d>>>()
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
                label: Some("deferred_transparent_pass"),
                color_attachments: &[overlay.albedo_color_attachment(Operations {
                    load: LoadOp::Load,
                    store: true,
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &overlay.albedo_depth.default_view,
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
                .get_resource::<DrawFunctions<Deferred<Transparent3d>>>()
                .unwrap()
                .write();
            let mut tracked_pass = TrackedRenderPass::new(render_pass);

            for item in &transparent_phase.items {
                let draw_function = draw_functions.get_mut(item.0.draw_function).unwrap();
                draw_function.draw(world, &mut tracked_pass, view_entity, item);
            }
        }

        Ok(())
    }
}
