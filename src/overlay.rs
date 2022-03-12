use crate::{Volume, OVERLAY_SHADER_HANDLE};
use bevy::{
    core::FloatOrd,
    ecs::system::{lifetimeless::SRes, SystemParamItem},
    pbr::{
        DrawMesh, MaterialPipeline, MeshPipelineKey, SetMaterialBindGroup, SetMeshBindGroup,
        SetMeshViewBindGroup, SpecializedMaterial,
    },
    prelude::*,
    reflect::TypeUuid,
    render::{
        render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin, RenderAssets},
        render_component::ExtractComponentPlugin,
        render_graph::{self, SlotInfo, SlotType},
        render_phase::{
            AddRenderCommand, CachedPipelinePhaseItem, DrawFunctionId, DrawFunctions,
            EntityPhaseItem, PhaseItem, RenderPhase, SetItemPipeline, TrackedRenderPass,
        },
        render_resource::*,
        renderer::RenderDevice,
        view::{ExtractedView, ViewDepthTexture, ViewTarget},
        RenderApp, RenderStage,
    },
};

pub struct OverlayPlugin;

impl Plugin for OverlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_asset::<OverlayMaterial>()
            .add_plugin(ExtractComponentPlugin::<Handle<OverlayMaterial>>::default())
            .add_plugin(RenderAssetPlugin::<OverlayMaterial>::default());
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<DrawFunctions<Overlay>>()
                .init_resource::<MaterialPipeline<OverlayMaterial>>()
                .init_resource::<SpecializedPipelines<MaterialPipeline<OverlayMaterial>>>()
                .add_render_command::<Overlay, DrawMaterial<OverlayMaterial>>()
                .add_system_to_stage(RenderStage::Prepare, prepare_overlay_phase)
                .add_system_to_stage(RenderStage::Queue, queue_material_meshes);
        }
    }
}

#[derive(Debug, Clone, TypeUuid)]
#[uuid = "3eb25222-95fd-11ec-b909-0242ac120002"]
pub struct OverlayMaterial {
    pub irradiance_image: Handle<Image>,
    pub albedo_image: Handle<Image>,
}

#[derive(Clone)]
pub struct GpuOverlayMaterial {
    bind_group: BindGroup,
}

#[allow(clippy::type_complexity)]
impl RenderAsset for OverlayMaterial {
    type ExtractedAsset = OverlayMaterial;
    type PreparedAsset = GpuOverlayMaterial;
    type Param = (
        SRes<RenderDevice>,
        SRes<MaterialPipeline<Self>>,
        SRes<RenderAssets<Image>>,
    );
    fn extract_asset(&self) -> Self::ExtractedAsset {
        self.clone()
    }

    fn prepare_asset(
        material: Self::ExtractedAsset,
        (render_device, material_pipeline, images): &mut SystemParamItem<Self::Param>,
    ) -> Result<Self::PreparedAsset, PrepareAssetError<Self::ExtractedAsset>> {
        let irradiance = if let Some(result) = images.get(&material.irradiance_image) {
            result
        } else {
            return Err(PrepareAssetError::RetryNextUpdate(material));
        };

        let albedo = if let Some(result) = images.get(&material.albedo_image) {
            result
        } else {
            return Err(PrepareAssetError::RetryNextUpdate(material));
        };

        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&irradiance.texture_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&irradiance.sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&albedo.texture_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(&albedo.sampler),
                },
            ],
            label: Some("overlay_bind_group"),
            layout: &material_pipeline.material_layout,
        });

        Ok(GpuOverlayMaterial { bind_group })
    }
}

impl SpecializedMaterial for OverlayMaterial {
    type Key = ();

    fn key(_material: &<Self as RenderAsset>::PreparedAsset) -> Self::Key {}

    fn vertex_shader(_asset_server: &AssetServer) -> Option<Handle<Shader>> {
        Some(OVERLAY_SHADER_HANDLE.typed())
    }

    fn fragment_shader(_asset_server: &AssetServer) -> Option<Handle<Shader>> {
        Some(OVERLAY_SHADER_HANDLE.typed())
    }

    fn bind_group(render_asset: &<Self as RenderAsset>::PreparedAsset) -> &BindGroup {
        &render_asset.bind_group
    }

    fn bind_group_layout(render_device: &RenderDevice) -> BindGroupLayout {
        render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("overlay_layout"),
        })
    }

    fn specialize(_key: Self::Key, descriptor: &mut RenderPipelineDescriptor) {
        descriptor.fragment.as_mut().unwrap().targets[0].blend = Some(BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::Zero,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
        })
    }

    fn alpha_mode(_material: &<Self as RenderAsset>::PreparedAsset) -> AlphaMode {
        AlphaMode::Blend
    }
}

pub fn prepare_overlay_phase(mut commands: Commands, views: Query<Entity, With<Volume>>) {
    for view in views.iter() {
        commands
            .entity(view)
            .insert(RenderPhase::<Overlay>::default());
    }
}

#[allow(clippy::too_many_arguments)]
pub fn queue_material_meshes(
    draw_functions: Res<DrawFunctions<Overlay>>,
    material_pipeline: Res<MaterialPipeline<OverlayMaterial>>,
    mut pipelines: ResMut<SpecializedPipelines<MaterialPipeline<OverlayMaterial>>>,
    mut pipeline_cache: ResMut<RenderPipelineCache>,
    msaa: Res<Msaa>,
    render_meshes: Res<RenderAssets<Mesh>>,
    render_materials: Res<RenderAssets<OverlayMaterial>>,
    material_meshes: Query<(Entity, &Handle<OverlayMaterial>, &Handle<Mesh>)>,
    mut phases: Query<&mut RenderPhase<Overlay>>,
) {
    for mut phase in phases.iter_mut() {
        let draw_function = draw_functions
            .read()
            .get_id::<DrawMaterial<OverlayMaterial>>()
            .unwrap();

        let mesh_key = MeshPipelineKey::from_msaa_samples(msaa.samples);

        for (entity, material, mesh) in material_meshes.iter() {
            if let Some(material) = render_materials.get(material) {
                let mut mesh_key = mesh_key;
                if let Some(mesh) = render_meshes.get(mesh) {
                    if mesh.has_tangents {
                        mesh_key |= MeshPipelineKey::VERTEX_TANGENTS;
                    }
                    mesh_key |= MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);
                }

                mesh_key |= MeshPipelineKey::TRANSPARENT_MAIN_PASS;

                let specialized_key = OverlayMaterial::key(material);
                let pipeline_id = pipelines.specialize(
                    &mut pipeline_cache,
                    &material_pipeline,
                    (mesh_key, specialized_key),
                );

                phase.add(Overlay {
                    distance: 0.0,
                    entity,
                    pipeline: pipeline_id,
                    draw_function,
                });
            }
        }
    }
}

pub struct Overlay {
    distance: f32,
    entity: Entity,
    pipeline: CachedPipelineId,
    draw_function: DrawFunctionId,
}

impl PhaseItem for Overlay {
    type SortKey = FloatOrd;

    fn sort_key(&self) -> Self::SortKey {
        FloatOrd(self.distance)
    }

    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }
}

impl EntityPhaseItem for Overlay {
    fn entity(&self) -> Entity {
        self.entity
    }
}

impl CachedPipelinePhaseItem for Overlay {
    fn cached_pipeline(&self) -> CachedPipelineId {
        self.pipeline
    }
}

type DrawMaterial<M> = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMaterialBindGroup<M, 1>,
    SetMeshBindGroup<2>,
    DrawMesh,
);

pub struct OverlayPassNode {
    query: QueryState<
        (
            &'static RenderPhase<Overlay>,
            &'static ViewTarget,
            &'static ViewDepthTexture,
        ),
        With<ExtractedView>,
    >,
}

impl OverlayPassNode {
    pub const IN_VIEW: &'static str = "view";

    pub fn new(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl render_graph::Node for OverlayPassNode {
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
        let (phase, target, depth) = match self.query.get_manual(world, view_entity) {
            Ok(query) => query,
            Err(_) => return Ok(()),
        };

        let pass_descriptor = RenderPassDescriptor {
            label: Some("tracing_pass"),
            color_attachments: &[target.get_color_attachment(Operations {
                load: LoadOp::Load,
                store: true,
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &depth.view,
                depth_ops: Some(Operations {
                    load: LoadOp::Load,
                    store: false,
                }),
                stencil_ops: None,
            }),
        };

        let draw_functions = world.get_resource::<DrawFunctions<Overlay>>().unwrap();
        let render_pass = render_context
            .command_encoder
            .begin_render_pass(&pass_descriptor);
        let mut draw_functions = draw_functions.write();
        let mut tracked_pass = TrackedRenderPass::new(render_pass);
        for item in &phase.items {
            let draw_function = draw_functions.get_mut(item.draw_function).unwrap();
            draw_function.draw(world, &mut tracked_pass, view_entity, item);
        }

        Ok(())
    }
}
