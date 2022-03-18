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
        texture::{BevyDefault, CachedTexture, TextureCache},
        view::{ExtractedView, ViewDepthTexture, ViewTarget},
        RenderApp, RenderStage,
    },
};

pub struct OverlayPlugin;

impl Plugin for OverlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_asset::<OverlayMaterial>()
            .add_plugin(ExtractComponentPlugin::<Handle<OverlayMaterial>>::default())
            .add_plugin(RenderAssetPlugin::<OverlayMaterial>::default())
            .init_resource::<ScreenOverlay>()
            .add_startup_system(setup);

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<DrawFunctions<Overlay>>()
                .init_resource::<MaterialPipeline<OverlayMaterial>>()
                .init_resource::<SpecializedPipelines<MaterialPipeline<OverlayMaterial>>>()
                .add_render_command::<Overlay, DrawMaterial<OverlayMaterial>>()
                .add_system_to_stage(RenderStage::Extract, extract_screen_overlay)
                .add_system_to_stage(RenderStage::Prepare, prepare_screen_overlay)
                .add_system_to_stage(RenderStage::Prepare, prepare_overlay_phase)
                .add_system_to_stage(RenderStage::Queue, queue_material_meshes);
        }
    }
}

fn setup(
    mut commands: Commands,
    overlay: Res<ScreenOverlay>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<OverlayMaterial>>,
) {
    let irradiance_resolve = overlay.irradiance_resolve.clone();
    let albedo_resolve = overlay.albedo_resolve.clone();

    commands.spawn_bundle(MaterialMeshBundle {
        mesh: meshes.add(shape::Quad::new(Vec2::ZERO).into()),
        material: materials.add(OverlayMaterial {
            irradiance_image: irradiance_resolve,
            albedo_image: albedo_resolve,
        }),
        ..Default::default()
    });
}

fn extract_screen_overlay(mut commands: Commands, screen_overlay: Res<ScreenOverlay>) {
    commands.insert_resource(screen_overlay.clone());
}

fn prepare_screen_overlay(
    mut commands: Commands,
    images: Res<RenderAssets<Image>>,
    overlay: Res<ScreenOverlay>,
    msaa: Res<Msaa>,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
) {
    let irradiance_depth = texture_cache.get(
        &render_device,
        TextureDescriptor {
            label: Some("volume_overlay_depth_texture"),
            size: overlay.irradiance_size,
            mip_level_count: 1,
            sample_count: msaa.samples,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        },
    );

    let albedo_depth = texture_cache.get(
        &render_device,
        TextureDescriptor {
            label: Some("volume_overlay_depth_texture"),
            size: overlay.albedo_size,
            mip_level_count: 1,
            sample_count: msaa.samples,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        },
    );

    let retrieve_textures = || {
        let irradiance = images.get(&overlay.irradiance)?.texture_view.clone();
        let irradiance_resolve = images
            .get(&overlay.irradiance_resolve)?
            .texture_view
            .clone();
        let albedo = images.get(&overlay.albedo)?.texture_view.clone();
        let albedo_resolve = images.get(&overlay.albedo_resolve)?.texture_view.clone();
        Some((irradiance, irradiance_resolve, albedo, albedo_resolve))
    };

    if let Some((irradiance, irradiance_resolve, albedo, albedo_resolve)) = retrieve_textures() {
        commands.insert_resource(GpuScreenOverlay {
            irradiance,
            irradiance_resolve,
            irradiance_depth,
            albedo,
            albedo_resolve,
            albedo_depth,
        });
    }
}

#[derive(Debug, Clone)]
pub struct ScreenOverlay {
    pub irradiance_size: Extent3d,
    pub irradiance: Handle<Image>,
    pub irradiance_resolve: Handle<Image>,

    pub albedo_size: Extent3d,
    pub albedo: Handle<Image>,
    pub albedo_resolve: Handle<Image>,
}

pub struct GpuScreenOverlay {
    pub irradiance: TextureView,
    pub irradiance_resolve: TextureView,
    pub irradiance_depth: CachedTexture,

    pub albedo: TextureView,
    pub albedo_resolve: TextureView,
    pub albedo_depth: CachedTexture,
}

impl FromWorld for ScreenOverlay {
    fn from_world(world: &mut World) -> Self {
        let windows = world.get_resource::<Windows>().unwrap();
        let window = windows.get_primary().unwrap();
        let width = window.width() as u32;
        let height = window.height() as u32;

        let msaa = world.get_resource::<Msaa>().unwrap();
        let samples = msaa.samples;

        let mut images = world.get_resource_mut::<Assets<Image>>().unwrap();

        let irradiance_size = Extent3d {
            width: width >> 1,
            height: height >> 1,
            depth_or_array_layers: 1,
        };
        let mut image = Image::new_fill(
            irradiance_size,
            TextureDimension::D2,
            &[0, 0, 0, 255],
            TextureFormat::bevy_default(),
        );
        image.texture_descriptor.usage = TextureUsages::COPY_DST
            | TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::TEXTURE_BINDING;

        image.texture_descriptor.sample_count = samples;
        let irradiance = images.add(image.clone());

        image.texture_descriptor.sample_count = 1;
        let irradiance_resolve = images.add(image);

        let albedo_size = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let mut image = Image::new_fill(
            albedo_size,
            TextureDimension::D2,
            &[0, 0, 0, 255],
            TextureFormat::bevy_default(),
        );
        image.texture_descriptor.usage = TextureUsages::COPY_DST
            | TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::TEXTURE_BINDING;
        image.sampler_descriptor.mag_filter = FilterMode::Linear;
        image.sampler_descriptor.min_filter = FilterMode::Linear;

        image.texture_descriptor.sample_count = samples;
        let albedo = images.add(image.clone());

        image.texture_descriptor.sample_count = 1;
        let albedo_resolve = images.add(image);

        Self {
            irradiance_size,
            irradiance,
            irradiance_resolve,
            albedo_size,
            albedo,
            albedo_resolve,
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
        Option<SRes<GpuScreenOverlay>>,
    );
    fn extract_asset(&self) -> Self::ExtractedAsset {
        self.clone()
    }

    fn prepare_asset(
        material: Self::ExtractedAsset,
        (render_device, material_pipeline, images, overlay): &mut SystemParamItem<Self::Param>,
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

        let _overlay = if let Some(overlay) = overlay {
            overlay
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

fn prepare_overlay_phase(mut commands: Commands, views: Query<Entity, With<Volume>>) {
    for view in views.iter() {
        commands
            .entity(view)
            .insert(RenderPhase::<Overlay>::default());
    }
}

#[allow(clippy::too_many_arguments)]
fn queue_material_meshes(
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
