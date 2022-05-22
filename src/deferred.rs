use crate::{utils::SimplePassDriver, volume::Volume, ALBEDO_SHADER_HANDLE};
use bevy::{
    core_pipeline::{node, AlphaMask3d, Opaque3d, Transparent3d},
    pbr::*,
    prelude::*,
    render::{
        camera::{ActiveCamera, Camera3d, CameraTypePlugin, RenderTarget},
        mesh::MeshVertexBufferLayout,
        render_asset::RenderAssets,
        render_graph::RenderGraph,
        render_phase::{DrawFunctions, RenderPhase, SetItemPipeline},
        render_resource::*,
        renderer::RenderDevice,
        texture::BevyDefault,
        view::ExtractedView,
        RenderApp, RenderStage,
    },
    transform::TransformSystem,
};

pub struct DeferredPlugin;
impl Plugin for DeferredPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(CameraTypePlugin::<DeferredCamera>::default())
            .add_startup_system(setup_deferred_camera)
            .add_system_to_stage(
                CoreStage::PostUpdate,
                update_deferred_camera.before(TransformSystem::TransformPropagate),
            );

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<DeferredPipeline>()
            .init_resource::<SpecializedMeshPipelines<DeferredPipeline>>()
            .add_system_to_stage(RenderStage::Extract, extract_deferred_camera_phases)
            .add_system_to_stage(
                RenderStage::Queue,
                queue_deferred_meshes.after(queue_material_meshes::<StandardMaterial>),
            );

        use crate::node::DEFERRED_PASS_DRIVER;
        use node::{CLEAR_PASS_DRIVER, MAIN_PASS_DEPENDENCIES};

        let driver = SimplePassDriver::<DeferredCamera>::new(&mut render_app.world);
        let mut graph = render_app.world.resource_mut::<RenderGraph>();
        graph.add_node(DEFERRED_PASS_DRIVER, driver);

        graph
            .add_node_edge(CLEAR_PASS_DRIVER, DEFERRED_PASS_DRIVER)
            .unwrap();
        graph
            .add_node_edge(MAIN_PASS_DEPENDENCIES, DEFERRED_PASS_DRIVER)
            .unwrap();
    }
}

#[derive(Default, Component)]
pub struct DeferredCamera;

pub fn setup_deferred_camera(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    windows: Res<Windows>,
) {
    let size = Extent3d {
        width: windows.primary().width() as u32,
        height: windows.primary().height() as u32,
        ..default()
    };
    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::bevy_default(),
            usage: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST,
        },
        ..default()
    };
    image.resize(size);
    let image_handle = images.add(image);

    let camera = Camera {
        target: RenderTarget::Image(image_handle),
        ..default()
    };

    commands.spawn_bundle(PerspectiveCameraBundle::<DeferredCamera> {
        camera,
        perspective_projection: default(),
        visible_entities: default(),
        frustum: default(),
        transform: default(),
        global_transform: default(),
        marker: default(),
    });
}

/// Sync deferred camera's transform with main camera.
pub fn update_deferred_camera(
    main_active: Res<ActiveCamera<Camera3d>>,
    deferred_active: Res<ActiveCamera<DeferredCamera>>,
    mut query: Query<&mut Transform>,
) {
    if let Some((main_camera, deferred_camera)) = main_active.get().zip(deferred_active.get()) {
        let [main_transform, mut deferred_transform] =
            query.many_mut([main_camera, deferred_camera]);
        *deferred_transform = *main_transform;
    }
}

pub fn extract_deferred_camera_phases(
    mut commands: Commands,
    active: Res<ActiveCamera<DeferredCamera>>,
) {
    if let Some(entity) = active.get() {
        commands.get_or_spawn(entity).insert_bundle((
            RenderPhase::<Opaque3d>::default(),
            RenderPhase::<AlphaMask3d>::default(),
            RenderPhase::<Transparent3d>::default(),
        ));
    }
}

pub struct DeferredPipeline {
    pub material_layout: BindGroupLayout,
    pub mesh_pipeline: MeshPipeline,
}

impl FromWorld for DeferredPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let mesh_pipeline = world.resource::<MeshPipeline>().clone();

        let material_layout = StandardMaterial::bind_group_layout(render_device);

        Self {
            material_layout,
            mesh_pipeline,
        }
    }
}

impl SpecializedMeshPipeline for DeferredPipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayout,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.mesh_pipeline.specialize(key, layout)?;
        descriptor.layout = Some(vec![
            self.mesh_pipeline.view_layout.clone(),
            self.material_layout.clone(),
            self.mesh_pipeline.mesh_layout.clone(),
        ]);

        let shader = ALBEDO_SHADER_HANDLE.typed();
        let fragment = descriptor.fragment.as_mut().unwrap();
        fragment.shader = shader;
        fragment.targets[0].blend = Some(BlendState::REPLACE);
        if key.contains(MeshPipelineKey::TRANSPARENT_MAIN_PASS) {
            fragment.shader_defs.push("TRANSPARENT_MAIN_PASS".into());
        }

        Ok(descriptor)
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn queue_deferred_meshes(
    transparent_draw_functions: Res<DrawFunctions<Transparent3d>>,
    deferred_pipeline: Res<DeferredPipeline>,
    material_meshes: Query<(&Handle<StandardMaterial>, &Handle<Mesh>)>,
    render_meshes: Res<RenderAssets<Mesh>>,
    render_materials: Res<RenderAssets<StandardMaterial>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<DeferredPipeline>>,
    mut pipeline_cache: ResMut<PipelineCache>,
    msaa: Res<Msaa>,
    volume: Res<Volume>,
    mut views: Query<
        (
            &mut RenderPhase<Opaque3d>,
            &mut RenderPhase<AlphaMask3d>,
            &mut RenderPhase<Transparent3d>,
        ),
        (With<ExtractedView>, With<DeferredCamera>),
    >,
) {
    for (mut opaque_phase, mut alpha_mask_phase, mut transparent_phase) in views.iter_mut() {
        if !volume.enabled {
            opaque_phase.items.clear();
            alpha_mask_phase.items.clear();
            transparent_phase.items.clear();
            continue;
        }

        for item in opaque_phase.items.iter_mut() {
            if let Ok((material_handle, mesh_handle)) = material_meshes.get(item.entity) {
                if let Some((_material, mesh)) = render_materials
                    .get(material_handle)
                    .zip(render_meshes.get(mesh_handle))
                {
                    let mut key = MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);
                    key |= MeshPipelineKey::from_msaa_samples(msaa.samples);

                    if let Ok(pipeline) = pipelines.specialize(
                        &mut pipeline_cache,
                        &deferred_pipeline,
                        key,
                        &mesh.layout,
                    ) {
                        item.pipeline = pipeline;
                    }
                }
            }
        }

        for item in alpha_mask_phase.items.iter_mut() {
            if let Ok((material_handle, mesh_handle)) = material_meshes.get(item.entity) {
                if let Some((_material, mesh)) = render_materials
                    .get(material_handle)
                    .zip(render_meshes.get(mesh_handle))
                {
                    let mut key = MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);
                    key |= MeshPipelineKey::from_msaa_samples(msaa.samples);

                    if let Ok(pipeline) = pipelines.specialize(
                        &mut pipeline_cache,
                        &deferred_pipeline,
                        key,
                        &mesh.layout,
                    ) {
                        item.pipeline = pipeline;
                    }
                }
            }
        }

        let draw_function = transparent_draw_functions
            .read()
            .get_id::<DrawMaterial<StandardMaterial>>()
            .unwrap();

        for item in transparent_phase.items.drain(..) {
            let Transparent3d {
                distance, entity, ..
            } = item;
            if let Ok((material_handle, mesh_handle)) = material_meshes.get(entity) {
                if let Some((_material, mesh)) = render_materials
                    .get(material_handle)
                    .zip(render_meshes.get(mesh_handle))
                {
                    let mut key = MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);
                    key |= MeshPipelineKey::from_msaa_samples(msaa.samples);
                    key |= MeshPipelineKey::TRANSPARENT_MAIN_PASS;

                    if let Ok(pipeline) = pipelines.specialize(
                        &mut pipeline_cache,
                        &deferred_pipeline,
                        key,
                        &mesh.layout,
                    ) {
                        opaque_phase.add(Opaque3d {
                            distance,
                            pipeline,
                            entity,
                            draw_function,
                        })
                    }
                }
            }
        }
    }
}

type DrawMaterial<M> = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMaterialBindGroup<M, 1>,
    SetMeshBindGroup<2>,
    DrawMesh,
);
