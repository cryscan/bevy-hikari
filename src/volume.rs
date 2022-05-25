use crate::{
    utils::{
        custom_camera::{extract_camera, prepare_lights},
        SimplePassDriver,
    },
    VOXEL_COUNT, VOXEL_RESOLUTION, VOXEL_SHADER_HANDLE,
};
use bevy::{
    core_pipeline::{node, AlphaMask3d, Opaque3d, Transparent3d},
    ecs::system::{lifetimeless::SRes, SystemParamItem},
    pbr::*,
    prelude::*,
    render::{
        camera::{CameraProjection, DepthCalculation, RenderTarget},
        mesh::MeshVertexBufferLayout,
        primitives::Frustum,
        render_asset::RenderAssets,
        render_graph::RenderGraph,
        render_phase::*,
        render_resource::{std140::AsStd140, std430::AsStd430, *},
        renderer::{RenderDevice, RenderQueue},
        texture::BevyDefault,
        view::{update_frusta, ExtractedView, VisibleEntities},
        RenderApp, RenderStage,
    },
    transform::TransformSystem,
};
use std::f32::consts::FRAC_PI_2;

pub struct VolumePlugin;
impl Plugin for VolumePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Volume>()
            .add_system(update_volume)
            .add_system_to_stage(
                CoreStage::PostUpdate,
                update_frusta::<VolumeProjection>.after(TransformSystem::TransformPropagate),
            );

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<VolumePipeline>()
            .init_resource::<SpecializedMeshPipelines<VolumePipeline>>()
            .init_resource::<VolumeMeta>()
            .add_render_command::<Transparent3d, DrawVoxelMesh>()
            .add_system_to_stage(RenderStage::Extract, extract_camera::<VolumeCamera>)
            .add_system_to_stage(RenderStage::Extract, extract_volume)
            .add_system_to_stage(RenderStage::Prepare, prepare_volume)
            .add_system_to_stage(
                RenderStage::Prepare,
                prepare_lights::<VolumeCamera>
                    .exclusive_system()
                    .after(RenderLightSystems::PrepareLights),
            )
            .add_system_to_stage(RenderStage::Queue, queue_volume_bind_groups)
            .add_system_to_stage(
                RenderStage::Queue,
                queue_voxel_meshes.after(queue_material_meshes::<StandardMaterial>),
            );

        use crate::node::VOXEL_PASS_DRIVER;
        use node::{CLEAR_PASS_DRIVER, MAIN_PASS_DEPENDENCIES, MAIN_PASS_DRIVER};

        let driver = SimplePassDriver::<VolumeCamera>::new(&mut render_app.world);
        let mut graph = render_app.world.resource_mut::<RenderGraph>();
        graph.add_node(VOXEL_PASS_DRIVER, driver);

        graph
            .add_node_edge(MAIN_PASS_DEPENDENCIES, VOXEL_PASS_DRIVER)
            .unwrap();
        graph
            .add_node_edge(CLEAR_PASS_DRIVER, VOXEL_PASS_DRIVER)
            .unwrap();
        graph
            .add_node_edge(MAIN_PASS_DRIVER, VOXEL_PASS_DRIVER)
            .unwrap();
    }
}

#[derive(Clone)]
pub struct Volume {
    pub enabled: bool,
    pub min: Vec3,
    pub max: Vec3,
    pub views: Option<[Entity; 3]>,
}

impl Volume {
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self {
            enabled: true,
            min,
            max,
            views: None,
        }
    }
}

impl Default for Volume {
    fn default() -> Self {
        Self {
            enabled: true,
            min: Vec3::new(-1.0, -1.0, -1.0),
            max: Vec3::new(1.0, 1.0, 1.0),
            views: None,
        }
    }
}

pub struct VolumeMeta {
    pub volume_uniform: UniformVec<GpuVolume>,
    pub directions_uniform: UniformVec<GpuDirections>,
    pub voxel_buffer: Buffer,
}

impl FromWorld for VolumeMeta {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let voxel_buffer = render_device.create_buffer(&BufferDescriptor {
            label: None,
            size: GpuVoxelBuffer::std430_size_static() as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        Self {
            volume_uniform: default(),
            directions_uniform: default(),
            voxel_buffer,
        }
    }
}

pub struct VolumeBindGroup(BindGroup);

#[derive(AsStd140)]
pub struct GpuVolume {
    pub min: Vec3,
    pub max: Vec3,
}

#[derive(AsStd430)]
pub struct GpuVoxelBuffer {
    data: [u32; VOXEL_COUNT],
}

#[derive(AsStd140)]
pub struct GpuDirections {
    pub data: [Vec3; 14],
}

impl Default for GpuDirections {
    fn default() -> Self {
        Self {
            data: [
                Vec3::new(1.0, 1.0, 1.0).normalize(),
                Vec3::new(-1.0, 1.0, 1.0).normalize(),
                Vec3::new(1.0, -1.0, 1.0).normalize(),
                Vec3::new(-1.0, -1.0, 1.0).normalize(),
                Vec3::new(1.0, 1.0, -1.0).normalize(),
                Vec3::new(-1.0, 1.0, -1.0).normalize(),
                Vec3::new(1.0, -1.0, -1.0).normalize(),
                Vec3::new(-1.0, -1.0, -1.0).normalize(),
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

#[derive(Component, Default)]
pub struct VolumeCamera;

/// Use custom projection to prevent [`camera_system`](bevy::render::camera::camera_system) from running
/// and updating the projection automatically.
#[derive(Component, Deref, DerefMut)]
pub struct VolumeProjection(pub OrthographicProjection);

impl CameraProjection for VolumeProjection {
    fn get_projection_matrix(&self) -> Mat4 {
        self.0.get_projection_matrix()
    }

    fn update(&mut self, width: f32, height: f32) {
        self.0.update(width, height);
    }

    fn depth_calculation(&self) -> DepthCalculation {
        self.0.depth_calculation()
    }

    fn far(&self) -> f32 {
        self.0.far()
    }
}

/// Setup cameras for the volume.
pub fn update_volume(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut volume: ResMut<Volume>,
) {
    if !volume.is_changed() {
        return;
    }

    let size = Extent3d {
        width: VOXEL_RESOLUTION as u32,
        height: VOXEL_RESOLUTION as u32,
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
            usage: TextureUsages::COPY_DST | TextureUsages::RENDER_ATTACHMENT,
        },
        ..default()
    };
    image.resize(size);
    let image_handle = images.add(image);

    let center = (volume.max + volume.min) / 2.0;
    let extent = (volume.max - volume.min) / 2.0;

    if let Some(views) = volume.views {
        for entity in views {
            commands.entity(entity).despawn();
        }
    }

    volume.views = Some(
        [
            Quat::IDENTITY,
            Quat::from_rotation_y(FRAC_PI_2),
            Quat::from_rotation_x(FRAC_PI_2),
        ]
        .map(|rotation| {
            let projection = OrthographicProjection {
                left: -extent.x,
                right: extent.x,
                bottom: -extent.y,
                top: extent.y,
                near: -extent.z,
                far: extent.z,
                ..default()
            };
            let camera = Camera {
                target: RenderTarget::Image(image_handle.clone()),
                projection_matrix: projection.get_projection_matrix(),
                near: -extent.z,
                far: extent.z,
                ..default()
            };
            let transform = Transform {
                translation: center,
                rotation,
                ..default()
            };
            commands
                .spawn_bundle((
                    camera,
                    VolumeProjection(projection),
                    transform,
                    VisibleEntities::default(),
                    Frustum::default(),
                    GlobalTransform::default(),
                    VolumeCamera::default(),
                ))
                .id()
        }),
    );
}

pub fn extract_volume(mut commands: Commands, volume: Res<Volume>) {
    if volume.is_added() || volume.is_changed() {
        commands.insert_resource(volume.clone());
    }
}

pub fn prepare_volume(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    volume: Res<Volume>,
    mut volume_meta: ResMut<VolumeMeta>,
) {
    if volume.is_added() || volume.is_changed() {
        volume_meta.volume_uniform.clear();
        volume_meta.volume_uniform.push(GpuVolume {
            min: volume.min,
            max: volume.max,
        });
        volume_meta
            .volume_uniform
            .write_buffer(&render_device, &render_queue);

        volume_meta.directions_uniform.clear();
        volume_meta.directions_uniform.push(default());
        volume_meta
            .directions_uniform
            .write_buffer(&render_device, &render_queue);
    }
}

pub struct VolumePipeline {
    pub material_layout: BindGroupLayout,
    pub volume_layout: BindGroupLayout,
    pub mesh_pipeline: MeshPipeline,
}

impl FromWorld for VolumePipeline {
    fn from_world(world: &mut World) -> Self {
        let mesh_pipeline = world.resource::<MeshPipeline>().clone();
        let render_device = world.resource::<RenderDevice>();

        let material_layout = StandardMaterial::bind_group_layout(render_device);

        let volume_layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(GpuVolume::std140_size_static() as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(
                            GpuVoxelBuffer::std430_size_static() as u64
                        ),
                    },
                    count: None,
                },
            ],
        });

        Self {
            material_layout,
            volume_layout,
            mesh_pipeline,
        }
    }
}

impl SpecializedMeshPipeline for VolumePipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayout,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let shader = VOXEL_SHADER_HANDLE.typed();

        let mut descriptor = self.mesh_pipeline.specialize(key, layout)?;
        descriptor.fragment.as_mut().unwrap().shader = shader;
        descriptor.layout = Some(vec![
            self.mesh_pipeline.view_layout.clone(),
            self.material_layout.clone(),
            self.mesh_pipeline.mesh_layout.clone(),
            self.volume_layout.clone(),
        ]);
        descriptor.primitive.cull_mode = None;

        Ok(descriptor)
    }
}

pub fn queue_volume_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    volume_meta: Res<VolumeMeta>,
    volume_pipeline: Res<VolumePipeline>,
) {
    let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &volume_pipeline.volume_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: volume_meta.volume_uniform.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 1,
                resource: volume_meta.voxel_buffer.as_entire_binding(),
            },
        ],
    });

    commands.insert_resource(VolumeBindGroup(bind_group));
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn queue_voxel_meshes(
    transparent_draw_functions: Res<DrawFunctions<Transparent3d>>,
    volume_pipeline: Res<VolumePipeline>,
    material_meshes: Query<(&Handle<StandardMaterial>, &Handle<Mesh>)>,
    render_meshes: Res<RenderAssets<Mesh>>,
    render_materials: Res<RenderAssets<StandardMaterial>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<VolumePipeline>>,
    mut pipeline_cache: ResMut<PipelineCache>,
    msaa: Res<Msaa>,
    volume: Res<Volume>,
    mut views: Query<
        (
            &mut RenderPhase<Opaque3d>,
            &mut RenderPhase<AlphaMask3d>,
            &mut RenderPhase<Transparent3d>,
        ),
        (With<ExtractedView>, With<VolumeCamera>),
    >,
) {
    let draw_function = transparent_draw_functions
        .read()
        .get_id::<DrawVoxelMesh>()
        .unwrap();

    for (mut opaque_phase, mut alpha_mask_phase, mut transparent_phase) in views.iter_mut() {
        transparent_phase.items.clear();

        if !volume.enabled {
            opaque_phase.items.clear();
            alpha_mask_phase.items.clear();
            continue;
        }

        let mut add_phase_item = |entity, distance| {
            if let Ok((material_handle, mesh_handle)) = material_meshes.get(entity) {
                if !render_materials.contains_key(material_handle) {
                    return;
                }

                if let Some(mesh) = render_meshes.get(mesh_handle) {
                    let mut key = MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);
                    key |= MeshPipelineKey::from_msaa_samples(msaa.samples);
                    key |= MeshPipelineKey::TRANSPARENT_MAIN_PASS;

                    if let Ok(pipeline) = pipelines.specialize(
                        &mut pipeline_cache,
                        &volume_pipeline,
                        key,
                        &mesh.layout,
                    ) {
                        transparent_phase.add(Transparent3d {
                            distance,
                            pipeline,
                            entity,
                            draw_function,
                        });
                    }
                }
            }
        };

        for item in opaque_phase.items.drain(..) {
            add_phase_item(item.entity, item.distance);
        }
        for item in alpha_mask_phase.items.drain(..) {
            add_phase_item(item.entity, item.distance);
        }
    }
}

type DrawVoxelMesh = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMaterialBindGroup<StandardMaterial, 1>,
    SetMeshBindGroup<2>,
    SetVolumeBindGroup<3>,
    DrawMesh,
);

pub struct SetVolumeBindGroup<const I: usize>;
impl<const I: usize> EntityRenderCommand for SetVolumeBindGroup<I> {
    type Param = SRes<VolumeBindGroup>;

    fn render<'w>(
        _view: Entity,
        _item: Entity,
        volume_bind_group: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(I, &volume_bind_group.into_inner().0, &[]);
        RenderCommandResult::Success
    }
}
