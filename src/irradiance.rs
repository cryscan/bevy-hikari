use crate::{
    mipmap::MipmapMeta,
    utils::{
        custom_camera::{extract_phases, update_transform},
        SimplePassDriver,
    },
    volume::{GpuDirections, GpuVolume, GpuVoxelBuffer, Volume, VolumeMeta},
    NotGiReceiver, IRRADIANCE_SHADER_HANDLE,
};
use bevy::{
    core_pipeline::{node, AlphaMask3d, Opaque3d, RenderTargetClearColors, Transparent3d},
    ecs::system::lifetimeless::SRes,
    pbr::*,
    prelude::*,
    render::{
        camera::{CameraTypePlugin, ExtractedCamera, RenderTarget},
        mesh::MeshVertexBufferLayout,
        render_asset::RenderAssets,
        render_graph::RenderGraph,
        render_phase::{
            AddRenderCommand, DrawFunctions, EntityRenderCommand, RenderCommandResult, RenderPhase,
            SetItemPipeline,
        },
        render_resource::{std140::AsStd140, std430::AsStd430, *},
        renderer::RenderDevice,
        texture::TextureCache,
        view::{ExtractedView, ViewTarget},
        RenderApp, RenderStage,
    },
    transform::TransformSystem,
};
use itertools::Itertools;

pub struct IrradiancePlugin;
impl Plugin for IrradiancePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(CameraTypePlugin::<IrradianceCamera>::default())
            .add_startup_system(setup_irradiance_camera)
            .add_system_to_stage(
                CoreStage::PostUpdate,
                update_transform::<IrradianceCamera>.before(TransformSystem::TransformPropagate),
            );

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<IrradiancePipeline>()
            .init_resource::<SpecializedMeshPipelines<IrradiancePipeline>>()
            .add_render_command::<Opaque3d, DrawIrradiance>()
            .add_render_command::<AlphaMask3d, DrawIrradiance>()
            .add_render_command::<Transparent3d, DrawIrradiance>()
            .add_system_to_stage(RenderStage::Extract, extract_phases::<IrradianceCamera>)
            .add_system_to_stage(
                RenderStage::Prepare,
                prepare_irradiance_view_target.exclusive_system().at_end(),
            )
            .add_system_to_stage(RenderStage::Queue, queue_irradiance_bind_groups)
            .add_system_to_stage(
                RenderStage::Queue,
                queue_irradiance_meshes.after(queue_material_meshes::<StandardMaterial>),
            );

        use crate::node::{IRRADIANCE_PASS_DRIVER, MIPMAP_PASS};
        use node::{CLEAR_PASS_DRIVER, MAIN_PASS_DEPENDENCIES};

        let driver = SimplePassDriver::<IrradianceCamera>::new(&mut render_app.world);
        let mut graph = render_app.world.resource_mut::<RenderGraph>();
        graph.add_node(IRRADIANCE_PASS_DRIVER, driver);

        graph
            .add_node_edge(CLEAR_PASS_DRIVER, IRRADIANCE_PASS_DRIVER)
            .unwrap();
        graph
            .add_node_edge(MAIN_PASS_DEPENDENCIES, IRRADIANCE_PASS_DRIVER)
            .unwrap();
        graph
            .add_node_edge(MIPMAP_PASS, IRRADIANCE_PASS_DRIVER)
            .unwrap();
    }
}

const IRRADIANCE_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

#[derive(Default, Component)]
pub struct IrradianceCamera;

pub fn setup_irradiance_camera(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut clear_colors: ResMut<RenderTargetClearColors>,
    windows: Res<Windows>,
) {
    let size = Extent3d {
        width: windows.primary().width() as u32 >> 1,
        height: windows.primary().height() as u32 >> 1,
        ..default()
    };
    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: IRRADIANCE_TEXTURE_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_DST,
        },
        ..default()
    };
    image.resize(size);
    let image_handle = images.add(image);

    let target = RenderTarget::Image(image_handle);
    clear_colors.insert(target.clone(), Color::NONE);
    let camera = Camera {
        target,
        ..default()
    };

    commands.spawn_bundle(PerspectiveCameraBundle::<IrradianceCamera> {
        camera,
        perspective_projection: default(),
        visible_entities: default(),
        frustum: default(),
        transform: default(),
        global_transform: default(),
        marker: default(),
    });
}

pub struct IrradianceBindGroup(BindGroup);

bitflags::bitflags! {
    #[repr(transparent)]
    pub struct IrradiancePipelineKey: u32 {
        const NONE = 0;
        const NOT_GI_RECEIVER = (1 << 0);
    }
}

pub struct IrradiancePipeline {
    pub material_layout: BindGroupLayout,
    pub irradiance_layout: BindGroupLayout,
    pub mesh_pipeline: MeshPipeline,
}

impl FromWorld for IrradiancePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let mesh_pipeline = world.resource::<MeshPipeline>().clone();

        let material_layout = StandardMaterial::bind_group_layout(render_device);

        let irradiance_layout_entries = (0u32..6)
            .map(|direction| BindGroupLayoutEntry {
                binding: direction,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            })
            .chain([
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 7,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(
                            GpuVoxelBuffer::std430_size_static() as u64
                        ),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 8,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(
                            GpuDirections::std140_size_static() as u64
                        ),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 9,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(GpuVolume::std140_size_static() as u64),
                    },
                    count: None,
                },
            ])
            .collect_vec();
        let irradiance_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &irradiance_layout_entries,
            });

        Self {
            material_layout,
            irradiance_layout,
            mesh_pipeline,
        }
    }
}

impl SpecializedMeshPipeline for IrradiancePipeline {
    type Key = (MeshPipelineKey, IrradiancePipelineKey);

    fn specialize(
        &self,
        (mesh_key, irradiance_key): Self::Key,
        layout: &MeshVertexBufferLayout,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.mesh_pipeline.specialize(mesh_key, layout)?;
        descriptor.layout = Some(vec![
            self.mesh_pipeline.view_layout.clone(),
            self.material_layout.clone(),
            self.mesh_pipeline.mesh_layout.clone(),
            self.irradiance_layout.clone(),
        ]);

        let shader = IRRADIANCE_SHADER_HANDLE.typed();
        let fragment = descriptor.fragment.as_mut().unwrap();
        fragment.shader = shader;
        if mesh_key.contains(MeshPipelineKey::TRANSPARENT_MAIN_PASS)
            || irradiance_key.contains(IrradiancePipelineKey::NOT_GI_RECEIVER)
        {
            fragment.shader_defs.push("NOT_GI_RECEIVER".into());
        }

        let mut target = &mut fragment.targets[0];
        target.format = IRRADIANCE_TEXTURE_FORMAT;

        Ok(descriptor)
    }
}

pub fn prepare_irradiance_view_target(
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    msaa: Res<Msaa>,
    mut cameras: Query<(&ExtractedCamera, &mut ViewTarget), With<IrradianceCamera>>,
) {
    if msaa.samples == 1 {
        return;
    }

    for (camera, mut view_target) in cameras.iter_mut() {
        if let Some(size) = camera.physical_size {
            let sampled_texture = texture_cache.get(
                &render_device,
                TextureDescriptor {
                    label: None,
                    size: Extent3d {
                        width: size.x,
                        height: size.y,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: msaa.samples,
                    dimension: TextureDimension::D2,
                    format: IRRADIANCE_TEXTURE_FORMAT,
                    usage: TextureUsages::RENDER_ATTACHMENT,
                },
            );
            view_target.sampled_target = Some(sampled_texture.default_view);
        }
    }
}

pub fn queue_irradiance_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    volume_meta: Res<VolumeMeta>,
    mipmap_meta: Res<MipmapMeta>,
    irradiance_pipeline: Res<IrradiancePipeline>,
) {
    let anisotropic_texture_views = mipmap_meta
        .anisotropic_textures
        .iter()
        .map(|texture| texture.create_view(&TextureViewDescriptor::default()))
        .collect_vec();
    let bind_group_entries = anisotropic_texture_views
        .iter()
        .enumerate()
        .map(|(index, texture_view)| BindGroupEntry {
            binding: index as u32,
            resource: BindingResource::TextureView(texture_view),
        })
        .chain([
            BindGroupEntry {
                binding: 6,
                resource: BindingResource::Sampler(&mipmap_meta.sampler),
            },
            BindGroupEntry {
                binding: 7,
                resource: volume_meta.voxel_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 8,
                resource: volume_meta.directions_uniform.binding().unwrap(),
            },
            BindGroupEntry {
                binding: 9,
                resource: volume_meta.volume_uniform.binding().unwrap(),
            },
        ])
        .collect_vec();

    let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &irradiance_pipeline.irradiance_layout,
        entries: &bind_group_entries,
    });
    commands.insert_resource(IrradianceBindGroup(bind_group));
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn queue_irradiance_meshes(
    opaque_draw_functions: Res<DrawFunctions<Opaque3d>>,
    alpha_mask_draw_functions: Res<DrawFunctions<AlphaMask3d>>,
    transparent_draw_functions: Res<DrawFunctions<Transparent3d>>,
    irradiance_pipeline: Res<IrradiancePipeline>,
    material_meshes: Query<(&Handle<StandardMaterial>, &Handle<Mesh>)>,
    render_meshes: Res<RenderAssets<Mesh>>,
    render_materials: Res<RenderAssets<StandardMaterial>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<IrradiancePipeline>>,
    mut pipeline_cache: ResMut<PipelineCache>,
    msaa: Res<Msaa>,
    volume: Res<Volume>,
    mut views: Query<
        (
            &mut RenderPhase<Opaque3d>,
            &mut RenderPhase<AlphaMask3d>,
            &mut RenderPhase<Transparent3d>,
        ),
        (With<ExtractedView>, With<IrradianceCamera>),
    >,
    query: Query<(), (With<NotGiReceiver>, Without<IrradianceCamera>)>,
) {
    for (mut opaque_phase, mut alpha_mask_phase, mut transparent_phase) in views.iter_mut() {
        if !volume.enabled {
            opaque_phase.items.clear();
            alpha_mask_phase.items.clear();
            transparent_phase.items.clear();
            continue;
        }

        let draw_opaque = opaque_draw_functions
            .read()
            .get_id::<DrawIrradiance>()
            .unwrap();
        let draw_alpha_mask = alpha_mask_draw_functions
            .read()
            .get_id::<DrawIrradiance>()
            .unwrap();
        let draw_transparent = transparent_draw_functions
            .read()
            .get_id::<DrawIrradiance>()
            .unwrap();

        for item in opaque_phase.items.iter_mut() {
            if let Ok((material_handle, mesh_handle)) = material_meshes.get(item.entity) {
                if let Some((_material, mesh)) = render_materials
                    .get(material_handle)
                    .zip(render_meshes.get(mesh_handle))
                {
                    let mut mesh_key =
                        MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);
                    mesh_key |= MeshPipelineKey::from_msaa_samples(msaa.samples);

                    let mut irradiance_key = IrradiancePipelineKey::NONE;
                    if query.contains(item.entity) {
                        irradiance_key |= IrradiancePipelineKey::NOT_GI_RECEIVER;
                    }

                    if let Ok(pipeline) = pipelines.specialize(
                        &mut pipeline_cache,
                        &irradiance_pipeline,
                        (mesh_key, irradiance_key),
                        &mesh.layout,
                    ) {
                        item.pipeline = pipeline;
                        item.draw_function = draw_opaque;
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
                    let mut mesh_key =
                        MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);
                    mesh_key |= MeshPipelineKey::from_msaa_samples(msaa.samples);

                    let mut irradiance_key = IrradiancePipelineKey::NONE;
                    if query.contains(item.entity) {
                        irradiance_key |= IrradiancePipelineKey::NOT_GI_RECEIVER;
                    }

                    if let Ok(pipeline) = pipelines.specialize(
                        &mut pipeline_cache,
                        &irradiance_pipeline,
                        (mesh_key, irradiance_key),
                        &mesh.layout,
                    ) {
                        item.pipeline = pipeline;
                        item.draw_function = draw_alpha_mask;
                    }
                }
            }
        }

        for item in transparent_phase.items.iter_mut() {
            if let Ok((material_handle, mesh_handle)) = material_meshes.get(item.entity) {
                if let Some((_material, mesh)) = render_materials
                    .get(material_handle)
                    .zip(render_meshes.get(mesh_handle))
                {
                    let mut mesh_key =
                        MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);
                    mesh_key |= MeshPipelineKey::from_msaa_samples(msaa.samples);
                    mesh_key |= MeshPipelineKey::TRANSPARENT_MAIN_PASS;

                    let mut irradiance_key = IrradiancePipelineKey::NONE;
                    if query.contains(item.entity) {
                        irradiance_key |= IrradiancePipelineKey::NOT_GI_RECEIVER;
                    }

                    if let Ok(pipeline) = pipelines.specialize(
                        &mut pipeline_cache,
                        &irradiance_pipeline,
                        (mesh_key, irradiance_key),
                        &mesh.layout,
                    ) {
                        item.pipeline = pipeline;
                        item.draw_function = draw_transparent;
                    }
                }
            }
        }
    }
}

type DrawIrradiance = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMaterialBindGroup<StandardMaterial, 1>,
    SetMeshBindGroup<2>,
    SetIrradianceBindGroup<3>,
    DrawMesh,
);

pub struct SetIrradianceBindGroup<const I: usize>;
impl<const I: usize> EntityRenderCommand for SetIrradianceBindGroup<I> {
    type Param = SRes<IrradianceBindGroup>;

    fn render<'w>(
        _view: Entity,
        _item: Entity,
        irradiance_bind_group: bevy::ecs::system::SystemParamItem<'w, '_, Self::Param>,
        pass: &mut bevy::render::render_phase::TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(I, &irradiance_bind_group.into_inner().0, &[]);
        RenderCommandResult::Success
    }
}
