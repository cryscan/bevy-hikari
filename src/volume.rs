use crate::{
    extract_custom_cameras, GiRenderLayers, SimplePassDriver, VOXEL_COUNT, VOXEL_SHADER_HANDLE,
    VOXEL_SIZE,
};
use bevy::{
    core_pipeline::node,
    ecs::system::{lifetimeless::SRes, SystemParamItem},
    pbr::{
        MaterialPipeline, RenderLightSystems, StandardMaterialFlags, StandardMaterialUniformData,
        ViewShadowBindings,
    },
    prelude::*,
    reflect::TypeUuid,
    render::{
        camera::{ActiveCamera, Camera3d, CameraProjection, DepthCalculation, RenderTarget},
        mesh::{skinning::SkinnedMesh, MeshVertexBufferLayout},
        primitives::Frustum,
        render_asset::{PrepareAssetError, RenderAsset, RenderAssets},
        render_graph::RenderGraph,
        render_resource::{
            std140::{AsStd140, Std140},
            std430::AsStd430,
            *,
        },
        renderer::{RenderDevice, RenderQueue},
        texture::BevyDefault,
        view::{update_frusta, RenderLayers, VisibleEntities},
        RenderApp, RenderStage,
    },
    transform::TransformSystem,
};
use std::f32::consts::FRAC_PI_2;

pub struct VolumePlugin;
impl Plugin for VolumePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Volume>()
            .add_plugin(MaterialPlugin::<VoxelMaterial>::default())
            .add_startup_system(setup_volume)
            .add_system(create_voxel_mesh.exclusive_system().before_commands())
            .add_system_to_stage(
                CoreStage::PostUpdate,
                update_frusta::<VolumeProjection>.after(TransformSystem::TransformPropagate),
            );

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<VolumeMeta>()
            .add_system_to_stage(RenderStage::Extract, extract_custom_cameras::<VolumeCamera>)
            .add_system_to_stage(RenderStage::Extract, extract_volume)
            .add_system_to_stage(RenderStage::Prepare, prepare_volume)
            .add_system_to_stage(
                RenderStage::Prepare,
                prepare_volume_lights
                    .exclusive_system()
                    .after(RenderLightSystems::PrepareLights),
            );

        render_app
            .world
            .resource_scope(|world, mut graph: Mut<RenderGraph>| {
                use crate::node::VOXEL_PASS_DRIVER;
                use node::{CLEAR_PASS_DRIVER, MAIN_PASS_DEPENDENCIES, MAIN_PASS_DRIVER};

                let driver = SimplePassDriver::<VolumeCamera>::new(world);
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
            });
    }
}

#[derive(Clone)]
pub struct Volume {
    pub enabled: bool,
    pub min: Vec3,
    pub max: Vec3,
    pub views: Option<[Entity; 3]>,
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
            volume_uniform: Default::default(),
            voxel_buffer,
        }
    }
}

#[derive(AsStd140)]
pub struct GpuVolume {
    pub min: Vec3,
    pub max: Vec3,
}

#[derive(AsStd430)]
pub struct GpuVoxelBuffer {
    data: [u32; VOXEL_COUNT],
}

#[derive(Component, Default)]
pub struct VolumeCamera;

#[derive(Default, Clone, TypeUuid)]
#[uuid = "e0c8e218-4a3e-4113-a231-fe39e993f6f5"]
pub struct VoxelMaterial {
    pub base_color: Color,
    pub base_color_texture: Option<Handle<Image>>,
    pub emissive: Color,
    pub emissive_texture: Option<Handle<Image>>,
    pub perceptual_roughness: f32,
    pub metallic: f32,
    pub metallic_roughness_texture: Option<Handle<Image>>,
    pub reflectance: f32,
    pub unlit: bool,
    pub alpha_mode: AlphaMode,
}

impl From<StandardMaterial> for VoxelMaterial {
    fn from(material: StandardMaterial) -> Self {
        let StandardMaterial {
            base_color,
            base_color_texture,
            emissive,
            emissive_texture,
            perceptual_roughness,
            metallic,
            metallic_roughness_texture,
            reflectance,
            unlit,
            alpha_mode,
            ..
        } = material;

        Self {
            base_color,
            base_color_texture,
            emissive,
            emissive_texture,
            perceptual_roughness,
            metallic,
            metallic_roughness_texture,
            reflectance,
            unlit,
            alpha_mode,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuVoxelMaterial {
    pub buffer: Buffer,
    pub bind_group: BindGroup,
}

impl RenderAsset for VoxelMaterial {
    type ExtractedAsset = VoxelMaterial;
    type PreparedAsset = GpuVoxelMaterial;
    type Param = (
        SRes<RenderDevice>,
        SRes<MaterialPipeline<VoxelMaterial>>,
        SRes<RenderAssets<Image>>,
        SRes<VolumeMeta>,
    );

    fn extract_asset(&self) -> Self::ExtractedAsset {
        self.clone()
    }

    fn prepare_asset(
        material: Self::ExtractedAsset,
        (render_device, pipeline, images, volume_meta): &mut SystemParamItem<Self::Param>,
    ) -> Result<Self::PreparedAsset, PrepareAssetError<Self::ExtractedAsset>> {
        let (base_color_texture_view, base_color_sampler) = match pipeline
            .mesh_pipeline
            .get_image_texture(images, &material.base_color_texture)
        {
            Some(result) => result,
            None => return Err(PrepareAssetError::RetryNextUpdate(material)),
        };

        let (emissive_texture_view, emissive_sampler) = match pipeline
            .mesh_pipeline
            .get_image_texture(images, &material.emissive_texture)
        {
            Some(result) => result,
            None => return Err(PrepareAssetError::RetryNextUpdate(material)),
        };

        let (metallic_roughness_texture_view, metallic_roughness_sampler) = match pipeline
            .mesh_pipeline
            .get_image_texture(images, &material.metallic_roughness_texture)
        {
            Some(result) => result,
            None => return Err(PrepareAssetError::RetryNextUpdate(material)),
        };

        let mut flags = StandardMaterialFlags::NONE;
        if material.base_color_texture.is_some() {
            flags |= StandardMaterialFlags::BASE_COLOR_TEXTURE;
        }
        if material.emissive_texture.is_some() {
            flags |= StandardMaterialFlags::EMISSIVE_TEXTURE;
        }
        if material.metallic_roughness_texture.is_some() {
            flags |= StandardMaterialFlags::METALLIC_ROUGHNESS_TEXTURE;
        }
        if material.unlit {
            flags |= StandardMaterialFlags::UNLIT;
        }

        let mut alpha_cutoff = 0.5;
        match material.alpha_mode {
            AlphaMode::Opaque => flags |= StandardMaterialFlags::ALPHA_MODE_OPAQUE,
            AlphaMode::Mask(cutoff) => {
                alpha_cutoff = cutoff;
                flags |= StandardMaterialFlags::ALPHA_MODE_MASK;
            }
            AlphaMode::Blend => flags |= StandardMaterialFlags::ALPHA_MODE_BLEND,
        };

        let volume_binding = match volume_meta.volume_uniform.binding() {
            Some(result) => result,
            None => return Err(PrepareAssetError::RetryNextUpdate(material)),
        };

        let value = StandardMaterialUniformData {
            base_color: material.base_color.as_linear_rgba_f32().into(),
            emissive: material.emissive.into(),
            roughness: material.perceptual_roughness,
            metallic: material.metallic,
            reflectance: material.reflectance,
            flags: flags.bits(),
            alpha_cutoff,
        };

        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: None,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            contents: value.as_std140().as_bytes(),
        });
        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.material_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(base_color_texture_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(base_color_sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(emissive_texture_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Sampler(emissive_sampler),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(metallic_roughness_texture_view),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::Sampler(metallic_roughness_sampler),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: volume_binding,
                },
                BindGroupEntry {
                    binding: 8,
                    resource: volume_meta.voxel_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(GpuVoxelMaterial { buffer, bind_group })
    }
}

impl Material for VoxelMaterial {
    fn fragment_shader(_asset_server: &AssetServer) -> Option<Handle<Shader>> {
        Some(VOXEL_SHADER_HANDLE.typed::<Shader>())
    }

    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayout,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.depth_stencil = Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: false,
            depth_compare: CompareFunction::Always,
            stencil: default(),
            bias: default(),
        });
        descriptor.primitive.cull_mode = None;
        Ok(())
    }

    fn alpha_mode(_material: &<Self as RenderAsset>::PreparedAsset) -> AlphaMode {
        AlphaMode::Blend
    }

    fn bind_group(material: &<Self as RenderAsset>::PreparedAsset) -> &BindGroup {
        &material.bind_group
    }

    fn bind_group_layout(render_device: &RenderDevice) -> BindGroupLayout {
        render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(
                            StandardMaterialUniformData::std140_size_static() as u64,
                        ),
                    },
                    count: None,
                },
                // Base Color Texture
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Base Color Texture Sampler
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                // Emissive Texture
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Emissive Texture Sampler
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                // Metallic Roughness Texture
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Metallic Roughness Texture Sampler
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                // Volume Uniform
                BindGroupLayoutEntry {
                    binding: 7,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(GpuVolume::std140_size_static() as u64),
                    },
                    count: None,
                },
                // Voxel Storage Buffer
                BindGroupLayoutEntry {
                    binding: 8,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
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
        })
    }
}

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
pub fn setup_volume(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut volume: ResMut<Volume>,
) {
    let size = Extent3d {
        width: VOXEL_SIZE as u32,
        height: VOXEL_SIZE as u32,
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
                    RenderLayers::from(GiRenderLayers::Voxel),
                ))
                .id()
        }),
    );
}

pub fn extract_volume(mut commands: Commands, volume: Res<Volume>) {
    commands.insert_resource(volume.clone());
}

pub fn prepare_volume(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    volume: Res<Volume>,
    mut volume_meta: ResMut<VolumeMeta>,
) {
    volume_meta.volume_uniform.clear();
    volume_meta.volume_uniform.push(GpuVolume {
        min: volume.min,
        max: volume.max,
    });
    volume_meta
        .volume_uniform
        .write_buffer(&render_device, &render_queue);
}

/// Hijack main camera's [`ViewShadowBindings`](bevy::pbr::ViewShadowBindings).
pub fn prepare_volume_lights(
    active: Res<ActiveCamera<Camera3d>>,
    main_camera_query: Query<&ViewShadowBindings, Without<VolumeCamera>>,
    mut volume_cameras: Query<&mut ViewShadowBindings, With<VolumeCamera>>,
) {
    if let Some(main_camera) = active.get() {
        if let Ok(ViewShadowBindings {
            point_light_depth_texture,
            point_light_depth_texture_view,
            directional_light_depth_texture,
            directional_light_depth_texture_view,
        }) = main_camera_query.get(main_camera)
        {
            for mut volume_bindings in volume_cameras.iter_mut() {
                let (
                    point_light_depth_texture,
                    point_light_depth_texture_view,
                    directional_light_depth_texture,
                    directional_light_depth_texture_view,
                ) = (
                    point_light_depth_texture.clone(),
                    point_light_depth_texture_view.clone(),
                    directional_light_depth_texture.clone(),
                    directional_light_depth_texture_view.clone(),
                );

                *volume_bindings = ViewShadowBindings {
                    point_light_depth_texture,
                    point_light_depth_texture_view,
                    directional_light_depth_texture,
                    directional_light_depth_texture_view,
                };
            }
        }
    }
}

/// Attach any standard material mesh with a voxel material copy.
pub fn create_voxel_mesh(
    mut commands: Commands,
    mesh_query: Query<
        (
            Entity,
            &Handle<StandardMaterial>,
            &Handle<Mesh>,
            Option<&SkinnedMesh>,
        ),
        Added<Handle<StandardMaterial>>,
    >,
    standard_materials: Res<Assets<StandardMaterial>>,
    mut voxel_materials: ResMut<Assets<VoxelMaterial>>,
) {
    for (entity, standard_material, mesh, maybe_skinned_mesh) in mesh_query.iter() {
        let standard_material = match standard_materials.get(standard_material) {
            Some(material) => material.clone(),
            None => continue,
        };
        let material = voxel_materials.add(standard_material.into());

        let child = commands
            .spawn_bundle(MaterialMeshBundle {
                mesh: mesh.clone(),
                material,
                ..Default::default()
            })
            .insert(RenderLayers::from(GiRenderLayers::Voxel))
            .id();
        if let Some(skinned_mesh) = maybe_skinned_mesh {
            commands.entity(child).insert(skinned_mesh.clone());
        }

        commands.entity(entity).add_child(child);
    }
}