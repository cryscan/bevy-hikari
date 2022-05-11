use crate::{
    extract_cameras_manual, SimplePassDriver, VOXEL_COUNT, VOXEL_LAYER, VOXEL_MIPMAP_LEVEL_COUNT,
    VOXEL_SHADER_HANDLE, VOXEL_SIZE,
};
use bevy::{
    core_pipeline::node,
    ecs::system::{lifetimeless::SRes, SystemParamItem, SystemState},
    pbr::{
        MaterialPipeline, RenderLightSystems, StandardMaterialFlags, StandardMaterialUniformData,
        ViewShadowBindings,
    },
    prelude::*,
    reflect::TypeUuid,
    render::{
        camera::{ActiveCamera, Camera3d, RenderTarget},
        mesh::{skinning::SkinnedMesh, MeshVertexBufferLayout},
        render_asset::{PrepareAssetError, RenderAsset, RenderAssets},
        render_graph::RenderGraph,
        render_resource::{
            std140::{AsStd140, Std140},
            std430::AsStd430,
            *,
        },
        renderer::{RenderDevice, RenderQueue},
        texture::{BevyDefault, CachedTexture, TextureCache},
        view::RenderLayers,
        RenderApp, RenderStage,
    },
};
use std::f32::consts::FRAC_PI_2;

pub struct VolumePlugin;
impl Plugin for VolumePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Volume>()
            .add_plugin(MaterialPlugin::<VoxelMaterial>::default())
            .add_startup_system(setup_volume)
            .add_system(attach_voxel_mesh.exclusive_system().before_commands());

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<VolumeMeta>()
            .add_system_to_stage(RenderStage::Extract, extract_cameras_manual::<VolumeCamera>)
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
            min: Vec3::new(-5.0, -5.0, -5.0),
            max: Vec3::new(5.0, 5.0, 5.0),
            views: None,
        }
    }
}

pub struct VolumeMeta {
    pub volume_uniform: UniformVec<GpuVolume>,
    pub voxel_buffer: Buffer,
    pub voxel_texture: CachedTexture,
    pub anisotropic_textures: [CachedTexture; 6],
    pub sampler: Sampler,
}

impl FromWorld for VolumeMeta {
    fn from_world(world: &mut World) -> Self {
        let (render_device, mut texture_cache) =
            SystemState::<(Res<RenderDevice>, ResMut<TextureCache>)>::new(world).get_mut(world);

        let voxel_buffer = render_device.create_buffer(&BufferDescriptor {
            label: None,
            size: GpuVoxelBuffer::std430_size_static() as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let voxel_texture = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: VOXEL_SIZE as u32,
                    height: VOXEL_SIZE as u32,
                    depth_or_array_layers: VOXEL_SIZE as u32,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            },
        );

        let size = (VOXEL_SIZE >> 1) as u32;
        let anisotropic_textures = [(); 6].map(|_| {
            texture_cache.get(
                &render_device,
                TextureDescriptor {
                    label: None,
                    size: Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: size,
                    },
                    mip_level_count: VOXEL_MIPMAP_LEVEL_COUNT as u32,
                    sample_count: 1,
                    dimension: TextureDimension::D3,
                    format: TextureFormat::Rgba16Float,
                    usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                },
            )
        });

        let sampler = render_device.create_sampler(&SamplerDescriptor {
            label: None,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });

        Self {
            volume_uniform: Default::default(),
            voxel_buffer,
            voxel_texture,
            anisotropic_textures,
            sampler,
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
            normal_map_texture: _,
            flip_normal_map_y: _,
            occlusion_texture: _,
            double_sided: _,
            cull_mode: _,
            unlit,
            alpha_mode,
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
    let extent = volume.max - volume.min;

    let camera = Camera {
        target: RenderTarget::Image(image_handle),
        near: -extent.z / 2.0,
        far: extent.z / 2.0,
        ..default()
    };

    volume.views = Some(
        [
            Quat::IDENTITY,
            Quat::from_rotation_y(FRAC_PI_2),
            Quat::from_rotation_x(FRAC_PI_2),
        ]
        .map(|rotation| {
            let camera = camera.clone();
            let transform = Transform {
                translation: center,
                rotation,
                scale: extent / (VOXEL_SIZE as f32),
            };
            commands
                .spawn_bundle(OrthographicCameraBundle {
                    camera,
                    orthographic_projection: default(),
                    visible_entities: default(),
                    frustum: default(),
                    transform,
                    global_transform: default(),
                    marker: VolumeCamera,
                })
                .insert(RenderLayers::layer(VOXEL_LAYER))
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
                *volume_bindings = ViewShadowBindings {
                    point_light_depth_texture: point_light_depth_texture.clone(),
                    point_light_depth_texture_view: point_light_depth_texture_view.clone(),
                    directional_light_depth_texture: directional_light_depth_texture.clone(),
                    directional_light_depth_texture_view: directional_light_depth_texture_view
                        .clone(),
                };
            }
        }
    }
}

/// Attach any standard material mesh with a voxel material copy.
pub fn attach_voxel_mesh(
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
            .insert(RenderLayers::layer(VOXEL_LAYER))
            .id();
        if let Some(skinned_mesh) = maybe_skinned_mesh {
            commands.entity(child).insert(skinned_mesh.clone());
        }

        commands.entity(entity).add_child(child);
    }
}
