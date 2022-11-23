use self::{
    instance::InstancePlugin,
    material::{MaterialPlugin, MaterialTextures},
    mesh::MeshPlugin,
};
use bevy::{
    ecs::system::{lifetimeless::SRes, SystemParamItem},
    pbr::MeshPipeline,
    prelude::*,
    render::{
        mesh::VertexAttributeValues,
        render_asset::RenderAssets,
        render_phase::{EntityRenderCommand, RenderCommandResult, TrackedRenderPass},
        render_resource::*,
        renderer::RenderDevice,
        RenderApp, RenderStage,
    },
};
use bvh::{
    aabb::{Bounded, AABB},
    bounding_hierarchy::BHShape,
    bvh::BVH,
};
use itertools::Itertools;
use std::num::NonZeroU32;

pub mod instance;
pub mod material;
pub mod mesh;

pub use instance::{
    DynamicInstanceIndex, GenericInstancePlugin, InstanceIndex, InstanceRenderAssets,
    PreviousMeshUniform,
};
pub use material::{GenericMaterialPlugin, MaterialRenderAssets};
pub use mesh::MeshRenderAssets;

pub struct MeshMaterialPlugin;
impl Plugin for MeshMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(MeshPlugin)
            .add_plugin(MaterialPlugin)
            .add_plugin(InstancePlugin)
            .add_plugin(GenericMaterialPlugin::<StandardMaterial>::default())
            .add_plugin(GenericInstancePlugin::<StandardMaterial>::default());

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<MeshMaterialBindGroupLayout>()
                .init_resource::<TextureBindGroupLayout>()
                .add_system_to_stage(
                    RenderStage::Prepare,
                    prepare_texture_bind_group_layout.label(MeshMaterialSystems::PrepareAssets),
                )
                .add_system_to_stage(RenderStage::Queue, queue_mesh_material_bind_group);
        }
    }
}

#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct GpuVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct GpuPrimitive {
    /// Global positions of vertices.
    pub vertices: [Vec3; 3],
    /// Indices of vertices in the vertex buffer (offset not applied).
    pub indices: [u32; 3],
    /// Index of the node in the node buffer (offset not applied).
    node_index: u32,
}

impl Bounded for GpuPrimitive {
    fn aabb(&self) -> AABB {
        AABB::empty()
            .grow(&self.vertices[0].to_array().into())
            .grow(&self.vertices[1].to_array().into())
            .grow(&self.vertices[2].to_array().into())
    }
}

impl BHShape for GpuPrimitive {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index as u32;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index as usize
    }
}

#[derive(Debug, Default, Clone, ShaderType)]
pub struct GpuInstance {
    pub min: Vec3,
    pub material: u32,
    pub max: Vec3,
    node_index: u32,
    pub transform: Mat4,
    pub inverse_transpose_model: Mat4,
    pub mesh: GpuMeshIndex,
}

impl Bounded for GpuInstance {
    fn aabb(&self) -> AABB {
        AABB {
            min: self.min.to_array().into(),
            max: self.max.to_array().into(),
        }
    }
}

impl BHShape for GpuInstance {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index as u32;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index as usize
    }
}

#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct GpuNode {
    pub min: Vec3,
    pub entry_index: u32,
    pub max: Vec3,
    pub exit_index: u32,
}

impl GpuNode {
    fn pack(aabb: &AABB, entry_index: u32, exit_index: u32, primitive_index: u32) -> Self {
        let entry_index = if entry_index == u32::MAX {
            primitive_index | 0x80000000
        } else {
            entry_index
        };
        let min = aabb.min.to_array().into();
        let max = aabb.max.to_array().into();
        Self {
            min,
            entry_index,
            max,
            exit_index,
        }
    }
}

#[derive(Debug, Default, Clone, ShaderType)]
pub struct GpuStandardMaterial {
    pub base_color: Vec4,
    pub base_color_texture: u32,

    pub emissive: Vec4,
    pub emissive_texture: u32,

    pub perceptual_roughness: f32,
    pub metallic: f32,
    pub metallic_roughness_texture: u32,
    pub reflectance: f32,

    pub normal_map_texture: u32,
    pub occlusion_texture: u32,
}

#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct GpuAliasEntry {
    /// The probability of choosing the other one in the bucket.
    pub prob: f32,
    /// The index of the other one in the bucket.
    pub index: u32,
}

#[derive(Debug, Default, Clone, ShaderType)]
pub struct GpuEmissive {
    pub emissive: Vec4,
    pub position: Vec3,
    pub radius: f32,
    pub instance: u32,
    pub alias_table: UVec2,
    pub surface_area: f32,
    node_index: u32,
}

impl Bounded for GpuEmissive {
    fn aabb(&self) -> AABB {
        AABB {
            min: self.position - self.radius,
            max: self.position + self.radius,
        }
    }
}

impl BHShape for GpuEmissive {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index as u32;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index as usize
    }
}

#[derive(Default, ShaderType)]
pub struct GpuVertexBuffer {
    #[size(runtime)]
    pub data: Vec<GpuVertex>,
}

#[derive(Default, ShaderType)]
pub struct GpuPrimitiveBuffer {
    #[size(runtime)]
    pub data: Vec<GpuPrimitive>,
}

#[derive(Default, ShaderType)]
pub struct GpuNodeBuffer {
    pub count: u32,
    #[size(runtime)]
    pub data: Vec<GpuNode>,
}

#[derive(Default, ShaderType)]
pub struct GpuInstanceBuffer {
    #[size(runtime)]
    pub data: Vec<GpuInstance>,
}

#[derive(Default, ShaderType)]
pub struct GpuStandardMaterialBuffer {
    #[size(runtime)]
    pub data: Vec<GpuStandardMaterial>,
}

#[derive(Default, ShaderType)]
pub struct GpuAliasTableBuffer {
    #[size(runtime)]
    pub data: Vec<GpuAliasEntry>,
}

#[derive(Default, ShaderType)]
pub struct GpuEmissiveBuffer {
    #[size(runtime)]
    pub data: Vec<GpuEmissive>,
}

#[derive(Debug)]
pub enum PrepareMeshError {
    MissingAttributePosition,
    MissingAttributeNormal,
    MissingAttributeUV,
    IncompatiblePrimitiveTopology,
    NoPrimitive,
}

#[derive(Default, Clone)]
pub struct GpuMesh {
    pub vertices: Vec<GpuVertex>,
    pub primitives: Vec<GpuPrimitive>,
    pub nodes: Vec<GpuNode>,
}

impl GpuMesh {
    pub fn transformed_primitive_areas(&self, transform: Mat4) -> Vec<f32> {
        self.primitives
            .iter()
            .map(|primitive| {
                let [v0, v1, v2] = [0, 1, 2]
                    .map(|id| self.vertices[primitive.indices[id] as usize])
                    .map(|v| transform.transform_point3(v.position));
                0.5 * (v1 - v0).cross(v2 - v0).length().abs()
            })
            .collect()
    }

    pub fn build_alias_table(&self, transform: Mat4) -> Vec<GpuAliasEntry> {
        let primitive_count = self.primitives.len();
        let areas = self.transformed_primitive_areas(transform);
        let surface_area: f32 = areas.iter().sum();

        if primitive_count == 0 {
            vec![]
        } else {
            let mean_area = surface_area / (primitive_count as f32);
            let probabilities = areas
                .iter()
                .enumerate()
                .map(|(id, area)| (id, area / mean_area));
            let mut over: Vec<_> = probabilities.clone().filter(|prob| prob.1 > 1.0).collect();
            let mut under: Vec<_> = probabilities.filter(|prob| prob.1 < 1.0).collect();

            let mut alias_table: Vec<_> = (0..primitive_count)
                .map(|id| GpuAliasEntry {
                    prob: 0.0,
                    index: id as u32,
                })
                .collect();

            while !under.is_empty() && !over.is_empty() {
                let mut over_bucket = over.pop().unwrap();
                let under_bucket = under.pop().unwrap();

                // Pour some part of `over_bucket` into `under_bucket` to equalize the later.
                let delta = 1.0 - under_bucket.1;
                over_bucket.1 -= delta;
                assert!(over_bucket.1 >= 0.0);

                if over_bucket.1 > 1.0 {
                    over.push(over_bucket);
                } else if over_bucket.1 < 1.0 {
                    under.push(over_bucket);
                }

                alias_table[under_bucket.0] = GpuAliasEntry {
                    prob: delta,
                    index: over_bucket.0 as u32,
                };
            }

            alias_table
        }
    }
}

impl TryFrom<Mesh> for GpuMesh {
    type Error = PrepareMeshError;

    fn try_from(mesh: Mesh) -> Result<Self, Self::Error> {
        let positions = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(VertexAttributeValues::as_float3)
            .ok_or(PrepareMeshError::MissingAttributePosition)?;
        let normals = mesh
            .attribute(Mesh::ATTRIBUTE_NORMAL)
            .and_then(VertexAttributeValues::as_float3)
            .ok_or(PrepareMeshError::MissingAttributeNormal)?;
        let uvs = mesh
            .attribute(Mesh::ATTRIBUTE_UV_0)
            .and_then(|attribute| match attribute {
                VertexAttributeValues::Float32x2(value) => Some(value),
                _ => None,
            })
            .ok_or(PrepareMeshError::MissingAttributeUV)?;

        let mut vertices = vec![];
        for (position, normal, uv) in itertools::multizip((positions, normals, uvs)) {
            vertices.push(GpuVertex {
                position: Vec3::from_slice(position),
                normal: Vec3::from_slice(normal),
                uv: Vec2::from_slice(uv),
            });
        }

        let indices: Vec<_> = match mesh.indices() {
            Some(indices) => indices.iter().collect(),
            None => vertices.iter().enumerate().map(|(id, _)| id).collect(),
        };

        let mut primitives = match mesh.primitive_topology() {
            PrimitiveTopology::TriangleList => {
                let mut primitives = vec![];
                for chunk in &indices.iter().chunks(3) {
                    let (v0, v1, v2) = chunk
                        .cloned()
                        .next_tuple()
                        .ok_or(PrepareMeshError::IncompatiblePrimitiveTopology)?;
                    let vertices = [v0, v1, v2]
                        .map(|id| vertices[id])
                        .map(|vertex| vertex.position);
                    let indices = [v0, v1, v2].map(|id| id as u32);
                    primitives.push(GpuPrimitive {
                        vertices,
                        indices,
                        node_index: 0,
                    });
                }
                Ok(primitives)
            }
            PrimitiveTopology::TriangleStrip => {
                let mut primitives = vec![];
                for (id, (v0, v1, v2)) in indices.iter().cloned().tuple_windows().enumerate() {
                    let indices = if id & 1 == 0 {
                        [v0, v1, v2]
                    } else {
                        [v1, v0, v2]
                    };
                    let vertices = indices.map(|id| vertices[id]).map(|vertex| vertex.position);
                    let indices = indices.map(|id| id as u32);
                    primitives.push(GpuPrimitive {
                        vertices,
                        indices,
                        node_index: 0,
                    })
                }
                Ok(primitives)
            }
            _ => Err(PrepareMeshError::IncompatiblePrimitiveTopology),
        }?;

        if primitives.is_empty() {
            return Err(PrepareMeshError::NoPrimitive);
        }

        let bvh = BVH::build(&mut primitives);
        let nodes = bvh.flatten_custom(&GpuNode::pack);

        Ok(Self {
            vertices,
            primitives,
            nodes,
        })
    }
}

/// Offsets (and length for nodes) of the mesh in the universal buffer.
/// This is known only when [`MeshAssetState`] isn't [`Dirty`](MeshAssetState::Dirty).
#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct GpuMeshIndex {
    pub vertex: u32,
    pub primitive: u32,
    pub node: UVec2,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
pub enum MeshMaterialSystems {
    PrepareTextures,
    PrepareAssets,
    PrepareInstances,
}

#[derive(Resource, Deref, DerefMut)]
pub struct MeshMaterialBindGroupLayout(pub BindGroupLayout);

impl FromWorld for MeshMaterialBindGroupLayout {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Vertices
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuVertexBuffer::min_size()),
                    },
                    count: None,
                },
                // Primitives
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuPrimitiveBuffer::min_size()),
                    },
                    count: None,
                },
                // Asset nodes
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuNodeBuffer::min_size()),
                    },
                    count: None,
                },
                // Alias Table
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuAliasTableBuffer::min_size()),
                    },
                    count: None,
                },
                // Instances
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuInstanceBuffer::min_size()),
                    },
                    count: None,
                },
                // Instance nodes
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuNodeBuffer::min_size()),
                    },
                    count: None,
                },
                // Materials
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuStandardMaterialBuffer::min_size()),
                    },
                    count: None,
                },
                // Emissive nodes
                BindGroupLayoutEntry {
                    binding: 7,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuNodeBuffer::min_size()),
                    },
                    count: None,
                },
                // Emissives
                BindGroupLayoutEntry {
                    binding: 8,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuEmissiveBuffer::min_size()),
                    },
                    count: None,
                },
            ],
        });

        Self(layout)
    }
}

#[derive(Resource)]
pub struct TextureBindGroupLayout {
    pub layout: BindGroupLayout,
    pub texture_count: u32,
}

impl FromWorld for TextureBindGroupLayout {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Textures
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Samplers
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        Self {
            layout,
            texture_count: 0,
        }
    }
}

fn prepare_texture_bind_group_layout(
    render_device: Res<RenderDevice>,
    textures: Res<MaterialTextures>,
    mut texture_layout: ResMut<TextureBindGroupLayout>,
) {
    let texture_count = textures.data.len() as u32;
    let layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            // Textures
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::all(),
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: NonZeroU32::new(texture_count as u32),
            },
            // Samplers
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::all(),
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: NonZeroU32::new(texture_count as u32),
            },
        ],
    });

    *texture_layout = TextureBindGroupLayout {
        layout,
        texture_count,
    }
}

#[derive(Resource)]
pub struct MeshMaterialBindGroup {
    pub mesh_material: BindGroup,
    pub texture: BindGroup,
}

#[allow(clippy::too_many_arguments)]
fn queue_mesh_material_bind_group(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mesh_pipeline: Res<MeshPipeline>,
    meshes: Res<MeshRenderAssets>,
    textures: Res<MaterialTextures>,
    materials: Res<MaterialRenderAssets>,
    instances: Res<InstanceRenderAssets>,
    images: Res<RenderAssets<Image>>,
    mesh_material_layout: Res<MeshMaterialBindGroupLayout>,
    texture_layout: Res<TextureBindGroupLayout>,
) {
    if let (
        Some(vertex_binding),
        Some(primitive_binding),
        Some(asset_node_binding),
        Some(instance_binding),
        Some(instance_node_binding),
        Some(material_binding),
        Some(emissive_binding),
        Some(emissive_node_binding),
        Some(alias_table_binding),
    ) = (
        meshes.vertex_buffer.binding(),
        meshes.primitive_buffer.binding(),
        meshes.node_buffer.binding(),
        instances.instance_buffer.binding(),
        instances.instance_node_buffer.binding(),
        materials.binding(),
        instances.emissive_buffer.binding(),
        instances.emissive_node_buffer.binding(),
        instances.alias_table_buffer.binding(),
    ) {
        let mesh_material = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &mesh_material_layout.0,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: vertex_binding,
                },
                BindGroupEntry {
                    binding: 1,
                    resource: primitive_binding,
                },
                BindGroupEntry {
                    binding: 2,
                    resource: asset_node_binding,
                },
                BindGroupEntry {
                    binding: 3,
                    resource: alias_table_binding,
                },
                BindGroupEntry {
                    binding: 4,
                    resource: instance_binding,
                },
                BindGroupEntry {
                    binding: 5,
                    resource: instance_node_binding,
                },
                BindGroupEntry {
                    binding: 6,
                    resource: material_binding,
                },
                BindGroupEntry {
                    binding: 7,
                    resource: emissive_node_binding,
                },
                BindGroupEntry {
                    binding: 8,
                    resource: emissive_binding,
                },
            ],
        });

        let images = textures.data.iter().map(|handle| {
            images
                .get(handle)
                .unwrap_or(&mesh_pipeline.dummy_white_gpu_image)
        });
        let textures: Vec<_> = images.clone().map(|image| &*image.texture_view).collect();
        let samplers: Vec<_> = images.map(|image| &*image.sampler).collect();

        let texture = if !textures.is_empty() {
            render_device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &texture_layout.layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureViewArray(textures.as_slice()),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::SamplerArray(samplers.as_slice()),
                    },
                ],
            })
        } else {
            let dummy_white_gpu_image = &mesh_pipeline.dummy_white_gpu_image;
            render_device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &texture_layout.layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&dummy_white_gpu_image.texture_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&dummy_white_gpu_image.sampler),
                    },
                ],
            })
        };

        commands.insert_resource(MeshMaterialBindGroup {
            mesh_material,
            texture,
        });
    } else {
        commands.remove_resource::<MeshMaterialBindGroup>();
    }
}

pub struct SetMeshMaterialBindGroup<const I: usize>;
impl<const I: usize> EntityRenderCommand for SetMeshMaterialBindGroup<I> {
    type Param = SRes<MeshMaterialBindGroup>;

    fn render<'w>(
        _view: Entity,
        _item: Entity,
        bind_group: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(I, &bind_group.into_inner().mesh_material, &[]);

        RenderCommandResult::Success
    }
}
