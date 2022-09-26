use self::{
    instance::{GenericInstancePlugin, InstancePlugin},
    material::{GenericMaterialPlugin, MaterialPlugin},
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
    DynamicInstanceIndex, InstanceIndex, InstanceRenderAssets, PreviousMeshUniform,
};
pub use material::MaterialRenderAssets;
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
                .add_system_to_stage(
                    RenderStage::Prepare,
                    prepare_texture_bind_group_layout.after(MeshMaterialSystems::PrepareAssets),
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

#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct GpuInstance {
    pub min: Vec3,
    pub max: Vec3,
    pub transform: Mat4,
    pub inverse_transpose_model: Mat4,
    pub slice: GpuMeshSlice,
    pub material: GpuStandardMaterialOffset,
    node_index: u32,
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

#[derive(Debug, Default, Clone, ShaderType)]
pub struct GpuNode {
    pub min: Vec3,
    pub max: Vec3,
    pub entry_index: u32,
    pub exit_index: u32,
    pub primitive_index: u32,
}

#[derive(Debug, Default, Clone, Copy, ShaderType)]
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
pub struct GpuStandardMaterialOffset {
    pub value: u32,
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

#[derive(Debug)]
pub enum PrepareMeshError {
    MissingAttributePosition,
    MissingAttributeNormal,
    MissingAttributeUV,
    IncompatiblePrimitiveTopology,
}

#[derive(Default, Clone)]
pub struct GpuMesh {
    pub vertices: Vec<GpuVertex>,
    pub primitives: Vec<GpuPrimitive>,
    pub nodes: Vec<GpuNode>,
}

impl GpuMesh {
    pub fn from_mesh(mesh: Mesh) -> Result<Self, PrepareMeshError> {
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

        let bvh = BVH::build(&mut primitives);
        let nodes = bvh.flatten_custom(&|aabb, entry_index, exit_index, primitive_index| GpuNode {
            min: aabb.min.to_array().into(),
            max: aabb.max.to_array().into(),
            entry_index,
            exit_index,
            primitive_index,
        });

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
pub struct GpuMeshSlice {
    pub vertex: u32,
    pub primitive: u32,
    pub node_offset: u32,
    pub node_len: u32,
}

pub trait IntoStandardMaterial: Material {
    /// Coverts a [`Material`] into a [`StandardMaterial`].
    /// Any new textures should be registered into [`MaterialRenderAssets`].
    fn into_standard_material(self, render_assets: &mut MaterialRenderAssets) -> StandardMaterial;
}

impl IntoStandardMaterial for StandardMaterial {
    fn into_standard_material(self, render_assets: &mut MaterialRenderAssets) -> Self {
        if let Some(texture) = &self.base_color_texture {
            render_assets.textures.insert(texture.clone_weak());
        }
        self
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
pub enum MeshMaterialSystems {
    PrePrepareAssets,
    PrepareAssets,
    PrepareInstances,
    PostPrepareInstances,
}

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
                // Instances
                BindGroupLayoutEntry {
                    binding: 3,
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
                    binding: 4,
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
                    binding: 5,
                    visibility: ShaderStages::all(),
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(GpuStandardMaterialBuffer::min_size()),
                    },
                    count: None,
                },
            ],
        });

        Self(layout)
    }
}

pub struct TextureBindGroupLayout {
    pub layout: BindGroupLayout,
    pub count: usize,
}

fn prepare_texture_bind_group_layout(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    materials: Res<MaterialRenderAssets>,
) {
    let count = materials.textures.len();
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
                count: NonZeroU32::new(count as u32),
            },
            // Samplers
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::all(),
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: NonZeroU32::new(count as u32),
            },
        ],
    });
    commands.insert_resource(TextureBindGroupLayout { layout, count });
}

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
    ) = (
        meshes.vertex_buffer.binding(),
        meshes.primitive_buffer.binding(),
        meshes.node_buffer.binding(),
        instances.instance_buffer.binding(),
        instances.node_buffer.binding(),
        materials.buffer.binding(),
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
                    resource: instance_binding,
                },
                BindGroupEntry {
                    binding: 4,
                    resource: instance_node_binding,
                },
                BindGroupEntry {
                    binding: 5,
                    resource: material_binding,
                },
            ],
        });

        let images = materials.textures.iter().map(|handle| {
            images
                .get(handle)
                .unwrap_or(&mesh_pipeline.dummy_white_gpu_image)
        });
        let textures: Vec<_> = images.clone().map(|image| &*image.texture_view).collect();
        let samplers: Vec<_> = images.map(|image| &*image.sampler).collect();

        let texture = render_device.create_bind_group(&BindGroupDescriptor {
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
        });

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
