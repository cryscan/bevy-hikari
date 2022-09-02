use crate::transform::PreviousGlobalTransform;
use bevy::{
    prelude::*,
    render::{
        extract_component::UniformComponentPlugin,
        mesh::VertexAttributeValues,
        render_resource::{PrimitiveTopology, ShaderType, StorageBuffer},
        renderer::{RenderDevice, RenderQueue},
        Extract, RenderApp, RenderStage,
    },
    utils::{HashMap, HashSet},
};
use bvh::{aabb::Bounded, bounding_hierarchy::BHShape, bvh::BVH};
use itertools::Itertools;
use std::collections::BTreeMap;

pub struct BindlessMeshPlugin;
impl Plugin for BindlessMeshPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(UniformComponentPlugin::<PreviousMeshUniform>::default());

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<BindlessMeshes>()
                .init_resource::<BindlessMeshMeta>()
                .init_resource::<BindlessMeshesUpdated>()
                .add_system_to_stage(RenderStage::Extract, extract_mesh_assets.exclusive_system())
                .add_system_to_stage(RenderStage::Extract, extract_meshes)
                .add_system_to_stage(RenderStage::Prepare, prepare_mesh_assets);
        }
    }
}

#[derive(Default, Clone, Copy, ShaderType)]
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
    fn aabb(&self) -> bvh::aabb::AABB {
        bvh::aabb::AABB::empty()
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

#[derive(Default, Clone, ShaderType)]
pub struct GpuNode {
    pub min: Vec3,
    pub max: Vec3,
    pub entry_index: u32,
    pub exit_index: u32,
    pub face_index: u32,
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
    #[size(runtime)]
    pub data: Vec<GpuNode>,
}

#[derive(Default)]
pub struct BindlessMeshMeta {
    pub vertex_buffer: StorageBuffer<GpuVertexBuffer>,
    pub primitive_buffer: StorageBuffer<GpuPrimitiveBuffer>,
    pub node_buffer: StorageBuffer<GpuNodeBuffer>,
}

impl BindlessMeshMeta {
    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        self.vertex_buffer.write_buffer(device, queue);
        self.primitive_buffer.write_buffer(device, queue);
        self.node_buffer.write_buffer(device, queue);
    }
}

/// GPU representation of assets of type [`BindlessMesh`].
#[derive(Debug, Clone)]
pub struct BindlessMeshOffset {
    pub vertex_offset: usize,
    pub primitive_offset: usize,
    pub node_offset: usize,
}

#[derive(Debug)]
pub enum BindlessMeshError {
    MissingAttributePosition,
    MissingAttributeNormal,
    MissingAttributeUV,
    IncompatiblePrimitiveTopology,
}

/// [`BindlessMesh`] only exists in the render world,
/// which is extracted from the [`Mesh`] asset.
#[derive(Default, Clone)]
pub struct BindlessMesh {
    pub vertices: Vec<GpuVertex>,
    pub primitives: Vec<GpuPrimitive>,
    pub nodes: Vec<GpuNode>,
}

impl BindlessMesh {
    pub fn from_mesh(mesh: &Mesh) -> Result<Self, BindlessMeshError> {
        let positions = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(VertexAttributeValues::as_float3)
            .ok_or(BindlessMeshError::MissingAttributePosition)?;
        let normals = mesh
            .attribute(Mesh::ATTRIBUTE_NORMAL)
            .and_then(VertexAttributeValues::as_float3)
            .ok_or(BindlessMeshError::MissingAttributeNormal)?;
        let uvs = mesh
            .attribute(Mesh::ATTRIBUTE_UV_0)
            .and_then(|attribute| match attribute {
                VertexAttributeValues::Float32x2(value) => Some(value),
                _ => None,
            })
            .ok_or(BindlessMeshError::MissingAttributeUV)?;

        let mut vertices = vec![];
        for (position, normal, uv) in itertools::multizip((positions, normals, uvs)) {
            vertices.push(GpuVertex {
                position: Vec3::from_slice(position),
                normal: Vec3::from_slice(normal),
                uv: Vec2::from_slice(uv),
            });
        }

        let indices = match mesh.indices() {
            Some(indices) => indices.iter().collect_vec(),
            None => vertices.iter().enumerate().map(|(id, _)| id).collect_vec(),
        };

        let mut faces = match mesh.primitive_topology() {
            PrimitiveTopology::TriangleList => {
                let mut faces = vec![];
                for chunk in &indices.iter().chunks(3) {
                    let (v0, v1, v2) = chunk
                        .cloned()
                        .next_tuple()
                        .ok_or(BindlessMeshError::IncompatiblePrimitiveTopology)?;
                    let vertices = [v0, v1, v2]
                        .map(|id| vertices[id])
                        .map(|vertex| vertex.position);
                    let indices = [v0, v1, v2].map(|id| id as u32);
                    faces.push(GpuPrimitive {
                        vertices,
                        indices,
                        node_index: 0,
                    });
                }
                Ok(faces)
            }
            PrimitiveTopology::TriangleStrip => {
                let mut faces = vec![];
                for (id, (v0, v1, v2)) in indices.iter().cloned().tuple_windows().enumerate() {
                    let indices = if id & 1 == 0 {
                        [v0, v1, v2]
                    } else {
                        [v1, v0, v2]
                    };
                    let vertices = indices.map(|id| vertices[id]).map(|vertex| vertex.position);
                    let indices = indices.map(|id| id as u32);
                    faces.push(GpuPrimitive {
                        vertices,
                        indices,
                        node_index: 0,
                    })
                }
                Ok(faces)
            }
            _ => Err(BindlessMeshError::IncompatiblePrimitiveTopology),
        }?;

        let bvh = BVH::build(&mut faces);
        let nodes = bvh.flatten_custom(&|aabb, entry_index, exit_index, face_index| GpuNode {
            min: aabb.min.to_array().into(),
            max: aabb.max.to_array().into(),
            entry_index,
            exit_index,
            face_index,
        });

        Ok(Self {
            vertices,
            primitives: faces,
            nodes,
        })
    }
}

#[derive(Default, Deref, DerefMut)]
pub struct BindlessMeshesUpdated(bool);

#[derive(Default)]
pub struct BindlessMeshes {
    pub assets: BTreeMap<Handle<Mesh>, BindlessMesh>,
    pub offsets: HashMap<Handle<Mesh>, BindlessMeshOffset>,
}

fn extract_mesh_assets(
    mut commands: Commands,
    mut events: Extract<EventReader<AssetEvent<Mesh>>>,
    mut meta: ResMut<BindlessMeshMeta>,
    mut meshes: ResMut<BindlessMeshes>,
    assets: Extract<Res<Assets<Mesh>>>,
) {
    let mut changed_assets = HashSet::default();
    let mut removed = Vec::new();
    for event in events.iter() {
        match event {
            AssetEvent::Created { handle } | AssetEvent::Modified { handle } => {
                changed_assets.insert(handle.clone_weak());
            }
            AssetEvent::Removed { handle } => {
                changed_assets.remove(handle);
                removed.push(handle.clone_weak());
            }
        }
    }

    let mut extracted = Vec::new();
    for handle in changed_assets.drain() {
        if let Some(mesh) = assets
            .get(&handle)
            .and_then(|mesh| BindlessMesh::from_mesh(mesh).ok())
        {
            extracted.push((handle, mesh));
        }
    }

    let updated = !extracted.is_empty() || !removed.is_empty();

    for (handle, mesh) in extracted {
        meshes.assets.insert(handle, mesh);
    }
    for handle in removed {
        meshes.assets.remove(&handle);
    }

    if updated {
        let mut offsets = Vec::new();

        meta.vertex_buffer.get_mut().data.clear();
        meta.primitive_buffer.get_mut().data.clear();
        meta.node_buffer.get_mut().data.clear();

        for (handle, mesh) in meshes.assets.iter() {
            let vertex_offset = meta.vertex_buffer.get().data.len();
            meta.vertex_buffer
                .get_mut()
                .data
                .append(&mut mesh.vertices.clone());

            let primitive_offset = meta.primitive_buffer.get().data.len();
            meta.primitive_buffer
                .get_mut()
                .data
                .append(&mut mesh.primitives.clone());

            let node_offset = meta.node_buffer.get().data.len();
            meta.node_buffer
                .get_mut()
                .data
                .append(&mut mesh.nodes.clone());

            offsets.push((
                handle.clone_weak(),
                BindlessMeshOffset {
                    vertex_offset,
                    primitive_offset,
                    node_offset,
                },
            ));
        }

        for (handle, offset) in offsets {
            meshes.offsets.insert(handle, offset);
        }
    }

    commands.insert_resource(BindlessMeshesUpdated(updated));
}

fn prepare_mesh_assets(
    updated: Res<BindlessMeshesUpdated>,
    mut meta: ResMut<BindlessMeshMeta>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if **updated {
        meta.write_buffer(&render_device, &render_queue);
    }
}

// #[derive(Default, Component, Clone, ShaderType)]
// pub struct BindlessMeshUniform {
//     pub id: u32,
//     pub vertex_offset: u32,
//     pub primitive_offset: u32,
//     pub node_offset: u32,
// }

#[derive(Default, Component, Clone, ShaderType)]
pub struct PreviousMeshUniform {
    pub transform: Mat4,
    pub inverse_transpose_model: Mat4,
}

#[allow(clippy::type_complexity)]
fn extract_meshes(
    mut commands: Commands,
    query: Extract<Query<(Entity, &PreviousGlobalTransform), With<Handle<Mesh>>>>,
) {
    for (entity, transform) in query.iter() {
        // if let Some(offset) = meshes.offsets.get(handle) {
        //     let uniform = BindlessMeshUniform {
        //         id: entity.id(),
        //         vertex_offset: offset.vertex_offset as u32,
        //         primitive_offset: offset.primitive_offset as u32,
        //         node_offset: offset.node_offset as u32,
        //     };
        // }
        let transform = transform.compute_matrix();
        let uniform = PreviousMeshUniform {
            transform,
            inverse_transpose_model: transform.inverse().transpose(),
        };
        commands.get_or_spawn(entity).insert(uniform);
    }
}
