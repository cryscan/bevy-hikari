use crate::transform::PreviousGlobalTransform;
use bevy::{
    math::Vec3A,
    prelude::*,
    render::{
        extract_component::UniformComponentPlugin,
        mesh::VertexAttributeValues,
        primitives::Aabb,
        render_resource::{PrimitiveTopology, ShaderType, StorageBuffer},
        renderer::{RenderDevice, RenderQueue},
        Extract, RenderApp, RenderStage,
    },
    utils::{HashMap, HashSet},
};
use bvh::{
    aabb::{Bounded, AABB},
    bounding_hierarchy::BHShape,
    bvh::BVH,
};
use itertools::Itertools;
use std::collections::BTreeMap;

pub struct BindlessMeshPlugin;
impl Plugin for BindlessMeshPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(UniformComponentPlugin::<PreviousMeshUniform>::default())
            .add_event::<BindlessMeshInstanceEvent>()
            .add_system(mesh_instance_system);

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<BindlessMeshes>()
                .init_resource::<BindlessMeshInstances>()
                .init_resource::<BindlessMeshBottomMeta>()
                .init_resource::<BindlessMeshTopMeta>()
                .init_resource::<BindlessMeshState>()
                .add_system_to_stage(RenderStage::Extract, extract_mesh_assets.exclusive_system())
                .add_system_to_stage(RenderStage::Extract, extract_meshes)
                .add_system_to_stage(RenderStage::Extract, extract_mesh_instances)
                .add_system_to_stage(RenderStage::Prepare, prepare_mesh_assets)
                .add_system_to_stage(RenderStage::Prepare, prepare_mesh_instances);
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
    /// Packed value of vertex, primitive, node offsets and node length.
    pub data: UVec4,

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

#[derive(Default, ShaderType)]
pub struct GpuInstanceBuffer {
    #[size(runtime)]
    pub data: Vec<GpuInstance>,
}

/// Bottom-level acceleration structure.
#[derive(Default)]
pub struct BindlessMeshBottomMeta {
    pub vertex_buffer: StorageBuffer<GpuVertexBuffer>,
    pub primitive_buffer: StorageBuffer<GpuPrimitiveBuffer>,
    pub node_buffer: StorageBuffer<GpuNodeBuffer>,
}

impl BindlessMeshBottomMeta {
    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        self.vertex_buffer.write_buffer(device, queue);
        self.primitive_buffer.write_buffer(device, queue);
        self.node_buffer.write_buffer(device, queue);
    }
}

/// Top-level acceleration structure.
#[derive(Default)]
pub struct BindlessMeshTopMeta {
    pub instance_buffer: StorageBuffer<GpuInstanceBuffer>,
    pub node_buffer: StorageBuffer<GpuNodeBuffer>,
}

impl BindlessMeshTopMeta {
    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        self.instance_buffer.write_buffer(device, queue);
        self.node_buffer.write_buffer(device, queue);
    }
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

        let mut primitives = match mesh.primitive_topology() {
            PrimitiveTopology::TriangleList => {
                let mut primitives = vec![];
                for chunk in &indices.iter().chunks(3) {
                    let (v0, v1, v2) = chunk
                        .cloned()
                        .next_tuple()
                        .ok_or(BindlessMeshError::IncompatiblePrimitiveTopology)?;
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
            _ => Err(BindlessMeshError::IncompatiblePrimitiveTopology),
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

#[derive(Debug, Clone)]
pub struct BindlessMeshOffset {
    pub vertex_offset: usize,
    pub primitive_offset: usize,
    pub node_offset: usize,
    pub node_length: usize,
}

#[derive(Default)]
pub struct BindlessMeshState {
    pub asset_updated: bool,
    pub instance_updated: bool,
}

#[derive(Default)]
pub struct BindlessMeshes {
    pub assets: BTreeMap<Handle<Mesh>, BindlessMesh>,
    pub offsets: HashMap<Handle<Mesh>, BindlessMeshOffset>,
}

fn extract_mesh_assets(
    mut events: Extract<EventReader<AssetEvent<Mesh>>>,
    mut meta: ResMut<BindlessMeshBottomMeta>,
    mut meshes: ResMut<BindlessMeshes>,
    mut state: ResMut<BindlessMeshState>,
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
            let node_length = mesh.nodes.len();
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
                    node_length,
                },
            ));
        }

        for (handle, offset) in offsets {
            meshes.offsets.insert(handle, offset);
        }
    }

    state.asset_updated = updated;
}

fn prepare_mesh_assets(
    state: Res<BindlessMeshState>,
    mut meta: ResMut<BindlessMeshBottomMeta>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if state.asset_updated {
        meta.write_buffer(&render_device, &render_queue);
    }
}

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
        let transform = transform.compute_matrix();
        let uniform = PreviousMeshUniform {
            transform,
            inverse_transpose_model: transform.inverse().transpose(),
        };
        commands.get_or_spawn(entity).insert(uniform);
    }
}

#[derive(Default, Deref, DerefMut)]
pub struct BindlessMeshInstances(BTreeMap<Entity, GpuInstance>);

pub enum BindlessMeshInstanceEvent {
    Created(Entity, Handle<Mesh>),
    Changed(Entity, Handle<Mesh>),
    Removed(Entity),
}

#[allow(clippy::type_complexity)]
fn mesh_instance_system(
    mut events: EventWriter<BindlessMeshInstanceEvent>,
    removed: RemovedComponents<Handle<Mesh>>,
    mut set: ParamSet<(
        Query<(Entity, &Handle<Mesh>), Added<Handle<Mesh>>>,
        Query<(Entity, &Handle<Mesh>), Changed<Transform>>,
    )>,
) {
    for entity in removed.iter() {
        events.send(BindlessMeshInstanceEvent::Removed(entity));
    }
    for (entity, handle) in &set.p0() {
        events.send(BindlessMeshInstanceEvent::Created(
            entity,
            handle.clone_weak(),
        ));
    }
    for (entity, handle) in &set.p1() {
        events.send(BindlessMeshInstanceEvent::Changed(
            entity,
            handle.clone_weak(),
        ));
    }
}

fn extract_mesh_instances(
    mut events: Extract<EventReader<BindlessMeshInstanceEvent>>,
    meshes: Res<BindlessMeshes>,
    mut instances: ResMut<BindlessMeshInstances>,
    mut state: ResMut<BindlessMeshState>,
    query: Extract<Query<(&Aabb, &GlobalTransform)>>,
) {
    let mut changed_instances = vec![];
    let mut removed = vec![];

    for event in events.iter() {
        match event {
            BindlessMeshInstanceEvent::Created(entity, handle)
            | BindlessMeshInstanceEvent::Changed(entity, handle) => {
                changed_instances.push((*entity, handle.clone_weak()))
            }
            BindlessMeshInstanceEvent::Removed(entity) => removed.push(*entity),
        }
    }

    let updated = !changed_instances.is_empty() || !removed.is_empty();

    for entity in removed {
        instances.remove(&entity);
    }
    for (entity, handle) in changed_instances {
        if let (Ok((aabb, transform)), Some(offset)) =
            (query.get(entity), meshes.offsets.get(&handle))
        {
            let transform = transform.compute_matrix();
            let center = transform.transform_point3a(aabb.center);
            let vertices = (0..8i32)
                .map(|index| {
                    let x = 2 * (index & 1) - 1;
                    let y = 2 * ((index >> 1) & 1) - 1;
                    let z = 2 * ((index >> 2) & 1) - 1;
                    let vertex = aabb.half_extents * Vec3A::new(x as f32, y as f32, z as f32);
                    transform.transform_vector3a(vertex)
                })
                .collect_vec();

            let mut min = Vec3A::ZERO;
            let mut max = Vec3A::ZERO;
            for vertex in vertices {
                min = min.min(vertex);
                max = max.max(vertex);
            }
            min += center;
            max += center;

            instances.insert(
                entity,
                GpuInstance {
                    min: min.into(),
                    max: max.into(),
                    transform,
                    inverse_transpose_model: transform.inverse().transpose(),
                    data: UVec4::new(
                        offset.vertex_offset as u32,
                        offset.primitive_offset as u32,
                        offset.node_offset as u32,
                        offset.node_length as u32,
                    ),
                    node_index: 0,
                },
            );
        }
    }

    state.instance_updated = updated;
}

fn prepare_mesh_instances(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut meta: ResMut<BindlessMeshTopMeta>,
    instances: Res<BindlessMeshInstances>,
    state: Res<BindlessMeshState>,
) {
    if state.instance_updated {
        let mut instances = instances.values().cloned().collect_vec();
        let bvh = BVH::build(&mut instances);
        let nodes = bvh.flatten_custom(&|aabb, entry_index, exit_index, primitive_index| GpuNode {
            min: aabb.min.to_array().into(),
            max: aabb.max.to_array().into(),
            entry_index,
            exit_index,
            primitive_index,
        });

        meta.instance_buffer.get_mut().data = instances;
        meta.node_buffer.get_mut().data = nodes;
        meta.write_buffer(&render_device, &render_queue);
    }
}
