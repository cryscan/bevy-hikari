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

pub struct MeshPlugin;
impl Plugin for MeshPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(UniformComponentPlugin::<PreviousMeshUniform>::default())
            .add_event::<MeshInstanceEvent>()
            .add_system(mesh_instance_system);

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<GpuMeshes>()
                .init_resource::<GpuMeshInstances>()
                .init_resource::<MeshRenderAssets>()
                .init_resource::<MeshAssetState>()
                .init_resource::<MeshInstanceState>()
                .add_system_to_stage(RenderStage::Extract, extract_mesh_assets)
                .add_system_to_stage(RenderStage::Extract, extract_mesh_instances)
                .add_system_to_stage(RenderStage::Extract, extract_meshes)
                .add_system_to_stage(
                    RenderStage::Prepare,
                    prepare_mesh_assets.label(MeshSystems::PrepareMeshAssets),
                )
                .add_system_to_stage(
                    RenderStage::Prepare,
                    prepare_mesh_instances
                        .label(MeshSystems::PrepareMeshInstances)
                        .after(MeshSystems::PrepareMeshAssets),
                );
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
pub enum MeshSystems {
    PrepareMeshAssets,
    PrepareMeshInstances,
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
    pub count: u32,
    #[size(runtime)]
    pub data: Vec<GpuNode>,
}

#[derive(Default, ShaderType)]
pub struct GpuInstanceBuffer {
    #[size(runtime)]
    pub data: Vec<GpuInstance>,
}

/// Acceleration structures on GPU.
#[derive(Default)]
pub struct MeshRenderAssets {
    pub vertex_buffer: StorageBuffer<GpuVertexBuffer>,
    pub primitive_buffer: StorageBuffer<GpuPrimitiveBuffer>,
    pub asset_node_buffer: StorageBuffer<GpuNodeBuffer>,
    pub instance_buffer: StorageBuffer<GpuInstanceBuffer>,
    pub instance_node_buffer: StorageBuffer<GpuNodeBuffer>,
}

impl MeshRenderAssets {
    pub fn clear_assets(&mut self) {
        self.vertex_buffer.get_mut().data.clear();
        self.primitive_buffer.get_mut().data.clear();
        self.asset_node_buffer.get_mut().data.clear();
        self.asset_node_buffer.get_mut().count = 0;
    }

    pub fn clear_instances(&mut self) {
        self.instance_buffer.get_mut().data.clear();
        self.instance_node_buffer.get_mut().data.clear();
        self.instance_node_buffer.get_mut().count = 0;
    }

    pub fn write_assets(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        self.vertex_buffer.write_buffer(device, queue);
        self.primitive_buffer.write_buffer(device, queue);
        self.asset_node_buffer.write_buffer(device, queue);
    }

    pub fn write_instances(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        self.instance_buffer.write_buffer(device, queue);
        self.instance_node_buffer.write_buffer(device, queue);
    }
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
    pub fn from_mesh(mesh: &Mesh) -> Result<Self, PrepareMeshError> {
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

#[derive(Debug, Default, PartialEq, Eq)]
pub enum MeshAssetState {
    /// No updates for all mesh assets.
    #[default]
    Clean,
    /// There are upcoming updates but mesh assets haven't been prepared.
    Dirty,
    /// There were asset updates and mesh assets have been prepared.
    Updated,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub enum MeshInstanceState {
    #[default]
    Clean,
    Dirty,
}

/// Holds all GPU representatives of mesh assets.
#[derive(Default)]
pub struct GpuMeshes {
    pub assets: BTreeMap<Handle<Mesh>, GpuMesh>,
    pub slices: HashMap<Handle<Mesh>, GpuMeshSlice>,
}

#[derive(Default)]
pub struct ExtractedMeshes {
    extracted: Vec<(Handle<Mesh>, Mesh)>,
    removed: Vec<Handle<Mesh>>,
}

fn extract_mesh_assets(
    mut commands: Commands,
    mut events: Extract<EventReader<AssetEvent<Mesh>>>,
    mut asset_state: ResMut<MeshAssetState>,
    meshes: Extract<Res<Assets<Mesh>>>,
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

    let mut extracted_assets = Vec::new();
    for handle in changed_assets.drain() {
        if let Some(mesh) = meshes.get(&handle) {
            extracted_assets.push((handle, mesh.clone()));
        }
    }

    *asset_state = if !extracted_assets.is_empty() || !removed.is_empty() {
        MeshAssetState::Dirty
    } else {
        MeshAssetState::Clean
    };

    commands.insert_resource(ExtractedMeshes {
        extracted: extracted_assets,
        removed,
    });
}

fn prepare_mesh_assets(
    mut extracted_assets: ResMut<ExtractedMeshes>,
    mut asset_state: ResMut<MeshAssetState>,
    mut meshes: ResMut<GpuMeshes>,
    mut render_assets: ResMut<MeshRenderAssets>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if *asset_state == MeshAssetState::Clean {
        return;
    }

    for handle in extracted_assets.removed.drain(..) {
        meshes.assets.remove(&handle);
        meshes.slices.remove(&handle);
    }
    for (handle, mesh) in extracted_assets.extracted.drain(..) {
        let mesh = GpuMesh::from_mesh(&mesh).unwrap();
        meshes.assets.insert(handle, mesh);
    }

    let mut slices = vec![];

    meshes.slices.clear();
    render_assets.clear_assets();
    for (handle, mesh) in &meshes.assets {
        let vertex = render_assets.vertex_buffer.get().data.len();
        let primitive = render_assets.primitive_buffer.get().data.len();
        let node = render_assets.asset_node_buffer.get().data.len();

        render_assets
            .vertex_buffer
            .get_mut()
            .data
            .append(&mut mesh.vertices.clone());
        render_assets
            .primitive_buffer
            .get_mut()
            .data
            .append(&mut mesh.primitives.clone());
        render_assets
            .asset_node_buffer
            .get_mut()
            .data
            .append(&mut mesh.nodes.clone());

        slices.push((
            handle.clone_weak(),
            GpuMeshSlice {
                vertex: vertex as u32,
                primitive: primitive as u32,
                node_offset: node as u32,
                node_len: mesh.nodes.len() as u32,
            },
        ));
    }
    render_assets.write_assets(&render_device, &render_queue);

    for (handle, slice) in slices {
        meshes.slices.insert(handle, slice);
    }

    *asset_state = MeshAssetState::Updated;
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
pub struct GpuMeshInstances(BTreeMap<Entity, (Handle<Mesh>, GpuInstance)>);

pub enum MeshInstanceEvent {
    Created(Entity, Handle<Mesh>),
    Modified(Entity, Handle<Mesh>),
    Removed(Entity),
}

#[allow(clippy::type_complexity)]
fn mesh_instance_system(
    mut events: EventWriter<MeshInstanceEvent>,
    removed: RemovedComponents<Handle<Mesh>>,
    mut set: ParamSet<(
        Query<(Entity, &Handle<Mesh>), Added<Handle<Mesh>>>,
        Query<(Entity, &Handle<Mesh>), Changed<Transform>>,
    )>,
) {
    for entity in removed.iter() {
        events.send(MeshInstanceEvent::Removed(entity));
    }
    for (entity, handle) in &set.p0() {
        events.send(MeshInstanceEvent::Created(entity, handle.clone_weak()));
    }
    for (entity, handle) in &set.p1() {
        events.send(MeshInstanceEvent::Modified(entity, handle.clone_weak()));
    }
}

fn extract_mesh_instances(
    mut events: Extract<EventReader<MeshInstanceEvent>>,
    mut instances: ResMut<GpuMeshInstances>,
    mut instance_state: ResMut<MeshInstanceState>,
    query: Extract<Query<(&Aabb, &GlobalTransform)>>,
) {
    let mut extracted_instances = vec![];
    let mut removed = vec![];

    for event in events.iter() {
        match event {
            MeshInstanceEvent::Created(entity, handle)
            | MeshInstanceEvent::Modified(entity, handle) => {
                extracted_instances.push((*entity, handle.clone_weak()))
            }
            MeshInstanceEvent::Removed(entity) => removed.push(*entity),
        }
    }

    *instance_state = if !removed.is_empty() || !extracted_instances.is_empty() {
        MeshInstanceState::Dirty
    } else {
        MeshInstanceState::Clean
    };

    for entity in removed {
        instances.remove(&entity);
    }

    for (entity, handle) in extracted_instances {
        if let Ok((aabb, transform)) = query.get(entity) {
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
                (
                    handle,
                    GpuInstance {
                        min: min.into(),
                        max: max.into(),
                        transform,
                        inverse_transpose_model: transform.inverse().transpose(),
                        slice: Default::default(),
                        node_index: 0,
                    },
                ),
            );
        }
    }
}

fn prepare_mesh_instances(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut render_assets: ResMut<MeshRenderAssets>,
    instances: Res<GpuMeshInstances>,
    meshes: Res<GpuMeshes>,
    asset_state: Res<MeshAssetState>,
    instance_state: Res<MeshInstanceState>,
) {
    if *asset_state == MeshAssetState::Clean && *instance_state == MeshInstanceState::Clean {
        return;
    }
    if *asset_state == MeshAssetState::Dirty {
        panic!("Mesh assets must be prepared before instances!");
    }

    let mut instances = instances
        .values()
        .cloned()
        .map(|(handle, mut instance)| {
            let slice = meshes.slices.get(&handle).unwrap();
            instance.slice = *slice;
            instance
        })
        .collect_vec();

    if !instances.is_empty() {
        let bvh = BVH::build(&mut instances);
        let nodes = bvh.flatten_custom(&|aabb, entry_index, exit_index, primitive_index| GpuNode {
            min: aabb.min.to_array().into(),
            max: aabb.max.to_array().into(),
            entry_index,
            exit_index,
            primitive_index,
        });
        render_assets.instance_buffer.get_mut().data = instances;
        render_assets.instance_node_buffer.get_mut().count = nodes.len() as u32;
        render_assets.instance_node_buffer.get_mut().data = nodes;
        render_assets.write_instances(&render_device, &render_queue);
    }
}
