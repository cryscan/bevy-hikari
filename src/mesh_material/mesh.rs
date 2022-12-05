use super::{
    GpuMesh, GpuMeshIndex, GpuNode, GpuNodeBuffer, GpuPrimitive, GpuPrimitiveBuffer,
    GpuPrimitiveCompact, GpuVertex, GpuVertexBuffer, GpuVertexCompact, MeshMaterialSystems,
};
use bevy::{
    prelude::*,
    render::{
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        Extract, RenderApp, RenderStage,
    },
    utils::{HashMap, HashSet},
};
use std::collections::BTreeMap;

pub struct MeshPlugin;
impl Plugin for MeshPlugin {
    fn build(&self, app: &mut App) {
        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<GpuMeshes>()
                .init_resource::<MeshRenderAssets>()
                .add_system_to_stage(RenderStage::Extract, extract_mesh_assets)
                .add_system_to_stage(
                    RenderStage::Prepare,
                    prepare_mesh_assets.label(MeshMaterialSystems::PrepareAssets),
                );
        }
    }
}

/// Acceleration structures on GPU.
#[derive(Default, Resource)]
pub struct MeshRenderAssets {
    pub vertex_buffer: StorageBuffer<GpuVertexBuffer>,
    pub primitive_buffer: StorageBuffer<GpuPrimitiveBuffer>,
    pub node_buffer: StorageBuffer<GpuNodeBuffer>,
}

impl MeshRenderAssets {
    pub fn set(
        &mut self,
        vertices: Vec<GpuVertex>,
        primitives: Vec<GpuPrimitive>,
        nodes: Vec<GpuNode>,
    ) {
        self.vertex_buffer.get_mut().data =
            vertices.into_iter().map(GpuVertexCompact::from).collect();
        self.primitive_buffer.get_mut().data = primitives
            .into_iter()
            .map(GpuPrimitiveCompact::from)
            .collect();

        self.node_buffer.get_mut().count = nodes.len() as u32;
        self.node_buffer.get_mut().data = nodes;
    }

    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        self.vertex_buffer.write_buffer(device, queue);
        self.primitive_buffer.write_buffer(device, queue);
        self.node_buffer.write_buffer(device, queue);
    }
}

/// Holds all GPU representatives of mesh assets.
#[derive(Default, Resource, Deref, DerefMut)]
pub struct GpuMeshes(HashMap<Handle<Mesh>, (GpuMesh, GpuMeshIndex)>);

#[derive(Default, Resource)]
pub struct ExtractedMeshes {
    extracted: Vec<(Handle<Mesh>, Mesh)>,
    removed: Vec<Handle<Mesh>>,
}

fn extract_mesh_assets(
    mut commands: Commands,
    mut events: Extract<EventReader<AssetEvent<Mesh>>>,
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
        if let Some(mesh) = assets.get(&handle) {
            extracted.push((handle, mesh.clone()));
        }
    }

    commands.insert_resource(ExtractedMeshes { extracted, removed });
}

fn prepare_mesh_assets(
    mut extracted_assets: ResMut<ExtractedMeshes>,
    mut assets: Local<BTreeMap<Handle<Mesh>, GpuMesh>>,
    mut meshes: ResMut<GpuMeshes>,
    mut render_assets: ResMut<MeshRenderAssets>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if extracted_assets.removed.is_empty() && extracted_assets.extracted.is_empty() {
        return;
    }

    for handle in extracted_assets.removed.drain(..) {
        assets.remove(&handle);
        meshes.remove(&handle);
    }
    for (handle, mesh) in extracted_assets.extracted.drain(..) {
        match mesh.try_into() {
            Ok(mesh) => {
                info!("Loaded mesh {}", assets.len());
                assets.insert(handle, mesh);
            }
            Err(_err) => {
                #[cfg(feature = "warn_mesh_load")]
                warn!("Encounter an error when loading mesh: {:#?}", _err);
            }
        }
    }

    let mut vertices = vec![];
    let mut primitives = vec![];
    let mut nodes = vec![];

    for (handle, mesh) in assets.iter() {
        let vertex = vertices.len() as u32;
        let primitive = primitives.len() as u32;
        let node = UVec2::new(nodes.len() as u32, mesh.nodes.len() as u32);

        {
            let mut mesh = mesh.clone();
            vertices.append(&mut mesh.vertices);
            primitives.append(&mut mesh.primitives);
            nodes.append(&mut mesh.nodes);
        }

        let index = GpuMeshIndex {
            vertex,
            primitive,
            node,
        };

        meshes.insert(handle.clone_weak(), (mesh.clone(), index));
    }
    render_assets.set(vertices, primitives, nodes);
    render_assets.write_buffer(&render_device, &render_queue);
}
