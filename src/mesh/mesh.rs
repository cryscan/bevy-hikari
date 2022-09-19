use super::{
    GpuMesh, GpuMeshSlice, GpuNodeBuffer, GpuPrimitiveBuffer, GpuVertexBuffer, MeshMaterialSystems,
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
                .init_resource::<GpuMeshSlices>()
                .init_resource::<MeshRenderAssets>()
                .init_resource::<MeshAssetState>()
                .add_system_to_stage(RenderStage::Extract, extract_mesh_assets)
                .add_system_to_stage(
                    RenderStage::Prepare,
                    prepare_mesh_assets
                        .label(MeshMaterialSystems::PrepareAssets)
                        .after(MeshMaterialSystems::PrePrepareAssets),
                );
        }
    }
}

/// Acceleration structures on GPU.
#[derive(Default)]
pub struct MeshRenderAssets {
    pub vertex_buffer: StorageBuffer<GpuVertexBuffer>,
    pub primitive_buffer: StorageBuffer<GpuPrimitiveBuffer>,
    pub node_buffer: StorageBuffer<GpuNodeBuffer>,
}

impl MeshRenderAssets {
    pub fn clear(&mut self) {
        self.vertex_buffer.get_mut().data.clear();
        self.primitive_buffer.get_mut().data.clear();
        self.node_buffer.get_mut().data.clear();
        self.node_buffer.get_mut().count = 0;
    }

    pub fn write_buffer(&mut self, device: &RenderDevice, queue: &RenderQueue) {
        self.vertex_buffer.write_buffer(device, queue);
        self.primitive_buffer.write_buffer(device, queue);
        self.node_buffer.write_buffer(device, queue);
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum MeshAssetState {
    /// No updates for all mesh assets.
    #[default]
    Clean,
    /// There are upcoming updates but mesh assets haven't been prepared.
    Dirty,
    /// There were asset updates and mesh assets have been prepared.
    Updated,
}

#[derive(Default, Deref, DerefMut)]
pub struct GpuMeshes(BTreeMap<Handle<Mesh>, GpuMesh>);

/// Holds all GPU representatives of mesh assets.
#[derive(Default, Deref, DerefMut)]
pub struct GpuMeshSlices(HashMap<Handle<Mesh>, GpuMeshSlice>);

#[derive(Default)]
pub struct ExtractedMeshes {
    extracted: Vec<(Handle<Mesh>, Mesh)>,
    removed: Vec<Handle<Mesh>>,
}

fn extract_mesh_assets(
    mut commands: Commands,
    mut events: Extract<EventReader<AssetEvent<Mesh>>>,
    mut state: ResMut<MeshAssetState>,
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

    *state = if !extracted.is_empty() || !removed.is_empty() {
        MeshAssetState::Dirty
    } else {
        MeshAssetState::Clean
    };

    commands.insert_resource(ExtractedMeshes { extracted, removed });
}

fn prepare_mesh_assets(
    mut extracted_assets: ResMut<ExtractedMeshes>,
    mut asset_state: ResMut<MeshAssetState>,
    mut meshes: ResMut<GpuMeshes>,
    mut slices: ResMut<GpuMeshSlices>,
    mut render_assets: ResMut<MeshRenderAssets>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if *asset_state == MeshAssetState::Clean {
        return;
    }

    for handle in extracted_assets.removed.drain(..) {
        meshes.remove(&handle);
        slices.remove(&handle);
    }
    for (handle, mesh) in extracted_assets.extracted.drain(..) {
        meshes.insert(handle, GpuMesh::from_mesh(mesh).unwrap());
    }

    render_assets.clear();
    for (handle, mesh) in meshes.iter() {
        let vertex = render_assets.vertex_buffer.get().data.len() as u32;
        let primitive = render_assets.primitive_buffer.get().data.len() as u32;
        let node_offset = render_assets.node_buffer.get().data.len() as u32;
        let node_len = mesh.nodes.len() as u32;

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
            .node_buffer
            .get_mut()
            .data
            .append(&mut mesh.nodes.clone());

        slices.insert(
            handle.clone_weak(),
            GpuMeshSlice {
                vertex,
                primitive,
                node_offset,
                node_len,
            },
        );
    }
    render_assets.write_buffer(&render_device, &render_queue);

    *asset_state = MeshAssetState::Updated;
}
