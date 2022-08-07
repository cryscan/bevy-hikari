use bevy::{
    ecs::system::{
        lifetimeless::{SRes, SResMut},
        SystemParamItem,
    },
    prelude::*,
    reflect::TypeUuid,
    render::{
        mesh::MeshVertexBufferLayout,
        render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin},
        render_resource::{BufferUsages, BufferVec, IndexFormat, PrimitiveTopology},
        renderer::{RenderDevice, RenderQueue},
        RenderApp,
    },
};
use bvh::aabb::{Bounded, AABB};
use std::collections::BTreeMap;

pub struct GiPlugin;

impl Plugin for GiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TriangleTable>()
            .add_asset::<BatchMesh>()
            .add_plugin(RenderAssetPlugin::<BatchMesh>::default());

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<BatchMeshMeta>();
        }
    }
}

pub struct Triangle {
    /// Global positions of vertices.
    pub vertices: [[f32; 3]; 3],
    /// Indices of vertices in the universal vertex buffer.
    pub indices: [usize; 3],
    /// Instance id the triangle belongs to.
    pub instance: usize,
}

impl Bounded for Triangle {
    fn aabb(&self) -> AABB {
        let mut aabb = AABB::empty();
        for vertex in self.vertices {
            aabb.grow_mut(&vertex.into());
        }
        aabb
    }
}

pub type TriangleTable = BTreeMap<Entity, Vec<Triangle>>;

pub struct BatchMeshMeta {
    pub index_buffer: BufferVec<u8>,
    pub vertex_buffer: BufferVec<u8>,
}

impl Default for BatchMeshMeta {
    fn default() -> Self {
        Self {
            index_buffer: BufferVec::new(BufferUsages::COPY_DST | BufferUsages::UNIFORM),
            vertex_buffer: BufferVec::new(BufferUsages::COPY_DST | BufferUsages::UNIFORM),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuBatchMesh {
    pub vertex_offset: u32,
    pub buffer_info: GpuBatchBufferInfo,
    pub primitive_topology: PrimitiveTopology,
    pub layout: MeshVertexBufferLayout,
}

#[derive(Debug, Clone)]
pub enum GpuBatchBufferInfo {
    Indexed {
        offset: u32,
        count: u32,
        index_format: IndexFormat,
    },
    NonIndexed {
        vertex_count: u32,
    },
}

#[derive(Debug, TypeUuid, Clone, Deref, DerefMut)]
#[uuid = "d5cd37e2-e015-4415-bc67-cfb7ceba0b26"]
pub struct BatchMesh(Mesh);

impl<T: Into<Mesh>> From<T> for BatchMesh {
    fn from(t: T) -> Self {
        Self(t.into())
    }
}

impl RenderAsset for BatchMesh {
    type ExtractedAsset = Self;
    type PreparedAsset = GpuBatchMesh;
    type Param = (
        SRes<RenderDevice>,
        SRes<RenderQueue>,
        SResMut<BatchMeshMeta>,
    );

    fn extract_asset(&self) -> Self::ExtractedAsset {
        self.clone()
    }

    fn prepare_asset(
        mesh: Self::ExtractedAsset,
        (render_device, render_queue, mesh_meta): &mut SystemParamItem<Self::Param>,
    ) -> Result<Self::PreparedAsset, PrepareAssetError<Self::ExtractedAsset>> {
        let vertex_offset = mesh_meta.vertex_buffer.len() as u32;
        for value in mesh.get_vertex_buffer_data() {
            mesh_meta.vertex_buffer.push(value);
        }
        mesh_meta
            .vertex_buffer
            .write_buffer(render_device, render_queue);

        let buffer_info = mesh.get_index_buffer_bytes().map_or(
            GpuBatchBufferInfo::NonIndexed {
                vertex_count: mesh.count_vertices() as u32,
            },
            |data| {
                let offset = mesh_meta.index_buffer.len() as u32;
                for value in data {
                    mesh_meta.index_buffer.push(*value);
                }
                GpuBatchBufferInfo::Indexed {
                    offset,
                    count: mesh.indices().unwrap().len() as u32,
                    index_format: mesh.indices().unwrap().into(),
                }
            },
        );

        let primitive_topology = mesh.primitive_topology();
        let layout = mesh.get_mesh_vertex_buffer_layout();

        Ok(GpuBatchMesh {
            vertex_offset,
            buffer_info,
            primitive_topology,
            layout,
        })
    }
}
