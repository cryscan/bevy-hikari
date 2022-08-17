use bevy::{
    ecs::system::{
        lifetimeless::{SRes, SResMut},
        SystemParamItem,
    },
    prelude::*,
    reflect::TypeUuid,
    render::{
        render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin},
        render_resource::{ShaderType, StorageBuffer},
        renderer::{RenderDevice, RenderQueue},
        RenderApp,
    },
};
use bvh::{aabb::Bounded, bounding_hierarchy::BHShape};

pub struct GiPlugin;

impl Plugin for GiPlugin {
    fn build(&self, app: &mut App) {
        app.add_asset::<BatchMesh>()
            .add_plugin(RenderAssetPlugin::<BatchMesh>::default());

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<BatchMeshMeta>();
        }
    }
}

#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct GpuFace {
    /// Global positions of vertices.
    pub vertices: [Vec3; 3],
    /// Indices of vertices in the vertex buffer (offset not applied).
    pub indices: [u32; 3],
    /// Index of the material of the face.
    pub material: u32,
    /// Index of the node in the node buffer (offset not applied).
    node_index: u32,
}

impl Bounded for GpuFace {
    fn aabb(&self) -> bvh::aabb::AABB {
        bvh::aabb::AABB::empty()
            .grow(&self.vertices[0].to_array().into())
            .grow(&self.vertices[1].to_array().into())
            .grow(&self.vertices[2].to_array().into())
    }
}

impl BHShape for GpuFace {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index as u32;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index as usize
    }
}

#[derive(Default, Clone, Copy, ShaderType)]
pub struct GpuVertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
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
pub struct GpuFaceBuffer {
    #[size(runtime)]
    pub data: Vec<GpuFace>,
}

#[derive(Default, ShaderType)]
pub struct GpuNodeBuffer {
    #[size(runtime)]
    pub data: Vec<GpuNode>,
}

#[derive(Default)]
pub struct BatchMeshMeta {
    pub vertex_buffer: StorageBuffer<GpuVertexBuffer>,
    pub face_buffer: StorageBuffer<GpuFaceBuffer>,
    pub node_buffer: StorageBuffer<GpuNodeBuffer>,
}

#[derive(Debug, Clone)]
pub struct GpuBatchMesh {
    pub vertex_offset: u32,
    pub face_offset: u32,
    pub node_offset: u32,
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
        _mesh: Self::ExtractedAsset,
        (_render_device, _render_queue, _mesh_meta): &mut SystemParamItem<Self::Param>,
    ) -> Result<Self::PreparedAsset, PrepareAssetError<Self::ExtractedAsset>> {
        /*
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
        */
        todo!()
    }
}
