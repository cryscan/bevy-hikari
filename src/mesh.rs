use bevy::{
    asset::AddAsset,
    ecs::system::{
        lifetimeless::{SRes, SResMut},
        SystemParamItem,
    },
    prelude::*,
    reflect::TypeUuid,
    render::{
        mesh::{GpuMesh, VertexAttributeValues},
        primitives::Aabb,
        render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin},
        render_resource::{PrimitiveTopology, ShaderType, StorageBuffer},
        renderer::{RenderDevice, RenderQueue},
        view::{NoFrustumCulling, VisibilitySystems},
        RenderApp,
    },
};
use bvh::{aabb::Bounded, bounding_hierarchy::BHShape, bvh::BVH};
use itertools::Itertools;

pub struct BatchMeshPlugin;

impl Plugin for BatchMeshPlugin {
    fn build(&self, app: &mut App) {
        app.add_asset::<BatchMesh>()
            .add_plugin(RenderAssetPlugin::<BatchMesh>::default())
            .add_system_to_stage(
                CoreStage::PostUpdate,
                calculate_bounds.before(VisibilitySystems::CheckVisibility),
            );

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
    pub mesh: GpuMesh,
    /// Offset to the global buffers in [`BatchMeshMeta`].
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

#[derive(Debug)]
pub enum BatchMeshPrepareError {
    MissAttributePosition,
    MissAttributeNormal,
    MissAttributeUV,
    IncompatiblePrimitiveTopology,
}

impl BatchMesh {
    pub fn prepare_resources(
        &self,
    ) -> Result<(Vec<GpuVertex>, Vec<GpuFace>), BatchMeshPrepareError> {
        let positions = self
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(VertexAttributeValues::as_float3)
            .ok_or(BatchMeshPrepareError::MissAttributePosition)?;
        let normals = self
            .attribute(Mesh::ATTRIBUTE_NORMAL)
            .and_then(VertexAttributeValues::as_float3)
            .ok_or(BatchMeshPrepareError::MissAttributeNormal)?;
        let uvs = self
            .attribute(Mesh::ATTRIBUTE_UV_0)
            .and_then(|attribute| match attribute {
                VertexAttributeValues::Float32x2(value) => Some(value),
                _ => None,
            })
            .ok_or(BatchMeshPrepareError::MissAttributeUV)?;

        let mut vertices = vec![];
        for (position, normal, uv) in itertools::multizip((positions, normals, uvs)) {
            vertices.push(GpuVertex {
                position: Vec3::from_slice(position),
                normal: Vec3::from_slice(normal),
                uv: Vec2::from_slice(uv),
            });
        }

        let indices = match self.indices() {
            Some(indices) => indices.iter().collect_vec(),
            None => vertices.iter().enumerate().map(|(id, _)| id).collect_vec(),
        };

        let faces = match self.primitive_topology() {
            PrimitiveTopology::TriangleList => {
                let mut faces = vec![];
                for chunk in &indices.iter().chunks(3) {
                    let (v0, v1, v2) = chunk
                        .cloned()
                        .next_tuple()
                        .ok_or(BatchMeshPrepareError::IncompatiblePrimitiveTopology)?;
                    let vertices = [v0, v1, v2]
                        .map(|id| vertices[id])
                        .map(|vertex| vertex.position);
                    let indices = [v0, v1, v2].map(|id| id as u32);
                    faces.push(GpuFace {
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
                    faces.push(GpuFace {
                        vertices,
                        indices,
                        node_index: 0,
                    })
                }
                Ok(faces)
            }
            _ => Err(BatchMeshPrepareError::IncompatiblePrimitiveTopology),
        }?;

        Ok((vertices, faces))
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
        extracted_asset: Self::ExtractedAsset,
        (render_device, render_queue, mesh_meta): &mut SystemParamItem<Self::Param>,
    ) -> Result<Self::PreparedAsset, PrepareAssetError<Self::ExtractedAsset>> {
        let mesh =
            match <Mesh as RenderAsset>::prepare_asset(extracted_asset.0.clone(), render_device) {
                Ok(mesh) => mesh,
                Err(_) => return Err(PrepareAssetError::RetryNextUpdate(extracted_asset)),
            };

        let (mut vertices, mut faces) = extracted_asset.prepare_resources().unwrap();

        let vertex_offset = mesh_meta.vertex_buffer.get().data.len() as u32;
        mesh_meta.vertex_buffer.get_mut().data.append(&mut vertices);
        mesh_meta
            .vertex_buffer
            .write_buffer(render_device, render_queue);

        let bvh = BVH::build(&mut faces);
        let mut nodes = bvh.flatten_custom(&|aabb, entry_index, exit_index, face_index| GpuNode {
            min: aabb.min.to_array().into(),
            max: aabb.max.to_array().into(),
            entry_index,
            exit_index,
            face_index,
        });

        let face_offset = mesh_meta.face_buffer.get().data.len() as u32;
        mesh_meta.face_buffer.get_mut().data.append(&mut faces);
        mesh_meta
            .face_buffer
            .write_buffer(render_device, render_queue);

        let node_offset = mesh_meta.node_buffer.get().data.len() as u32;
        mesh_meta.node_buffer.get_mut().data.append(&mut nodes);
        mesh_meta
            .node_buffer
            .write_buffer(render_device, render_queue);

        Ok(GpuBatchMesh {
            mesh,
            vertex_offset,
            face_offset,
            node_offset,
        })
    }
}

#[allow(clippy::type_complexity)]
pub fn calculate_bounds(
    mut commands: Commands,
    meshes: Res<Assets<BatchMesh>>,
    without_aabb: Query<(Entity, &Handle<BatchMesh>), (Without<Aabb>, Without<NoFrustumCulling>)>,
) {
    for (entity, mesh_handle) in &without_aabb {
        if let Some(mesh) = meshes.get(mesh_handle) {
            if let Some(aabb) = mesh.compute_aabb() {
                commands.entity(entity).insert(aabb);
            }
        }
    }
}
