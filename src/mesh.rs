use bevy::{
    asset::AddAsset,
    ecs::system::{
        lifetimeless::{Read, SQuery, SRes, SResMut},
        SystemParamItem,
    },
    math::Mat3A,
    pbr::MeshUniform,
    prelude::*,
    reflect::TypeUuid,
    render::{
        mesh::{
            GpuBufferInfo::{Indexed, NonIndexed},
            GpuMesh, VertexAttributeValues,
        },
        primitives::Aabb,
        render_asset::{PrepareAssetError, RenderAsset, RenderAssetPlugin, RenderAssets},
        render_phase::{EntityRenderCommand, RenderCommandResult, TrackedRenderPass},
        render_resource::{PrimitiveTopology, ShaderType, StorageBuffer},
        renderer::{RenderDevice, RenderQueue},
        view::{NoFrustumCulling, VisibilitySystems},
        Extract, RenderApp, RenderStage,
    },
};
use bvh::{aabb::Bounded, bounding_hierarchy::BHShape, bvh::BVH};
use itertools::Itertools;

pub struct BoundedMeshPlugin;

impl Plugin for BoundedMeshPlugin {
    fn build(&self, app: &mut App) {
        app.add_asset::<BoundedMesh>()
            .add_plugin(RenderAssetPlugin::<BoundedMesh>::default())
            .add_system_to_stage(
                CoreStage::PostUpdate,
                calculate_bounds.before(VisibilitySystems::CheckVisibility),
            );

        if let Ok(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<BoundedMeshMeta>()
                .add_system_to_stage(RenderStage::Extract, extract_batch_meshes);
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
pub struct BoundedMeshMeta {
    pub vertex_buffer: StorageBuffer<GpuVertexBuffer>,
    pub face_buffer: StorageBuffer<GpuFaceBuffer>,
    pub node_buffer: StorageBuffer<GpuNodeBuffer>,
}

#[derive(Debug, Clone)]
pub struct GpuBoundedMesh {
    pub mesh: GpuMesh,
    /// Offset to the global buffers in [`BoundedMeshMeta`].
    pub vertex_offset: u32,
    pub face_offset: u32,
    pub node_offset: u32,
}

#[derive(Debug, TypeUuid, Clone, Deref, DerefMut)]
#[uuid = "d5cd37e2-e015-4415-bc67-cfb7ceba0b26"]
pub struct BoundedMesh(Mesh);

impl<T: Into<Mesh>> From<T> for BoundedMesh {
    fn from(t: T) -> Self {
        Self(t.into())
    }
}

#[derive(Debug)]
pub enum BoundedMeshPrepareError {
    MissAttributePosition,
    MissAttributeNormal,
    MissAttributeUV,
    IncompatiblePrimitiveTopology,
}

impl BoundedMesh {
    pub fn prepare_resources(
        &self,
    ) -> Result<(Vec<GpuVertex>, Vec<GpuFace>), BoundedMeshPrepareError> {
        let positions = self
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .and_then(VertexAttributeValues::as_float3)
            .ok_or(BoundedMeshPrepareError::MissAttributePosition)?;
        let normals = self
            .attribute(Mesh::ATTRIBUTE_NORMAL)
            .and_then(VertexAttributeValues::as_float3)
            .ok_or(BoundedMeshPrepareError::MissAttributeNormal)?;
        let uvs = self
            .attribute(Mesh::ATTRIBUTE_UV_0)
            .and_then(|attribute| match attribute {
                VertexAttributeValues::Float32x2(value) => Some(value),
                _ => None,
            })
            .ok_or(BoundedMeshPrepareError::MissAttributeUV)?;

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
                        .ok_or(BoundedMeshPrepareError::IncompatiblePrimitiveTopology)?;
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
            _ => Err(BoundedMeshPrepareError::IncompatiblePrimitiveTopology),
        }?;

        Ok((vertices, faces))
    }
}

impl RenderAsset for BoundedMesh {
    type ExtractedAsset = Self;
    type PreparedAsset = GpuBoundedMesh;
    type Param = (
        SRes<RenderDevice>,
        SRes<RenderQueue>,
        SResMut<BoundedMeshMeta>,
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

        Ok(GpuBoundedMesh {
            mesh,
            vertex_offset,
            face_offset,
            node_offset,
        })
    }
}

pub struct DrawBoundedMesh;

impl EntityRenderCommand for DrawBoundedMesh {
    type Param = (
        SRes<RenderAssets<BoundedMesh>>,
        SQuery<Read<Handle<BoundedMesh>>>,
    );

    #[inline]
    fn render<'w>(
        _view: Entity,
        item: Entity,
        (meshes, mesh_query): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let mesh_handle = mesh_query.get(item).unwrap();
        if let Some(gpu_mesh) = meshes
            .into_inner()
            .get(mesh_handle)
            .map(|gpu_bounded_mesh| &gpu_bounded_mesh.mesh)
        {
            pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
            match &gpu_mesh.buffer_info {
                Indexed {
                    buffer,
                    count,
                    index_format,
                } => {
                    pass.set_index_buffer(buffer.slice(..), 0, *index_format);
                    pass.draw_indexed(0..*count, 0, 0..1);
                }
                NonIndexed { vertex_count } => pass.draw(0..*vertex_count, 0..1),
            }
            RenderCommandResult::Success
        } else {
            RenderCommandResult::Failure
        }
    }
}

#[allow(clippy::type_complexity)]
pub fn calculate_bounds(
    mut commands: Commands,
    meshes: Res<Assets<BoundedMesh>>,
    without_aabb: Query<(Entity, &Handle<BoundedMesh>), (Without<Aabb>, Without<NoFrustumCulling>)>,
) {
    for (entity, mesh_handle) in &without_aabb {
        if let Some(mesh) = meshes.get(mesh_handle) {
            if let Some(aabb) = mesh.compute_aabb() {
                commands.entity(entity).insert(aabb);
            }
        }
    }
}

// NOTE: These must match the bit flags in bevy_pbr2/src/render/mesh.wgsl!
bitflags::bitflags! {
    #[repr(transparent)]
    struct MeshFlags: u32 {
        const SHADOW_RECEIVER            = (1 << 0);
        // Indicates the sign of the determinant of the 3x3 model matrix. If the sign is positive,
        // then the flag should be set, else it should not be set.
        const SIGN_DETERMINANT_MODEL_3X3 = (1 << 31);
        const NONE                       = 0;
        const UNINITIALIZED              = 0xFFFF;
    }
}
pub fn extract_batch_meshes(
    mut commands: Commands,
    mut prev_mesh_commands_len: Local<usize>,
    query: Extract<Query<(Entity, &GlobalTransform, &Handle<BoundedMesh>)>>,
) {
    let mut mesh_commands = Vec::with_capacity(*prev_mesh_commands_len);

    for (entity, transform, handle) in query.iter() {
        let transform = transform.compute_matrix();

        let mut flags = MeshFlags::SHADOW_RECEIVER;
        if Mat3A::from_mat4(transform).determinant().is_sign_positive() {
            flags |= MeshFlags::SIGN_DETERMINANT_MODEL_3X3;
        }

        let uniform = MeshUniform {
            transform,
            inverse_transpose_model: transform.inverse().transpose(),
            flags: flags.bits,
        };

        mesh_commands.push((entity, (handle.clone_weak(), uniform)));
    }

    *prev_mesh_commands_len = mesh_commands.len();
    commands.insert_or_spawn_batch(mesh_commands);
}
