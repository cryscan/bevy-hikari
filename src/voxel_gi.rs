use bevy::{
    math::Vec3Swizzles,
    pbr::{MeshPipeline, MeshPipelineKey},
    prelude::*,
    reflect::TypeUuid,
    render::{
        camera::CameraProjection,
        render_resource::{std140::AsStd140, *},
        renderer::RenderDevice,
        RenderApp, RenderStage,
    },
};

pub const MAX_FRAGMENTS: usize = 4194304;
pub const MAX_OCTREE_SIZE: usize = 524288;

pub const VOXEL_SHADER_HANDLE: HandleUntyped =
    HandleUntyped::weak_from_u64(Shader::TYPE_UUID, 14750151725749984738);

#[derive(Default)]
pub struct VoxelGIPlugin;

impl Plugin for VoxelGIPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Volume {
            min: Vec3::new(-5.0, -5.0, -5.0),
            max: Vec3::new(5.0, 5.0, 5.0),
        });

        let mut shaders = app.world.get_resource_mut::<Assets<Shader>>().unwrap();
        shaders.set_untracked(
            VOXEL_SHADER_HANDLE,
            Shader::from_wgsl(include_str!("shaders/voxel.wgsl")),
        );

        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => return,
        };

        render_app
            .init_resource::<VoxelGIPipeline>()
            .init_resource::<SpecializedPipelines<VoxelGIPipeline>>()
            .add_system_to_stage(
                RenderStage::Extract,
                extract_volume.label(VoxelGISystems::ExtractVolume),
            )
            .add_system_to_stage(
                RenderStage::Prepare,
                prepare_volume.label(VoxelGISystems::PrepareVolume),
            );
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
pub enum VoxelGISystems {
    ExtractVolume,
    PrepareVolume,
    Queue,
}

pub struct Volume {
    min: Vec3,
    max: Vec3,
}

pub struct ExtractedVolume {
    min: Vec3,
    max: Vec3,
    projections: [OrthographicProjection; 3],
}

#[derive(AsStd140, Clone, Copy)]
struct GpuVolume {
    min: Vec3,
    max: Vec3,
    projection: Mat4,
}

#[derive(AsStd140)]
struct GpuList {
    data: [u32; MAX_FRAGMENTS],
    counter: u32,
}

#[derive(AsStd140, Clone, Copy)]
struct GpuNode {
    children: u32,
}

#[derive(AsStd140)]
struct GpuOctree {
    nodes: [GpuNode; MAX_OCTREE_SIZE],
    levels: [u32; 8],
    node_counter: u32,
    level_counter: u32,
}

pub struct VoxelGIPipeline {
    shader: Handle<Shader>,
    bind_group_layout: BindGroupLayout,

    mesh_pipeline: MeshPipeline,

    init_pipeline: ComputePipeline,
    mark_nodes_pipeline: ComputePipeline,
    expand_nodes_pipeline: ComputePipeline,
    expand_final_level_nodes_pipeline: ComputePipeline,
    build_mipmap_pipeline: ComputePipeline,
}

impl FromWorld for VoxelGIPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();
        let bind_group_layout =
            render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: BufferSize::new(
                                GpuVolume::std140_size_static() as u64
                            ),
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: BufferSize::new(GpuList::std140_size_static() as u64),
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: BufferSize::new(
                                GpuOctree::std140_size_static() as u64
                            ),
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::ReadWrite,
                            format: TextureFormat::Rgba8Unorm,
                            view_dimension: TextureViewDimension::D1,
                        },
                        count: None,
                    },
                ],
            });

        let shader = render_device.create_shader_module(&ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(include_str!("shaders/octree.wgsl").into()),
        });
        let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("octree_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let make_compute_pipeline = |entry_point: &str| -> ComputePipeline {
            let label = format!("{entry_point}_pipeline");

            render_device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(&label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point,
            })
        };

        let init_pipeline = make_compute_pipeline("init");
        let mark_nodes_pipeline = make_compute_pipeline("mark_nodes");
        let expand_nodes_pipeline = make_compute_pipeline("expand_nodes");
        let expand_final_level_nodes_pipeline = make_compute_pipeline("expand_final_level_nodes");
        let build_mipmap_pipeline = make_compute_pipeline("build_mipmap");

        let asset_sever = world.get_resource::<AssetServer>().unwrap();
        let shader = asset_sever.get_handle(VOXEL_SHADER_HANDLE);

        let mesh_pipeline = world.get_resource::<MeshPipeline>().unwrap().clone();

        Self {
            shader,
            bind_group_layout,
            mesh_pipeline,
            init_pipeline,
            mark_nodes_pipeline,
            expand_nodes_pipeline,
            expand_final_level_nodes_pipeline,
            build_mipmap_pipeline,
        }
    }
}

impl SpecializedPipeline for VoxelGIPipeline {
    type Key = MeshPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut descriptor = self.mesh_pipeline.specialize(key);
        descriptor.vertex.shader = self.shader.clone();
        descriptor.fragment.as_mut().unwrap().shader = self.shader.clone();
        descriptor.layout = Some(vec![
            self.mesh_pipeline.view_layout.clone(),
            self.mesh_pipeline.mesh_layout.clone(),
            self.bind_group_layout.clone(),
        ]);

        descriptor
    }
}

fn extract_volume(mut commands: Commands, volume: Res<Volume>) {
    let min = volume.min;
    let max = volume.max;

    commands.insert_resource(ExtractedVolume {
        min,
        max,
        projections: [
            make_projection(min, max),
            make_projection(min.yzx(), max.yxz()),
            make_projection(min.zxy(), max.zxy()),
        ],
    });
}

fn prepare_volume(mut commands: Commands, volume: Res<ExtractedVolume>) {
    commands.insert_resource(GpuVolume {
        min: volume.min,
        max: volume.max,
        projection: volume.projections[0].get_projection_matrix(),
    })
}

fn make_projection(min: Vec3, max: Vec3) -> OrthographicProjection {
    OrthographicProjection {
        left: min.x,
        right: max.x,
        bottom: min.y,
        top: max.y,
        near: min.z,
        far: max.z,
        ..Default::default()
    }
}
