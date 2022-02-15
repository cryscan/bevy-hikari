use bevy::{
    pbr::{MeshPipeline, MeshPipelineKey},
    prelude::*,
    render::{
        render_resource::{std140::AsStd140, *},
        renderer::RenderDevice,
    },
};

pub const MAX_FRAGMENTS: usize = 4194304;
pub const MAX_OCTREE_SIZE: usize = 524288;

#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemLabel)]
pub enum VoxelConeTracingSystems {
    ExtractVolumes,
    PrepareVolumes,
    Queue,
}

#[derive(Component)]
pub struct Volume;

#[derive(Component)]
pub struct ExtractedVolume {
    boundary: [Vec3; 2],
    projections: [Mat4; 3],
}

#[derive(AsStd140, Clone, Copy)]
struct GpuVolume {
    min: Vec3,
    max: Vec3,
}

#[derive(AsStd140)]
struct GpuFragmentList {
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

fn make_bind_group_layout(render_device: &RenderDevice) -> BindGroupLayout {
    render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("voxel_bind_group"),
        entries: &[
            // Volume
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(GpuVolume::std140_size_static() as u64),
                },
                count: None,
            },
            // Fragments
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(GpuFragmentList::std140_size_static() as u64),
                },
                count: None,
            },
            // Octree
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(GpuOctree::std140_size_static() as u64),
                },
                count: None,
            },
            // Texture
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
    })
}

pub struct VoxelPipeline {
    shader: Handle<Shader>,
    voxel_layout: BindGroupLayout,
    mesh_pipeline: MeshPipeline,
}

impl FromWorld for VoxelPipeline {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.get_resource::<AssetServer>().unwrap();
        let shader = asset_server.load("shaders/voxel.wgsl");

        let mesh_pipeline = world.get_resource::<MeshPipeline>().unwrap().clone();

        let render_device = world.get_resource::<RenderDevice>().unwrap();
        let voxel_layout = make_bind_group_layout(render_device);

        Self {
            shader,
            voxel_layout,
            mesh_pipeline,
        }
    }
}

impl SpecializedPipeline for VoxelPipeline {
    type Key = MeshPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut descriptor = self.mesh_pipeline.specialize(key);
        descriptor.vertex.shader = self.shader.clone();
        descriptor.fragment.as_mut().unwrap().shader = self.shader.clone();
        descriptor.multisample.count = 4;
        descriptor.layout = Some(vec![
            self.mesh_pipeline.view_layout.clone(),
            self.mesh_pipeline.mesh_layout.clone(),
            self.voxel_layout.clone(),
        ]);

        descriptor
    }
}

pub struct OctreePipeline {
    init_pipeline: ComputePipeline,
    mark_nodes_pipeline: ComputePipeline,
    expand_nodes_pipeline: ComputePipeline,
    expand_final_level_nodes_pipeline: ComputePipeline,
    build_mipmap_pipeline: ComputePipeline,
}

impl FromWorld for OctreePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();

        let shader_source = include_str!("../assets/shaders/octree.wgsl");
        let shader = render_device.create_shader_module(&ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = make_bind_group_layout(render_device);
        let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("octree_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let init_pipeline = render_device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("octree_init_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "init",
        });
        let mark_nodes_pipeline =
            render_device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("octree_mark_nodes_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "mark_nodes_pipeline",
            });
        let expand_nodes_pipeline =
            render_device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("octree_expand_nodes_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "expand_nodes_pipeline",
            });
        let expand_final_level_nodes_pipeline =
            render_device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("octree_expand_final_level_nodes_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "expand_final_level_nodes_pipeline",
            });
        let build_mipmap_pipeline =
            render_device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("octree_build_mipmap_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "build_mipmap_pipeline",
            });

        Self {
            init_pipeline,
            mark_nodes_pipeline,
            expand_nodes_pipeline,
            expand_final_level_nodes_pipeline,
            build_mipmap_pipeline,
        }
    }
}

fn prepare_volume() {}