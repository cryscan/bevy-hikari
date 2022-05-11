#define_import_path bevy_hikari::volume_struct

let VOXEL_SIZE: u32 = 256u;
let VOXEL_COUNT: u32 = 16777216u;

struct Volume {
    min: vec3<f32>;
    max: vec3<f32>;
};

struct VoxelBuffer {
    data: array<atomic<u32>, VOXEL_COUNT>;
};
