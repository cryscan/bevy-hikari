#define_import_path bevy_hikari::volume_struct

let CLUSTER_SIZE: u32 = 8u;

struct Volume {
    min: vec3<f32>;
    max: vec3<f32>;
    resolution: u32;
};

struct Clusters {
    data: array<u32>;
}

struct VoxelBuffer {
    data: array<atomic<u32>>;
};

let SAMPLE_INDICES = array<vec3<i32>, 8>(
    vec3<i32>(0, 0, 0),
    vec3<i32>(1, 0, 0),
    vec3<i32>(0, 1, 0),
    vec3<i32>(1, 1, 0),
    vec3<i32>(0, 0, 1),
    vec3<i32>(1, 0, 1),
    vec3<i32>(0, 1, 1),
    vec3<i32>(1, 1, 1),
);

struct Mipmap {
    direction: u32;
};