struct VoxelBuffer {
    data: array<atomic<u32>, 16777216>;
};

[[group(0), binding(0)]]
var texture_in: texture_3d<f32>;
[[group(0), binding(1)]]
var texture_out: texture_storage_3d<rgba16float, write>;
[[group(0), binding(2)]]
var<storage, read_write> voxel_buffer: VoxelBuffer;

let INDICES = array<vec3<i32>, 8>(
    vec3<i32>(0, 0, 0),
    vec3<i32>(1, 0, 0),
    vec3<i32>(0, 1, 0),
    vec3<i32>(1, 1, 0),
    vec3<i32>(0, 0, 1),
    vec3<i32>(1, 0, 1),
    vec3<i32>(0, 1, 1),
    vec3<i32>(1, 1, 1),
);

fn sample_voxel(id: vec3<u32>, index: vec3<i32>) -> vec4<f32> {
    var location = vec3<i32>(id) * 2 + index;
    return textureLoad(texture_in, location, 0);
}

fn mipmap(id: vec3<u32>, dir: i32) {
    if (any(vec3<i32>(id) >= textureDimensions(texture_out))) {
        return;
    }

    var samples: array<vec4<f32>, 8>;
    samples[0] = sample_voxel(id, INDICES[0]);
    samples[1] = sample_voxel(id, INDICES[1]);
    samples[2] = sample_voxel(id, INDICES[2]);
    samples[3] = sample_voxel(id, INDICES[3]);
    samples[4] = sample_voxel(id, INDICES[4]);
    samples[5] = sample_voxel(id, INDICES[5]);
    samples[6] = sample_voxel(id, INDICES[6]);
    samples[7] = sample_voxel(id, INDICES[7]);

    var color = vec4<f32>(0.);
    if (dir == 0) {
        // +X
        color = color + samples[0] + (1. - samples[0].a) * samples[1];
        color = color + samples[2] + (1. - samples[2].a) * samples[3];
        color = color + samples[4] + (1. - samples[4].a) * samples[5];
        color = color + samples[6] + (1. - samples[6].a) * samples[7];
    } else if (dir == 1) {
        // -X
        color = color + samples[1] + (1. - samples[1].a) * samples[0];
        color = color + samples[3] + (1. - samples[3].a) * samples[2];
        color = color + samples[5] + (1. - samples[5].a) * samples[4];
        color = color + samples[7] + (1. - samples[7].a) * samples[6];
    } else if (dir == 2) {
        // +Y
        color = color + samples[0] + (1. - samples[0].a) * samples[2];
        color = color + samples[1] + (1. - samples[1].a) * samples[3];
        color = color + samples[4] + (1. - samples[4].a) * samples[6];
        color = color + samples[5] + (1. - samples[5].a) * samples[7];
    } else if (dir == 3) {
        // -Y
        color = color + samples[2] + (1. - samples[2].a) * samples[0];
        color = color + samples[3] + (1. - samples[3].a) * samples[1];
        color = color + samples[6] + (1. - samples[6].a) * samples[4];
        color = color + samples[7] + (1. - samples[7].a) * samples[5];
    } else if (dir == 4) {
        // +Z
        color = color + samples[0] + (1. - samples[0].a) * samples[4];
        color = color + samples[1] + (1. - samples[1].a) * samples[5];
        color = color + samples[2] + (1. - samples[2].a) * samples[6];
        color = color + samples[3] + (1. - samples[3].a) * samples[7];
    } else if (dir == 5) {
        // -Z
        color = color + samples[4] + (1. - samples[4].a) * samples[0];
        color = color + samples[5] + (1. - samples[5].a) * samples[1];
        color = color + samples[6] + (1. - samples[6].a) * samples[2];
        color = color + samples[7] + (1. - samples[7].a) * samples[3];
    }

    color = color * 0.25;
    textureStore(texture_out, vec3<i32>(id), color);
}

[[stage(compute), workgroup_size(8, 8, 8)]]
fn mipmap_0([[builtin(global_invocation_id)]] id: vec3<u32>) {
    mipmap(id, 0);
}

[[stage(compute), workgroup_size(8, 8, 8)]]
fn mipmap_1([[builtin(global_invocation_id)]] id: vec3<u32>) {
    mipmap(id, 1);
}

[[stage(compute), workgroup_size(8, 8, 8)]]
fn mipmap_2([[builtin(global_invocation_id)]] id: vec3<u32>) {
    mipmap(id, 2);
}

[[stage(compute), workgroup_size(8, 8, 8)]]
fn mipmap_3([[builtin(global_invocation_id)]] id: vec3<u32>) {
    mipmap(id, 3);
}

[[stage(compute), workgroup_size(8, 8, 8)]]
fn mipmap_4([[builtin(global_invocation_id)]] id: vec3<u32>) {
    mipmap(id, 4);
}

[[stage(compute), workgroup_size(8, 8, 8)]]
fn mipmap_5([[builtin(global_invocation_id)]] id: vec3<u32>) {
    mipmap(id, 5);
}

fn linear_index(index: vec3<i32>) -> i32 {
    var spatial = vec3<u32>(index);
    var morton = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let coord = (vec3<u32>(index) >> vec3<u32>(i)) & vec3<u32>(1u);
        let offset = 3u * i;

        morton = morton | (coord.x << offset);
        morton = morton | (coord.y << (offset + 1u));
        morton = morton | (coord.z << (offset + 2u));
    }

    return i32(morton);
}

fn unpack_color(voxel: u32) -> vec4<f32> {
    let unpacked =  unpack4x8unorm(voxel);
    let multiplier = unpacked.a * 255.0;
    let alpha = min(1.0, multiplier);
    return vec4<f32>(multiplier * unpacked.rgb, alpha);
}

[[stage(compute), workgroup_size(8, 8, 8)]]
fn clear([[builtin(global_invocation_id)]] id: vec3<u32>) {
    let coords = vec3<i32>(id);
    if (all(coords < textureDimensions(texture_out))) {
        let index = linear_index(coords);
        let voxel = &voxel_buffer.data[index];
        atomicStore(voxel, 0u);
    }
}

[[stage(compute), workgroup_size(8, 8, 8)]]
fn fill([[builtin(global_invocation_id)]] id: vec3<u32>) {
    let coords = vec3<i32>(id);
    if (all(coords < textureDimensions(texture_out))) {
        let index = linear_index(coords);
        let voxel = &voxel_buffer.data[index];
        let color = unpack_color(atomicLoad(voxel));
        textureStore(texture_out, coords, color);
    }
}