struct Voxel {
    top: u32;
    bot: u32;
};

struct VoxelBuffer {
    data: array<Voxel>;
};

[[group(0), binding(0)]]
var texture_in: texture_3d<f32>;
[[group(0), binding(1)]]
var texture_out: texture_storage_3d<rgba8unorm, write>;
[[group(0), binding(2)]]
var<storage, read_write> voxel_buffer: VoxelBuffer;

var<workgroup> samples: array<vec3<f32>, 8>;

fn mipmap(id: vec3<u32>, dir: i32) {
    if (any(vec3<i32>(id) >= textureDimensions(texture_out))) {
        return;
    }
    
    let in_dims = textureDimensions(texture_in);
    let out_dims = textureDimensions(texture_out);

    var indices: array<vec3<i32>, 8>;
    indices[0] = vec3<i32>(0, 0, 0);
    indices[1] = vec3<i32>(1, 0, 0);
    indices[2] = vec3<i32>(0, 1, 0);
    indices[3] = vec3<i32>(1, 1, 0);
    indices[4] = vec3<i32>(0, 0, 1);
    indices[5] = vec3<i32>(1, 0, 1);
    indices[6] = vec3<i32>(0, 1, 1);
    indices[7] = vec3<i32>(1, 1, 1);

    var samples: array<vec4<f32>, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        var index = vec3<i32>(id) * 2 + indices[i];
        index = vec3<i32>(index.xy, index.z % in_dims.z);
        samples[i] = textureLoad(texture_in, index, 0);
    }

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

[[stage(compute), workgroup_size(4, 4, 4)]]
fn mipmap_0([[builtin(global_invocation_id)]] id: vec3<u32>) {
    mipmap(id, 0);
}

[[stage(compute), workgroup_size(4, 4, 4)]]
fn mipmap_1([[builtin(global_invocation_id)]] id: vec3<u32>) {
    mipmap(id, 1);
}

[[stage(compute), workgroup_size(4, 4, 4)]]
fn mipmap_2([[builtin(global_invocation_id)]] id: vec3<u32>) {
    mipmap(id, 2);
}

[[stage(compute), workgroup_size(4, 4, 4)]]
fn mipmap_3([[builtin(global_invocation_id)]] id: vec3<u32>) {
    mipmap(id, 3);
}

[[stage(compute), workgroup_size(4, 4, 4)]]
fn mipmap_4([[builtin(global_invocation_id)]] id: vec3<u32>) {
    mipmap(id, 4);
}

[[stage(compute), workgroup_size(4, 4, 4)]]
fn mipmap_5([[builtin(global_invocation_id)]] id: vec3<u32>) {
    mipmap(id, 5);
}

fn linear_index(index: vec3<i32>) -> i32 {
    let dims = textureDimensions(texture_out);
    return index.x + index.y * dims.x + index.z * dims.x * dims.y;
}

[[stage(compute), workgroup_size(4, 4, 4)]]
fn clear([[builtin(global_invocation_id)]] id: vec3<u32>) {
    let coords = vec3<i32>(id);
    if (all(coords < textureDimensions(texture_out))) {
        let index = linear_index(coords);
        let voxel = &voxel_buffer.data[index];
        (*voxel).top = 0u;
        (*voxel).bot = 0u;
    }
}

[[stage(compute), workgroup_size(4, 4, 4)]]
fn fill([[builtin(global_invocation_id)]] id: vec3<u32>) {
    let coords = vec3<i32>(id);
    if (all(coords < textureDimensions(texture_out))) {
        let index = linear_index(coords);
        let voxel = &voxel_buffer.data[index];
        
        var color: vec4<f32>;
        let top = (*voxel).top;
        let bot = (*voxel).bot;
        let mask = 0xffffu;
        color.r = f32(top >> 16u) / 255.;
        color.g = f32(top & mask) / 255.;
        color.b = f32(bot >> 16u) / 255.;
        color.a = f32(bot & mask) / 255.;

        if (color.a > 1.0) {
            color = color / color.a;
        }

        textureStore(texture_out, coords, color);
    }
}