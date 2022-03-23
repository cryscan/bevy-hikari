[[group(0), binding(0)]]
var texture_out_0: texture_storage_3d<rgba16float, write>;
[[group(0), binding(1)]]
var texture_out_1: texture_storage_3d<rgba16float, write>;
[[group(0), binding(2)]]
var texture_out_2: texture_storage_3d<rgba16float, write>;
[[group(0), binding(3)]]
var texture_out_3: texture_storage_3d<rgba16float, write>;
[[group(0), binding(4)]]
var texture_out_4: texture_storage_3d<rgba16float, write>;
[[group(0), binding(5)]]
var texture_out_5: texture_storage_3d<rgba16float, write>;
[[group(0), binding(6)]]
var texture_in: texture_3d<f32>;

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
    let location = vec3<i32>(id) * 2 + index;
    return textureLoad(texture_in, location, 0);
}

fn take_samples(id: vec3<u32>) -> array<vec4<f32>, 8> {
    var samples: array<vec4<f32>, 8>;
    samples[0] = sample_voxel(id, INDICES[0]);
    samples[1] = sample_voxel(id, INDICES[1]);
    samples[2] = sample_voxel(id, INDICES[2]);
    samples[3] = sample_voxel(id, INDICES[3]);
    samples[4] = sample_voxel(id, INDICES[4]);
    samples[5] = sample_voxel(id, INDICES[5]);
    samples[6] = sample_voxel(id, INDICES[6]);
    samples[7] = sample_voxel(id, INDICES[7]);
    return samples;
}

[[stage(compute), workgroup_size(8, 8, 8)]]
fn mipmap([[builtin(global_invocation_id)]] invocation_id: vec3<u32>) {
    let dims = textureDimensions(texture_out_0);
    let id = vec3<u32>(invocation_id.xy, invocation_id.z % u32(dims.z));
    let dir = invocation_id.z / u32(dims.z);

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
    if (dir == 0u) {
        // +X
        color = color + samples[0] + (1. - samples[0].a) * samples[1];
        color = color + samples[2] + (1. - samples[2].a) * samples[3];
        color = color + samples[4] + (1. - samples[4].a) * samples[5];
        color = color + samples[6] + (1. - samples[6].a) * samples[7];
        color = color * 0.25;
        textureStore(texture_out_0, vec3<i32>(id), color);
    } else if (dir == 1u) {
        // -X
        color = color + samples[1] + (1. - samples[1].a) * samples[0];
        color = color + samples[3] + (1. - samples[3].a) * samples[2];
        color = color + samples[5] + (1. - samples[5].a) * samples[4];
        color = color + samples[7] + (1. - samples[7].a) * samples[6];
        color = color * 0.25;
        textureStore(texture_out_1, vec3<i32>(id), color);
    } else if (dir == 2u) {
        // +Y
        color = color + samples[0] + (1. - samples[0].a) * samples[2];
        color = color + samples[1] + (1. - samples[1].a) * samples[3];
        color = color + samples[4] + (1. - samples[4].a) * samples[6];
        color = color + samples[5] + (1. - samples[5].a) * samples[7];
        color = color * 0.25;
        textureStore(texture_out_2, vec3<i32>(id), color);
    } else if (dir == 3u) {
        // -Y
        color = color + samples[2] + (1. - samples[2].a) * samples[0];
        color = color + samples[3] + (1. - samples[3].a) * samples[1];
        color = color + samples[6] + (1. - samples[6].a) * samples[4];
        color = color + samples[7] + (1. - samples[7].a) * samples[5];
        color = color * 0.25;
        textureStore(texture_out_3, vec3<i32>(id), color);
    } else if (dir == 4u) {
        // +Z
        color = color + samples[0] + (1. - samples[0].a) * samples[4];
        color = color + samples[1] + (1. - samples[1].a) * samples[5];
        color = color + samples[2] + (1. - samples[2].a) * samples[6];
        color = color + samples[3] + (1. - samples[3].a) * samples[7];
        color = color * 0.25;
        textureStore(texture_out_4, vec3<i32>(id), color);
    } else if (dir == 5u) {
        // -Z
        color = color + samples[4] + (1. - samples[4].a) * samples[0];
        color = color + samples[5] + (1. - samples[5].a) * samples[1];
        color = color + samples[6] + (1. - samples[6].a) * samples[2];
        color = color + samples[7] + (1. - samples[7].a) * samples[3];
        color = color * 0.25;
        textureStore(texture_out_5, vec3<i32>(id), color);
    }
}