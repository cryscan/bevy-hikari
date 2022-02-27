[[group(0), binding(0)]]
var texture_in: texture_3d<f32>;
[[group(0), binding(1)]]
var texture_out: texture_storage_3d<rgba8unorm, write>;
[[group(0), binding(2)]]
var texture_sampler: sampler;

[[stage(compute), workgroup_size(4, 4, 24)]]
fn mipmap([[builtin(global_invocation_id)]] id: vec3<u32>) {
    if (any(vec3<i32>(id) >= textureDimensions(texture_out))) {
        return;
    }
    
    let in_dims = vec3<f32>(textureDimensions(texture_in));
    let out_dims = vec3<f32>(textureDimensions(texture_out));
    var coords = vec3<f32>(id) / out_dims;
    let dir = u32(floor(coords.z * 6.0));

    var indices: array<vec3<f32>, 8>;
    indices[0] = vec3<f32>(0., 0., 0.);
    indices[1] = vec3<f32>(1., 0., 0.);
    indices[2] = vec3<f32>(0., 1., 0.);
    indices[3] = vec3<f32>(1., 1., 0.);
    indices[4] = vec3<f32>(0., 0., 1.);
    indices[5] = vec3<f32>(1., 0., 1.);
    indices[6] = vec3<f32>(0., 1., 1.);
    indices[7] = vec3<f32>(1., 1., 1.);

    let blocks = in_dims.z / max(in_dims.x, in_dims.y);
    var samples: array<vec4<f32>, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        var sample_coords = coords + indices[i] / out_dims;
        sample_coords = (6. / blocks) * sample_coords;
        samples[i] = textureSampleLevel(texture_in, texture_sampler, sample_coords, 0.0);
    }

    var color = vec4<f32>(0.);
    if (dir == 0u) {    
        // +X
        color = color + samples[0] + (1. - samples[0].a) * samples[1];
        color = color + samples[2] + (1. - samples[2].a) * samples[3];
        color = color + samples[4] + (1. - samples[4].a) * samples[5];
        color = color + samples[6] + (1. - samples[6].a) * samples[7];
    } else if (dir == 1u) {
        // -X
        color = color + samples[1] + (1. - samples[1].a) * samples[0];
        color = color + samples[3] + (1. - samples[3].a) * samples[2];
        color = color + samples[5] + (1. - samples[5].a) * samples[4];
        color = color + samples[7] + (1. - samples[7].a) * samples[6];
    } else if (dir == 2u) {
        // +Y
        color = color + samples[0] + (1. - samples[0].a) * samples[2];
        color = color + samples[1] + (1. - samples[1].a) * samples[3];
        color = color + samples[4] + (1. - samples[4].a) * samples[6];
        color = color + samples[5] + (1. - samples[5].a) * samples[7];
    } else if (dir == 3u) {
        // -Y
        color = color + samples[2] + (1. - samples[2].a) * samples[0];
        color = color + samples[3] + (1. - samples[3].a) * samples[1];
        color = color + samples[6] + (1. - samples[6].a) * samples[4];
        color = color + samples[7] + (1. - samples[7].a) * samples[5];
    } else if (dir == 4u) {
        // +Z
        color = color + samples[0] + (1. - samples[0].a) * samples[4];
        color = color + samples[1] + (1. - samples[1].a) * samples[5];
        color = color + samples[2] + (1. - samples[2].a) * samples[6];
        color = color + samples[3] + (1. - samples[3].a) * samples[7];
    } else if (dir == 5u) {
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
fn clear([[builtin(global_invocation_id)]] id: vec3<u32>) {
    if (all(vec3<i32>(id) >= textureDimensions(texture_out))) {
        let clear_color = vec4<f32>(0.0);
        textureStore(texture_out, vec3<i32>(id), clear_color);
    }
}