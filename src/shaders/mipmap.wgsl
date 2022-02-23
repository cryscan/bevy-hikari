[[group(0), binding(0)]]
var voxel_texture_in: texture_storage_3d<rgba8unorm, read>;
[[group(0), binding(1)]]
var voxel_texture_out: texture_storage_3d<rgba8unorm, write>;

[[stage(compute), workgroup_size(4, 4, 4)]]
fn mipmap([[builtin(global_invocation_id)]] id: vec3<u32>) {
    if (any(vec3<i32>(id) >= textureDimensions(voxel_texture_out))) {
        return;
    }

    var colors: array<vec4<f32>, 8>;
    var alpha: f32 = 0.0;

    for (var i = 0u; i < 8u; i = i + 1u) {
        let offset = vec3<u32>(i & 1u, (i >> 1u) & 1u, (i >> 2u) & 1u);
        let coord = vec3<i32>(id * 2u + offset);
        colors[i] = textureLoad(voxel_texture_in, coord);
        alpha = alpha + colors[i].a;
    }

    var output_color = vec4<f32>(0.0);

    if (alpha > 0.0) {
        var color = vec3<f32>(0.0);
        for (var i = 0u; i < 8u; i = i + 1u) {
            color = color + colors[i].rgb * colors[i].a;
        }
        output_color = vec4<f32>(color / alpha, alpha);
    }

    textureStore(voxel_texture_out, vec3<i32>(id), output_color);
}