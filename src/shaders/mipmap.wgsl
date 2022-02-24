[[group(0), binding(0)]]
var voxel_texture_in: texture_3d<f32>;
[[group(0), binding(1)]]
var voxel_texture_out: texture_storage_3d<rgba8unorm, write>;
[[group(0), binding(2)]]
var voxel_texture_sampler: sampler;

[[stage(compute), workgroup_size(4, 4, 4)]]
fn mipmap([[builtin(global_invocation_id)]] id: vec3<u32>) {
    let dims = textureDimensions(voxel_texture_out);
    if (any(vec3<i32>(id) >= dims)) {
        return;
    }
    
    let coords = (vec3<f32>(id) + vec3<f32>(0.5)) / vec3<f32>(dims);
    let color = textureSampleLevel(voxel_texture_in, voxel_texture_sampler, coords, 0.0);

    textureStore(voxel_texture_out, vec3<i32>(id), color);
}

[[stage(compute), workgroup_size(4, 4, 4)]]
fn clear([[builtin(global_invocation_id)]] id: vec3<u32>) {
    if (any(vec3<i32>(id) >= textureDimensions(voxel_texture_out))) {
        return;
    }
    
    let clear_color = vec4<f32>(0.0);
    textureStore(voxel_texture_out, vec3<i32>(id), clear_color);
}