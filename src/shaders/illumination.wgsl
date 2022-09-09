#import bevy_pbr::mesh_view_bindings
#import bevy_hikari::ray_tracing_bindings

@group(2) @binding(0)
var depth_texture: texture_depth_2d;
@group(2) @binding(1)
var depth_sampler: sampler;
@group(2) @binding(2)
var normal_velocity_texture: texture_2d<f32>;
@group(2) @binding(3)
var normal_velocity_sampler: sampler;

struct Input {
    @builtin(global_invocation_id) invocation_id: vec3<u32>;
    @builtin(num_workgroups) workgroups: vec3<u32>,
};

@compute @workgroup_size(8, 8, 1)
fn direct(in: Input) {
    let uv = vec2<f32>(in.invocation_id.xy) / vec2<f32>(in.workgroups.xy * 8u);
    let depth = textureSample(depth_texture, depth_sampler, uv);
    let normal_velocity = textureSample(normal_velocity_texture, normal_velocity_sampler, uv);
}