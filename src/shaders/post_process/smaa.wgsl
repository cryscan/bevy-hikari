#import bevy_hikari::mesh_view_bindings
#import bevy_hikari::deferred_bindings
#import bevy_hikari::utils

#import bevy_core_pipeline::fullscreen_vertex_shader

@group(2) @binding(0)
var nearest_sampler: sampler;
@group(2) @binding(1)
var linear_sampler: sampler;

#ifdef BLIT
@group(3) @binding(0)
var render_texture: texture_2d<f32>;

@fragment 
fn blit(uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(render_texture, nearest_sampler, uv);
}
#else
@group(3) @binding(0)
var previous_render_texture: texture_2d<f32>;
@group(3) @binding(1)
var render_texture: texture_2d<f32>;
@group(3) @binding(2)
var nearest_velocity_texture: texture_2d<f32>;

@group(4) @binding(0)
var output_texture: texture_storage_2d<rgba16float, read_write>;

let TAU: f32 = 6.283185307;

// The following 3 functions are from Playdead
// https://github.com/playdeadgames/temporal/blob/master/Assets/Shaders/TemporalReprojection.shader
fn RGB_to_YCoCg(rgb: vec3<f32>) -> vec3<f32> {
    let y = (rgb.r / 4.0) + (rgb.g / 2.0) + (rgb.b / 4.0);
    let co = (rgb.r / 2.0) - (rgb.b / 2.0);
    let cg = (-rgb.r / 4.0) + (rgb.g / 2.0) - (rgb.b / 4.0);
    return vec3<f32>(y, co, cg);
}

fn YCoCg_to_RGB(ycocg: vec3<f32>) -> vec3<f32> {
    let r = ycocg.x + ycocg.y - ycocg.z;
    let g = ycocg.x + ycocg.z;
    let b = ycocg.x - ycocg.y - ycocg.z;
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn clip_towards_aabb_center(previous_color: vec3<f32>, current_color: vec3<f32>, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> vec3<f32> {
    let p_clip = 0.5 * (aabb_max + aabb_min);
    let e_clip = 0.5 * (aabb_max - aabb_min);
    let v_clip = previous_color - p_clip;
    let v_unit = v_clip.xyz / e_clip;
    let a_unit = abs(v_unit);
    let ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));
    return select(previous_color, p_clip + v_clip / ma_unit, ma_unit > 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn smaa_tu4x(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    // In this implementation, a thread computes 4 output pixels in a quad.
    // One of the pixel (c) on the diagonal can be fetched from render_texture,
    // the other (p) is reprojected from previous_render_texture.
    // The two left (p', c') are extrapolated using differential blend method.

    // Odd frame:
    //     -1  0  1  2
    //    +--+--+--+--+
    // -1 |  |  |cn|  |
    //  0 |  |p |c'|pe|
    //  1 |cw|p'|c |  |
    //  2 |  |ps|  |  |
    //    +--+--+--+--+

    // Even frame:
    //     -1  0  1  2
    //    +--+--+--+--+
    // -1 |  |  |pn|  |
    //  0 |  |c |p'|ce|
    //  1 |pw|c'|p |  |
    //  2 |  |cs|  |  |
    //    +--+--+--+--+
}

@compute @workgroup_size(8, 8, 1)
fn smaa_tu4x_extrapolation(@builtin(global_invocation_id) invocation_id: vec3<u32>) {}
#endif