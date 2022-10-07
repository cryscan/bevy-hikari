@group(0) @binding(0)
var direct_render_texture: texture_2d<f32>;
@group(0) @binding(1)
var direct_render_sampler: sampler;
@group(0) @binding(2)
var indirect_render_texture: texture_2d<f32>;
@group(0) @binding(3)
var indirect_render_sampler: sampler;

#import bevy_hikari::deferred_bindings

// luminance coefficients from Rec. 709.
// https://en.wikipedia.org/wiki/Rec._709
fn luminance(v: vec3<f32>) -> f32 {
    return dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn change_luminance(c_in: vec3<f32>, l_out: f32) -> vec3<f32> {
    let l_in = luminance(c_in);
    return c_in * (l_out / l_in);
}

fn reinhard_luminance(color: vec3<f32>) -> vec3<f32> {
    let l_old = luminance(color);
    let l_new = l_old / (1.0 + l_old);
    return change_luminance(color, l_new);
}

fn tone_mapping(in: vec4<f32>) -> vec4<f32> {
    // tone_mapping
    return vec4<f32>(reinhard_luminance(in.rgb), in.a);
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) position: vec3<f32>,
};

@vertex
fn vertex(@location(0) position: vec3<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(position, 1.0);
    out.position = position;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    var uv = 0.5 * in.position.xy + 0.5;
    uv.y = 1.0 - uv.y;
    
    let position = textureSample(position_texture, position_sampler, uv);
    let direct_color = textureSample(direct_render_texture, direct_render_sampler, uv).rgb;
    let indirect_color = textureSample(indirect_render_texture, indirect_render_sampler, uv).rgb;
    let color = direct_color + indirect_color;

    return tone_mapping(vec4<f32>(color, position.a));
}