#import bevy_hikari::utils

@group(0) @binding(0)
var input_texture: texture_storage_2d<rgba16float, read_write>;

// Source: Advanced VR Rendering, GDC 2015, Alex Vlachos, Valve, Slide 49
// https://media.steampowered.com/apps/valve/2015/Alex_Vlachos_Advanced_VR_Rendering_GDC2015.pdf
fn screen_space_dither(coords: vec2<f32>) -> vec3<f32> {
    var dither = vec3<f32>(dot(vec2<f32>(171.0, 231.0), coords)).xxx;
    dither = fract(dither.rgb / vec3<f32>(103.0, 71.0, 97.0)) - 0.5;
    return dither / 255.0;
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

struct FragmentOutput {
    @location(0) color: vec4<f32>,
}

@fragment
fn fragment(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;

    out.color = textureLoad(input_texture, vec2<i32>(in.clip_position.xy));
    out.color += vec4<f32>(screen_space_dither(in.clip_position.xy), 0.0);
    return out;
}