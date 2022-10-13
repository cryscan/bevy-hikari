@group(0) @binding(0)
var direct_render_texture: texture_2d<f32>;
@group(0) @binding(1)
var direct_render_sampler: sampler;
@group(0) @binding(2)
var indirect_render_texture: texture_2d<f32>;
@group(0) @binding(3)
var indirect_render_sampler: sampler;
@group(0) @binding(4)
var previous_direct_render_texture: texture_2d<f32>;
@group(0) @binding(5)
var previous_direct_render_sampler: sampler;
@group(0) @binding(6)
var previous_indirect_render_texture: texture_2d<f32>;
@group(0) @binding(7)
var previous_indirect_render_sampler: sampler;

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

fn fetch_previous_color(uv: vec2<f32>) -> vec4<f32> {
    let direct = textureSample(previous_direct_render_texture, previous_direct_render_sampler, uv);
    let indirect = textureSample(previous_indirect_render_texture, previous_indirect_render_sampler, uv);
    return tone_mapping(direct + indirect);
    
}

fn fetch_color(coords: vec2<i32>) -> vec4<f32> {
    let direct = textureLoad(direct_render_texture, coords, 0);
    let indirect = textureLoad(indirect_render_texture, coords, 0);
    return tone_mapping(direct + indirect);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    var uv = 0.5 * in.position.xy + 0.5;
    uv.y = 1.0 - uv.y;
    let coords = vec2<i32>(in.clip_position.xy);

    let velocity = textureLoad(velocity_texture, coords, 0).xy;
    let previous_uv = uv - velocity;
    let previous_color = fetch_previous_color(previous_uv);

    var antialiased = previous_color.rgb;
    var mix_rate = min(previous_color.w, 0.5);

    var colors: array<vec3<f32>, 9>;
    colors[0] = fetch_color(coords).rgb;
    antialiased = sqrt(mix(antialiased * antialiased, colors[0] * colors[0], mix_rate));

    colors[1] = fetch_color(coords + vec2<i32>(1, 0)).rgb;
    colors[2] = fetch_color(coords + vec2<i32>(-1, 0)).rgb;
    colors[3] = fetch_color(coords + vec2<i32>(0, 1)).rgb;
    colors[4] = fetch_color(coords + vec2<i32>(0, -1)).rgb;
    colors[5] = fetch_color(coords + vec2<i32>(1, 1)).rgb;
    colors[6] = fetch_color(coords + vec2<i32>(-1, 1)).rgb;
    colors[7] = fetch_color(coords + vec2<i32>(1, -1)).rgb;
    colors[8] = fetch_color(coords + vec2<i32>(-1, -1)).rgb;

    var color = vec4<f32>(0.0);
    return color;
}