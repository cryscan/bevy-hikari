#import bevy_pbr::mesh_view_bind_group
#import bevy_pbr::mesh_struct

[[group(2), binding(0)]]
var<uniform> mesh: Mesh;

struct IrradianceSample {
    irradiance: vec4<f32>;
    z: f32;
};

[[group(1), binding(0)]]
var irradiance_texture: texture_2d<f32>;
[[group(1), binding(1)]]
var irradiance_sampler: sampler;
[[group(1), binding(2)]]
var irradiance_depth: texture_depth_2d;
[[group(1), binding(3)]]
var irradiance_depth_sampler: sampler;
[[group(1), binding(4)]]
var albedo_texture: texture_2d<f32>;
[[group(1), binding(5)]]
var albedo_sampler: sampler;
[[group(1), binding(6)]]
var albedo_depth: texture_depth_multisampled_2d;

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] uv: vec2<f32>;
};

[[stage(vertex)]]
fn vertex([[builtin(vertex_index)]] id: u32) -> VertexOutput {
    var position: vec2<f32>;
    position.x = -2.0 * f32(id / 2u) + 1.0;
    position.y = -2.0 * f32(((id + 1u) % 4u) / 2u) + 1.0;

    var out: VertexOutput;
    out.uv = position * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    out.clip_position = vec4<f32>(position, 0.1, 1.0);
    return out;
}

fn compute_index(dims: vec2<i32>, uv: vec2<f32>) -> vec2<i32> {
    let scale = vec2<f32>(dims - vec2<i32>(1));
    return vec2<i32>(uv * scale);
}

fn biased_uv(dims: vec2<i32>, uv: vec2<f32>, bias: vec2<i32>) -> vec2<f32> {
    let scale = vec2<f32>(dims);
    return uv + vec2<f32>(bias) / scale;
}

fn linear_z(depth: f32) -> f32 {
    let b = -view.near / (view.far - view.near);
    let a = -b * view.far;
    return a / (1.0 - depth - b);
}

fn sample_irradiance(
    uv: vec2<f32>,
    bias: vec2<i32>
) -> IrradianceSample {
    var sample: IrradianceSample;

    let sample_uv = biased_uv(textureDimensions(irradiance_texture), uv, bias);
    sample.irradiance = textureSample(irradiance_texture, irradiance_sampler, sample_uv);

    let depth = textureSample(irradiance_depth, irradiance_depth_sampler, sample_uv);
    sample.z = linear_z(depth);

    return sample;
}

[[stage(fragment)]]
fn fragment(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    // var irradiance = textureSample(irradiance_texture, irradiance_sampler, in.uv);

    let albedo_index = compute_index(textureDimensions(albedo_depth), in.uv);

    var irradiance = vec4<f32>(0.0);
    var samples: array<IrradianceSample, 9>;

    // let n = textureNumSamples(albedo_depth);
    let n = 1;

    samples[0] = sample_irradiance(in.uv, vec2<i32>(0, 0));

    samples[1] = sample_irradiance(in.uv, vec2<i32>(1, 0));
    samples[2] = sample_irradiance(in.uv, vec2<i32>(-1, 0));
    samples[3] = sample_irradiance(in.uv, vec2<i32>(0, 1));
    samples[4] = sample_irradiance(in.uv, vec2<i32>(0, -1));

    samples[5] = sample_irradiance(in.uv, vec2<i32>(1, 1));
    samples[6] = sample_irradiance(in.uv, vec2<i32>(-1, 1));
    samples[7] = sample_irradiance(in.uv, vec2<i32>(1, -1));
    samples[8] = sample_irradiance(in.uv, vec2<i32>(-1, -1));

    for (var i = 0; i < n; i = i + 1) {
        let ref_z = linear_z(textureLoad(albedo_depth, albedo_index, i));

        var color = vec4<f32>(0.0);
        var total_weight = 0.0;

        for (var j = 0; j < 9; j = j + 1) {
            var weight = 0.0;
            if (j == 0) {
                weight = 0.5;
            } else if (j < 5) {
                weight = 1.0 / 9.0;
            } else {
                weight = 1.0 / 36.0;
            }
            weight = weight / (1.0 + pow(1000.0 * (samples[j].z - ref_z), 2.0));

            total_weight = total_weight + weight;
            color = color + samples[j].irradiance * weight;
        }
        color = color / total_weight;
        irradiance = irradiance + color;
    }
    irradiance = irradiance / f32(n);

    var base_color = textureSample(albedo_texture, albedo_sampler, in.uv);

    return irradiance * base_color;
}