#import bevy_pbr::mesh_view_bind_group
#import bevy_pbr::mesh_struct

[[group(2), binding(0)]]
var<uniform> mesh: Mesh;

struct Volume {
    min: vec3<f32>;
    max: vec3<f32>;
};

[[group(1), binding(0)]]
var<uniform> volume: Volume;
[[group(1), binding(1)]]
var voxel_texture: texture_3d<f32>;
[[group(1), binding(2)]]
var voxel_texture_sampler: sampler;

let DIRECTIONAL_LIGHT_SHADOW_CONE_HALF_ANGLE: f32 = 0.01;

fn out_of_volume(position: vec3<f32>) -> bool {
    return any(position < volume.min) || any(position > volume.max);
}

fn max_component(v: vec3<f32>) -> f32 {
    return max(max(v.x, v.y), v.z);
}

fn cone(origin: vec3<f32>, direction: vec3<f32>, half_angle: f32, normal: vec3<f32>) -> vec4<f32> {
    var color = vec4<f32>(0.0);

    let dims = vec3<f32>(textureDimensions(voxel_texture));
    let extends = volume.max - volume.min;

    let extend = max_component(extends);
    let dim = max_component(dims);
    let max_level = log2(dim);

    let bias = extend / dim;
    var distance = bias / max_component(direction);
    let normal_bias = bias * normal / max_component(normal);

    loop {
        let position = origin + normal_bias + distance * direction;
        if (out_of_volume(position) || color.a >= 1.0) {
            break;
        }

        let coords = (position - volume.min) / extends;

        let radius = distance * sin(half_angle);
        let unit_radius = radius / extend;
        let level = clamp(max_level + log2(unit_radius * 2.0), 0.0, max_level);

        let sample = textureSampleLevel(voxel_texture, voxel_texture_sampler, coords, level);
        color = color + (1.0 - color.a) * sample;

        let step_size = min(radius, bias);
        distance = distance + step_size / max_component(direction);
    }

    return color;
}

struct FragmentInput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] world_position: vec4<f32>;
    [[location(1)]] world_normal: vec3<f32>;
};

[[stage(fragment)]]
fn fragment(in: FragmentInput) -> [[location(0)]] vec4<f32> {
    // For each directional light, shoot a shadow cone
    var occlusion = 0.0;
    for (var i = 0u; i < lights.n_directional_lights; i = i + 1u) {
        let light = lights.directional_lights[i];
        let direction = normalize(light.direction_to_light);
        let color = cone(in.world_position.xyz, direction, DIRECTIONAL_LIGHT_SHADOW_CONE_HALF_ANGLE, in.world_normal);
        occlusion = occlusion + color.a;
    }
    occlusion = occlusion / f32(min(lights.n_directional_lights, 1u));
    
    return vec4<f32>(0.0, 0.0, 0.0, occlusion);
}