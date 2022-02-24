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

let PI: f32 = 3.141592653589793;

fn out_of_volume(position: vec3<f32>) -> bool {
    return any(position < volume.min) || any(position > volume.max);
}

fn max_component(v: vec3<f32>) -> f32 {
    return max(max(v.x, v.y), v.z);
}

fn normal_basis(n: vec3<f32>) -> mat3x3<f32> {
    var b: vec3<f32>;
    var t: vec3<f32>;
    
    if (abs(n.y) > 0.999) {
        b = vec3<f32>(1., 0., 0.);
        t = vec3<f32>(0., 0., 1.);
    } else {
    	b = normalize(cross(n, vec3<f32>(0., 1., 0.)));
    	t = normalize(cross(b, n));
    }
    return mat3x3<f32>(t, b, n);
}

fn cone(origin: vec3<f32>, direction: vec3<f32>, half_angle: f32) -> vec4<f32> {
    var color = vec4<f32>(0.0);

    let dims = vec3<f32>(textureDimensions(voxel_texture));
    let extends = volume.max - volume.min;

    let extend = max_component(extends);
    let dim = max_component(dims);
    let max_level = log2(dim);

    let unit = extend / dim;
    let step_factor = 1.0 / max_component(direction);

    var distance = unit * step_factor;

    loop {
        let position = origin + distance * direction;
        if (out_of_volume(position) || color.a >= 1.0) {
            break;
        }

        let radius = distance * sin(half_angle);
        let coords = (position - volume.min) / extends;

        let diameter = 2.0 * radius / extend;
        let level = clamp(max_level + log2(diameter), 0.0, max_level);

        let sample = textureSampleLevel(voxel_texture, voxel_texture_sampler, coords, level);
        color = color + (1.0 - color.a) * sample;

        let step_size = max(radius, unit);
        distance = distance + step_size * step_factor;
    }

    return color;
}

fn cone_90(origin: vec3<f32>, direction: vec3<f32>) -> vec4<f32> {
    var color = vec4<f32>(0.0);

    let dims = vec3<f32>(textureDimensions(voxel_texture));
    let extends = volume.max - volume.min;
    let unit = max_component(extends / dims);
    let step_factor = 1.0 / max_component(direction);

    let max_level = log2(max_component(dims));
    var distance: f32 = 0.0;

    for (var level = 0u; level <= u32(max_level); level = level + 1u) {
        distance = distance + unit * pow(2.0, f32(level)) * step_factor;
        let position = origin + distance * direction;

        if (out_of_volume(position) || color.a >= 1.0) {
            break;
        }

        let coords = (position - volume.min) / extends;
        let sample = textureSampleLevel(voxel_texture, voxel_texture_sampler, coords, f32(level));
        color = color + sample * (1.0 - color.a);
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
    let dims = vec3<f32>(textureDimensions(voxel_texture));
    let extends = volume.max - volume.min;
    let unit = extends / dims;

    let position = in.world_position.xyz;
    let normal = normalize(in.world_normal);
    let origin = position + 2.0 * normal * unit / max_component(normal);
    let tbn = normal_basis(normal);

    let coords = (position - volume.min) / extends;

    var color = vec4<f32>(0.);
    color = color + cone(origin, normal, PI / 6.0) * 0.25;
    color = color + cone(origin, tbn * vec3<f32>(0.0, 0.866025, 0.5), PI / 6.0) * 0.15;
    color = color + cone(origin, tbn * vec3<f32>(0.823639, 0.267617, 0.5), PI / 6.0) * 0.15;
    color = color + cone(origin, tbn * vec3<f32>(0.509037, -0.700629, 0.5), PI / 6.0) * 0.15;
    color = color + cone(origin, tbn * vec3<f32>(-0.509037, -0.700629, 0.5), PI / 6.0) * 0.15;
    color = color + cone(origin, tbn * vec3<f32>(-0.823638, 0.267617, 0.5), PI / 6.0) * 0.15;

    return color;
}