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

var<private> voxel_size: f32;
var<private> max_level: f32;

let PI: f32 = 3.141592653589793;

fn max_component(v: vec3<f32>) -> f32 {
    return max(max(v.x, v.y), v.z);
}

fn normalize_position(v: vec3<f32>) -> vec3<f32> {
    return (v - volume.min) / (volume.max - volume.min);
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

fn cone(origin: vec3<f32>, direction: vec3<f32>, ratio: f32) -> vec4<f32> {
    var color = vec4<f32>(0.0);
    var distance = voxel_size;

    loop {
        let position = origin + distance * direction;
        if (any(position < vec3<f32>(0.)) || any(position > vec3<f32>(1.)) || color.a >= 1.0) {
            break;
        }

        let diameter = distance * ratio;
        let level = clamp(max_level + log2(diameter), 0.0, max_level);
        let sample = textureSampleLevel(voxel_texture, voxel_texture_sampler, position, level);
        color = color + (1.0 - color.a) * sample;

        let step_size = max(diameter, voxel_size);
        distance = distance + step_size;
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
    voxel_size = 1.0 / max_component(dims);
    max_level = log2(max_component(dims));
    
    let position = normalize_position(in.world_position.xyz / in.world_position.w);
    let normal = normalize(in.world_normal);
    let origin = position + normal * voxel_size * 2.0;

    let tbn = normal_basis(normal);

    var color = vec4<f32>(0.);
    color = color + cone(origin, normal, 1.0);
    color = color + cone(origin, normalize(normal + tbn[0]), 1.0) * 0.707;
    color = color + cone(origin, normalize(normal - tbn[0]), 1.0) * 0.707;
    color = color + cone(origin, normalize(normal + tbn[1]), 1.0) * 0.707;
    color = color + cone(origin, normalize(normal - tbn[1]), 1.0) * 0.707;
    
    return vec4<f32>(color);
}