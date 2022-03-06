#import bevy_pbr::mesh_view_bind_group
#import bevy_pbr::mesh_struct

struct StandardMaterial {
    base_color: vec4<f32>;
    emissive: vec4<f32>;
    perceptual_roughness: f32;
    metallic: f32;
    reflectance: f32;
    // 'flags' is a bit field indicating various options. u32 is 32 bits so we have up to 32 options.
    flags: u32;
    alpha_cutoff: f32;
};

[[group(1), binding(0)]]
var<uniform> material: StandardMaterial;
[[group(1), binding(1)]]
var base_color_texture: texture_2d<f32>;
[[group(1), binding(2)]]
var base_color_sampler: sampler;
[[group(1), binding(3)]]
var emissive_texture: texture_2d<f32>;
[[group(1), binding(4)]]
var emissive_sampler: sampler;
[[group(1), binding(5)]]
var metallic_roughness_texture: texture_2d<f32>;
[[group(1), binding(6)]]
var metallic_roughness_sampler: sampler;
[[group(1), binding(7)]]
var occlusion_texture: texture_2d<f32>;
[[group(1), binding(8)]]
var occlusion_sampler: sampler;
[[group(1), binding(9)]]
var normal_map_texture: texture_2d<f32>;
[[group(1), binding(10)]]
var normal_map_sampler: sampler;

[[group(2), binding(0)]]
var<uniform> mesh: Mesh;

struct Volume {
    min: vec3<f32>;
    max: vec3<f32>;
};

[[group(3), binding(0)]]
var anisotropic_texture_0: texture_3d<f32>;
[[group(3), binding(1)]]
var anisotropic_texture_1: texture_3d<f32>;
[[group(3), binding(2)]]
var anisotropic_texture_2: texture_3d<f32>;
[[group(3), binding(3)]]
var anisotropic_texture_3: texture_3d<f32>;
[[group(3), binding(4)]]
var anisotropic_texture_4: texture_3d<f32>;
[[group(3), binding(5)]]
var anisotropic_texture_5: texture_3d<f32>;
[[group(3), binding(6)]]
var<uniform> volume: Volume;
[[group(3), binding(7)]]
var voxel_texture: texture_3d<f32>;
[[group(3), binding(8)]]
var texture_sampler: sampler;

var<private> voxel_size: f32;
var<private> max_level: f32;

let PI: f32 = 3.141592653589793;
let SQRT3: f32 = 1.732050808;

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

fn compute_roughness(perceptual_roughness: f32) -> f32 {
    let clamped = clamp(perceptual_roughness, 0.089, 1.0);
    return clamped * clamped;
}

fn cone(origin: vec3<f32>, direction: vec3<f32>, ratio: f32) -> vec4<f32> {
    var color = vec4<f32>(0.0);
    var distance = voxel_size * SQRT3;

    loop {
        let position = origin + distance * direction;
        if (any(position < vec3<f32>(0.)) || any(position > vec3<f32>(1., 1., 1.)) || color.a >= 1.0) {
            break;
        }

        let diameter = distance * ratio;
        let level = clamp(max_level + log2(diameter), 0.0, max_level);

        let weight = direction * direction;
        var face: vec3<u32>;
        face.x = u32(direction.x < 0.);
        face.y = u32(direction.y < 0.) + 2u;
        face.z = u32(direction.z < 0.) + 4u;

        let anisotropic_level = max(level - 1., 0.);

        var sample = vec4<f32>(0.);
        if (direction.x > 0.0) {
            sample = sample + weight.x * textureSampleLevel(anisotropic_texture_0, texture_sampler, position, anisotropic_level);
        } else {
            sample = sample + weight.x * textureSampleLevel(anisotropic_texture_1, texture_sampler, position, anisotropic_level);
        }
        if (direction.y > 0.0) {
            sample = sample + weight.y * textureSampleLevel(anisotropic_texture_2, texture_sampler, position, anisotropic_level);
        } else {
            sample = sample + weight.y * textureSampleLevel(anisotropic_texture_3, texture_sampler, position, anisotropic_level);
        }
        if (direction.z > 0.0) {
            sample = sample + weight.z * textureSampleLevel(anisotropic_texture_4, texture_sampler, position, anisotropic_level);
        } else {
            sample = sample + weight.z * textureSampleLevel(anisotropic_texture_5, texture_sampler, position, anisotropic_level);
        }

        if (level < 1.0) {
            let base_sample = textureSampleLevel(voxel_texture, texture_sampler, position, 0.0);
            sample = mix(base_sample, sample, level);
        }

        color = color + (1.0 - color.a) * sample;

        let step_size = max(diameter / 2.0, voxel_size);
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
    let N = normalize(in.world_normal);
    let origin = position + N * voxel_size * SQRT3;

    let tbn = normal_basis(N);
    let T = tbn[0];
    let B = tbn[1];

    let ratio = 1.0;
    var color = vec4<f32>(0.);
    color = color + cone(origin, N, ratio);
    // color = color + cone(origin, tbn * vec3<f32>(0.0, 0.866025, 0.5), ratio) * 0.15;
    // color = color + cone(origin, tbn * vec3<f32>(0.823639, 0.267617, 0.5), ratio) * 0.15;
    // color = color + cone(origin, tbn * vec3<f32>(0.509037, -0.700629, 0.5), ratio) * 0.15;
    // color = color + cone(origin, tbn * vec3<f32>(-0.509037, -0.700629, 0.5), ratio) * 0.15;
    // color = color + cone(origin, tbn * vec3<f32>(-0.823639, 0.267617, 0.5), ratio) * 0.15;

    color = color + cone(origin, normalize(N + T + B), ratio) * 0.707;
    color = color + cone(origin, normalize(N - T + B), ratio) * 0.707;
    color = color + cone(origin, normalize(N + T - B), ratio) * 0.707;
    color = color + cone(origin, normalize(N - T - B), ratio) * 0.707;
    
    var V: vec3<f32>;
    let is_orthographic = view.projection[3].w == 1.0;
    if (is_orthographic) {
        V = normalize(vec3<f32>(view.view_proj[0].z, view.view_proj[1].z, view.view_proj[2].z));
    } else {
        V = normalize(view.world_position.xyz - in.world_position.xyz);
    }

    let R = -reflect(V, N);

    let roughness = compute_roughness(material.perceptual_roughness);
    color = color + cone(origin, R, roughness);
    
    return vec4<f32>(color * 0.1);
}