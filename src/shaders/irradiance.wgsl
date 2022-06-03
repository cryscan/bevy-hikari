#import bevy_pbr::mesh_view_bind_group
#import bevy_pbr::mesh_struct
#import bevy_hikari::volume_struct
#import bevy_hikari::standard_material

[[group(2), binding(0)]]
var<uniform> mesh: Mesh;

[[group(3), binding(0)]]
var voxel_texture_0: texture_3d<f32>;
[[group(3), binding(1)]]
var voxel_texture_1: texture_3d<f32>;
[[group(3), binding(2)]]
var voxel_texture_2: texture_3d<f32>;
[[group(3), binding(3)]]
var voxel_texture_3: texture_3d<f32>;
[[group(3), binding(4)]]
var voxel_texture_4: texture_3d<f32>;
[[group(3), binding(5)]]
var voxel_texture_5: texture_3d<f32>;
[[group(3), binding(6)]]
var voxel_texture: texture_3d<f32>;
[[group(3), binding(7)]]
var texture_sampler: sampler;
[[group(3), binding(8)]]
var<uniform> volume: Volume;

let PI: f32 = 3.141592653589793;

// luminance coefficients from Rec. 709.
// https://en.wikipedia.org/wiki/Rec._709
fn luminance(v: vec3<f32>) -> f32 {
    return dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn reinhard_luminance(color: vec3<f32>) -> vec3<f32> {
    let lum = luminance(color);
    let factor = 1.0 / (1.0 + lum);
    // return change_luminance(color, l_new);
    return color * factor;
}

fn max_component(v: vec3<f32>) -> f32 {
    return max(max(v.x, v.y), v.z);
}

fn normalized_position(v: vec3<f32>) -> vec3<f32> {
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

fn cone(origin: vec3<f32>, direction: vec3<f32>, ratio: f32, max_distance: f32, alpha_distance: f32) -> vec4<f32> {
    let voxel_size = 1.0 / f32(VOXEL_RESOLUTION);
    let max_level = f32(VOXEL_LEVELS);

    var color = vec4<f32>(0.0);
    var alpha = 0.0;
    var distance = voxel_size;

    loop {
        let position = origin + distance * direction;
        if (any(position < vec3<f32>(0.)) || any(position > vec3<f32>(1., 1., 1.)) || color.a >= 1.0 || distance > max_distance) {
            break;
        }

        let diameter = distance * ratio;
        let level = clamp(max_level + log2(diameter), 0.0, max_level);

        let weight = direction * direction;
        let anisotropic_level = max(level - 1., 0.);

        var sample = vec4<f32>(0.);
        if (direction.x > 0.0) {
            sample = sample + weight.x * textureSampleLevel(voxel_texture_0, texture_sampler, position, anisotropic_level);
        } else {
            sample = sample + weight.x * textureSampleLevel(voxel_texture_1, texture_sampler, position, anisotropic_level);
        }
        if (direction.y > 0.0) {
            sample = sample + weight.y * textureSampleLevel(voxel_texture_2, texture_sampler, position, anisotropic_level);
        } else {
            sample = sample + weight.y * textureSampleLevel(voxel_texture_3, texture_sampler, position, anisotropic_level);
        }
        if (direction.z > 0.0) {
            sample = sample + weight.z * textureSampleLevel(voxel_texture_4, texture_sampler, position, anisotropic_level);
        } else {
            sample = sample + weight.z * textureSampleLevel(voxel_texture_5, texture_sampler, position, anisotropic_level);
        }

        if (level < 1.0) {
            sample = mix(textureSampleLevel(voxel_texture, texture_sampler, position, 0.0), sample, level);
        }

        color = color + (1.0 - color.a) * sample;

        if (distance < alpha_distance) {
            alpha = color.a;
        }

        let step_size = max(diameter / 2.0, voxel_size);
        distance = distance + step_size;
    }

    color.a = alpha;
    return color;
}

// fn cone_single(origin: vec3<f32>, direction: vec3<f32>, ratio: f32, max_distance: f32, alpha_distance: f32) -> vec4<f32> {
//     let voxel_size = 1.0 / f32(VOXEL_RESOLUTION);
//     let max_level = f32(VOXEL_LEVELS);

//     var color = vec4<f32>(0.0);
//     var alpha = 0.0;
//     var distance = voxel_size;

//     loop {
//         let position = origin + distance * direction;
//         if (any(position < vec3<f32>(0.)) || any(position > vec3<f32>(1., 1., 1.)) || color.a >= 1.0 || distance > max_distance) {
//             break;
//         }

//         let diameter = distance * ratio;
//         let level = clamp(max_level + log2(diameter), 0.0, max_level);

//         let anisotropic_level = max(level - 1., 0.);

//         var sample = vec4<f32>(0.);
//         if (direction.x > 0.5) {
//             sample = sample + textureSampleLevel(voxel_texture_0, texture_sampler, position, anisotropic_level);
//         } else if (direction.x < -0.5) {
//             sample = sample + textureSampleLevel(voxel_texture_1, texture_sampler, position, anisotropic_level);
//         }
//         if (direction.y > 0.5) {
//             sample = sample + textureSampleLevel(voxel_texture_2, texture_sampler, position, anisotropic_level);
//         } else if (direction.y < -0.5) {
//             sample = sample + textureSampleLevel(voxel_texture_3, texture_sampler, position, anisotropic_level);
//         }
//         if (direction.z > 0.5) {
//             sample = sample + textureSampleLevel(voxel_texture_4, texture_sampler, position, anisotropic_level);
//         } else if (direction.z < -0.5) {
//             sample = sample + textureSampleLevel(voxel_texture_5, texture_sampler, position, anisotropic_level);
//         }

//         if (level < 1.0) {
//             let base_sample = sample_voxel(position);
//             sample = mix(base_sample, sample, level);
//         }

//         color = color + (1.0 - color.a) * sample;

//         if (distance < alpha_distance) {
//             alpha = color.a;
//         }

//         let step_size = max(diameter / 2.0, voxel_size);
//         distance = distance + step_size;
//     }

//     color.a = alpha;
//     return color;
// }

struct FragmentInput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] world_position: vec4<f32>;
    [[location(1)]] world_normal: vec3<f32>;
    [[location(2)]] uv: vec2<f32>;
};

[[stage(fragment)]]
fn fragment(in: FragmentInput) -> [[location(0)]] vec4<f32> {
    let voxel_size = 1.0 / f32(VOXEL_RESOLUTION);

    let position = normalized_position(in.world_position.xyz / in.world_position.w);
    let N = normalize(in.world_normal);
    let origin = position + N * voxel_size;

#ifdef NOT_GI_RECEIVER
    return vec4<f32>(0.0);
#else
    let ratio = 1.0;
    let max_distance = 1.0;
    let alpha_distance = 0.02;
    var color = vec4<f32>(0.);

    // for (var i = 0u; i < 8u; i = i + 1u) {
    //     let direction = directions.data[i];
    //     let factor = dot(N, direction);
    //     if (factor > 0.0) {
    //         color = color + cone(origin, direction, ratio, max_distance, alpha_distance) * factor;
    //     }
    // }
    // for (var i = 8u; i < 14u; i = i + 1u) {
    //     let direction = directions.data[i];
    //     let factor = dot(N, direction);
    //     if (factor > 0.0) {
    //         color = color + cone_single(origin, direction, ratio, max_distance, alpha_distance) * factor;
    //     }
    // }

    let tbn = normal_basis(N);
    color = color + cone(origin, N, ratio, max_distance, alpha_distance);
    color = color + cone(origin, vec3<f32>(0.707, 0.0, 0.707) * tbn, ratio, max_distance, alpha_distance) * 0.707;
    color = color + cone(origin, vec3<f32>(-0.707, 0.0, 0.707) * tbn, ratio, max_distance, alpha_distance) * 0.707;
    color = color + cone(origin, vec3<f32>(0.0, 0.707, 0.707) * tbn, ratio, max_distance, alpha_distance) * 0.707;
    color = color + cone(origin, vec3<f32>(0.0, -0.707, 0.707) * tbn, ratio, max_distance, alpha_distance) * 0.707;
    color = color * 0.2;

    // let roughness = clamp(material.perceptual_roughness, 0.01, 1.0);
    // if (roughness < 0.5) {
    //     var V: vec3<f32>;
    //     let is_orthographic = view.projection[3].w == 1.0;
    //     if (is_orthographic) {
    //         V = normalize(vec3<f32>(view.view_proj[0].z, view.view_proj[1].z, view.view_proj[2].z));
    //     } else {
    //         V = normalize(view.world_position.xyz - in.world_position.xyz);
    //     }

    //     let R = -reflect(V, N);

    //     color = color + cone(origin, R, roughness, 1.0);
    // }

    return color;
#endif
}