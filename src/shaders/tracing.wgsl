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

let STANDARD_MATERIAL_FLAGS_BASE_COLOR_TEXTURE_BIT: u32 = 1u;
let STANDARD_MATERIAL_FLAGS_EMISSIVE_TEXTURE_BIT: u32 = 2u;
let STANDARD_MATERIAL_FLAGS_METALLIC_ROUGHNESS_TEXTURE_BIT: u32 = 4u;
let STANDARD_MATERIAL_FLAGS_OCCLUSION_TEXTURE_BIT: u32 = 8u;
let STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT: u32 = 16u;
let STANDARD_MATERIAL_FLAGS_UNLIT_BIT: u32 = 32u;
let STANDARD_MATERIAL_FLAGS_ALPHA_MODE_OPAQUE: u32 = 64u;
let STANDARD_MATERIAL_FLAGS_ALPHA_MODE_MASK: u32 = 128u;
let STANDARD_MATERIAL_FLAGS_ALPHA_MODE_BLEND: u32 = 256u;

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

fn cone(origin: vec3<f32>, direction: vec3<f32>, ratio: f32, max_distance: f32) -> vec4<f32> {
    var color = vec4<f32>(0.0);
    var distance = voxel_size * SQRT3;

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

fn cone_single(origin: vec3<f32>, direction: vec3<f32>, ratio: f32, max_distance: f32) -> vec4<f32> {
    var color = vec4<f32>(0.0);
    var distance = voxel_size * SQRT3;

    loop {
        let position = origin + distance * direction;
        if (any(position < vec3<f32>(0.)) || any(position > vec3<f32>(1., 1., 1.)) || color.a >= 1.0 || distance > max_distance) {
            break;
        }

        let diameter = distance * ratio;
        let level = clamp(max_level + log2(diameter), 0.0, max_level);

        let anisotropic_level = max(level - 1., 0.);

        var sample = vec4<f32>(0.);
        if (direction.x > 0.5) {
            sample = sample + textureSampleLevel(anisotropic_texture_0, texture_sampler, position, anisotropic_level);
        } else if (direction.x < -0.5) {
            sample = sample + textureSampleLevel(anisotropic_texture_1, texture_sampler, position, anisotropic_level);
        }
        if (direction.y > 0.5) {
            sample = sample + textureSampleLevel(anisotropic_texture_2, texture_sampler, position, anisotropic_level);
        } else if (direction.y < -0.5) {
            sample = sample + textureSampleLevel(anisotropic_texture_3, texture_sampler, position, anisotropic_level);
        }
        if (direction.z > 0.5) {
            sample = sample + textureSampleLevel(anisotropic_texture_4, texture_sampler, position, anisotropic_level);
        } else if (direction.z < -0.5) {
            sample = sample + textureSampleLevel(anisotropic_texture_5, texture_sampler, position, anisotropic_level);
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
    [[location(2)]] uv: vec2<f32>;
};

[[stage(fragment)]]
fn fragment(in: FragmentInput) -> [[location(0)]] vec4<f32> {
    let dims = vec3<f32>(textureDimensions(voxel_texture));
    voxel_size = 1.0 / max_component(dims);
    max_level = log2(max_component(dims));
    
    let position = normalize_position(in.world_position.xyz / in.world_position.w);
    let N = normalize(in.world_normal);
    let origin = position + N * voxel_size * SQRT3;
    
    var base_color: vec4<f32> = material.base_color;
    if ((material.flags & STANDARD_MATERIAL_FLAGS_BASE_COLOR_TEXTURE_BIT) != 0u) {
        base_color = base_color * textureSample(base_color_texture, base_color_sampler, in.uv);
    }

    if ((material.flags & STANDARD_MATERIAL_FLAGS_ALPHA_MODE_MASK) != 0u) {
        if (base_color.a >= material.alpha_cutoff) {
            base_color.a = 1.0;
        } else {
            discard;
        }
    }

#ifdef NOT_GI_RECEIVER
    return vec4<f32>(0.0);
#else

    var directions: array<vec3<f32>, 14>;
    directions[0] = vec3<f32>(1.0, 1.0, 1.0);
    directions[1] = vec3<f32>(1.0, -1.0, 1.0);
    directions[2] = vec3<f32>(1.0, 1.0, -1.0);
    directions[3] = vec3<f32>(1.0, -1.0, -1.0);
    directions[4] = vec3<f32>(-1.0, 1.0, 1.0);
    directions[5] = vec3<f32>(-1.0, -1.0, 1.0);
    directions[6] = vec3<f32>(-1.0, 1.0, -1.0);
    directions[7] = vec3<f32>(-1.0, -1.0, -1.0);
    directions[8] = vec3<f32>(1.0, 0.0, 0.0);
    directions[9] = vec3<f32>(0.0, 1.0, 0.0);
    directions[10] = vec3<f32>(0.0, 0.0, 1.0);
    directions[11] = vec3<f32>(-1.0, 0.0, 0.0);
    directions[12] = vec3<f32>(0.0, -1.0, 0.0);
    directions[13] = vec3<f32>(0.0, 0.0, -1.0);

#ifdef AMBIENT_OCCLUSION

    let ratio = 1.0;
    var color = vec4<f32>(0.);
    for (var i = 0u; i < 8u; i = i + 1u) {
        let direction = normalize(directions[i]);
        let factor = dot(N, direction);
        if (factor > 0.0) {
            color = color + cone(origin, direction, ratio, 0.02) * factor;
        }
    }
    return color * 0.2;

#else

    let ratio = 1.0;
    var color = vec4<f32>(0.);
    for (var i = 0u; i < 8u; i = i + 1u) {
        let direction = normalize(directions[i]);
        let factor = dot(N, direction);
        if (factor > 0.0) {
            color = color + cone(origin, direction, ratio, 0.3) * factor;
        }
    }
    for (var i = 8u; i < 14u; i = i + 1u) {
        let direction = directions[i];
        let factor = dot(N, direction);
        if (factor > 0.0) {
            color = color + cone_single(origin, direction, ratio, 0.3) * factor;
        }
    }
    color = color * 0.2;
    
    var V: vec3<f32>;
    let is_orthographic = view.projection[3].w == 1.0;
    if (is_orthographic) {
        V = normalize(vec3<f32>(view.view_proj[0].z, view.view_proj[1].z, view.view_proj[2].z));
    } else {
        V = normalize(view.world_position.xyz - in.world_position.xyz);
    }

    let R = -reflect(V, N);

    let roughness = clamp(material.perceptual_roughness, 0.1, 1.0);
    if (roughness < 0.5) {
        color = color + cone(origin, R, roughness, 1.0);
    }
    
    color = vec4<f32>(pow(color.rgb, vec3<f32>(1.0 / 2.2)), color.a);

    var output_color = color;
    return vec4<f32>(output_color.rgb, base_color.a);

#endif
#endif
}