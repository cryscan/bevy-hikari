#import bevy_pbr::mesh_view_bindings
#import bevy_pbr::utils
#import bevy_pbr::lighting

#import bevy_hikari::mesh_material_bindings
#import bevy_hikari::deferred_bindings

#ifdef NO_TEXTURE
@group(3) @binding(0)
var textures: texture_2d<f32>;
@group(3) @binding(1)
var samplers: sampler;
#else
@group(3) @binding(0)
var textures: binding_array<texture_2d<f32>>;
@group(3) @binding(1)
var samplers: binding_array<sampler>;
#endif

struct Frame {
    number: u32,
    kernel: mat3x3<f32>,
};

@group(4) @binding(0)
var<uniform> frame: Frame;
@group(4) @binding(1)
var noise_texture: binding_array<texture_2d<f32>>;
@group(4) @binding(2)
var noise_sampler: sampler;

@group(5) @binding(0)
var denoised_texture_0: texture_storage_2d<rgba16float, read_write>;
@group(5) @binding(1)
var denoised_texture_1: texture_storage_2d<rgba16float, read_write>;
@group(5) @binding(2)
var denoised_texture_2: texture_storage_2d<rgba16float, read_write>;
@group(5) @binding(3)
var denoised_texture_3: texture_storage_2d<rgba16float, read_write>;
@group(5) @binding(4)
var render_texture: texture_storage_2d<rgba16float, read_write>;

// 64 Bytes
struct PackedReservoir {
    radiance: vec2<u32>,            // RGBA16F
    random: vec2<u32>,              // RGBA16F
    visible_position: vec4<f32>,    // RGBA32F
    sample_position: vec4<f32>,     // RGBA32F
    visible_normal: u32,            // RGBA8SN
    sample_normal: u32,             // RGBA8SN
    reservoir: vec2<u32>,           // RGBA16F
};

struct Reservoirs {
    data: array<PackedReservoir>,
};

@group(6) @binding(0)
var<storage, read> previous_reservoir_buffer: Reservoirs;
@group(6) @binding(1)
var<storage, read_write> reservoir_buffer: Reservoirs;

let TAU: f32 = 6.283185307;
let F32_EPSILON: f32 = 1.1920929E-7;
let F32_MAX: f32 = 3.402823466E+38;
let U32_MAX: u32 = 0xFFFFFFFFu;
let BVH_LEAF_FLAG: u32 = 0x80000000u;

let RAY_BIAS: f32 = 0.02;
let DISTANCE_MAX: f32 = 65535.0;
let VALIDATION_INTERVAL: u32 = 16u;
let NOISE_TEXTURE_COUNT: u32 = 64u;
let GOLDEN_RATIO: f32 = 1.618033989;

let DONT_SAMPLE_DIRECTIONAL_LIGHT: u32 = 0xFFFFFFFFu;
let DONT_SAMPLE_EMISSIVE: u32 = 0x80000000u;
let SAMPLE_ALL_EMISSIVE: u32 = 0xFFFFFFFFu;

let SOLAR_ANGLE: f32 = 0.130899694;

let MAX_TEMPORAL_REUSE_COUNT: f32 = 50.0;
let SPATIAL_REUSE_COUNT: u32 = 1u;
let SPATIAL_REUSE_RANGE: f32 = 30.0;

fn hash(value: u32) -> u32 {
    var state = value;
    state = state ^ 2747636419u;
    state = state * 2654435769u;
    state = state ^ state >> 16u;
    state = state * 2654435769u;
    state = state ^ state >> 16u;
    state = state * 2654435769u;
    return state;
}

fn random_float(value: u32) -> f32 {
    return f32(hash(value)) / 4294967295.0;
}

fn normal_basis(n: vec3<f32>) -> mat3x3<f32> {
    let s = min(sign(n.z) * 2.0 + 1.0, 1.0);
    let u = -1.0 / (s + n.z);
    let v = n.x * n.y * u;
    let t = vec3<f32>(1.0 + s * n.x * n.x * u, s * v, -s * n.x);
    let b = vec3<f32>(v, s + n.y * n.y * u, -n.y);
    return mat3x3<f32>(t, b, n);
}

// https://en.wikipedia.org/wiki/Halton_sequence#Implementation_in_pseudocode
fn halton(base: u32, index: u32) -> f32 {
    var result = 0.;
    var f = 1.;
    for (var id = index; id > 0u; id /= base) {
        f = f / f32(base);
        result += f * f32(id % base);
    }
    return result;
}

fn frame_jitter() -> vec2<f32> {
    let index = frame.number % 16u + 7u;
    let delta = vec2<f32>(halton(2u, index), halton(3u, index));
    return delta;
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    inv_direction: vec3<f32>,
};

struct Aabb {
    min: vec3<f32>,
    max: vec3<f32>,
};

struct Intersection {
    uv: vec2<f32>,
    distance: f32,
};

struct Hit {
    intersection: Intersection,
    instance_index: u32,
    primitive_index: u32,
};

struct Surface {
    base_color: vec4<f32>,
    emissive: vec4<f32>,
    reflectance: f32,
    metallic: f32,
    roughness: f32,
    occlusion: f32,
};

struct HitInfo {
    position: vec4<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
    instance_index: u32,
    material_index: u32,
};

struct LightCandidate {
    // Orientation + cos(half_angle)
    cone: vec4<f32>, 
    direction: vec3<f32>,
    max_distance: f32,
    min_distance: f32,
    directional_index: u32,
    emissive_index: u32,
    p: f32,
};

struct Sample {
    radiance: vec4<f32>,
    random: vec4<f32>,
    visible_position: vec4<f32>,
    visible_normal: vec3<f32>,
    visible_instance: u32,
    sample_position: vec4<f32>,
    sample_normal: vec3<f32>,
};

struct Reservoir {
    s: Sample,
    count: f32,
    w: f32,
    w_sum: f32,
    w2_sum: f32,
};

struct RestirOutput {
    radiance: vec3<f32>,
    output: vec3<f32>,
};

fn instance_position_world_to_local(instance: Instance, world_position: vec3<f32>) -> vec3<f32> {
    let inverse_model = transpose(instance.inverse_transpose_model);
    let position = inverse_model * vec4<f32>(world_position, 1.0);
    return position.xyz / position.w;
}

fn instance_direction_world_to_local(instance: Instance, world_direction: vec3<f32>) -> vec3<f32> {
    let inverse_model = transpose(instance.inverse_transpose_model);
    let direction = inverse_model * vec4<f32>(world_direction, 0.0);
    return direction.xyz;
}

fn intersects_aabb(ray: Ray, aabb: Aabb) -> f32 {
    let t1 = (aabb.min - ray.origin) * ray.inv_direction;
    let t2 = (aabb.max - ray.origin) * ray.inv_direction;

    var t_min = min(t1.x, t2.x);
    var t_max = max(t1.x, t2.x);

    t_min = max(t_min, min(t1.y, t2.y));
    t_max = min(t_max, max(t1.y, t2.y));

    t_min = max(t_min, min(t1.z, t2.z));
    t_max = min(t_max, max(t1.z, t2.z));

    var t: f32 = F32_MAX;
    if (t_max >= t_min && t_max >= 0.0) {
        t = t_min;
    }
    return t;
}

fn intersects_triangle(ray: Ray, tri: array<vec3<f32>, 3>) -> Intersection {
    var result: Intersection;
    result.distance = F32_MAX;

    // let a = tri[0];
    // let b = tri[1];
    // let c = tri[2];

    let ab = tri[1] - tri[0];
    let ac = tri[2] - tri[0];

    let u_vec = cross(ray.direction, ac);
    let det = dot(ab, u_vec);
    if (abs(det) < F32_EPSILON) {
        return result;
    }

    let inv_det = 1.0 / det;
    let ao = ray.origin - tri[0];
    let u = dot(ao, u_vec) * inv_det;
    if (u < 0.0 || u > 1.0) {
        result.uv = vec2<f32>(u, 0.0);
        return result;
    }

    let v_vec = cross(ao, ab);
    let v = dot(ray.direction, v_vec) * inv_det;
    result.uv = vec2<f32>(u, v);
    if (v < 0.0 || u + v > 1.0) {
        return result;
    }

    let distance = dot(ac, v_vec) * inv_det;
    if (distance > F32_EPSILON) {
        result.distance = distance;
    }

    return result;
}

fn traverse_bottom(hit: ptr<function, Hit>, ray: Ray, slice: Slice, early_distance: f32) -> bool {
    var intersected = false;
    var index = 0u;
    for (; index < slice.node_len;) {
        let node_index = slice.node_offset + index;
        let node = asset_node_buffer.data[node_index];
        var aabb: Aabb;
        if (node.entry_index >= BVH_LEAF_FLAG) {
            let primitive_index = slice.primitive + node.entry_index - BVH_LEAF_FLAG;
            let vertices = primitive_buffer.data[primitive_index].vertices;

            aabb.min = min(vertices[0], min(vertices[1], vertices[2]));
            aabb.max = max(vertices[0], max(vertices[1], vertices[2]));

            if (intersects_aabb(ray, aabb) < (*hit).intersection.distance) {
                let intersection = intersects_triangle(ray, vertices);
                if (intersection.distance < (*hit).intersection.distance) {
                    (*hit).intersection = intersection;
                    (*hit).primitive_index = primitive_index;
                    intersected = true;

                    if (intersection.distance < early_distance) {
                        return intersected;
                    }
                }
            }

            index = node.exit_index;
        } else {
            aabb.min = node.min;
            aabb.max = node.max;

            if (intersects_aabb(ray, aabb) < (*hit).intersection.distance) {
                index = node.entry_index;
            } else {
                index = node.exit_index;
            }
        }
    }

    return intersected;
}

fn traverse_top(ray: Ray, max_distance: f32, early_distance: f32) -> Hit {
    var hit: Hit;
    hit.intersection.distance = max_distance;
    hit.instance_index = U32_MAX;
    hit.primitive_index = U32_MAX;

    var index = 0u;
    for (; index < instance_node_buffer.count;) {
        let node = instance_node_buffer.data[index];
        var aabb: Aabb;

        if (node.entry_index >= BVH_LEAF_FLAG) {
            let instance_index = node.entry_index - BVH_LEAF_FLAG;
            let instance = instance_buffer.data[instance_index];
            aabb.min = instance.min;
            aabb.max = instance.max;

            if (intersects_aabb(ray, aabb) < hit.intersection.distance) {
                var r: Ray;
                r.origin = instance_position_world_to_local(instance, ray.origin);
                r.direction = instance_direction_world_to_local(instance, ray.direction);
                r.inv_direction = 1.0 / r.direction;

                if (traverse_bottom(&hit, r, instance.slice, early_distance)) {
                    hit.instance_index = instance_index;
                    if (hit.intersection.distance < early_distance) {
                        return hit;
                    }
                }
            }

            index = node.exit_index;
        } else {
            aabb.min = node.min;
            aabb.max = node.max;

            if (intersects_aabb(ray, aabb) < hit.intersection.distance) {
                index = node.entry_index;
            } else {
                index = node.exit_index;
            }
        }
    }

    return hit;
}

fn sample_cosine_hemisphere(rand: vec2<f32>) -> vec3<f32> {
    let r = sqrt(rand.x);
    let theta = 2.0 * PI * rand.y;
    var direction = vec3<f32>(r * cos(theta), r * sin(theta), 0.0);
    direction.z = sqrt(1.0 - dot(direction.xy, direction.xy));
    return direction;
}

fn sample_uniform_cone(rand: vec2<f32>, cos_angle: f32) -> vec3<f32> {
    let z = 1.0 - (1.0 - cos_angle) * rand.x;  // [cos(angle), 1.0]
    let theta = 2.0 * PI * rand.y;
    let r = sqrt(1.0 - z * z);
    return vec3<f32>(r * cos(theta), r * sin(theta), z);
}

// Choose a light source based on luminance
fn select_light_candidate(
    rand: vec4<f32>,
    position: vec3<f32>,
    normal: vec3<f32>,
    instance: u32,
) -> LightCandidate {
    var candidate: LightCandidate;
    candidate.max_distance = F32_MAX;
    candidate.min_distance = DISTANCE_MAX;
    candidate.directional_index = 0u;
    candidate.emissive_index = DONT_SAMPLE_EMISSIVE;

    var directional = lights.directional_lights[0];
    let cone = vec4<f32>(directional.direction_to_light, cos(SOLAR_ANGLE));
    let direction = normal_basis(cone.xyz) * sample_uniform_cone(rand.zw, cone.w);

    candidate.cone = cone;
    candidate.direction = direction;

    var sum_flux = luminance(directional.color.rgb) / TAU;
    var selected_flux = sum_flux;

    var rand_1d = fract(rand.x + rand.y);

    // for (var id = 0u; id < lights.n_directional_lights; id += 1u) {
    //     let directional = lights.directional_lights[id];
    //     let cone = vec4<f32>(directional.direction_to_light, cos(SOLAR_ANGLE));
    //     let direction = normal_basis(cone.xyz) * sample_uniform_cone(rand.zw, cone.w);

    //     let flux = luminance(directional.color.rgb) / TAU;
    //     sum_flux += flux;
    //     rand_1d = fract(rand_1d + GOLDEN_RATIO);
    //     if (rand_1d <= flux / sum_flux) {
    //         candidate.directional_index = id;
    //         candidate.emissive_index = DONT_SAMPLE_EMISSIVE;
    //         selected_flux = flux;
    //     }
    // }

    for (var id = 0u; id < light_source_buffer.count; id += 1u) {
        let source = light_source_buffer.data[id];
        let delta = source.position - position;
        let d2 = dot(delta, delta);
        let r2 = source.radius * source.radius;
        let d = sqrt(d2);

        let cone = vec4<f32>(normalize(delta), sqrt(max(d2 - r2, 0.0) / max(d2, 0.0001)));
        let sin = sqrt(1.0 - cone.w * cone.w);
        if (instance == source.instance || dot(cone.xyz, normal) < -sin) {
            continue;
        }
        let direction = normal_basis(cone.xyz) * sample_uniform_cone(rand.zw, cone.w);

        let flux = 255.0 * source.emissive.a * luminance(source.emissive.rgb) * (1.0 - cone.w);
        sum_flux += flux;
        rand_1d = fract(rand_1d + GOLDEN_RATIO);
        if (rand_1d <= flux / sum_flux) {
            candidate.directional_index = DONT_SAMPLE_DIRECTIONAL_LIGHT;
            candidate.emissive_index = source.instance;
            candidate.cone = cone;
            candidate.direction = direction;
            candidate.max_distance = d + source.radius;
            candidate.min_distance = d - source.radius;
            selected_flux = flux;
        }
    }

    candidate.p = selected_flux / sum_flux;
    candidate.p *= 0.5 / (1.0 - candidate.cone.w);
    return candidate;
}

fn light_candidate_pdf(candidate: LightCandidate, direction: vec3<f32>) -> f32 {
    return candidate.p * saturate(sign(dot(direction, candidate.cone.xyz) - candidate.cone.w));
}

// NOTE: Correctly calculates the view vector depending on whether
// the projection is orthographic or perspective.
fn calculate_view(
    world_position: vec4<f32>,
    is_orthographic: bool,
) -> vec3<f32> {
    var V: vec3<f32>;
    if (is_orthographic) {
        // Orthographic view vector
        V = normalize(vec3<f32>(view.view_proj[0].z, view.view_proj[1].z, view.view_proj[2].z));
    } else {
        // Only valid for a perpective projection
        V = normalize(view.world_position.xyz - world_position.xyz);
    }
    return V;
}

#ifdef NO_TEXTURE
fn retreive_surface(material_index: u32, uv: vec2<f32>) -> Surface {
    var surface: Surface;
    let material = material_buffer.data[material_index];

    surface.base_color = material.base_color;
    surface.emissive = material.emissive;
    surface.metallic = material.metallic;
    surface.occlusion = 1.0;
    surface.roughness = perceptualRoughnessToRoughness(material.perceptual_roughness);
    surface.reflectance = material.reflectance;

    return surface;
}

fn retreive_emissive(material_index: u32, uv: vec2<f32>) -> vec4<f32> {
    var emissive = material_buffer.data[material_index].emissive;
    return emissive;
}
#else
fn retreive_surface(material_index: u32, uv: vec2<f32>) -> Surface {
    var surface: Surface;
    let material = material_buffer.data[material_index];

    surface.base_color = material.base_color;
    var id = material.base_color_texture;
    if (id != U32_MAX) {
        surface.base_color *= textureSampleLevel(textures[id], samplers[id], uv, 0.0);
    }

    surface.emissive = material.emissive;
    id = material.emissive_texture;
    if (id != U32_MAX) {
        surface.emissive *= textureSampleLevel(textures[id], samplers[id], uv, 0.0);
    }

    surface.metallic = material.metallic;
    id = material.metallic_roughness_texture;
    if (id != U32_MAX) {
        surface.metallic *= textureSampleLevel(textures[id], samplers[id], uv, 0.0).r;
    }

    surface.occlusion = 1.0;
    id = material.occlusion_texture;
    if (id != U32_MAX) {
        surface.occlusion = textureSampleLevel(textures[id], samplers[id], uv, 0.0).r;
    }

    surface.roughness = perceptualRoughnessToRoughness(material.perceptual_roughness);
    surface.reflectance = material.reflectance;

    return surface;
}

fn retreive_emissive(material_index: u32, uv: vec2<f32>) -> vec4<f32> {
    let material = material_buffer.data[material_index];

    var emissive = material.emissive;
    let id = material.emissive_texture;
    if (id != U32_MAX) {
        emissive *= textureSampleLevel(textures[id], samplers[id], uv, 0.0);
    }

    return emissive;
}
#endif

fn hit_info(ray: Ray, hit: Hit) -> HitInfo {
    var info: HitInfo;
    info.instance_index = hit.instance_index;
    info.material_index = U32_MAX;

    if (hit.instance_index != U32_MAX) {
        let instance = instance_buffer.data[hit.instance_index];
        let indices = primitive_buffer.data[hit.primitive_index].indices;

        let v0 = vertex_buffer.data[(instance.slice.vertex + indices[0])];
        let v1 = vertex_buffer.data[(instance.slice.vertex + indices[1])];
        let v2 = vertex_buffer.data[(instance.slice.vertex + indices[2])];
        let uv = hit.intersection.uv;
        info.uv = v0.uv + uv.x * (v1.uv - v0.uv) + uv.y * (v2.uv - v0.uv);
        info.normal = v0.normal + uv.x * (v1.normal - v0.normal) + uv.y * (v2.normal - v0.normal);

        // info.surface = retreive_surface(instance.material, info.uv);
        info.position = vec4<f32>(ray.origin + ray.direction * hit.intersection.distance, 1.0);
        info.material_index = instance.material;
    } else {
        info.position = vec4<f32>(ray.origin + ray.direction * DISTANCE_MAX, 0.0);
    }

    return info;
}

fn empty_sample() -> Sample {
    var s: Sample;
    s.radiance = vec4<f32>(0.0);
    s.random = vec4<f32>(0.0);
    s.visible_position = vec4<f32>(0.0);
    s.visible_normal = vec3<f32>(0.0);
    s.sample_position = vec4<f32>(0.0);
    s.sample_normal = vec3<f32>(0.0);
    return s;
}

// fn load_previous_reservoir(coords: vec2<i32>) -> Reservoir {
//     var r: Reservoir;

//     let reservoir = textureLoad(previous_reservoir_texture, coords);
//     r.count = reservoir.x;
//     r.w_sum = reservoir.y;
//     r.w2_sum = reservoir.z;
//     r.w = reservoir.w;

//     r.s.radiance = textureLoad(previous_radiance_texture, coords);
//     r.s.random = textureLoad(previous_random_texture, coords);
//     r.s.visible_position = textureLoad(previous_visible_position_texture, coords);
//     r.s.visible_normal = textureLoad(previous_visible_normal_texture, coords).xyz;
//     r.s.visible_id = textureLoad(previous_visible_id_texture, coords).xy;
//     r.s.sample_position = textureLoad(previous_sample_position_texture, coords);
//     r.s.sample_normal = textureLoad(previous_sample_normal_texture, coords).xyz;

//     return r;
// }

// fn store_reservoir(coords: vec2<i32>, r: Reservoir) {
//     let reservoir = vec4<f32>(r.count, r.w_sum, r.w2_sum, r.w);
//     textureStore(reservoir_texture, coords, reservoir);

//     textureStore(radiance_texture, coords, r.s.radiance);
//     textureStore(random_texture, coords, r.s.random);
//     textureStore(visible_position_texture, coords, r.s.visible_position);
//     textureStore(visible_normal_texture, coords, vec4<f32>(r.s.visible_normal, 0.0));
//     textureStore(visible_id_texture, coords, vec4<u32>(r.s.visible_id, 0u, 0u));
//     textureStore(sample_position_texture, coords, r.s.sample_position);
//     textureStore(sample_normal_texture, coords, vec4<f32>(r.s.sample_normal, 0.0));
// }

fn unpack_reservoir(packed: PackedReservoir) -> Reservoir {
    var r: Reservoir;

    var t0: vec2<f32>;
    var t1: vec2<f32>;

    t0 = unpack2x16float(packed.reservoir.x);
    t1 = unpack2x16float(packed.reservoir.y);
    r.count = t0.x;
    r.w = t0.y;
    r.w_sum = t1.x;
    r.w2_sum = t1.y;

    t0 = unpack2x16float(packed.radiance.x);
    t1 = unpack2x16float(packed.radiance.y);
    r.s.radiance = vec4<f32>(t0, t1);

    t0 = unpack2x16unorm(packed.random.x);
    t1 = unpack2x16unorm(packed.random.y);
    r.s.random = vec4<f32>(t0, t1);

    r.s.visible_position = packed.visible_position;
    r.s.sample_position = packed.sample_position;

    r.s.visible_normal = normalize(unpack4x8snorm(packed.visible_normal).xyz);
    r.s.sample_normal = normalize(unpack4x8snorm(packed.sample_normal).xyz);

    r.s.visible_instance = ((packed.visible_normal >> 24u) & 0xFFu) << 8u;
    r.s.visible_instance |= (packed.sample_normal >> 24u) & 0xFFu;

    return r;
}

fn load_previous_reservoir(index: i32) -> Reservoir {
    var r: Reservoir;
    let packed = previous_reservoir_buffer.data[index];
    return unpack_reservoir(packed);
}

fn load_reservoir(index: i32) -> Reservoir {
    var r: Reservoir;
    let packed = reservoir_buffer.data[index];
    return unpack_reservoir(packed);
}

fn store_reservoir(index: i32, r: Reservoir) {
    var packed: PackedReservoir;

    var t0: u32;
    var t1: u32;

    t0 = pack2x16float(vec2<f32>(r.count, r.w));
    t1 = pack2x16float(vec2<f32>(r.w_sum, r.w2_sum));
    packed.reservoir = vec2<u32>(t0, t1);

    t0 = pack2x16float(r.s.radiance.xy);
    t1 = pack2x16float(r.s.radiance.zw);
    packed.radiance = vec2<u32>(t0, t1);

    t0 = pack2x16unorm(r.s.random.xy);
    t1 = pack2x16unorm(r.s.random.zw);
    packed.random = vec2<u32>(t0, t1);

    packed.visible_position = r.s.visible_position;
    packed.sample_position = r.s.sample_position;

    packed.visible_normal = pack4x8snorm(vec4<f32>(r.s.visible_normal, 0.0));
    packed.sample_normal = pack4x8snorm(vec4<f32>(r.s.sample_normal, 0.0));

    packed.visible_normal |= ((r.s.visible_instance >> 8u) & 0xFFu) << 24u;
    packed.sample_normal |= (r.s.visible_instance & 0xFFu) << 24u;

    reservoir_buffer.data[index] = packed;
}

fn set_reservoir(r: ptr<function, Reservoir>, s: Sample, w_new: f32) {
    (*r).count = 1.0;
    (*r).w_sum = w_new;
    (*r).w2_sum = w_new * w_new;
    (*r).s = s;
}

fn update_reservoir(
    r: ptr<function, Reservoir>,
    s: Sample,
    w_new: f32,
) {
    (*r).w_sum += w_new;
    (*r).w2_sum += w_new * w_new;
    (*r).count = (*r).count + 1.0;

    let rand = fract(dot(s.random, vec4<f32>(1.0)));
    if (rand < w_new / (*r).w_sum) {
        (*r).s = s;
    }
}

fn merge_reservoir(r: ptr<function, Reservoir>, other: Reservoir, p: f32) {
    let count = (*r).count;
    update_reservoir(r, other.s, p * other.w * other.count);
    (*r).count = count + other.count;
}

fn lit(
    radiance: vec3<f32>,
    diffuse_color: vec3<f32>,
    roughness: f32,
    F0: vec3<f32>,
    L: vec3<f32>,
    N: vec3<f32>,
    V: vec3<f32>,
) -> vec3<f32> {
    let H = normalize(L + V);
    let NoL = saturate(dot(N, L));
    let NoH = saturate(dot(N, H));
    let LoH = saturate(dot(L, H));
    let NdotV = max(dot(N, V), 0.0001);

    let R = reflect(-V, N);

    let diffuse = diffuse_color * Fd_Burley(roughness, NdotV, NoL, LoH);
    let specular_intensity = 1.0;
    let specular_light = specular(F0, roughness, H, NdotV, NoL, NoH, LoH, specular_intensity);

    return (specular_light + diffuse) * radiance * NoL;
}

fn ambient(
    diffuse_color: vec3<f32>,
    roughness: f32,
    occlusion: f32,
    F0: vec3<f32>,
    N: vec3<f32>,
    V: vec3<f32>,
) -> vec3<f32> {
    let NdotV = max(dot(N, V), 0.0001);

    let diffuse_ambient = EnvBRDFApprox(diffuse_color, 1.0, NdotV);
    let specular_ambient = EnvBRDFApprox(F0, roughness, NdotV);
    return occlusion * (diffuse_ambient + specular_ambient) * lights.ambient_color.rgb;
}

fn input_radiance(
    ray: Ray,
    info: HitInfo,
    directional_index: u32,
    emissive_index: u32,
) -> vec4<f32> {
    var radiance = vec3<f32>(0.0);
    var ambient = 0.0;

    if (info.instance_index == U32_MAX) {
        // Ray hits nothing, input radiance could be either directional or ambient
        ambient = 1.0;
        radiance = lights.ambient_color.rgb;

        if (directional_index < lights.n_directional_lights) {
            let directional = lights.directional_lights[directional_index];
            let cos_angle = dot(ray.direction, directional.direction_to_light);
            let cos_solar_angle = cos(SOLAR_ANGLE);

            ambient = saturate(sign(cos_solar_angle - cos_angle));
            radiance = directional.color.rgb / (TAU * (1.0 - cos_solar_angle));
            radiance *= 1.0 - ambient;
        }
    } else {
        // Input radiance is emissive, but bounced radiance is not added here
        if (emissive_index == SAMPLE_ALL_EMISSIVE || emissive_index == info.instance_index) {
            let emissive = retreive_emissive(info.material_index, info.uv);
            radiance = 255.0 * emissive.a * emissive.rgb;
        }
    }

    return vec4<f32>(radiance, 1.0 - ambient);
}

fn shading(
    V: vec3<f32>,
    N: vec3<f32>,
    L: vec3<f32>,
    surface: Surface,
    input_radiance: vec4<f32>,
) -> vec3<f32> {
    let base_color = surface.base_color.rgb;
    let reflectance = surface.reflectance;
    let roughness = surface.roughness;
    let metallic = surface.metallic;
    let occlusion = surface.occlusion;

    let F0 = 0.16 * reflectance * reflectance * (1.0 - metallic) + base_color * metallic;
    let diffuse_color = base_color * (1.0 - metallic);

    let lit_radiance = lit(input_radiance.rgb, diffuse_color, roughness, F0, L, N, V);
    let ambient_radiance = ambient(diffuse_color, roughness, occlusion, F0, N, V);
    return mix(lit_radiance, ambient_radiance, 1.0 - input_radiance.a);
}

fn env_brdf(
    V: vec3<f32>,
    N: vec3<f32>,
    surface: Surface,
) -> vec3<f32> {
    let base_color = surface.base_color.rgb;
    let reflectance = surface.reflectance;
    let roughness = surface.roughness;
    let metallic = surface.metallic;
    let occlusion = surface.occlusion;

    let NdotV = max(dot(N, V), 0.0001);
    let F0 = 0.16 * reflectance * reflectance * (1.0 - metallic) + base_color * metallic;
    let diffuse_color = base_color * (1.0 - metallic);

    let diffuse_ambient = EnvBRDFApprox(diffuse_color, 1.0, NdotV);
    let specular_ambient = EnvBRDFApprox(F0, roughness, NdotV);
    return occlusion * (diffuse_ambient + specular_ambient);
}

fn temporal_restir(
    r: ptr<function, Reservoir>,
    previous_uv: vec2<f32>,
    V: vec3<f32>,
    surface: Surface,
    s: Sample,
    pdf: f32,
    max_sample_count: f32
) -> RestirOutput {
    var out: RestirOutput;
    out.radiance = shading(
        V,
        s.visible_normal,
        normalize(s.sample_position.xyz - s.visible_position.xyz),
        surface,
        s.radiance
    );
    let w_new = luminance(out.radiance) / pdf;

    let uv_miss = any(abs(previous_uv - 0.5) > vec2<f32>(0.5));

    let depth_ratio = (*r).s.visible_position.w / s.visible_position.w;
    let depth_miss = depth_ratio > 2.0 || depth_ratio < 0.5;

    let instance_miss = (*r).s.visible_instance != s.visible_instance;
    let normal_miss = dot(s.visible_normal, (*r).s.visible_normal) < 0.866;

    if (uv_miss || depth_miss || instance_miss || normal_miss) {
        set_reservoir(r, s, w_new);
    } else {
        update_reservoir(r, s, w_new);
    }

    // Clamp...
    (*r).w_sum *= max_sample_count / max((*r).count, max_sample_count);
    (*r).w2_sum *= max_sample_count / max((*r).count, max_sample_count);
    (*r).count = min((*r).count, max_sample_count);

    out.radiance = shading(
        V,
        (*r).s.visible_normal,
        normalize((*r).s.sample_position.xyz - (*r).s.visible_position.xyz),
        surface,
        (*r).s.radiance
    );
    (*r).w = (*r).w_sum / max((*r).count * luminance(out.radiance), 0.0001);
    out.output = (*r).w * out.radiance;

    return out;
}

@compute @workgroup_size(8, 8, 1)
fn direct_lit(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let size = textureDimensions(render_texture);
    let uv = (vec2<f32>(invocation_id.xy) + frame_jitter()) / vec2<f32>(size);
    let coords = vec2<i32>(invocation_id.xy);

    var s = empty_sample();

    let deferred_coords = vec2<i32>(uv * vec2<f32>(textureDimensions(position_texture)));
    let position_depth = textureLoad(position_texture, deferred_coords, 0);
    let position = vec4<f32>(position_depth.xyz, 1.0);
    let depth = position_depth.w;

    if (depth < F32_EPSILON) {
        var r: Reservoir;
        set_reservoir(&r, s, 0.0);
        store_reservoir(coords.x + size.x * coords.y, r);

        textureStore(albedo_texture, coords, vec4<f32>(0.0));
        textureStore(render_texture, coords, vec4<f32>(0.0));

        return;
    }
    let view_direction = calculate_view(position, view.projection[3].w == 1.0);

    let normal = textureLoad(normal_texture, deferred_coords, 0).xyz;
    let instance_material = textureLoad(instance_material_texture, deferred_coords, 0).xy;
    let object_uv = textureLoad(uv_texture, deferred_coords, 0).xy;
    let velocity = textureLoad(velocity_texture, deferred_coords, 0).xy;

    let noise_id = frame.number % NOISE_TEXTURE_COUNT;
    let noise_size = textureDimensions(noise_texture[noise_id]);
    let noise_uv = (vec2<f32>(invocation_id.xy) + f32(frame.number) + 0.5) / vec2<f32>(noise_size);
    s.random = textureSampleLevel(noise_texture[noise_id], noise_sampler, noise_uv, 0.0);
    s.random = fract(s.random + f32(frame.number) * GOLDEN_RATIO);

    s.visible_position = vec4<f32>(position.xyz, depth);
    s.visible_normal = normal;
    s.visible_instance = instance_material.x;

    let surface = retreive_surface(instance_material.y, object_uv);
    textureStore(albedo_texture, coords, vec4<f32>(env_brdf(view_direction, normal, surface), 1.0));

    var ray: Ray;
    var hit: Hit;
    var info: HitInfo;

    let candidate = select_light_candidate(s.random, position.xyz, normal, instance_material.x);

    // Direct light sampling
    ray.origin = position.xyz + normal * RAY_BIAS;
    ray.direction = candidate.direction;
    ray.inv_direction = 1.0 / ray.direction;

    if (dot(candidate.direction, normal) > 0.0) {
        hit = traverse_top(ray, candidate.max_distance, candidate.min_distance);
        info = hit_info(ray, hit);

        s.sample_position = info.position;
        s.sample_normal = info.normal;

        s.radiance = input_radiance(ray, info, candidate.directional_index, candidate.emissive_index);
    } else {
        s.sample_position = vec4<f32>(ray.origin + DISTANCE_MAX * ray.direction, 0.0);
        s.sample_normal = -ray.direction;
    }

    // ReSTIR: Temporal
    var previous_uv = uv - velocity;
    let previous_coords = vec2<i32>(previous_uv * vec2<f32>(size));
    var r = load_previous_reservoir(previous_coords.x + size.x * previous_coords.y);
    let restir = temporal_restir(&r, previous_uv, view_direction, surface, s, candidate.p, 4.0 * MAX_TEMPORAL_REUSE_COUNT);
    store_reservoir(coords.x + size.x * coords.y, r);

    var output_color = 255.0 * surface.emissive.a * surface.emissive.rgb;
    output_color += restir.output;

    textureStore(render_texture, coords, vec4<f32>(output_color, 1.0));

    // let variance = (r.w2_sum - r.w_sum * r.w_sum / r.count) / r.count;
    // textureStore(variance_texture, coords, vec4<f32>(variance, variance / r.count, 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn indirect_lit_ambient(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let size = textureDimensions(render_texture);
    let uv = (vec2<f32>(invocation_id.xy) + frame_jitter()) / vec2<f32>(size);
    let coords = vec2<i32>(invocation_id.xy);

    var s = empty_sample();

    let deferred_coords = vec2<i32>(uv * vec2<f32>(textureDimensions(position_texture)));
    let position_depth = textureLoad(position_texture, deferred_coords, 0);
    let position = vec4<f32>(position_depth.xyz, 1.0);
    let depth = position_depth.w;

    if (depth < F32_EPSILON) {
        var r: Reservoir;
        set_reservoir(&r, s, 0.0);
        store_reservoir(coords.x + size.x * coords.y, r);

        textureStore(render_texture, coords, vec4<f32>(0.0));

        return;
    }
    let view_direction = calculate_view(position, view.projection[3].w == 1.0);

    let normal = normalize(textureLoad(normal_texture, deferred_coords, 0).xyz);
    let instance_material = textureLoad(instance_material_texture, deferred_coords, 0).xy;
    let object_uv = textureLoad(uv_texture, deferred_coords, 0).xy;
    let velocity = textureLoad(velocity_texture, deferred_coords, 0).xy;

    let noise_id = frame.number % NOISE_TEXTURE_COUNT;
    let noise_size = textureDimensions(noise_texture[noise_id]);
    let noise_uv = (vec2<f32>(invocation_id.xy) + f32(frame.number) + 0.5) / vec2<f32>(noise_size);
    s.random = textureSampleLevel(noise_texture[noise_id], noise_sampler, noise_uv, 0.0);
    s.random = fract(s.random + f32(frame.number) * GOLDEN_RATIO);

    s.visible_position = vec4<f32>(position.xyz, depth);
    s.visible_normal = normal;
    s.visible_instance = instance_material.x;

    var ray: Ray;
    var hit: Hit;
    var info: HitInfo;
    var surface: Surface;

    ray.origin = position.xyz + normal * RAY_BIAS;
    ray.direction = normal_basis(normal) * sample_cosine_hemisphere(s.random.xy);
    ray.inv_direction = 1.0 / ray.direction;
    let p1 = dot(ray.direction, normal);

    hit = traverse_top(ray, F32_MAX, 0.0);
    info = hit_info(ray, hit);
    s.sample_position = info.position;
    s.sample_normal = info.normal;

    // Only ambient radiance
    s.radiance = input_radiance(ray, info, DONT_SAMPLE_DIRECTIONAL_LIGHT, DONT_SAMPLE_EMISSIVE);

    // Second bounce: from sample position
    var p2 = 0.5;
    var bounce_radiance = vec3<f32>(0.0);
    if (hit.instance_index != U32_MAX) {
        let bounce_candidate = select_light_candidate(
            s.random,
            info.position.xyz,
            info.normal,
            info.instance_index
        );

        var bounce_ray: Ray;
        var bounce_info: HitInfo;

        bounce_ray.origin = info.position.xyz + info.normal * RAY_BIAS;
        bounce_ray.direction = bounce_candidate.direction;
        bounce_ray.inv_direction = 1.0 / bounce_ray.direction;
        p2 = bounce_candidate.p;

        if (dot(bounce_candidate.direction, info.normal) > 0.0) {
            hit = traverse_top(bounce_ray, bounce_candidate.max_distance, bounce_candidate.min_distance);
            bounce_info = hit_info(bounce_ray, hit);

            surface = retreive_surface(info.material_index, info.uv);
            let radiance = input_radiance(bounce_ray, bounce_info, bounce_candidate.directional_index, bounce_candidate.emissive_index);
            s.radiance += vec4<f32>(shading(-ray.direction, info.normal, bounce_ray.direction, surface, radiance), 0.0);
        }
    }

    surface = retreive_surface(instance_material.y, object_uv);

    // ReSTIR: Temporal
    let previous_uv = uv - velocity;
    let previous_coords = vec2<i32>(previous_uv * vec2<f32>(size));
    var r = load_previous_reservoir(previous_coords.x + size.x * previous_coords.y);
    let restir = temporal_restir(&r, previous_uv, view_direction, surface, s, p1 * p2, MAX_TEMPORAL_REUSE_COUNT);
    store_reservoir(coords.x + size.x * coords.y, r);

    let output_color = restir.output;
    textureStore(render_texture, coords, vec4<f32>(output_color, 1.0));

    // According to the ReSTIR PT papar, the variance of ReSTIR estimation is proportional to the variance of average w
    // let variance = (r.w2_sum - r.w_sum * r.w_sum / r.count) / max(1.0, r.count);
    // textureStore(variance_texture, coords, vec4<f32>(variance, variance / r.count, 0.0, 0.0));
}

// Normal-weighting function (4.4.1)
fn normal_weight(n0: vec3<f32>, n1: vec3<f32>) -> f32 {
    let exponent = 64.0;
    return pow(max(0.0, dot(n0, n1)), exponent);
}

// Depth-weighting function (4.4.2)
fn depth_weight(d0: f32, d1: f32, gradient: vec2<f32>, offset: vec2<i32>) -> f32 {
    let eps = 0.001;
    return exp((-abs(d0 - d1)) / (abs(dot(gradient, vec2<f32>(offset))) + eps));
}

// Luminance-weighting function (4.4.3)
fn luminance_weight(l0: f32, l1: f32, variance: f32) -> f32 {
    let strictness = 4.0;
    let eps = 0.001;
    return exp((-abs(l0 - l1)) / (strictness * variance + eps));
}

fn instance_weight(i0: u32, i1: u32) -> f32 {
    return f32(i0 == i1);
}

@compute @workgroup_size(8, 8, 1)
fn denoise_atrous(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let render_size = textureDimensions(render_texture);
    let output_size = textureDimensions(denoised_texture_0);

    let output_uv = (vec2<f32>(invocation_id.xy) + 0.5) / vec2<f32>(output_size);
    let output_coords = vec2<i32>(invocation_id.xy);

    let depth = textureLoad(position_texture, output_coords, 0).w;
    let depth_gradient = textureLoad(depth_gradient_texture, output_coords, 0).xy;
    let normal = normalize(textureLoad(normal_texture, output_coords, 0).xyz);
    let instance = textureLoad(instance_material_texture, output_coords, 0).x;

    let albedo = textureLoad(albedo_texture, output_coords);
    var irradiance = textureLoad(render_texture, output_coords).rgb / max(albedo.rgb, vec3<f32>(0.01));
    irradiance *= max(sign(albedo.rgb - vec3<f32>(0.01)), vec3<f32>(0.0));
    let lum = luminance(irradiance);

    let r = load_reservoir(output_coords.x + output_size.x * output_coords.y);

    var irradiance_sum = vec3<f32>(0.0);
    var w_sum = 0.0;

#ifdef DENOISER_LEVEL_0
    // Pass 0, stride 8
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let offset = vec2<i32>(x, y);
            let sample_coords = output_coords + offset * 8;
            if (any(sample_coords < vec2<i32>(0)) || any(sample_coords >= output_size)) {
                continue;
            }

            let sample_albedo = textureLoad(albedo_texture, sample_coords).rgb;
            irradiance = textureLoad(render_texture, sample_coords).rgb / max(sample_albedo, vec3<f32>(0.01));
            irradiance *= max(sign(sample_albedo - vec3<f32>(0.01)), vec3<f32>(0.0));

            let sample_normal = textureLoad(normal_texture, sample_coords, 0).xyz;
            let sample_depth = textureLoad(position_texture, sample_coords, 0).w;
            let sample_instance = textureLoad(instance_material_texture, sample_coords, 0).x;
            let sample_variance = 1.0 / clamp(r.w2_sum, 1.0, 10.0);
            let sample_luminance = luminance(irradiance);

            let w_normal = normal_weight(normal, sample_normal);
            let w_depth = depth_weight(depth, sample_depth, depth_gradient, offset);
            let w_instance = instance_weight(instance, sample_instance);
            let w_luminance = luminance_weight(lum, sample_luminance, sample_variance);

            let w = saturate(w_normal * w_depth * w_instance * w_luminance) * frame.kernel[y + 1][x + 1];

            irradiance_sum += irradiance * w;
            w_sum += w;
        }
    }

    w_sum = max(w_sum, 0.0001);
    textureStore(denoised_texture_0, output_coords, vec4<f32>(irradiance_sum / w_sum, w_sum));
#endif

#ifdef DENOISER_LEVEL_1
    // Pass 1, stride 4
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let offset = vec2<i32>(x, y);
            let sample_coords = output_coords + offset * 4;
            if (any(sample_coords < vec2<i32>(0)) || any(sample_coords >= output_size)) {
                continue;
            }

            irradiance = textureLoad(denoised_texture_0, sample_coords).rgb;
            let sample_normal = textureLoad(normal_texture, sample_coords, 0).xyz;
            let sample_depth = textureLoad(position_texture, sample_coords, 0).w;
            let sample_instance = textureLoad(instance_material_texture, sample_coords, 0).x;
            let sample_variance = 1.0 / clamp(r.w2_sum, 1.0, 10.0);
            let sample_luminance = luminance(irradiance);

            let w_normal = normal_weight(normal, sample_normal);
            let w_depth = depth_weight(depth, sample_depth, depth_gradient, offset);            
            let w_instance = instance_weight(instance, sample_instance);
            let w_luminance = luminance_weight(lum, sample_luminance, sample_variance);

            let w = saturate(w_normal * w_depth * w_instance * w_luminance) * frame.kernel[y + 1][x + 1];

            irradiance_sum += irradiance * w;
            w_sum += w;
        }
    }

    w_sum = max(w_sum, 0.0001);
    textureStore(denoised_texture_1, output_coords, vec4<f32>(irradiance_sum / w_sum, w_sum));
#endif

#ifdef DENOISER_LEVEL_2
    // Pass 2, stride 2
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let offset = vec2<i32>(x, y);
            let sample_coords = output_coords + offset * 2;
            if (any(sample_coords < vec2<i32>(0)) || any(sample_coords >= output_size)) {
                continue;
            }

            irradiance = textureLoad(denoised_texture_1, sample_coords).rgb;
            let sample_normal = textureLoad(normal_texture, sample_coords, 0).xyz;
            let sample_depth = textureLoad(position_texture, sample_coords, 0).w;
            let sample_instance = textureLoad(instance_material_texture, sample_coords, 0).x;
            let sample_variance = 1.0 / clamp(r.w2_sum, 1.0, 10.0);
            let sample_luminance = luminance(irradiance);

            let w_normal = normal_weight(normal, sample_normal);
            let w_depth = depth_weight(depth, sample_depth, depth_gradient, offset);
            let w_instance = instance_weight(instance, sample_instance);
            let w_luminance = luminance_weight(lum, sample_luminance, sample_variance);

            let w = saturate(w_normal * w_depth * w_instance * w_luminance) * frame.kernel[y + 1][x + 1];

            irradiance_sum += irradiance * w;
            w_sum += w;
        }
    }

    w_sum = max(w_sum, 0.0001);
    textureStore(denoised_texture_2, output_coords, vec4<f32>(irradiance_sum / w_sum, w_sum));
#endif

#ifdef DENOISER_LEVEL_3
    // Pass 3, stride 1
    for (var y = -1; y <= 1; y += 1) {
        for (var x = -1; x <= 1; x += 1) {
            let offset = vec2<i32>(x, y);
            let sample_coords = output_coords + offset * 1;
            if (any(sample_coords < vec2<i32>(0)) || any(sample_coords >= output_size)) {
                continue;
            }

            irradiance = textureLoad(denoised_texture_2, sample_coords).rgb;
            let sample_normal = textureLoad(normal_texture, sample_coords, 0).xyz;
            let sample_depth = textureLoad(position_texture, sample_coords, 0).w;
            let sample_instance = textureLoad(instance_material_texture, sample_coords, 0).x;
            let sample_variance = 1.0 / clamp(r.w2_sum, 1.0, 10.0);
            let sample_luminance = luminance(irradiance);

            let w_normal = normal_weight(normal, sample_normal);
            let w_depth = depth_weight(depth, sample_depth, depth_gradient, offset);
            let w_instance = instance_weight(instance, sample_instance);
            let w_luminance = luminance_weight(lum, sample_luminance, sample_variance);

            let w = saturate(w_normal * w_depth * w_instance * w_luminance) * frame.kernel[y + 1][x + 1];

            irradiance_sum += irradiance * w;
            w_sum += w;
        }
    }

    w_sum = max(w_sum, 0.0001);
    let color = vec4<f32>(albedo.rgb * irradiance_sum / w_sum, albedo.a);
    textureStore(denoised_texture_3, output_coords, color);
#endif
}