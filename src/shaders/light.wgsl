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
    kernel: array<vec3<f32>, 25>,
};

@group(4) @binding(0)
var<uniform> frame: Frame;
@group(4) @binding(1)
var render_texture: texture_storage_2d<rgba16float, write>;
@group(4) @binding(2)
var noise_texture: binding_array<texture_2d<f32>>;
@group(4) @binding(3)
var noise_sampler: sampler;

@group(5) @binding(0)
var reservoir_texture: texture_storage_2d<rgba32float, read_write>;
@group(5) @binding(1)
var radiance_texture: texture_storage_2d<rgba16float, read_write>;
@group(5) @binding(2)
var random_texture: texture_storage_2d<rgba16float, read_write>;
@group(5) @binding(3)
var visible_position_texture: texture_storage_2d<rgba32float, read_write>;
@group(5) @binding(4)
var visible_normal_texture: texture_storage_2d<rgba8snorm, read_write>;
@group(5) @binding(5)
var sample_position_texture: texture_storage_2d<rgba32float, read_write>;
@group(5) @binding(6)
var sample_normal_texture: texture_storage_2d<rgba8snorm, read_write>;

@group(6) @binding(0)
var previous_reservoir_texture: texture_storage_2d<rgba32float, read_write>;
@group(6) @binding(1)
var previous_radiance_texture: texture_storage_2d<rgba16float, read_write>;
@group(6) @binding(2)
var previous_random_texture: texture_storage_2d<rgba16float, read_write>;
@group(6) @binding(3)
var previous_visible_position_texture: texture_storage_2d<rgba32float, read_write>;
@group(6) @binding(4)
var previous_visible_normal_texture: texture_storage_2d<rgba8snorm, read_write>;
@group(6) @binding(5)
var previous_sample_position_texture: texture_storage_2d<rgba32float, read_write>;
@group(6) @binding(6)
var previous_sample_normal_texture: texture_storage_2d<rgba8snorm, read_write>;

let TAU: f32 = 6.283185307;
let F32_EPSILON: f32 = 1.1920929E-7;
let F32_MAX: f32 = 3.402823466E+38;
let U32_MAX: u32 = 0xFFFFFFFFu;
let BVH_LEAF_FLAG: u32 = 0x80000000u;

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
    directional_index: u32,
    emissive_index: u32,
    p: f32,
};

struct Sample {
    radiance: vec4<f32>,
    random: vec4<f32>,
    visible_position: vec4<f32>,
    visible_normal: vec3<f32>,
    sample_position: vec4<f32>,
    sample_normal: vec3<f32>,
};

struct Reservoir {
    s: Sample,
    w: f32,
    w_sum: f32,
    count: f32,
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

fn traverse_bottom(ray: Ray, slice: Slice, hit: ptr<function, Hit>) -> bool {
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

fn traverse_top(ray: Ray) -> Hit {
    var hit: Hit;
    hit.intersection.distance = F32_MAX;
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

                if (traverse_bottom(r, instance.slice, &hit)) {
                    hit.instance_index = instance_index;
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

fn load_previous_reservoir(coords: vec2<i32>) -> Reservoir {
    var r: Reservoir;

    let reservoir = textureLoad(previous_reservoir_texture, coords);
    r.w_sum = reservoir.r;
    r.count = reservoir.g;
    r.w = reservoir.b;

    r.s.radiance = textureLoad(previous_radiance_texture, coords);
    r.s.random = textureLoad(previous_random_texture, coords);
    r.s.visible_position = textureLoad(previous_visible_position_texture, coords);
    r.s.visible_normal = textureLoad(previous_visible_normal_texture, coords).rgb;
    r.s.sample_position = textureLoad(previous_sample_position_texture, coords);
    r.s.sample_normal = textureLoad(previous_sample_normal_texture, coords).rgb;

    return r;
}

fn store_reservoir(coords: vec2<i32>, r: Reservoir) {
    let reservoir = vec4<f32>(r.w_sum, r.count, r.w, 0.0);
    textureStore(reservoir_texture, coords, reservoir);

    textureStore(radiance_texture, coords, r.s.radiance);
    textureStore(random_texture, coords, r.s.random);
    textureStore(visible_position_texture, coords, r.s.visible_position);
    textureStore(visible_normal_texture, coords, vec4<f32>(r.s.visible_normal, 0.0));
    textureStore(sample_position_texture, coords, r.s.sample_position);
    textureStore(sample_normal_texture, coords, vec4<f32>(r.s.sample_normal, 0.0));
}

fn set_reservoir(r: ptr<function, Reservoir>, s: Sample, w_new: f32) {
    (*r).count = 1.0;
    (*r).w_sum = w_new;
    (*r).s = s;
}

fn update_reservoir(
    invocation_id: vec3<u32>,
    r: ptr<function, Reservoir>,
    s: Sample,
    w_new: f32,
) {
    (*r).w_sum += w_new;
    (*r).count = (*r).count + 1.0;

    let rand = random_float(invocation_id.x << 16u ^ invocation_id.y ^ hash(frame.number));
    if (rand < w_new / (*r).w_sum) {
        (*r).s = s;
    }
}

fn merge_reservoir(invocation_id: vec3<u32>, r: ptr<function, Reservoir>, other: Reservoir, p: f32) {
    let count = (*r).count;
    update_reservoir(invocation_id, r, other.s, p * other.w * other.count);
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

        if (directional_index < lights.n_directional_lights) {
            let directional = lights.directional_lights[directional_index];
            let cos_angle = dot(ray.direction, directional.direction_to_light);
            let cos_solar_angle = cos(SOLAR_ANGLE);

            let directional_condition = saturate(sign(cos_angle - cos_solar_angle));
            radiance = directional.color.rgb / (TAU * (1.0 - cos_solar_angle));
            radiance *= directional_condition;
            ambient = 1.0 - directional_condition;
        }
    } else {
        // Input radiance is emissive, but bounced radiance is not added here
        if (emissive_index == SAMPLE_ALL_EMISSIVE || emissive_index == info.instance_index) {
            let emissive = retreive_emissive(info.material_index, info.uv);
            radiance = 255.0 * emissive.a * emissive.rgb;
        }
    }

    return vec4<f32>(radiance, ambient);
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
    return mix(lit_radiance, ambient_radiance, input_radiance.a);
}

@compute @workgroup_size(8, 8, 1)
fn direct_lit(@builtin(global_invocation_id) invocation_id: vec3<u32>,) {
    let size = textureDimensions(render_texture);
    let uv = (vec2<f32>(invocation_id.xy) + 0.5) / vec2<f32>(size);
    let coords = vec2<i32>(invocation_id.xy);

    var s = empty_sample();

    let position = textureSampleLevel(position_texture, position_sampler, uv, 0.0);
    if (position.w < 0.5) {
        var r: Reservoir;
        set_reservoir(&r, s, 0.0);
        store_reservoir(coords, r);
        textureStore(render_texture, coords, vec4<f32>(0.0));
        return;
    }
    let ndc = view.view_proj * position;
    let depth = ndc.z / ndc.w;
    let view_direction = calculate_view(position, view.projection[3].w == 1.0);

    let normal = textureSampleLevel(normal_texture, normal_sampler, uv, 0.0).xyz;
    let instance_material = textureLoad(instance_material_texture, coords, 0).xy;
    let object_uv = textureSampleLevel(uv_texture, uv_sampler, uv, 0.0).xy;
    let velocity = textureSampleLevel(velocity_texture, velocity_sampler, uv, 0.0).xy * 0.0001;
    var surface: Surface;

    let noise_id = frame.number % NOISE_TEXTURE_COUNT;
    let noise_size = textureDimensions(noise_texture[noise_id]).xy;
    let noise_uv = (vec2<f32>(invocation_id.xy) + f32(frame.number) + 0.5) / vec2<f32>(noise_size);
    s.random = textureSampleLevel(noise_texture[noise_id], noise_sampler, noise_uv, 0.0);
    s.random = fract(s.random + f32(frame.number) * GOLDEN_RATIO);

    s.visible_position = vec4<f32>(position.xyz, depth);
    s.visible_normal = normal;

    var ray: Ray;
    var hit: Hit;
    var info: HitInfo;

    let candidate = select_light_candidate(s.random, position.xyz, normal, instance_material.x);

    // ray.origin = position.xyz + normal * light.shadow_normal_bias;
    // ray.direction = normal_basis(normal) * sample_cosine_hemisphere(s.random.xy);
    // ray.inv_direction = 1.0 / ray.direction;
    // let p1 = mix(dot(ray.direction, normal), light_candidate_pdf(candidate, ray.direction), 0.5);
    // // let p1 = dot(ray.direction, normal);

    // var hit = traverse_top(ray);
    // info = hit_info(ray, hit);
    // s.sample_position = info.position;
    // s.sample_normal = info.normal;

    // // Second bounce: from sample position
    // var p2 = 0.5;
    // var bounce_radiance = vec3<f32>(0.0);
    // if (hit.instance_index != U32_MAX) {
    //     let bounce_candidate = select_light_candidate(
    //         fract(s.random + f32(frame.number) * GOLDEN_RATIO), 
    //         light, 
    //         info.position.xyz, 
    //         info.normal, 
    //         info.instance_index
    //     );

    //     var bounce_ray: Ray;
    //     var bounce_info: HitInfo;

    //     bounce_ray.origin = info.position.xyz + info.normal * light.shadow_normal_bias;
    //     bounce_ray.direction = bounce_candidate.direction;
    //     bounce_ray.inv_direction = 1.0 / bounce_ray.direction;
    //     p2 = bounce_candidate.p;

    //     if (dot(bounce_candidate.direction, info.normal) > 0.0) {
    //         hit = traverse_top(bounce_ray);
    //         bounce_info = hit_info(bounce_ray, hit);

    //         surface = retreive_surface(info.material_index, info.uv);
    //         bounce_radiance = shading(
    //             ray.direction,
    //             info.normal,
    //             bounce_ray,
    //             light,
    //             surface,
    //             bounce_info,
    //             (bounce_candidate.instance_index == DONT_SAMPLE_EMISSIVE),
    //             bounce_candidate.instance_index,
    //             vec3<f32>(0.0)
    //         );
    //     }
    // }

    surface = retreive_surface(instance_material.y, object_uv);
    // s.radiance += shading(
    //     view_direction,
    //     normal,
    //     ray,
    //     light,
    //     surface,
    //     info,
    //     true,
    //     SAMPLE_ALL_EMISSIVE,
    //     bounce_radiance
    // );

    // Direct light sampling
    ray.origin = position.xyz + normal * lights.directional_lights[0].shadow_normal_bias;
    ray.direction = candidate.direction;
    ray.inv_direction = 1.0 / ray.direction;
    // let p3 = mix(candidate.p, saturate(dot(ray.direction, normal)), 0.5);
    // let p3 = candidate.p;

    if (dot(candidate.direction, normal) > 0.0) {
        hit = traverse_top(ray);
        info = hit_info(ray, hit);

        s.sample_position = info.position;
        s.sample_normal = info.normal;

        s.radiance = input_radiance(ray, info, candidate.directional_index, candidate.emissive_index);
    } else {
        s.sample_position = vec4<f32>(ray.origin + DISTANCE_MAX * ray.direction, 0.0);
        s.sample_normal = -ray.direction;
    }

    // ReSTIR: Temporal
    let previous_uv = uv - velocity;
    let previous_coords = vec2<i32>(previous_uv * vec2<f32>(size));
    var r = load_previous_reservoir(previous_coords);

    // let p = luminance(s.radiance) / p3;    
    var output_radiance = shading(view_direction, s.visible_normal, ray.direction, surface, s.radiance);
    let p = luminance(output_radiance) / candidate.p;

    let uv_miss = any(abs(previous_uv - 0.5) > vec2<f32>(0.5));
    let depth_miss = abs(r.s.visible_position.w / s.visible_position.w - 1.0) > 0.1;
    let normal_miss = dot(s.visible_normal, r.s.visible_normal) < 0.866;
    if (uv_miss || depth_miss || normal_miss) {
        set_reservoir(&r, s, p);
    } else {
        update_reservoir(invocation_id, &r, s, p);
    }

    // Clamp...
    r.w_sum *= MAX_TEMPORAL_REUSE_COUNT / max(r.count, MAX_TEMPORAL_REUSE_COUNT);
    r.count = min(r.count, MAX_TEMPORAL_REUSE_COUNT);    
    
    output_radiance = shading(
        view_direction,
        r.s.visible_normal,
        normalize(r.s.sample_position.xyz - r.s.visible_position.xyz),
        surface,
        r.s.radiance
    );
    r.w = r.w_sum / max(r.count * luminance(output_radiance), 0.0001);

    store_reservoir(coords, r);

    var output_color = 255.0 * surface.emissive.a * surface.emissive.rgb;
    output_color += r.w * output_radiance;

    textureStore(render_texture, coords, vec4<f32>(output_color, 1.0));
}