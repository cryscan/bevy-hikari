#import bevy_hikari::mesh_view_bindings
#import bevy_hikari::utils
#import bevy_pbr::utils
#import bevy_pbr::lighting

#import bevy_hikari::mesh_material_bindings
#import bevy_hikari::deferred_bindings
#import bevy_hikari::reservoir_bindings

#import bevy_hikari::reservoir_functions

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

@group(4) @binding(0)
var noise_texture: binding_array<texture_2d<f32>>;
@group(4) @binding(1)
var noise_sampler: sampler;

@group(5) @binding(0)
var albedo_texture: texture_storage_2d<rgba16float, read_write>;
@group(5) @binding(1)
var variance_texture: texture_storage_2d<r32float, read_write>;
@group(5) @binding(2)
var render_texture: texture_storage_2d<rgba16float, read_write>;

let TAU: f32 = 6.283185307;
let INV_TAU: f32 = 0.159154943;

let F32_EPSILON: f32 = 1.1920929E-7;
let F32_MAX: f32 = 3.402823466E+38;
let U32_MAX: u32 = 0xFFFFFFFFu;
let BVH_LEAF_FLAG: u32 = 0x80000000u;

let RAY_BIAS: f32 = 0.02;
let DISTANCE_MAX: f32 = 65535.0;
let NOISE_TEXTURE_COUNT: u32 = 16u;
let GOLDEN_RATIO: f32 = 1.618033989;

let DONT_SAMPLE_DIRECTIONAL_LIGHT: u32 = 0xFFFFFFFFu;
let DONT_SAMPLE_EMISSIVE: u32 = 0x80000000u;
let SAMPLE_ALL_EMISSIVE: u32 = 0xFFFFFFFFu;

#ifdef INCLUDE_EMISSIVE
let SPATIAL_REUSE_COUNT: u32 = 8u;
let SPATIAL_REUSE_RANGE: f32 = 10.0;
#else
let SPATIAL_REUSE_COUNT: u32 = 16u;
let SPATIAL_REUSE_RANGE: f32 = 20.0;
#endif
let SPATIAL_REUSE_TAPS: u32 = 4u;

let DIRECT_VALIDATION_FRAME_SAMPLE_THRESHOLD: u32 = 16u;
let SPATIAL_VARIANCE_SAMPLE_THRESHOLD: u32 = 16u;

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
    emissive_index: u32,
    p: f32,
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
    if t_max >= t_min && t_max >= 0.0 {
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
    if abs(det) < F32_EPSILON {
        return result;
    }

    let inv_det = 1.0 / det;
    let ao = ray.origin - tri[0];
    let u = dot(ao, u_vec) * inv_det;
    if u < 0.0 || u > 1.0 {
        result.uv = vec2<f32>(u, 0.0);
        return result;
    }

    let v_vec = cross(ao, ab);
    let v = dot(ray.direction, v_vec) * inv_det;
    result.uv = vec2<f32>(u, v);
    if v < 0.0 || u + v > 1.0 {
        return result;
    }

    let distance = dot(ac, v_vec) * inv_det;
    if distance > F32_EPSILON {
        result.distance = distance;
    }

    return result;
}

fn traverse_bottom(hit: ptr<function, Hit>, ray: Ray, mesh: MeshIndex, early_distance: f32) -> bool {
    var intersected = false;
    var index = 0u;
    for (; index < mesh.node_len;) {
        let node_index = mesh.node_offset + index;
        let node = asset_node_buffer.data[node_index];
        var aabb: Aabb;
        if node.entry_index >= BVH_LEAF_FLAG {
            let primitive_index = mesh.primitive + node.entry_index - BVH_LEAF_FLAG;
            let vertices = primitive_buffer.data[primitive_index].vertices;

            aabb.min = min(vertices[0], min(vertices[1], vertices[2]));
            aabb.max = max(vertices[0], max(vertices[1], vertices[2]));

            if intersects_aabb(ray, aabb) < (*hit).intersection.distance {
                let intersection = intersects_triangle(ray, vertices);
                if intersection.distance < (*hit).intersection.distance {
                    (*hit).intersection = intersection;
                    (*hit).primitive_index = primitive_index;
                    intersected = true;

                    if intersection.distance < early_distance {
                        return intersected;
                    }
                }
            }

            index = node.exit_index;
        } else {
            aabb.min = node.min;
            aabb.max = node.max;

            if intersects_aabb(ray, aabb) < (*hit).intersection.distance {
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

        if node.entry_index >= BVH_LEAF_FLAG {
            let instance_index = node.entry_index - BVH_LEAF_FLAG;
            let instance = instance_buffer.data[instance_index];
            aabb.min = instance.min;
            aabb.max = instance.max;

            if intersects_aabb(ray, aabb) < hit.intersection.distance {
                var r: Ray;
                r.origin = instance_position_world_to_local(instance, ray.origin);
                r.direction = instance_direction_world_to_local(instance, ray.direction);
                r.inv_direction = 1.0 / r.direction;

                if traverse_bottom(&hit, r, instance.mesh, early_distance) {
                    hit.instance_index = instance_index;
                    if hit.intersection.distance < early_distance {
                        return hit;
                    }
                }
            }

            index = node.exit_index;
        } else {
            aabb.min = node.min;
            aabb.max = node.max;

            if intersects_aabb(ray, aabb) < hit.intersection.distance {
                index = node.entry_index;
            } else {
                index = node.exit_index;
            }
        }
    }

    return hit;
}

fn sample_uniform_disk(rand: vec2<f32>) -> vec2<f32> {
    let r = sqrt(rand.x);
    let theta = 2.0 * PI * rand.y;
    return vec2<f32>(r * cos(theta), r * sin(theta));

    // let t = 2.0 * rand - 1.0;
    // if (all(abs(t) < vec2<f32>(F32_EPSILON))) {
    //     return vec2<f32>(0.0);
    // }

    // var theta: f32;
    // var r: f32;
    // if (abs(t.x) > abs(t.y)) {
    //     r = t.x;
    //     theta = 0.25 * PI * (t.y / t.x);
    // } else {
    //     r = t.y;
    //     theta = 0.25 * PI * (2.0 - t.x / t.y);
    // }
    // return r * vec2<f32>(cos(theta), sin(theta));
}

// Cosine weight sampling, with pdf
fn sample_cosine_hemisphere(rand: vec2<f32>) -> vec4<f32> {
    let t = sample_uniform_disk(rand);
    let direction = vec3<f32>(t.x, t.y, sqrt(1.0 - dot(t, t)));
    let pdf = 2.0 * INV_TAU * direction.z;
    return vec4<f32>(direction, pdf);
}

// Samples a random direction in a cone with given half apex, also returns pdf
fn sample_uniform_cone(rand: vec2<f32>, cos_angle: f32) -> vec4<f32> {
    let z = 1.0 - (1.0 - cos_angle) * rand.x;  // [cos(angle), 1.0]
    let theta = TAU * rand.y;
    let r = sqrt(1.0 - z * z);
    let direction = vec3<f32>(r * cos(theta), r * sin(theta), z);
    let pdf = INV_TAU / (1.0 - cos_angle);
    return vec4<f32>(direction, pdf);
}

fn cone_pdf(cone: vec4<f32>, direction: vec3<f32>) -> f32 {
    return select(INV_TAU / (1.0 - cone.w), 0.0, (1.0 - cone.w < 0.0001) || (dot(direction, cone.xyz) < cone.w));
}

fn compute_directional_cone(directional: DirectionalLight) -> vec4<f32> {
    return vec4<f32>(directional.direction_to_light, cos(frame.solar_angle));
}

fn compute_emissive_cone(
    source: Emissive,
    position: vec3<f32>,
    normal: vec3<f32>,
) -> vec4<f32> {
    let delta = source.position - position;
    let d2 = dot(delta, delta);
    let r2 = source.radius * source.radius;
    let d = sqrt(d2);

    var cone: vec4<f32>;
    if d2 > r2 {
        cone = vec4<f32>(normalize(delta), sqrt((d2 - r2) / d2));
    } else {
        cone = vec4<f32>(normal, 0.0);
    }
    return cone;
}

fn compute_emissive_radiance(emissive: vec4<f32>) -> vec3<f32> {
    return 255.0 * emissive.a * emissive.rgb;
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
    candidate.emissive_index = DONT_SAMPLE_EMISSIVE;

    let directional = lights.directional_lights[0];
    var cone = compute_directional_cone(directional);
    var rand_sample = sample_uniform_cone(rand.zw, cone.w);
    candidate.cone = cone;
    candidate.direction = normal_basis(cone.xyz) * rand_sample.xyz;
    candidate.p = 1.0;

    if instance == DONT_SAMPLE_EMISSIVE {
        return candidate;
    }

    var lum = luminance(directional.color.rgb);
    var sum_lum = lum;

    var rand_1d = fract(dot(rand, vec4<f32>(1.0)));
    for (var id = 0u; id < emissive_buffer.count; id += 1u) {
        let source = emissive_buffer.data[id];
        if source.instance == instance {
            continue;
        }

        cone = compute_emissive_cone(source, position, normal);
        rand_sample = sample_uniform_cone(rand.zw, cone.w);
        let direction = normal_basis(cone.xyz) * rand_sample.xyz;
        if dot(direction, normal) < 0.0 {
            continue;
        }

        lum = luminance(compute_emissive_radiance(source.emissive));
        lum = lum * TAU * (1.0 - cone.w);
        sum_lum += lum;
        if rand_1d <= lum / max(sum_lum, 0.01) {
            candidate.cone = cone;
            candidate.direction = direction;
            candidate.emissive_index = source.instance;

            let dist = distance(source.position, position);
            candidate.min_distance = dist - source.radius;
            candidate.max_distance = dist + source.radius;
        }
    }

    // MIS
    if candidate.emissive_index != DONT_SAMPLE_EMISSIVE {
        var sum_pdf = select(luminance(directional.color.rgb) / sum_lum, 0.0, sum_lum < 0.01);
        var selected_pdf = 0.0;
        for (var id = 0u; id < emissive_buffer.count; id += 1u) {
            let source = emissive_buffer.data[id];
            if source.instance == instance {
                continue;
            }

            cone = compute_emissive_cone(source, position, normal);
            rand_sample = sample_uniform_cone(rand.zw, cone.w);
            let direction = normal_basis(cone.xyz) * rand_sample.xyz;
            if dot(direction, normal) < 0.0 {
                continue;
            }

            lum = luminance(compute_emissive_radiance(source.emissive));
            lum = lum * TAU * (1.0 - cone.w);

            var pdf = cone_pdf(cone, candidate.direction);
            pdf *= select(lum / sum_lum, 0.0, sum_lum < 0.01);
            if source.instance == candidate.emissive_index {
                selected_pdf = pdf;
            }
            sum_pdf += pdf;
        }

        candidate.p = cone_pdf(candidate.cone, candidate.direction);
        candidate.p *= select(selected_pdf / sum_pdf, 0.0, sum_pdf < 0.0001);
    }

    return candidate;
}

// NOTE: Correctly calculates the view vector depending on whether
// the projection is orthographic or perspective.
fn calculate_view(
    world_position: vec4<f32>,
    is_orthographic: bool,
) -> vec3<f32> {
    var V: vec3<f32>;
    if is_orthographic {
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
    if id != U32_MAX {
        surface.base_color *= textureSampleLevel(textures[id], samplers[id], uv, 0.0);
    }

    surface.emissive = material.emissive;
    id = material.emissive_texture;
    if id != U32_MAX {
        surface.emissive *= textureSampleLevel(textures[id], samplers[id], uv, 0.0);
    }

    surface.metallic = material.metallic;
    id = material.metallic_roughness_texture;
    if id != U32_MAX {
        surface.metallic *= textureSampleLevel(textures[id], samplers[id], uv, 0.0).r;
    }

    surface.occlusion = 1.0;
    id = material.occlusion_texture;
    if id != U32_MAX {
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
    if id != U32_MAX {
        emissive *= textureSampleLevel(textures[id], samplers[id], uv, 0.0);
    }

    return emissive;
}
#endif

fn hit_info(ray: Ray, hit: Hit) -> HitInfo {
    var info: HitInfo;
    info.instance_index = hit.instance_index;
    info.material_index = U32_MAX;

    if hit.instance_index != U32_MAX {
        let instance = instance_buffer.data[hit.instance_index];
        let indices = primitive_buffer.data[hit.primitive_index].indices;

        let v0 = vertex_buffer.data[(instance.mesh.vertex + indices[0])];
        let v1 = vertex_buffer.data[(instance.mesh.vertex + indices[1])];
        let v2 = vertex_buffer.data[(instance.mesh.vertex + indices[2])];
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
    sample_directional: bool,
    sample_emissive: bool,
    sample_ambient: bool,
) -> vec4<f32> {
    var radiance = vec3<f32>(0.0);
    var ambient = 0.0;

    if info.instance_index == U32_MAX {
        // Ray hits nothing, input radiance could be either directional or ambient
        let directional = lights.directional_lights[0];
        let cone = compute_directional_cone(directional);
        let hit_directional = dot(ray.direction, cone.xyz) >= cone.w;

        if sample_directional && hit_directional {
            radiance = directional.color.rgb;
            ambient = 0.0;
        } else {
            radiance = select(vec3<f32>(0.0), lights.ambient_color.rgb, sample_ambient);
            ambient = 1.0;
        }
    } else {
        // Input radiance is emissive, but bounced radiance is not added here
        if sample_emissive {
            let emissive = retreive_emissive(info.material_index, info.uv);
            radiance = compute_emissive_radiance(emissive);
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

fn check_previous_reservoir(
    r: ptr<function, Reservoir>,
    s: Sample,
) {
    let depth_ratio = (*r).s.visible_position.w / s.visible_position.w;
    let depth_miss = depth_ratio > 2.0 * (1.0 + s.random.x) || depth_ratio < 0.5 * s.random.y;

    let instance_miss = (*r).s.visible_instance != s.visible_instance;
    let normal_miss = dot(s.visible_normal, (*r).s.visible_normal) < 0.866;

    if (depth_miss || instance_miss || normal_miss) {
        (*r) = empty_reservoir();
    }
}

fn temporal_restir(
    r: ptr<function, Reservoir>,
    s: Sample,
    luminance: f32,
    max_sample_count: u32
) {
    update_reservoir(r, s, luminance);

    // Clamp...
    let m = f32(max_sample_count);
    if (*r).count > m {
        (*r).w_sum *= m / (*r).count;
        (*r).w2_sum *= m / (*r).count;
        (*r).count = m;
    }
}

fn compute_inv_jacobian(current_sample: Sample, neighbor_sample: Sample) -> f32 {
    var offset_b: vec3<f32> = neighbor_sample.sample_position.xyz - neighbor_sample.visible_position.xyz;
    var offset_a: vec3<f32> = neighbor_sample.sample_position.xyz - current_sample.visible_position.xyz;

    if dot(current_sample.visible_normal, offset_a) <= 0.0 {
        return 0.0;
    }

    let rb2: f32 = dot(offset_b, offset_b);
    let ra2: f32 = dot(offset_a, offset_a);
    offset_b = normalize(offset_b);
    offset_a = normalize(offset_a);
    let cos_a: f32 = dot(current_sample.visible_normal, offset_a);
    let cos_b: f32 = dot(neighbor_sample.visible_normal, offset_b);
    let cos_phi_a: f32 = -dot(offset_a, neighbor_sample.sample_normal);
    let cos_phi_b: f32 = -dot(offset_b, neighbor_sample.sample_normal);

    if cos_b <= 0.0 || cos_phi_b <= 0.0 {
        return 0.0;
    }

    if cos_a <= 0.0 || cos_phi_a <= 0.0 || ra2 <= 0.0 || rb2 <= 0.0 {
        return 0.0;
    }

    let denominator = rb2 * cos_phi_a;
    let numerator = ra2 * cos_phi_b;

    return select(clamp(numerator / denominator, 0.06, 1.0), 0.0, (denominator <= 0.0));
}

fn compute_jacobian(q: Sample, r: Sample) -> f32 {
    let normal: vec3<f32> = q.sample_normal;

    let cos_phi_1: f32 = abs(dot(normalize(r.visible_position.xyz - q.sample_position.xyz), normal));
    let cos_phi_2: f32 = abs(dot(normalize(q.visible_position.xyz - q.sample_position.xyz), normal));

    let term_1: f32 = cos_phi_1 / max(0.0001, cos_phi_2);

    var num: f32 = length((q.visible_position.xyz - q.sample_position.xyz));
    num *= num;

    var denom: f32 = length((r.visible_position.xyz - q.sample_position.xyz));
    denom *= denom;

    let term_2: f32 = num / max(denom, 0.0001);
    var jacobian: f32 = term_1 * term_2;

    jacobian = clamp(jacobian, 1.0, 50.0);
    return jacobian;
}

@compute @workgroup_size(8, 8, 1)
fn direct_lit(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let size = textureDimensions(render_texture);
    let coords = vec2<i32>(invocation_id.xy);
    let uv = coords_to_uv(coords, size);

    var s = empty_sample();

    let deferred_coords = vec2<i32>(uv * vec2<f32>(textureDimensions(position_texture)));
    let position_depth = textureLoad(position_texture, deferred_coords, 0);
    let position = vec4<f32>(position_depth.xyz, 1.0);
    let depth = position_depth.w;

    if depth < F32_EPSILON {
        var r: Reservoir;
        set_reservoir(&r, s, 0.0);
        store_reservoir(coords.x + size.x * coords.y, r);
        store_spatial_reservoir(coords.x + size.x * coords.y, r);

#ifndef INCLUDE_EMISSIVE
        textureStore(albedo_texture, coords, vec4<f32>(0.0));
#endif
        textureStore(variance_texture, coords, vec4<f32>(0.0));
        textureStore(render_texture, coords, vec4<f32>(0.0));

        return;
    }

    let normal = textureLoad(normal_texture, deferred_coords, 0).xyz;
    let instance_material = textureLoad(instance_material_texture, deferred_coords, 0).xy;
    let velocity_uv = textureLoad(velocity_uv_texture, deferred_coords, 0);

    let noise_id = frame.number % NOISE_TEXTURE_COUNT;
    let noise_size = textureDimensions(noise_texture[noise_id]);
    let noise_uv = (vec2<f32>(coords) + f32(frame.number) + 0.5) / vec2<f32>(noise_size);
    s.random = textureSampleLevel(noise_texture[noise_id], noise_sampler, noise_uv, 0.0);
    s.random = fract(s.random + f32(frame.number) * GOLDEN_RATIO);

    s.visible_position = vec4<f32>(position.xyz, depth);
    s.visible_normal = normal;
    s.visible_instance = instance_material.x;

    let surface = retreive_surface(instance_material.y, velocity_uv.zw);
    let view_direction = calculate_view(position, view.projection[3].w == 1.0);

#ifndef INCLUDE_EMISSIVE
    textureStore(albedo_texture, coords, vec4<f32>(env_brdf(view_direction, normal, surface), 1.0));
#endif

    var ray: Ray;
    var hit: Hit;
    var info: HitInfo;

    let previous_uv = uv - velocity_uv.xy;
    var r = load_previous_reservoir(previous_uv, size);
    check_previous_reservoir(&r, s);

#ifdef INCLUDE_EMISSIVE
    let validate_interval = frame.emissive_validate_interval;
    let select_light_instance = instance_material.x;
#else
    let validate_interval = frame.direct_validate_interval;
    let select_light_instance = DONT_SAMPLE_EMISSIVE;
#endif

    // Non-validation frame, or sample count too low
    if frame.number % validate_interval != 0u || r.count < f32(DIRECT_VALIDATION_FRAME_SAMPLE_THRESHOLD) {
        let candidate = select_light_candidate(
            s.random,
            s.visible_position.xyz,
            s.visible_normal,
            select_light_instance
        );

        // Direct light sampling
        ray.origin = position.xyz + normal * RAY_BIAS;
        ray.direction = candidate.direction;
        ray.inv_direction = 1.0 / ray.direction;

        var trace_condition = dot(candidate.direction, normal) > 0.0;
        trace_condition = trace_condition && candidate.p > 0.0;
#ifdef INCLUDE_EMISSIVE
        trace_condition = trace_condition && candidate.emissive_index != DONT_SAMPLE_EMISSIVE;
#endif

        if trace_condition {
            hit = traverse_top(ray, candidate.max_distance, candidate.min_distance);
            info = hit_info(ray, hit);

            s.sample_position = info.position;
            s.sample_normal = info.normal;

#ifdef INCLUDE_EMISSIVE
            // Don't sample directional light, sample emissive only
            s.radiance = input_radiance(ray, info, false, true, false);
#else
            // Sample directional light only, don't sample emissive
            s.radiance = input_radiance(ray, info, true, false, false);
#endif
        } else {
            s.sample_position = vec4<f32>(ray.origin + DISTANCE_MAX * ray.direction, 0.0);
            s.sample_normal = -ray.direction;
        }

        let sample_radiance = shading(
            view_direction,
            s.visible_normal,
            normalize(s.sample_position.xyz - s.visible_position.xyz),
            surface,
            s.radiance
        );
        let sample_luminance = select(luminance(sample_radiance) / candidate.p, 0.0, candidate.p < 0.0001);
        temporal_restir(&r, s, sample_luminance, frame.max_temporal_reuse_count);
    }

    // Validation frame
    if frame.number % validate_interval == 0u {
        let candidate = select_light_candidate(
            r.s.random,
            r.s.visible_position.xyz,
            r.s.visible_normal,
            select_light_instance
        );

        ray.origin = s.visible_position.xyz + s.visible_normal * RAY_BIAS;
        ray.direction = normalize(r.s.sample_position.xyz - s.visible_position.xyz);
        ray.inv_direction = 1.0 / ray.direction;

        var trace_condition = dot(candidate.direction, r.s.visible_normal) > 0.0;
        trace_condition = trace_condition && candidate.p > 0.0;
#ifdef INCLUDE_EMISSIVE
        trace_condition = trace_condition && candidate.emissive_index != DONT_SAMPLE_EMISSIVE;
#endif

        var validate_position: vec4<f32>;
        var validate_normal: vec3<f32>;
        var validate_radiance: vec4<f32>;

        if trace_condition {
            hit = traverse_top(ray, candidate.max_distance, candidate.min_distance);
            info = hit_info(ray, hit);

            validate_position = info.position;
            validate_normal = info.normal;

#ifdef INCLUDE_EMISSIVE
            validate_radiance = input_radiance(ray, info, false, true, false);
#else
            validate_radiance = input_radiance(ray, info, true, false, false);
#endif
        } else {
            validate_position = vec4<f32>(ray.origin + DISTANCE_MAX * ray.direction, 0.0);
            validate_normal = -ray.direction;
        }

        if r.count >= f32(DIRECT_VALIDATION_FRAME_SAMPLE_THRESHOLD) {
            // There is no new sample taken earlier this frame, so use the validate sample
            s.random = r.s.random;
            s.sample_position = validate_position;
            s.sample_normal = validate_normal;
            s.radiance = validate_radiance;
        }

        let luminance_ratio = luminance(validate_radiance.rgb) / max(luminance(r.s.radiance.rgb), 0.0001);
        if luminance_ratio > 1.25 || luminance_ratio < 0.8 {
            // Luminance miss, update reservoir
            let sample_radiance = shading(
                view_direction,
                s.visible_normal,
                normalize(s.sample_position.xyz - s.visible_position.xyz),
                surface,
                s.radiance
            );
            let w_new = select(luminance(sample_radiance) / candidate.p, 0.0, candidate.p < 0.0001);
            set_reservoir(&r, s, w_new);
        }
    }

    var out_radiance = shading(
        view_direction,
        r.s.visible_normal,
        normalize(r.s.sample_position.xyz - r.s.visible_position.xyz),
        surface,
        r.s.radiance
    );

    let total_lum = r.count * luminance(out_radiance);
    r.w = select(r.w_sum / total_lum, 0.0, total_lum < 0.0001);
    out_radiance *= r.w;

    if frame.suppress_temporal_reuse == 0u {
        store_reservoir(coords.x + size.x * coords.y, r);
    }

    var variance = r.w2_sum / r.count - pow(r.w_sum / r.count, 2.0);
    variance = select(variance / r.count, 0.0, r.count < 0.0001);
    textureStore(variance_texture, coords, vec4<f32>(variance));

    var out_color = out_radiance;
#ifdef INCLUDE_EMISSIVE
    out_color += compute_emissive_radiance(surface.emissive);
#endif
    textureStore(render_texture, coords, vec4<f32>(out_color, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn indirect_lit_ambient(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let deferred_size = textureDimensions(position_texture);
    let render_size = textureDimensions(render_texture);
    let reservoir_size = render_size;

    let coords = vec2<i32>(invocation_id.xy);
    let uv = coords_to_uv(coords, render_size);
    let deferred_coords = vec2<i32>(uv * vec2<f32>(deferred_size));

    let position_depth = textureLoad(position_texture, deferred_coords, 0);
    let position = vec4<f32>(position_depth.xyz, 1.0);
    let depth = position_depth.w;

    var s = empty_sample();
    var r = empty_reservoir();

    if depth < F32_EPSILON {
        store_reservoir(coords.x + reservoir_size.x * coords.y, r);
        store_spatial_reservoir(coords.x + reservoir_size.x * coords.y, r);

        textureStore(variance_texture, coords, vec4<f32>(0.0));
        textureStore(render_texture, coords, vec4<f32>(0.0));
        return;
    }

    let normal = normalize(textureLoad(normal_texture, deferred_coords, 0).xyz);
    let instance_material = textureLoad(instance_material_texture, deferred_coords, 0).xy;
    let velocity_uv = textureLoad(velocity_uv_texture, deferred_coords, 0);

    let noise_id = frame.number % NOISE_TEXTURE_COUNT;
    let noise_size = textureDimensions(noise_texture[noise_id]);
    let noise_uv = (vec2<f32>(coords) + f32(frame.number) + 0.5) / vec2<f32>(noise_size);
    s.random = textureSampleLevel(noise_texture[noise_id], noise_sampler, noise_uv, 0.0);
    s.random = fract(s.random + f32(frame.number) * GOLDEN_RATIO);

    s.visible_position = vec4<f32>(position.xyz, depth);
    s.visible_normal = normal;
    s.visible_instance = instance_material.x;

    var ray: Ray;
    var hit: Hit;
    var info: HitInfo;
    var pdf: f32;
    var surface: Surface;
    var bounce_sample = s;
    var color_transport = vec3<f32>(1.0);

    for (var n = 0u; n < frame.indirect_bounces && any(color_transport > vec3<f32>(0.01)); n += 1u) {
        var rand_sample = sample_cosine_hemisphere(bounce_sample.random.xy);
        ray.origin = bounce_sample.visible_position.xyz + bounce_sample.visible_normal * RAY_BIAS;
        ray.direction = normal_basis(bounce_sample.visible_normal) * rand_sample.xyz;
        ray.inv_direction = 1.0 / ray.direction;

        hit = traverse_top(ray, F32_MAX, 0.0);
        info = hit_info(ray, hit);

        if n == 0u {
            s.sample_position = info.position;
            s.sample_normal = info.normal;
        }

        bounce_sample.sample_position = info.position;
        bounce_sample.sample_normal = info.normal;

        // N bounce: from sample position
        if hit.instance_index != U32_MAX {
            var out_radiance = vec3<f32>(0.0);

            let candidate = select_light_candidate(
                bounce_sample.random,
                bounce_sample.sample_position.xyz,
                bounce_sample.sample_normal,
                info.instance_index
            );
            let sample_directional = (candidate.emissive_index == DONT_SAMPLE_EMISSIVE);
            let bounce_view_direction = normalize(bounce_sample.visible_position.xyz - bounce_sample.sample_position.xyz);

            surface = retreive_surface(info.material_index, info.uv);
            surface.roughness = 1.0;

            if dot(candidate.direction, bounce_sample.sample_normal) > 0.0 && candidate.p > 0.0 {
                ray.origin = bounce_sample.sample_position.xyz + bounce_sample.sample_normal * RAY_BIAS;
                ray.direction = candidate.direction;
                ray.inv_direction = 1.0 / ray.direction;

                hit = traverse_top(ray, candidate.max_distance, candidate.min_distance);
                info = hit_info(ray, hit);

                var in_radiance = input_radiance(ray, info, sample_directional, true, false);
                in_radiance = vec4<f32>(in_radiance.xyz, in_radiance.a);

                out_radiance = shading(
                    bounce_view_direction,
                    bounce_sample.sample_normal,
                    ray.direction,
                    surface,
                    in_radiance
                );
                out_radiance = out_radiance / candidate.p;
                out_radiance = select(out_radiance / rand_sample.w, vec3<f32>(0.0), rand_sample.w < 0.01);

                // Do radiance clamping
                let out_luminance = luminance(out_radiance);
                if out_luminance > frame.max_indirect_luminance {
                    out_radiance = out_radiance * frame.max_indirect_luminance / out_luminance;
                }

                s.radiance += vec4<f32>(color_transport * out_radiance, 1.0);
            }
            
            // Env BRDF approximates the reflection of the surface regardless of the input direction,
            // which may be a good choice for color transport.
            color_transport *= env_brdf(bounce_view_direction, bounce_sample.sample_normal, surface);

            bounce_sample.random = fract(bounce_sample.random + f32(frame.number) * GOLDEN_RATIO);
            bounce_sample.visible_position = bounce_sample.sample_position;
            bounce_sample.visible_normal = bounce_sample.sample_normal;
        } else {
            // Only ambient radiance
            var out_radiance = input_radiance(ray, info, false, false, true).rgb;
            s.radiance += vec4<f32>(color_transport * out_radiance, 0.0);
            break;
        }
    }

    surface = retreive_surface(instance_material.y, velocity_uv.zw);
    let view_direction = calculate_view(position, view.projection[3].w == 1.0);
    let sample_radiance = shading(
        view_direction,
        s.visible_normal,
        normalize(s.sample_position.xyz - s.visible_position.xyz),
        surface,
        s.radiance
    );
    let sample_luminance = luminance(sample_radiance);

    // ReSTIR: Temporal
    let previous_uv = uv - velocity_uv.xy;
    r = load_previous_reservoir(previous_uv, reservoir_size);
    check_previous_reservoir(&r, s);
    temporal_restir(&r, s, sample_luminance, frame.max_temporal_reuse_count);

    var out_radiance = shading(
        view_direction,
        r.s.visible_normal,
        normalize(r.s.sample_position.xyz - r.s.visible_position.xyz),
        surface,
        r.s.radiance
    );
    let total_lum = r.count * luminance(out_radiance);
    r.w = select(r.w_sum / total_lum, 0.0, total_lum < 0.0001);
    out_radiance *= r.w;

    if frame.suppress_temporal_reuse == 0u {
        store_reservoir(coords.x + reservoir_size.x * coords.y, r);
    }

    var variance = r.w2_sum / r.count - pow(r.w_sum / r.count, 2.0);
    variance = select(variance / r.count, 0.0, r.count < 0.0001);
    textureStore(variance_texture, coords, vec4<f32>(variance));

    textureStore(render_texture, coords, vec4<f32>(out_radiance, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn spatial_reuse(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let deferred_size = textureDimensions(position_texture);
    let render_size = textureDimensions(render_texture);
    let reservoir_size = render_size;

    let uv = (vec2<f32>(invocation_id.xy) + 0.5) / vec2<f32>(render_size);
    let coords = vec2<i32>(invocation_id.xy);
    let deferred_coords = vec2<i32>(uv * vec2<f32>(deferred_size));

    let position_depth = textureLoad(position_texture, deferred_coords, 0);
    let position = vec4<f32>(position_depth.xyz, 1.0);
    let depth = position_depth.w;

    var r = load_reservoir(coords.x + reservoir_size.x * coords.y);

    if depth < F32_EPSILON {
        store_spatial_reservoir(coords.x + reservoir_size.x * coords.y, r);
        textureStore(render_texture, coords, vec4<f32>(0.0));
        return;
    }

    let instance_material = textureLoad(instance_material_texture, deferred_coords, 0).xy;
    let velocity_uv = textureLoad(velocity_uv_texture, deferred_coords, 0);

    let surface = retreive_surface(instance_material.y, velocity_uv.zw);

    let use_spatial_variance = r.count <= f32(SPATIAL_VARIANCE_SAMPLE_THRESHOLD);

    // ReSTIR: Spatial
    let previous_uv = uv - velocity_uv.xy;

    var q = r;
    let s = q.s;
    r = load_previous_spatial_reservoir(previous_uv, reservoir_size);

    let view_direction = calculate_view(position, view.projection[3].w == 1.0);
    var out_radiance = shading(
        view_direction,
        s.visible_normal,
        normalize(s.sample_position.xyz - s.visible_position.xyz),
        surface,
        s.radiance
    );
    merge_reservoir(&r, q, luminance(out_radiance));

    let rot = mat2x2<f32>(
        vec2<f32>(0.707106781, 0.707106781),
        vec2<f32>(-0.707106781, 0.707106781)
    );
    var offset = sign(s.random.zw - 0.5);

    for (var i = 1u; i <= SPATIAL_REUSE_COUNT; i += 1u) {
        let offset_dist = mix(1.414213562, SPATIAL_REUSE_RANGE, f32(i) / f32(SPATIAL_REUSE_COUNT));
        offset = offset_dist * normalize(offset);

        let sample_coords = coords + vec2<i32>(offset);
        if any(sample_coords < vec2<i32>(0)) || any(sample_coords > reservoir_size) {
            continue;
        }

        let sample_depth = textureLoad(position_texture, sample_coords, 0).w;
        let depth_ratio = depth / sample_depth;
        if depth_ratio < 0.9 || depth_ratio > 1.1 {
            continue;
        }

        q = load_reservoir(sample_coords.x + reservoir_size.x * sample_coords.y);
        let normal_miss = dot(s.visible_normal, q.s.visible_normal) < 0.866;
        if q.count < F32_EPSILON || normal_miss {
            continue;
        }

        let sample_direction = normalize(q.s.sample_position.xyz - s.visible_position.xyz);
        if dot(sample_direction, s.visible_normal) < 0.0 {
            continue;
        }

        // Perform screen-space ray-marching the depth to reject samples
        let tap_interval = max(1.0, offset_dist / f32(SPATIAL_REUSE_TAPS + 1u));
        let tap_count = u32(offset_dist / tap_interval);
        var occluded = false;
        for (var j = 1u; j <= tap_count; j += 1u) {
            let tap_dist = f32(j) * tap_interval;
            let tap_offset = tap_dist * normalize(offset);

            let tap_uv = uv + tap_offset / vec2<f32>(render_size);
            let tap_coords = vec2<i32>(tap_uv * vec2<f32>(deferred_size));
            let tap_depth = textureLoad(position_texture, tap_coords, 0).w;

            let ref_depth = mix(depth, sample_depth, f32(j) / f32(tap_count + 1u));
            if tap_depth > ref_depth + 0.00001 {
                occluded = true;
                break;
            }
        }
        if occluded {
            continue;
        }

        // let inv_jac = select(1.0, compute_inv_jacobian(s, q.s), q.s.sample_position.w > 0.5);
        let jacobian = select(1.0, compute_jacobian(q.s, s), q.s.sample_position.w > 0.5);

        out_radiance = shading(
            view_direction,
            s.visible_normal,
            sample_direction,
            surface,
            q.s.radiance
        );
        merge_reservoir(&r, q, luminance(out_radiance) / jacobian);

        // offset = mix(1.25, 1.2, q.count / f32(frame.max_temporal_reuse_count)) * rot * offset;
        offset = rot * offset;
    }

    // Clamp...
    let m = f32(frame.max_spatial_reuse_count);
    if r.count > m {
        r.w_sum *= m / r.count;
        r.w2_sum *= m / r.count;
        r.count = m;
    }

    out_radiance = shading(
        view_direction,
        s.visible_normal,
        normalize(r.s.sample_position.xyz - s.visible_position.xyz),
        surface,
        r.s.radiance
    );
    let total_lum = r.count * luminance(out_radiance);
    r.w = select(r.w_sum / total_lum, 0.0, total_lum < 0.0001);

    if use_spatial_variance {
        var variance = r.w2_sum / r.count - pow(r.w_sum / r.count, 2.0);
        variance = select(variance / r.count, 0.0, r.count < 0.0001);
        textureStore(variance_texture, coords, vec4<f32>(variance));
    }

    var out_color = r.w * out_radiance;
#ifdef INCLUDE_EMISSIVE
    out_color += compute_emissive_radiance(surface.emissive);
#endif
    textureStore(render_texture, coords, vec4<f32>(out_color, 1.0));
}
