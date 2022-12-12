#import bevy_hikari::mesh_view_bindings
#import bevy_hikari::utils
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

// -------- RESERVOIR   --------
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
    lifetime: f32,
    w: f32,
    w_sum: f32,
    w2_sum: f32,
};

@group(6) @binding(0)
var<storage, read> previous_reservoir_buffer: Reservoirs;
@group(6) @binding(1)
var<storage, read_write> reservoir_buffer: Reservoirs;
@group(6) @binding(2)
var<storage, read_write> previous_spatial_reservoir_buffer: Reservoirs;
@group(6) @binding(3)
var<storage, read_write> spatial_reservoir_buffer: Reservoirs;

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

    var t2 = unpack4x8snorm(packed.visible_normal);
    r.s.visible_position = packed.visible_position;
    r.s.visible_normal = normalize(t2.xyz);
    r.lifetime = 127.0 * (1.0 + t2.w);

    t2 = unpack4x8snorm(packed.sample_normal);
    r.s.sample_position = vec4<f32>(packed.sample_position.xyz, t2.w);
    r.s.sample_normal = normalize(t2.xyz);
    r.s.visible_instance = u32(packed.sample_position.w);

    return r;
}

fn pack_reservoir(r: Reservoir) -> PackedReservoir {
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
    packed.sample_position = vec4<f32>(r.s.sample_position.xyz, f32(r.s.visible_instance));

    packed.visible_normal = pack4x8snorm(vec4<f32>(r.s.visible_normal, r.lifetime / 127.0 - 1.0));
    packed.sample_normal = pack4x8snorm(vec4<f32>(r.s.sample_normal, r.s.sample_position.w));

    return packed;
}

fn set_reservoir(r: ptr<function, Reservoir>, s: Sample, w_new: f32) {
    (*r).count = 1.0;
    (*r).lifetime = 0.0;
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
    if rand < w_new / (*r).w_sum {
        (*r).s = s;

        // trsh suggests that instead of substituting the sample in the reservoir,
        // merging the two with similar luminance works better.
        // let l1 = luminance(s.radiance.rgb);
        // let l2 = luminance((*r).s.radiance.rgb);
        // let ratio = l1 / max(l2, 0.0001);
        // var radiance = s.radiance;

        // if ratio > 0.8 && ratio < 1.25 {
        //     radiance = mix((*r).s.radiance, s.radiance, 0.5);
        // }

        // (*r).s = s;
        // (*r).s.radiance = radiance;
    }
}

fn merge_reservoir(r: ptr<function, Reservoir>, other: Reservoir, p: f32) {
    let count = (*r).count;
    update_reservoir(r, other.s, p * other.w * other.count);
    (*r).count = count + other.count;
}

fn load_previous_reservoir(uv: vec2<f32>, reservoir_size: vec2<i32>) -> Reservoir {
    var r: Reservoir;
    if all(abs(uv - 0.5) < vec2<f32>(0.5)) {
        let coords = vec2<i32>(uv * vec2<f32>(reservoir_size));
        let index = coords.x + reservoir_size.x * coords.y;
        let packed = previous_reservoir_buffer.data[index];
        r = unpack_reservoir(packed);
    }
    return r;
}

fn load_reservoir(index: i32) -> Reservoir {
    let packed = reservoir_buffer.data[index];
    return unpack_reservoir(packed);
}

fn store_reservoir(index: i32, r: Reservoir) {
    reservoir_buffer.data[index] = pack_reservoir(r);
}

fn load_previous_spatial_reservoir(uv: vec2<f32>, reservoir_size: vec2<i32>) -> Reservoir {
    var r: Reservoir;
    if all(abs(uv - 0.5) < vec2<f32>(0.5)) {
        let coords = vec2<i32>(uv * vec2<f32>(reservoir_size));
        let index = coords.x + reservoir_size.x * coords.y;
        let packed = previous_spatial_reservoir_buffer.data[index];
        r = unpack_reservoir(packed);
    }
    return r;
}

fn store_previous_spatial_reservoir(index: i32, r: Reservoir) {
    previous_spatial_reservoir_buffer.data[index] = pack_reservoir(r);
}

fn load_spatial_reservoir(index: i32) -> Reservoir {
    let packed = spatial_reservoir_buffer.data[index];
    return unpack_reservoir(packed);
}

fn store_spatial_reservoir(index: i32, r: Reservoir) {
    spatial_reservoir_buffer.data[index] = pack_reservoir(r);
}
// -------- RESERVOIR   --------

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
let POSITION_MISS_THRESHOLD: f32 = 0.5;
let MAX_VARIANCE: f32 = 10.0;

let DONT_EXCLUDE: u32 = 0xFFFFFFFFu;
let DONT_SAMPLE_DIRECTIONAL_LIGHT: u32 = 0xFFFFFFFFu;
let DONT_SAMPLE_EMISSIVE: u32 = 0x80000000u;
let SAMPLE_ALL_EMISSIVE: u32 = 0xFFFFFFFFu;

#ifdef EMISSIVE_LIT
let SPATIAL_REUSE_COUNT: u32 = 8u;
let SPATIAL_REUSE_RANGE: f32 = 10.0;
#else
let SPATIAL_REUSE_COUNT: u32 = 16u;
let SPATIAL_REUSE_RANGE: f32 = 20.0;
#endif
let SPATIAL_REUSE_TAPS: u32 = 4u;

let DIRECT_VALIDATION_FRAME_SAMPLE_THRESHOLD: u32 = 4u;
let SPATIAL_VARIANCE_SAMPLE_THRESHOLD: u32 = 4u;

// -------- TRACING     ---------
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
    direction: vec3<f32>,
    max_distance: f32,
    min_distance: f32,
    emissive_instance: u32,
    p: f32,
};

fn instance_position_world_to_local(instance: Instance, p: vec3<f32>) -> vec3<f32> {
    let inverse_model = transpose(instance.inverse_transpose_model);
    let position = inverse_model * vec4<f32>(p, 1.0);
    return position.xyz / position.w;
}

fn instance_direction_world_to_local(instance: Instance, p: vec3<f32>) -> vec3<f32> {
    let inverse_model = transpose(instance.inverse_transpose_model);
    let direction = inverse_model * vec4<f32>(p, 0.0);
    return direction.xyz;
}

fn instance_position_local_to_world(instance: Instance, p: vec3<f32>) -> vec3<f32> {
    let model = instance.model;
    let position = model * vec4<f32>(p, 1.0);
    return position.xyz / position.w;
}

fn instance_normal_local_to_world(instance: Instance, n: vec3<f32>) -> vec3<f32> {
    // NOTE: The mikktspace method of normal mapping requires that the world normal is
    // re-normalized in the vertex shader to match the way mikktspace bakes vertex tangents
    // and normal maps so that the exact inverse process is applied when shading. Blender, Unity,
    // Unreal Engine, Godot, and more all use the mikktspace method. Do not change this code
    // unless you really know what you are doing.
    // http://www.mikktspace.com/
    return normalize(
        mat3x3<f32>(
            instance.inverse_transpose_model[0].xyz,
            instance.inverse_transpose_model[1].xyz,
            instance.inverse_transpose_model[2].xyz
        ) * n
    );
}

fn inside_aabb(p: vec3<f32>, aabb: Aabb) -> bool {
    return all(p > aabb.min) && all(p < aabb.max);
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

fn intersects_triangle(ray: Ray, tri: array<PrimitiveVertex, 3>) -> Intersection {
    var result: Intersection;
    result.distance = F32_MAX;

    let ab = tri[1].position - tri[0].position;
    let ac = tri[2].position - tri[0].position;

    let u_vec = cross(ray.direction, ac);
    let det = dot(ab, u_vec);
    if abs(det) < F32_EPSILON {
        return result;
    }

    let inv_det = 1.0 / det;
    let ao = ray.origin - tri[0].position;
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
    for (; index < mesh.node.y;) {
        let node_index = mesh.node.x + index;
        let node = asset_node_buffer.data[node_index];
        var aabb: Aabb;
        if node.entry_index >= BVH_LEAF_FLAG {
            let primitive_index = mesh.primitive + node.entry_index - BVH_LEAF_FLAG;
            let vertices = primitive_buffer[primitive_index].vertices;

            aabb.min = min(vertices[0].position, min(vertices[1].position, vertices[2].position));
            aabb.max = max(vertices[0].position, max(vertices[1].position, vertices[2].position));

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
            index = select(
                node.exit_index,
                node.entry_index,
                intersects_aabb(ray, aabb) < (*hit).intersection.distance
            );
        }
    }

    return intersected;
}

fn traverse_top(ray: Ray, max_distance: f32, early_distance: f32, exclude_instance: u32) -> Hit {
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
            let instance = instance_buffer[instance_index];
            aabb.min = instance.min;
            aabb.max = instance.max;

            if instance_index != exclude_instance && intersects_aabb(ray, aabb) < hit.intersection.distance {
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
            index = select(
                node.exit_index,
                node.entry_index,
                intersects_aabb(ray, aabb) < hit.intersection.distance
            );
        }
    }

    return hit;
}

fn empty_hit_info(position: vec3<f32>, direction: vec3<f32>) -> HitInfo {
    var info: HitInfo;
    info.instance_index = U32_MAX;
    info.material_index = U32_MAX;
    info.position = vec4<f32>(position + direction * DISTANCE_MAX, 0.0);
    return info;
}

fn hit_info(ray: Ray, hit: Hit) -> HitInfo {
    var info: HitInfo;
    info.instance_index = hit.instance_index;
    info.material_index = U32_MAX;

    if hit.instance_index != U32_MAX {
        let instance = instance_buffer[hit.instance_index];
        let vertices = primitive_buffer[hit.primitive_index].vertices;

        let v0 = vertex_buffer[(instance.mesh.vertex + vertices[0].index)];
        let v1 = vertex_buffer[(instance.mesh.vertex + vertices[1].index)];
        let v2 = vertex_buffer[(instance.mesh.vertex + vertices[2].index)];
        let uv0 = vec2<f32>(v0.u, v0.v);
        let uv1 = vec2<f32>(v1.u, v1.v);
        let uv2 = vec2<f32>(v2.u, v2.v);
        let uv = hit.intersection.uv;
        info.uv = uv0 + uv.x * (uv1 - uv0) + uv.y * (uv2 - uv0);
        info.normal = v0.normal + uv.x * (v1.normal - v0.normal) + uv.y * (v2.normal - v0.normal);
        info.normal = instance_normal_local_to_world(instance, info.normal);

        info.position = vec4<f32>(ray.origin + ray.direction * hit.intersection.distance, 1.0);
        info.material_index = instance.material;
    } else {
        info.position = vec4<f32>(ray.origin + ray.direction * DISTANCE_MAX, 0.0);
    }

    return info;
}

// Use this only if the ray is a shadow ray, i.e., call after `select_light_candidate`.
fn occlude_hit_info(ray: Ray, hit: Hit, info: ptr<function, HitInfo>) {
    if hit.instance_index != U32_MAX {
        (*info).instance_index = hit.instance_index;
        (*info).material_index = U32_MAX;
        (*info).position = vec4<f32>(ray.origin + ray.direction * hit.intersection.distance, 1.0);
        (*info).normal = vec3<f32>(0.0);
    }
}
// -------- TRACING     --------

// -------- SAMPLING    --------
fn sample_uniform_disk(rand: vec2<f32>) -> vec2<f32> {
    let r = sqrt(rand.x);
    let theta = 2.0 * PI * rand.y;
    return vec2<f32>(r * cos(theta), r * sin(theta));
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

// https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#SamplingaTriangle
fn sample_uniform_triangle_barycentric(rand: vec2<f32>) -> vec2<f32> {
    let srx = sqrt(rand.x);
    return vec2<f32>(1.0 - srx, rand.y * srx);
}

fn cone_pdf(cone: vec4<f32>, direction: vec3<f32>) -> f32 {
    return select(INV_TAU / (1.0 - cone.w), 0.0, (cone.w - 1.0 > 0.0) || (dot(direction, cone.xyz) < cone.w));
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
    info: ptr<function, HitInfo>,
) -> LightCandidate {
    var candidate: LightCandidate;
    candidate.max_distance = F32_MAX;
    candidate.min_distance = DISTANCE_MAX;
    candidate.emissive_instance = DONT_SAMPLE_EMISSIVE;

    let directional = lights.directional_lights[0];
    let cone = compute_directional_cone(directional);
    let rand_direction = normal_basis(cone.xyz) * sample_uniform_cone(rand.zw, cone.w).xyz;
    candidate.direction = rand_direction;
    candidate.p = 1.0;

    *info = empty_hit_info(position, rand_direction);

    if instance == DONT_SAMPLE_EMISSIVE {
        return candidate;
    }

    // Traverse the LBVH to pick one emissive within range
    var emissive: Emissive;
    var count = 0.0;
    var index = 0u;
    var rand_1d = rand.x;
    for (; index < emissive_node_buffer.count;) {
        let node = emissive_node_buffer.data[index];
        var aabb: Aabb;

        if node.entry_index >= BVH_LEAF_FLAG {
            let emissive_index = node.entry_index - BVH_LEAF_FLAG;
            let current_emissive = emissive_buffer[emissive_index];
            aabb.min = current_emissive.position - current_emissive.radius;
            aabb.max = current_emissive.position + current_emissive.radius;

            if instance != current_emissive.instance && inside_aabb(position, aabb) {
                rand_1d = fract(rand_1d + GOLDEN_RATIO);
                count += 1.0;
                if rand_1d < 1.0 / count {
                    candidate.emissive_instance = current_emissive.instance;
                    emissive = current_emissive;
                }
            }

            index = node.exit_index;
        } else {
            aabb.min = node.min;
            aabb.max = node.max;
            index = select(
                node.exit_index,
                node.entry_index,
                inside_aabb(position, aabb)
            );
        }
    }

    if candidate.emissive_instance != DONT_SAMPLE_EMISSIVE {
        // Sample a point on the instance's surface
        // Select a primitive based using the alias table
        let alias_index = min(u32(rand.x * f32(emissive.alias_table.y)), emissive.alias_table.y - 1u);
        let alias_entry = alias_table_buffer[emissive.alias_table.x + alias_index];
        let primitive_index = select(alias_index, alias_entry.index, rand.y < alias_entry.prob);

        let emissive_instance = instance_buffer[candidate.emissive_instance];
        let v = primitive_buffer[emissive_instance.mesh.primitive + primitive_index].vertices;
        let b = sample_uniform_triangle_barycentric(rand.zw);
        let p = instance_position_local_to_world(emissive_instance, b.x * v[0].position + b.y * v[1].position + (1.0 - b.x - b.y) * v[2].position);

        // Cast a ray on the instance to find the hit
        var hit: Hit;
        hit.intersection.distance = F32_MAX;
        hit.instance_index = U32_MAX;
        hit.primitive_index = U32_MAX;

        var ray: Ray;
        ray.origin = position + normal * RAY_BIAS;
        ray.direction = normalize(p - position);

        var r: Ray;
        r.origin = instance_position_world_to_local(emissive_instance, ray.origin);
        r.direction = instance_direction_world_to_local(emissive_instance, ray.direction);
        r.inv_direction = 1.0 / r.direction;

        candidate.direction = ray.direction;
        if dot(candidate.direction, normal) > 0.0 && traverse_bottom(&hit, r, emissive_instance.mesh, 0.0) {
            hit.instance_index = emissive.instance;
            *info = hit_info(ray, hit);

            candidate.max_distance = hit.intersection.distance;
            candidate.min_distance = hit.intersection.distance - 0.1;
            let delta = (*info).position.xyz - position;

            candidate.p = dot(delta, delta) / (abs(dot(ray.direction, (*info).normal) * emissive.surface_area));
            candidate.p = candidate.p / count;
        } else {
            // Fallback to sample directional        
            *info = empty_hit_info(ray.origin, ray.direction);

            candidate.emissive_instance = DONT_SAMPLE_EMISSIVE;
            candidate.direction = rand_direction;
            candidate.p = 1.0;
        }
    }

    return candidate;
}
// -------- SAMPLING    --------

// -------- SHADING     --------
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
    let material = material_buffer[material_index];

    surface.base_color = material.base_color;
    surface.emissive = material.emissive;
    surface.metallic = material.metallic;
    surface.occlusion = 1.0;
    surface.roughness = perceptualRoughnessToRoughness(material.perceptual_roughness);
    surface.reflectance = material.reflectance;

    return surface;
}

fn retreive_emissive(material_index: u32, uv: vec2<f32>) -> vec4<f32> {
    var emissive = material_buffer[material_index].emissive;
    return emissive;
}
#else
fn retreive_surface(material_index: u32, uv: vec2<f32>) -> Surface {
    var surface: Surface;
    let material = material_buffer[material_index];

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
    let material = material_buffer[material_index];

    var emissive = material.emissive;
    let id = material.emissive_texture;
    if id != U32_MAX {
        emissive *= textureSampleLevel(textures[id], samplers[id], uv, 0.0);
    }

    return emissive;
}
#endif

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
    sample_emissive: u32,
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
        if sample_emissive == info.instance_index {
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
// -------- SHADING     --------

// -------- RESTIR      --------
// The lifetime of the reservoir is randomized per sample
fn reservoir_lifetime(r: Reservoir) -> f32 {
    return select(frame.max_reservoir_lifetime, F32_MAX, frame.max_reservoir_lifetime <= 1.0);
}

fn check_previous_reservoir(
    r: ptr<function, Reservoir>,
    s: Sample,
) -> bool {
    var depth_ratio = (*r).s.visible_position.w / s.visible_position.w;
    depth_ratio = select(depth_ratio, 1.0 / depth_ratio, depth_ratio < 1.0);
    let depth_miss = depth_ratio > 1.05 * (1.0 + 0.5 * s.random.x);
    // let position_miss = distance((*r).s.visible_position.xyz, s.visible_position.xyz) > POSITION_MISS_THRESHOLD;

    let instance_miss = (*r).s.visible_instance != s.visible_instance;
    let normal_miss = dot(s.visible_normal, (*r).s.visible_normal) < 0.9;

    if depth_miss || normal_miss || instance_miss {
        var q: Reservoir;
        (*r) = q;
        return false;
    } else {
        return true;
    }
}

fn temporal_restir(
    r: ptr<function, Reservoir>,
    s: Sample,
    w_new: f32,
    max_sample_count: u32
) {
    update_reservoir(r, s, w_new);

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
// -------- RESTIR  --------

@compute @workgroup_size(8, 8, 1)
fn full_screen_albedo(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let render_size = textureDimensions(albedo_texture);
    let deferred_size = textureDimensions(position_texture);
    let coords = vec2<i32>(invocation_id.xy);
    let uv = coords_to_uv(coords, render_size);

    let deferred_coords = jittered_deferred_coords(uv, deferred_size, render_size, frame.number);
    let position_depth = textureLoad(position_texture, deferred_coords, 0);
    let position = vec4<f32>(position_depth.xyz, 1.0);
    let depth = position_depth.w;

    if depth < F32_EPSILON {
        textureStore(albedo_texture, coords, vec4<f32>(0.0));
        return;
    }

    let normal = textureLoad(normal_texture, deferred_coords, 0).xyz;
    let instance_material = textureLoad(instance_material_texture, deferred_coords, 0).xy;
    let velocity_uv = textureLoad(velocity_uv_texture, deferred_coords, 0);

    let surface = retreive_surface(instance_material.y, velocity_uv.zw);
    let view_direction = calculate_view(position, view.projection[3].w == 1.0);
    textureStore(albedo_texture, coords, vec4<f32>(env_brdf(view_direction, normal, surface), 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn direct_lit(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let render_size = textureDimensions(render_texture);
    let deferred_size = textureDimensions(position_texture);
    let coords = vec2<i32>(invocation_id.xy);
    let uv = coords_to_uv(coords, render_size);

    var s: Sample;

    let deferred_coords = jittered_deferred_coords(uv, deferred_size, render_size, frame.number);
    let position_depth = textureLoad(position_texture, deferred_coords, 0);
    let position = vec4<f32>(position_depth.xyz, 1.0);
    let depth = position_depth.w;

    if depth < F32_EPSILON {
        var r: Reservoir;
        set_reservoir(&r, s, 0.0);
        store_reservoir(coords.x + render_size.x * coords.y, r);
        store_spatial_reservoir(coords.x + render_size.x * coords.y, r);
        store_previous_spatial_reservoir(coords.x + render_size.x * coords.y, r);

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

    var ray: Ray;
    var hit: Hit;
    var info: HitInfo;

    let previous_uv = jittered_deferred_uv(uv, render_size, frame.number) - velocity_uv.xy;
    var r: Reservoir = load_previous_reservoir(previous_uv, render_size);

    if !check_previous_reservoir(&r, s) && all(abs(previous_uv - 0.5) <= vec2<f32>(0.5)) {
        let previous_coords = vec2<i32>(previous_uv * vec2<f32>(render_size));
        store_previous_spatial_reservoir(previous_coords.x + render_size.x * previous_coords.y, r);
    }

#ifdef EMISSIVE_LIT
    let validate_interval = frame.emissive_validate_interval;
    let select_light_instance = instance_material.x;
    let sample_directional = false;
#else
    let validate_interval = frame.direct_validate_interval;
    let select_light_instance = DONT_SAMPLE_EMISSIVE;
    let sample_directional = true;
#endif

    // Non-validation frame, or sample count too low
    if frame.number % validate_interval != 0u || r.count < f32(DIRECT_VALIDATION_FRAME_SAMPLE_THRESHOLD) {
        let candidate = select_light_candidate(
            s.random,
            s.visible_position.xyz,
            s.visible_normal,
            select_light_instance,
            &info
        );

        // Direct light sampling
        ray.origin = position.xyz + normal * RAY_BIAS;
        ray.direction = candidate.direction;
        ray.inv_direction = 1.0 / ray.direction;

        var trace_condition = dot(candidate.direction, normal) > 0.0;
        trace_condition = trace_condition && candidate.p > 0.0;
#ifdef EMISSIVE_LIT
        trace_condition = trace_condition && candidate.emissive_instance != DONT_SAMPLE_EMISSIVE;
#endif

        if trace_condition {
            hit = traverse_top(ray, candidate.max_distance, candidate.min_distance, candidate.emissive_instance);
            occlude_hit_info(ray, hit, &info);

#ifdef EMISSIVE_LIT
            // Don't sample directional light, sample emissive only
            s.radiance = input_radiance(ray, info, false, candidate.emissive_instance, false);
#else
            // Sample directional light only, don't sample emissive
            s.radiance = input_radiance(ray, info, true, DONT_SAMPLE_EMISSIVE, false);
#endif
        }

        s.sample_position = info.position;
        s.sample_normal = info.normal;

        // let sample_radiance = shading(
        //     view_direction,
        //     s.visible_normal,
        //     normalize(s.sample_position.xyz - s.visible_position.xyz),
        //     surface,
        //     s.radiance
        // );
        let w_new = select(0.0, luminance(s.radiance.rgb) / candidate.p, candidate.p > 0.0);
        temporal_restir(&r, s, w_new, frame.max_temporal_reuse_count);
    }

    // Validation frame
    if frame.number % validate_interval == 0u {
        let candidate = select_light_candidate(
            r.s.random,
            r.s.visible_position.xyz,
            r.s.visible_normal,
            select_light_instance,
            &info
        );

        ray.origin = s.visible_position.xyz + s.visible_normal * RAY_BIAS;
        ray.direction = normalize(r.s.sample_position.xyz - s.visible_position.xyz);
        ray.inv_direction = 1.0 / ray.direction;

        var validate_radiance: vec4<f32>;

        var trace_condition = dot(candidate.direction, r.s.visible_normal) > 0.0;
        trace_condition = trace_condition && candidate.p > 0.0;
#ifdef EMISSIVE_LIT
        trace_condition = trace_condition && candidate.emissive_instance != DONT_SAMPLE_EMISSIVE;
#endif

        if trace_condition {
            hit = traverse_top(ray, candidate.max_distance, candidate.min_distance, candidate.emissive_instance);
            // info = hit_info(ray, hit);
            occlude_hit_info(ray, hit, &info);

#ifdef EMISSIVE_LIT
            validate_radiance = input_radiance(ray, info, false, candidate.emissive_instance, false);
#else
            validate_radiance = input_radiance(ray, info, true, DONT_SAMPLE_EMISSIVE, false);
#endif
        }

        if r.count >= f32(DIRECT_VALIDATION_FRAME_SAMPLE_THRESHOLD) {
            // There is no new sample taken earlier this frame, so use the validate sample
            s.random = r.s.random;
            s.sample_position = info.position;
            s.sample_normal = info.normal;
            s.radiance = validate_radiance;
        }

        let luminance_ratio = luminance(validate_radiance.rgb) / max(luminance(r.s.radiance.rgb), 0.0001);
        if luminance_ratio > 1.25 || luminance_ratio < 0.8 {
            if all(abs(previous_uv - 0.5) <= vec2<f32>(0.5)) {
                let previous_coords = vec2<i32>(previous_uv * vec2<f32>(render_size));
                store_previous_spatial_reservoir(previous_coords.x + render_size.x * previous_coords.y, r);
            }
            // Luminance miss, update reservoir
            // let sample_radiance = shading(
            //     view_direction,
            //     s.visible_normal,
            //     normalize(s.sample_position.xyz - s.visible_position.xyz),
            //     surface,
            //     s.radiance
            // );
            let w_new = select(0.0, luminance(s.radiance.rgb) / candidate.p, candidate.p > 0.0);
            set_reservoir(&r, s, w_new);
        }
    }

    let total_lum = r.count * luminance(r.s.radiance.rgb);
    r.w = select(0.0, r.w_sum / total_lum, total_lum > 0.0);

    r.s.visible_position = s.visible_position;
    r.s.visible_normal = s.visible_normal;

    r.lifetime += 1.0;

    var variance = r.w2_sum / r.count - pow(r.w_sum / r.count, 2.0);
    variance = select(variance / r.count, variance, r.count < 1.0);
    variance = min(variance, MAX_VARIANCE);
    textureStore(variance_texture, coords, vec4<f32>(variance));

    if frame.enable_temporal_reuse > 0u {
        store_reservoir(coords.x + render_size.x * coords.y, r);
    }

    let surface = retreive_surface(instance_material.y, velocity_uv.zw);
    let view_direction = calculate_view(position, view.projection[3].w == 1.0);

    // if frame.enable_spatial_reuse == 0u {
#ifdef RENDER_EMISSIVE
    var out_radiance = shading(
        view_direction,
        r.s.visible_normal,
        normalize(r.s.sample_position.xyz - r.s.visible_position.xyz),
        surface,
        r.s.radiance
    );
    out_radiance *= r.w;
    let out_color = out_radiance + compute_emissive_radiance(surface.emissive);
    textureStore(render_texture, coords, vec4<f32>(out_color, 1.0));
#else
    var out_radiance = shading(
        view_direction,
        r.s.visible_normal,
        normalize(r.s.sample_position.xyz - r.s.visible_position.xyz),
        surface,
        r.s.radiance
    );
    out_radiance *= r.w;
    let out_color = out_radiance;
    textureStore(render_texture, coords, vec4<f32>(out_color, 1.0));
#endif
    // }
}

@compute @workgroup_size(8, 8, 1)
fn indirect_lit_ambient(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let deferred_size = textureDimensions(position_texture);
    let render_size = textureDimensions(render_texture);

    let coords = vec2<i32>(invocation_id.xy);
    let uv = coords_to_uv(coords, render_size);
    let deferred_coords = jittered_deferred_coords(uv, deferred_size, render_size, frame.number);

    let position_depth = textureLoad(position_texture, deferred_coords, 0);
    let position = vec4<f32>(position_depth.xyz, 1.0);
    let depth = position_depth.w;

    var s: Sample;
    var r: Reservoir;

    if frame.indirect_bounces == 0u || depth < F32_EPSILON {
        store_reservoir(coords.x + render_size.x * coords.y, r);
        store_spatial_reservoir(coords.x + render_size.x * coords.y, r);
        store_previous_spatial_reservoir(coords.x + render_size.x * coords.y, r);

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

#ifdef MULTIPLE_BOUNCES
    var bounce_sample = s;
    var color_transport = vec3<f32>(1.0);

    for (var n = 0u; n < frame.indirect_bounces && any(color_transport > vec3<f32>(0.01)); n += 1u) {
        var rand_sample = sample_cosine_hemisphere(bounce_sample.random.xy);
        ray.origin = bounce_sample.visible_position.xyz + bounce_sample.visible_normal * RAY_BIAS;
        ray.direction = normal_basis(bounce_sample.visible_normal) * rand_sample.xyz;
        ray.inv_direction = 1.0 / ray.direction;

        hit = traverse_top(ray, F32_MAX, 0.0, DONT_EXCLUDE);
        info = hit_info(ray, hit);

        if n == 0u {
            s.sample_position = info.position;
            s.sample_normal = info.normal;
            pdf = rand_sample.w;
        }

        bounce_sample.sample_position = info.position;
        bounce_sample.sample_normal = info.normal;

        // N bounce: from sample position
        if hit.instance_index != U32_MAX {
            var out_radiance = vec3<f32>(0.0);

            surface = retreive_surface(info.material_index, info.uv);
            surface.roughness = 1.0;

            let candidate = select_light_candidate(
                bounce_sample.random,
                bounce_sample.sample_position.xyz,
                bounce_sample.sample_normal,
                info.instance_index,
                &info
            );
            let sample_directional = (candidate.emissive_instance == DONT_SAMPLE_EMISSIVE);
            let bounce_view_direction = normalize(bounce_sample.visible_position.xyz - bounce_sample.sample_position.xyz);

            if dot(candidate.direction, bounce_sample.sample_normal) > 0.0 && candidate.p > 0.0 {
                ray.origin = bounce_sample.sample_position.xyz + bounce_sample.sample_normal * RAY_BIAS;
                ray.direction = candidate.direction;
                ray.inv_direction = 1.0 / ray.direction;

                hit = traverse_top(ray, candidate.max_distance, candidate.min_distance, candidate.emissive_instance);
                // info = hit_info(ray, hit);
                occlude_hit_info(ray, hit, &info);

                var in_radiance = input_radiance(ray, info, sample_directional, candidate.emissive_instance, false);
                in_radiance = vec4<f32>(in_radiance.xyz, in_radiance.a);

                out_radiance = shading(
                    bounce_view_direction,
                    bounce_sample.sample_normal,
                    ray.direction,
                    surface,
                    in_radiance
                );
                out_radiance = out_radiance / candidate.p;
                if n > 0u {
                    out_radiance = select(out_radiance / rand_sample.w, vec3<f32>(0.0), rand_sample.w < 0.01);
                }

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
            var out_radiance = input_radiance(ray, info, false, DONT_SAMPLE_EMISSIVE, true).rgb;
            s.radiance += vec4<f32>(color_transport * out_radiance, 0.0);
            break;
        }
    }
#else
    var rand_sample = sample_cosine_hemisphere(s.random.xy);
    ray.origin = s.visible_position.xyz + s.visible_normal * RAY_BIAS;
    ray.direction = normal_basis(s.visible_normal) * rand_sample.xyz;
    ray.inv_direction = 1.0 / ray.direction;

    hit = traverse_top(ray, F32_MAX, 0.0, DONT_EXCLUDE);
    info = hit_info(ray, hit);

    s.sample_position = info.position;
    s.sample_normal = info.normal;
    pdf = rand_sample.w;

    if hit.instance_index != U32_MAX {
        var out_radiance = vec3<f32>(0.0);

        surface = retreive_surface(info.material_index, info.uv);
        surface.roughness = 1.0;

        let candidate = select_light_candidate(
            s.random,
            s.sample_position.xyz,
            s.sample_normal,
            info.instance_index,
            &info
        );
        let sample_directional = (candidate.emissive_instance == DONT_SAMPLE_EMISSIVE);

        if dot(candidate.direction, s.sample_normal) > 0.0 && candidate.p > 0.0 {
            ray.origin = s.sample_position.xyz + s.sample_normal * RAY_BIAS;
            ray.direction = candidate.direction;
            ray.inv_direction = 1.0 / ray.direction;

            hit = traverse_top(ray, candidate.max_distance, candidate.min_distance, candidate.emissive_instance);
            // info = hit_info(ray, hit);
            occlude_hit_info(ray, hit, &info);

            var in_radiance = input_radiance(ray, info, sample_directional, candidate.emissive_instance, false);
            in_radiance = vec4<f32>(in_radiance.xyz, in_radiance.a);

            out_radiance = shading(
                normalize(s.visible_position.xyz - s.sample_position.xyz),
                s.sample_normal,
                ray.direction,
                surface,
                in_radiance
            );
            out_radiance = out_radiance / candidate.p;
            s.radiance += vec4<f32>(out_radiance, 1.0);
        }
    } else {
        // Only ambient radiance
        var out_radiance = input_radiance(ray, info, false, DONT_SAMPLE_EMISSIVE, true).rgb;
        s.radiance += vec4<f32>(out_radiance, 0.0);
    }
#endif

    // ReSTIR: Temporal
    let previous_uv = jittered_deferred_uv(uv, render_size, frame.number) - velocity_uv.xy;
    r = load_previous_reservoir(previous_uv, render_size);

    if !check_previous_reservoir(&r, s) && all(abs(previous_uv - 0.5) <= vec2<f32>(0.5)) {
        let previous_coords = vec2<i32>(previous_uv * vec2<f32>(render_size));
        store_previous_spatial_reservoir(previous_coords.x + render_size.x * previous_coords.y, r);
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
    let w_new = select(0.0, luminance(sample_radiance) / pdf, pdf > 0.0);
    temporal_restir(&r, s, w_new, frame.max_temporal_reuse_count);

    let out_radiance = shading(
        view_direction,
        r.s.visible_normal,
        normalize(r.s.sample_position.xyz - r.s.visible_position.xyz),
        surface,
        r.s.radiance
    );
    let total_lum = r.count * luminance(out_radiance);
    r.w = select(0.0, r.w_sum / total_lum, total_lum > 0.0);

    r.s.visible_position = s.visible_position;
    r.s.visible_normal = s.visible_normal;

    r.lifetime += 1.0;

    var variance = r.w2_sum / r.count - pow(r.w_sum / r.count, 2.0);
    variance = select(variance / r.count, variance, r.count < 1.0);
    variance = min(variance, MAX_VARIANCE);
    textureStore(variance_texture, coords, vec4<f32>(variance));

    if frame.enable_temporal_reuse > 0u {
        store_reservoir(coords.x + render_size.x * coords.y, r);
    }

    if frame.enable_spatial_reuse == 0u {
        textureStore(render_texture, coords, vec4<f32>(out_radiance * r.w, 1.0));
    }
}

var<workgroup> shared_reservoir: array<array<Reservoir, 8u>, 8u>;
var<workgroup> shared_depth: array<array<f32, 8u>, 8u>;

@compute @workgroup_size(8, 8, 1)
fn spatial_reuse(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let deferred_size = textureDimensions(position_texture);
    let render_size = textureDimensions(render_texture);

    let coords = vec2<i32>(invocation_id.xy);
    let uv = coords_to_uv(coords, render_size);
    let deferred_coords = jittered_deferred_coords(uv, deferred_size, render_size, frame.number);

    let position_depth = textureLoad(position_texture, deferred_coords, 0);
    let position = vec4<f32>(position_depth.xyz, 1.0);
    let depth = position_depth.w;

    var r: Reservoir = load_reservoir(coords.x + render_size.x * coords.y);

    shared_depth[local_id.y][local_id.x] = depth;
    shared_reservoir[local_id.y][local_id.x] = r;
    workgroupBarrier();

    if depth < F32_EPSILON {
        store_spatial_reservoir(coords.x + render_size.x * coords.y, r);
        textureStore(render_texture, coords, vec4<f32>(0.0));
        return;
    }

    let instance_material = textureLoad(instance_material_texture, deferred_coords, 0).xy;
    let velocity_uv = textureLoad(velocity_uv_texture, deferred_coords, 0);

    let surface = retreive_surface(instance_material.y, velocity_uv.zw);

    let use_spatial_variance = r.count <= f32(SPATIAL_VARIANCE_SAMPLE_THRESHOLD);

    // ReSTIR: Spatial
    let previous_uv = jittered_deferred_uv(uv, render_size, frame.number) - velocity_uv.xy;

    var q = r;
    let s = q.s;

    if r.lifetime <= reservoir_lifetime(r) {
        r = load_previous_spatial_reservoir(previous_uv, render_size);
    }

    let view_direction = calculate_view(position, view.projection[3].w == 1.0);
#ifdef EMISSIVE_LIT
    merge_reservoir(&r, q, luminance(q.s.radiance.rgb));
#else
    var out_radiance = shading(
        view_direction,
        s.visible_normal,
        normalize(s.sample_position.xyz - s.visible_position.xyz),
        surface,
        s.radiance
    );
    merge_reservoir(&r, q, luminance(out_radiance));
#endif

    r.s.visible_position = s.visible_position;
    r.s.visible_normal = s.visible_normal;

    for (var i = 1u; i <= SPATIAL_REUSE_COUNT; i += 1u) {
        // Fibonacci spiral: http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
        let polar_offset = vec2<f32>(
            TAU * fract(f32(i) * GOLDEN_RATIO + dot(s.random, vec4<f32>(1.0)) + random_float(frame.number)),
            sqrt(f32(i) / f32(SPATIAL_REUSE_COUNT)) * SPATIAL_REUSE_RANGE
        );
        let offset = polar_offset.y * vec2<f32>(cos(polar_offset.x), sin(polar_offset.x));

        let sample_coords = vec2<i32>(offset + vec2<f32>(coords));
        let sample_uv = coords_to_uv(sample_coords, render_size);
        let sample_deferred_coords = jittered_deferred_coords(sample_uv, deferred_size, render_size, frame.number);
        if any(sample_uv < vec2<f32>(0.0)) || any(sample_uv > vec2<f32>(1.0)) {
            continue;
        }

        var sample_depth: f32;

        // Check if the sample location is in the shared memory
        if all(sample_coords / 8 == vec2<i32>(workgroup_id.xy)) {
            let local_sample_coords = sample_coords % 8;
            sample_depth = shared_depth[local_sample_coords.y][local_sample_coords.x];
            q = shared_reservoir[local_sample_coords.y][local_sample_coords.x];
        } else {
            sample_depth = textureLoad(position_texture, sample_deferred_coords, 0).w;
            q = load_reservoir(sample_coords.x + render_size.x * sample_coords.y);
        }

        let depth_ratio = depth / sample_depth;
        if depth_ratio < 0.9 || depth_ratio > 1.1 {
            continue;
        }

        let normal_miss = dot(s.visible_normal, q.s.visible_normal) < 0.866;
        if q.count < F32_EPSILON || normal_miss {
            continue;
        }

        let sample_direction = normalize(q.s.sample_position.xyz - s.visible_position.xyz);
        if dot(sample_direction, s.visible_normal) < 0.0 {
            continue;
        }

        // Perform screen-space ray-marching the depth to reject samples
        let tap_interval = max(1.0, polar_offset.y / f32(SPATIAL_REUSE_TAPS + 1u));
        let tap_count = u32(polar_offset.y / tap_interval);
        var occluded = false;
        for (var j = 1u; j <= tap_count; j += 1u) {
            let tap_dist = f32(j) * tap_interval;
            let tap_offset = tap_dist * normalize(offset);

            let tap_uv = uv + tap_offset / vec2<f32>(render_size);
            let tap_deferred_coords = jittered_deferred_coords(tap_uv, deferred_size, render_size, frame.number);
            let tap_depth = textureLoad(position_texture, tap_deferred_coords, 0).w;

            let ref_depth = mix(depth, sample_depth, f32(j) / f32(tap_count + 1u));
            if tap_depth > ref_depth + 0.00001 {
                occluded = true;
                break;
            }
        }
        if occluded {
            continue;
        }

        let jacobian = select(1.0, compute_jacobian(q.s, s), q.s.sample_position.w > 0.5);
#ifdef EMISSIVE_LIT
        merge_reservoir(&r, q, luminance(q.s.radiance.rgb) / jacobian);
#else
        let out_radiance = shading(
            view_direction,
            s.visible_normal,
            sample_direction,
            surface,
            q.s.radiance
        );
        merge_reservoir(&r, q, luminance(out_radiance) / jacobian);
#endif
    }

    // Clamp...
    let m = f32(frame.max_spatial_reuse_count);
    if r.count > m {
        r.w_sum *= m / r.count;
        r.w2_sum *= m / r.count;
        r.count = m;
    }

    let out_radiance = shading(
        view_direction,
        s.visible_normal,
        normalize(r.s.sample_position.xyz - s.visible_position.xyz),
        surface,
        r.s.radiance
    );
#ifdef EMISSIVE_LIT
    let total_lum = r.count * luminance(r.s.radiance.rgb);
#else
    let total_lum = r.count * luminance(out_radiance);
#endif
    r.w = select(0.0, r.w_sum / total_lum, total_lum > 0.0);

    r.lifetime += 1.0;

    store_spatial_reservoir(coords.x + render_size.x * coords.y, r);

    if use_spatial_variance {
        var variance = r.w2_sum / r.count - pow(r.w_sum / r.count, 2.0);
        variance = select(variance / r.count, variance, r.count < 1.0);
        variance = min(variance, MAX_VARIANCE);
        textureStore(variance_texture, coords, vec4<f32>(variance));
    }

#ifdef RENDER_EMISSIVE
    let out_color = r.w * out_radiance + compute_emissive_radiance(surface.emissive);
#else
    let out_color = r.w * out_radiance;
#endif
    textureStore(render_texture, coords, vec4<f32>(out_color, 1.0));
}
