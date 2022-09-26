#import bevy_pbr::mesh_view_bindings
#import bevy_pbr::utils
#import bevy_pbr::lighting

#import bevy_hikari::mesh_material_bindings
#import bevy_hikari::deferred_bindings

@group(3) @binding(0)
var textures: binding_array<texture_2d<f32>>;
@group(3) @binding(1)
var samplers: binding_array<sampler>;

struct Frame {
    number: u32,
    kernel: array<vec3<f32>, 25>,
};

@group(4) @binding(0)
var<uniform> frame: Frame;
@group(4) @binding(1)
var noise_texture: binding_array<texture_2d<f32>>;
@group(4) @binding(2)
var noise_sampler: sampler;

@group(5) @binding(0)
var render_texture: texture_storage_2d<rgba16float, write>;
@group(5) @binding(1)
var reservoir_texture: texture_storage_2d<rg32float, write>;
@group(5) @binding(2)
var radiance_texture: texture_storage_2d<rgba16float, write>;
@group(5) @binding(3)
var random_texture: texture_storage_2d<rgba16float, write>;
@group(5) @binding(4)
var visible_position_texture: texture_storage_2d<rgba32float, write>;
@group(5) @binding(5)
var visible_normal_texture: texture_storage_2d<rgba8snorm, write>;
@group(5) @binding(6)
var sample_position_texture: texture_storage_2d<rgba32float, write>;
@group(5) @binding(7)
var sample_normal_texture: texture_storage_2d<rgba8snorm, write>;
@group(5) @binding(8)
var previous_reservoir_textures: binding_array<texture_2d<f32>>;
@group(5) @binding(9)
var previous_reservoir_samplers: binding_array<sampler>;

let F32_EPSILON: f32 = 1.1920929E-7;
let F32_MAX: f32 = 3.402823466E+38;
let U32_MAX: u32 = 4294967295u;

let DISTANCE_MAX: f32 = 65535.0;
let VALIDATION_INTERVAL: u32 = 16u;
let NOISE_TEXTURE_COUNT: u32 = 64u;
let GOLDEN_RATIO: f32 = 1.618033989;
let SECOND_BOUNCE_CHANCE: f32 = 1.0;

let SOLAR_ANGLE: f32 = 0.523598776;

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
    surface: Surface,
    position: vec4<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
};

struct Sample {
    radiance: vec3<f32>,
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
        if (node.entry_index == U32_MAX) {
            let primitive_index = slice.primitive + node.primitive_index;
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

        if (node.entry_index == U32_MAX) {
            let instance_index = node.primitive_index;
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

fn cosine_sample_hemisphere(rand: vec2<f32>) -> vec3<f32> {
    let r = sqrt(rand.x);
    let theta = 2.0 * PI * rand.y;
    var direction = vec3<f32>(
        r * cos(theta),
        r * sin(theta),
        0.0
    );
    direction.z = sqrt(1.0 - dot(direction.xy, direction.xy));
    return direction;
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

fn retreive_surface(material_id: u32, uv: vec2<f32>) -> Surface {
    var surface: Surface;
    let material = material_buffer.data[material_id];

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

fn hit_info(ray: Ray, hit: Hit) -> HitInfo {
    var info: HitInfo;

    if (hit.instance_index != U32_MAX) {
        let instance = instance_buffer.data[hit.instance_index];
        let indices = primitive_buffer.data[hit.primitive_index].indices;

        let v0 = vertex_buffer.data[(instance.slice.vertex + indices[0])];
        let v1 = vertex_buffer.data[(instance.slice.vertex + indices[1])];
        let v2 = vertex_buffer.data[(instance.slice.vertex + indices[2])];
        let uv = hit.intersection.uv;
        info.uv = v0.uv + uv.x * (v1.uv - v0.uv) + uv.y * (v2.uv - v0.uv);
        info.normal = v0.normal + uv.x * (v1.normal - v0.normal) + uv.y * (v2.normal - v0.normal);

        info.surface = retreive_surface(instance.material, info.uv);
        info.position = vec4<f32>(ray.origin + ray.direction * hit.intersection.distance, 1.0);
    } else {
        info.position = vec4<f32>(ray.origin + ray.direction * DISTANCE_MAX, 0.0);
    }

    return info;
}

fn empty_sample() -> Sample {
    var s: Sample;
    s.radiance = vec3<f32>(0.0);
    s.random = vec4<f32>(0.0);
    s.visible_position = vec4<f32>(0.0);
    s.visible_normal = vec3<f32>(0.0);
    s.sample_position = vec4<f32>(0.0);
    s.sample_normal = vec3<f32>(0.0);
    return s;
}

fn sample_reservoir(uv: vec2<f32>) -> Reservoir {
    var r: Reservoir;

    let reservoir = textureSampleLevel(previous_reservoir_textures[0], previous_reservoir_samplers[0], uv, 0.0);
    r.w_sum = reservoir.r;
    r.count = reservoir.g;

    r.s.radiance = textureSampleLevel(previous_reservoir_textures[1], previous_reservoir_samplers[1], uv, 0.0).rgb;
    r.s.random = textureSampleLevel(previous_reservoir_textures[2], previous_reservoir_samplers[2], uv, 0.0);
    r.s.visible_position = textureSampleLevel(previous_reservoir_textures[3], previous_reservoir_samplers[3], uv, 0.0);
    r.s.visible_normal = textureSampleLevel(previous_reservoir_textures[4], previous_reservoir_samplers[4], uv, 0.0).rgb;
    r.s.sample_position = textureSampleLevel(previous_reservoir_textures[5], previous_reservoir_samplers[5], uv, 0.0);
    r.s.sample_normal = textureSampleLevel(previous_reservoir_textures[6], previous_reservoir_samplers[6], uv, 0.0).rgb;

    return r;
}

fn load_reservoir(coords: vec2<i32>) -> Reservoir {
    var r: Reservoir;

    let reservoir = textureLoad(reservoir_texture, coords);
    r.w_sum = reservoir.r;
    r.count = reservoir.g;

    r.s.radiance = textureLoad(radiance_texture, coords).rgb;
    r.s.random = textureLoad(random_texture, coords);
    r.s.visible_position = textureLoad(visible_position_texture, coords);
    r.s.visible_normal = textureLoad(visible_normal_texture, coords).rgb;
    r.s.sample_position = textureLoad(sample_position_texture, coords);
    r.s.sample_normal = textureLoad(sample_normal_texture, coords).rgb;

    return r;
}

fn store_reservoir(coords: vec2<i32>, r: Reservoir) {
    let reservoir = vec4<f32>(r.w_sum, r.count, 0.0, 0.0);
    textureStore(reservoir_texture, coords, reservoir);

    textureStore(radiance_texture, coords, vec4<f32>(r.s.radiance, 0.0));
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
    if (distance(s.visible_position.w, (*r).s.visible_position.w) > 0.001 || dot(s.visible_normal, (*r).s.visible_normal) < 0.866) {
        set_reservoir(r, s, w_new);
        return;
    }

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

fn shading(
    V: vec3<f32>,
    N: vec3<f32>,
    ray: Ray,
    light: DirectionalLight,
    surface: Surface,
    info: HitInfo,
    head_radiance: vec3<f32>,
) -> vec3<f32> {
    var radiance = vec3<f32>(0.0);

    let reflectance = surface.reflectance;
    let metallic = surface.metallic;
    let base_color = surface.base_color.rgb;

    let F0 = 0.16 * reflectance * reflectance * (1.0 - metallic) + base_color * metallic;
    let diffuse_color = base_color * (1.0 - metallic);

    if (info.position.w == 0.0) {
        // Directional and ambient
        if (dot(light.direction_to_light, ray.direction) > cos(SOLAR_ANGLE)) {
            radiance = lit(light.color.rgb, diffuse_color, surface.roughness, F0, ray.direction, N, V);
        } else {
            radiance = ambient(diffuse_color, surface.roughness, surface.occlusion, F0, N, V);
        }
    } else {
        // Emissive
        let emissive = 255.0 * info.surface.emissive.a * info.surface.emissive.rgb;
        radiance = lit(emissive + head_radiance, diffuse_color, surface.roughness, F0, ray.direction, N, V);
    }

    return radiance;
}

// var<workgroup> shared_reserviors: array<array<Reservoir, 12>, 12>;

@compute @workgroup_size(8, 8, 1)
fn direct_lit(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
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
    let instance_material = textureLoad(instance_material_texture, coords, 0);
    let velocity_uv = textureSampleLevel(velocity_uv_texture, velocity_uv_sampler, uv, 0.0);
    let surface = retreive_surface(instance_material.y, velocity_uv.zw);

    // let hashed_frame_number = hash(frame.number);
    // s.random.x = random_float(invocation_id.x * hash(invocation_id.y) ^ hashed_frame_number);
    // s.random.y = random_float(invocation_id.y * hash(invocation_id.x) ^ ~hashed_frame_number);
    // s.random.z = random_float(invocation_id.x ^ hash(invocation_id.y) * hashed_frame_number);
    // s.random.w = random_float(invocation_id.x ^ ~hash(invocation_id.y) * hashed_frame_number);

    let noise_id = frame.number % NOISE_TEXTURE_COUNT;
    let noise_size = textureDimensions(noise_texture[noise_id]).xy;
    let noise_uv = (vec2<f32>(invocation_id.xy) + f32(frame.number) + 0.5) / vec2<f32>(noise_size);
    let noise_temporal_offset = f32(frame.number);
    s.random = textureSampleLevel(noise_texture[noise_id], noise_sampler, noise_uv, 0.0);
    s.random = fract(s.random + noise_temporal_offset * GOLDEN_RATIO);

    s.radiance = 255.0 * surface.emissive.a * surface.emissive.rgb;
    s.visible_position = vec4<f32>(position.xyz, depth);
    s.visible_normal = normal;

    let light = lights.directional_lights[0];

    var ray: Ray;
    var bounce_ray: Ray;

    var info: HitInfo;
    var bounce_info: HitInfo;

    ray.origin = position.xyz + normal * light.shadow_normal_bias;
    ray.direction = normal_basis(normal) * cosine_sample_hemisphere(s.random.xy);
    ray.inv_direction = 1.0 / ray.direction;
    let p1 = dot(ray.direction, normal);

    var hit = traverse_top(ray);
    info = hit_info(ray, hit);
    s.sample_position = info.position;
    s.sample_normal = info.normal;

    // Second bounce: from sample position
    let b2_rand = random_float(workgroup_id.x + workgroup_id.y * num_workgroups.x + hash(frame.number));
    let b2_condition = max(0.0, sign(SECOND_BOUNCE_CHANCE - b2_rand));  // 1.0 if b2_rand < SECOND_BOUNCE_CHANCE
    s.random *= vec4<f32>(1.0, 1.0, b2_condition, b2_condition);

    var p2 = 1.0;
    var head_radiance = vec3<f32>(0.0);
    if (any(s.random.zw > vec2<f32>(0.0)) && hit.instance_index != U32_MAX) {
        bounce_ray.origin = info.position.xyz + info.normal * light.shadow_normal_bias;
        bounce_ray.direction = normal_basis(info.normal) * cosine_sample_hemisphere(s.random.zw);
        bounce_ray.inv_direction = 1.0 / bounce_ray.direction;
        p2 = dot(bounce_ray.direction, info.normal) * SECOND_BOUNCE_CHANCE;

        hit = traverse_top(bounce_ray);
        bounce_info = hit_info(bounce_ray, hit);
        head_radiance = shading(
            ray.direction,
            info.normal,
            bounce_ray,
            light,
            info.surface,
            bounce_info,
            vec3<f32>(0.0)
        );
    }

    s.radiance += shading(
        view_direction,
        normal,
        ray,
        light,
        surface,
        info,
        head_radiance
    );

    // ReSTIR: Temporal
    let previous_uv = uv - velocity_uv.xy * 0.01;
    var r = sample_reservoir(previous_uv);
    if (any(abs(previous_uv - 0.5) > vec2<f32>(0.5))) {
        r.s.visible_normal = vec3<f32>(0.0);
    }

    let p = luminance(s.radiance) / (p1 * p2);
    update_reservoir(invocation_id, &r, s, p);
    r.w = r.w_sum / (max(r.count, 1.0) * luminance(r.s.radiance));

    textureStore(render_texture, coords, vec4<f32>(r.s.radiance * r.w, 1.0));

    // Sample validation: is the temporally reused path xv-xs still valid?
    if (frame.number % VALIDATION_INTERVAL == 0u && distance(s.sample_position, r.s.sample_position) > 0.1) {
        ray.origin = position.xyz + light.shadow_normal_bias * normal;
        ray.direction = normal_basis(normal) * cosine_sample_hemisphere(r.s.random.xy);
        ray.inv_direction = 1.0 / ray.direction;
        hit = traverse_top(ray);
        info = hit_info(ray, hit);
        var valid_radiance = 255.0 * surface.emissive.a * surface.emissive.rgb;

        if (any(r.s.random.zw > vec2<f32>(0.0)) && hit.instance_index != U32_MAX) {
            bounce_ray.origin = info.position.xyz + info.normal * light.shadow_normal_bias;
            bounce_ray.direction = normal_basis(info.normal) * cosine_sample_hemisphere(r.s.random.zw);
            bounce_ray.inv_direction = 1.0 / bounce_ray.direction;

            hit = traverse_top(bounce_ray);
            bounce_info = hit_info(bounce_ray, hit);
            head_radiance = shading(
                ray.direction,
                info.normal,
                bounce_ray,
                light,
                info.surface,
                bounce_info,
                vec3<f32>(0.0)
            );
        }

        valid_radiance += shading(
            view_direction,
            normal,
            ray,
            light,
            surface,
            info,
            head_radiance
        );

        if (abs(luminance(r.s.radiance) - luminance(valid_radiance)) / luminance(r.s.radiance) > 0.1) {
            set_reservoir(&r, s, p);
        }
    }

    store_reservoir(coords, r);
}