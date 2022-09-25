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

@group(5) @binding(0)
var render_texture: texture_storage_2d<rgba16float, write>;
@group(5) @binding(1)
var reservoir_texture: texture_storage_2d<rgba16float, read_write>;
@group(5) @binding(2)
var radiance_texture: texture_storage_2d<rgba16float, read_write>;
@group(5) @binding(3)
var random_texture: texture_storage_2d<rgba8snorm, read_write>;
@group(5) @binding(4)
var visible_position_texture: texture_storage_2d<rgba32float, read_write>;
@group(5) @binding(5)
var visible_normal_texture: texture_storage_2d<rgba8snorm, read_write>;
@group(5) @binding(6)
var sample_position_texture: texture_storage_2d<rgba32float, read_write>;
@group(5) @binding(7)
var sample_normal_texture: texture_storage_2d<rgba8snorm, read_write>;
@group(5) @binding(8)
var previous_reservoir_textures: binding_array<texture_2d<f32>>;
@group(5) @binding(9)
var previous_reservoir_samplers: binding_array<sampler>;

let F32_EPSILON: f32 = 1.1920929E-7;
let F32_MAX: f32 = 3.402823466E+38;
let U32_MAX: u32 = 4294967295u;

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

struct Sample {
    radiance: vec3<f32>,
    random: vec3<f32>,
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
        if (node.entry_index == U32_MAX) {
            let primitive_index = slice.primitive + node.primitive_index;
            let primitive = &primitive_buffer.data[primitive_index];
            let intersection = intersects_triangle(ray, (*primitive).vertices);

            if (intersection.distance < (*hit).intersection.distance) {
                (*hit).intersection = intersection;
                (*hit).primitive_index = primitive_index;
                intersected = true;
            }

            index = node.exit_index;
        } else {
            var aabb: Aabb;
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

fn random_uniform_cone_vector(invocation_id: vec3<u32>, angle: f32) -> vec3<f32> {
    let hashed_frame_number = hash(frame.number);
    let x = random_float(invocation_id.x << 16u ^ invocation_id.y + hashed_frame_number);
    let y = random_float(invocation_id.y << 16u ^ invocation_id.x ^ hashed_frame_number);
    let r = sqrt(x);
    let a = sin(angle);
    let theta = 2.0 * PI * y;
    var rand = vec3<f32>(
        r * cos(theta) * a,
        r * sin(theta) * a,
        0.0
    );
    rand.z = sqrt(1.0 - dot(rand.xy, rand.xy));
    return rand;
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

fn new_sample() -> Sample {
    var s: Sample;
    s.radiance = vec3<f32>(0.0);
    s.random = vec3<f32>(0.0);
    s.visible_position = vec4<f32>(0.0);
    s.visible_normal = vec3<f32>(0.0);
    s.sample_position = vec4<f32>(0.0);
    s.sample_normal = vec3<f32>(0.0);
    return s;
}

fn sample_reservoir(uv: vec2<f32>) -> Reservoir {
    var r: Reservoir;

    let reservoir = textureSampleLevel(previous_reservoir_textures[0], previous_reservoir_samplers[0], uv, 0.0);
    r.w = reservoir.r;
    r.w_sum = reservoir.g;
    r.count = reservoir.b;

    r.s.radiance = textureSampleLevel(previous_reservoir_textures[1], previous_reservoir_samplers[1], uv, 0.0).rgb;
    r.s.random = textureSampleLevel(previous_reservoir_textures[2], previous_reservoir_samplers[2], uv, 0.0).rgb;
    r.s.visible_position = textureSampleLevel(previous_reservoir_textures[3], previous_reservoir_samplers[3], uv, 0.0);
    r.s.visible_normal = textureSampleLevel(previous_reservoir_textures[4], previous_reservoir_samplers[4], uv, 0.0).rgb;
    r.s.sample_position = textureSampleLevel(previous_reservoir_textures[5], previous_reservoir_samplers[5], uv, 0.0);
    r.s.sample_normal = textureSampleLevel(previous_reservoir_textures[6], previous_reservoir_samplers[6], uv, 0.0).rgb;

    return r;
}

fn load_reservoir(coords: vec2<i32>) -> Reservoir {
    var r: Reservoir;

    let reservoir = textureLoad(reservoir_texture, coords);
    r.w = reservoir.r;
    r.w_sum = reservoir.g;
    r.count = reservoir.b;

    r.s.radiance = textureLoad(radiance_texture, coords).rgb;
    r.s.random = textureLoad(random_texture, coords).rgb;
    r.s.visible_position = textureLoad(visible_position_texture, coords);
    r.s.visible_normal = textureLoad(visible_normal_texture, coords).rgb;
    r.s.sample_position = textureLoad(sample_position_texture, coords);
    r.s.sample_normal = textureLoad(sample_normal_texture, coords).rgb;

    return r;
}

fn store_reservoir(coords: vec2<i32>, r: Reservoir) {
    let reservoir = vec4<f32>(r.w, r.w_sum, r.count, 0.0);
    textureStore(reservoir_texture, coords, reservoir);

    textureStore(radiance_texture, coords, vec4<f32>(r.s.radiance, 0.0));
    textureStore(random_texture, coords, vec4<f32>(r.s.random, 0.0));
    textureStore(visible_position_texture, coords, r.s.visible_position);
    textureStore(visible_normal_texture, coords, vec4<f32>(r.s.visible_normal, 0.0));
    textureStore(sample_position_texture, coords, r.s.sample_position);
    textureStore(sample_normal_texture, coords, vec4<f32>(r.s.sample_normal, 0.0));
}

fn update_reservoir(
    invocation_id: vec3<u32>,
    r: ptr<function, Reservoir>,
    s: Sample,
    w_new: f32,
) {
    if (distance(s.visible_position.w, (*r).s.visible_position.w) > 0.001 || dot(s.visible_normal, (*r).s.visible_normal) < 0.866) {
        (*r).count = 0.0;
        (*r).w = 0.0;
        (*r).w_sum = 0.0;
        (*r).s = s;
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

fn shading(
    position: vec4<f32>,
    normal: vec3<f32>,
    surface: Surface,
    hit: Hit,
    ray: Ray,
    light: DirectionalLight,
    s: ptr<function, Sample>
) {
    if (hit.instance_index == U32_MAX) {
        // Direct and enviromental shading
        var ray_light = light;
        let N = normal;
        let V = calculate_view(position, view.projection[3].w == 1.0);

        let NdotV = max(dot(N, V), 0.0001);

        let reflectance = surface.reflectance;
        let metallic = surface.metallic;
        let F0 = 0.16 * reflectance * reflectance * (1.0 - metallic) + surface.base_color.rgb * metallic;

        let R = reflect(-V, N);

        let diffuse_color = surface.base_color.rgb * (1.0 - metallic);

        if (dot(light.direction_to_light, ray.direction) > cos(SOLAR_ANGLE)) {
            ray_light.direction_to_light = ray.direction;
            (*s).radiance += directional_light(ray_light, surface.roughness, NdotV, N, V, R, F0, diffuse_color);
        } else {
            let diffuse_ambient = EnvBRDFApprox(diffuse_color, 1.0, NdotV);
            let specular_ambient = EnvBRDFApprox(F0, surface.roughness, NdotV);
            (*s).radiance += surface.occlusion * (diffuse_ambient + specular_ambient) * lights.ambient_color.rgb;
        }
        (*s).sample_position = vec4<f32>(32767.0 * ray.direction + ray.origin, 0.0);
        (*s).sample_normal = -ray.direction;
    } else {
        // Emissive shading
        let instance = instance_buffer.data[hit.instance_index];
        let indices = primitive_buffer.data[hit.primitive_index].indices;

        let v0 = vertex_buffer.data[(instance.slice.vertex + indices[0])];
        let v1 = vertex_buffer.data[(instance.slice.vertex + indices[1])];
        let v2 = vertex_buffer.data[(instance.slice.vertex + indices[2])];
        let hit_uv = v0.uv + hit.intersection.uv.x * (v1.uv - v0.uv) + hit.intersection.uv.y * (v2.uv - v0.uv);
        let hit_normal = v0.normal + hit.intersection.uv.x * (v1.normal - v0.normal) + hit.intersection.uv.y * (v2.normal - v0.normal);

        let hit_surface = retreive_surface(instance.material, hit_uv);
        (*s).radiance += 255.0 * hit_surface.emissive.a * hit_surface.emissive.rgb;
        (*s).sample_position = vec4<f32>(hit.intersection.distance * ray.direction + ray.origin, 1.0);
        (*s).sample_normal = hit_normal;
    }
}

// var<workgroup> shared_reserviors: array<array<Reservoir, 12>, 12>;

@compute @workgroup_size(8, 8, 1)
fn direct_lit(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let size = textureDimensions(render_texture);
    let uv = (vec2<f32>(invocation_id.xy) + 0.5) / vec2<f32>(size);
    let coords = vec2<i32>(invocation_id.xy);
    var s = new_sample();

    let position = textureSampleLevel(position_texture, position_sampler, uv, 0.0);
    if (position.w < 0.5) {
        var r: Reservoir;
        r.s = s;
        r.count = 0.0;
        r.w = 0.0;
        r.w_sum = 0.0;
        store_reservoir(coords, r);
        textureStore(render_texture, coords, vec4<f32>(0.0));
        return;
    }
    let ndc = view.view_proj * position;
    let depth = ndc.z / ndc.w;

    let normal = textureSampleLevel(normal_texture, normal_sampler, uv, 0.0).xyz;
    let instance_material = textureLoad(instance_material_texture, coords, 0);
    let velocity_uv = textureSampleLevel(velocity_uv_texture, velocity_uv_sampler, uv, 0.0);
    let surface = retreive_surface(instance_material.y, velocity_uv.zw);

    s.random = random_uniform_cone_vector(invocation_id, PI / 2.0);
    s.random = normal_basis(normal) * s.random;
    s.radiance = 255.0 * surface.emissive.a * surface.emissive.rgb;
    s.visible_position = vec4<f32>(position.xyz, depth);
    s.visible_normal = normal;

    let light = lights.directional_lights[0];

    var ray: Ray;
    ray.origin = position.xyz + normal * light.shadow_normal_bias;
    ray.direction = s.random;
    ray.inv_direction = 1.0 / ray.direction;

    let hit = traverse_top(ray);
    shading(position, normal, surface, hit, ray, light, &s);

    // ReSTIR: Temporal
    let previous_uv = uv - velocity_uv.xy;
    var r = sample_reservoir(previous_uv);
    if (any(abs(previous_uv - 0.5) > vec2<f32>(0.5))) {
        r.s.visible_normal = vec3<f32>(0.0);
    }

    update_reservoir(invocation_id, &r, s, luminance(s.radiance));
    r.w = r.w_sum / (r.count * luminance(r.s.radiance) + 0.0001);

    // Sample validation: is the path xv-xs still valid?
    if (frame.number % 3u == 0u) {
        ray.origin = r.s.visible_position.xyz + light.shadow_normal_bias * r.s.visible_normal;
        ray.direction = r.s.random;
        ray.inv_direction = 1.0 / ray.direction;
        let hit_validate = traverse_top(ray);

        let sample_miss = (r.s.sample_position.w < 0.5);
        let validation_miss = (hit_validate.instance_index == U32_MAX);

        let old_distance = distance(ray.origin, r.s.sample_position.xyz);
        let distance_miss = (distance(hit_validate.intersection.distance, old_distance) > 0.1);

        // let old_radiance = r.s.radiance;
        // shading(position, normal, surface, hit_validate, ray, light, &s);
        // let radiance_miss = (distance(old_radiance, s.radiance) > 1.0);

        if ((sample_miss && !validation_miss) || (!sample_miss && distance_miss)) {
            r.count = 0.0;
            r.w = 0.0;
            r.w_sum = 0.0;
            r.s = s;
        }
    }

    store_reservoir(coords, r);
    // storageBarrier();

    // ReSTIR: Spatio
    // let kernel_index = (hash(workgroup_id.x) * hash(workgroup_id.y) ^ hash(frame.number)) % 25u;
    // let offset = vec2<i32>(frame.kernel[kernel_index].xy);

    // var z = 0.0;
    // var q_coords = min(size, max(vec2<i32>(0), coords + offset));
    // if (any(q_coords != coords)) {
    //     let q = load_reservoir(q_coords);
    //     if (dot(q.s.visible_normal, r.s.visible_normal) > 0.866 && distance(q.s.visible_position.w, r.s.visible_position.w) < 0.001) {
    //         let x2q = q.s.sample_position.xyz;
    //         let x1q = q.s.visible_position.xyz;
    //         let nq = q.s.sample_normal;
    //         let x1r = r.s.visible_position.xyz;

    //         let xqq = x1q - x2q;
    //         let xqr = x1r - x2q;

    //         let cos_phi_q = abs(dot(nq, normalize(xqq)));
    //         let cos_phi_r = abs(dot(nq, normalize(xqr)));

    //         let dqq = dot(xqq, xqq);
    //         let dqr = dot(xqr, xqr);

    //         let inv_jacobian = (cos_phi_q * dqr) / (cos_phi_r * dqq + 0.0001);
    //         var p = luminance(q.s.radiance) * inv_jacobian;

    //         // Trace a ray to check.
    //         let delta = x2q - ray.origin;
    //         ray.direction = normalize(delta);
    //         ray.inv_direction = 1.0 / ray.direction;

    //         let hit = traverse_top(ray);
    //         if (hit.instance_index == U32_MAX || distance(hit.intersection.distance, length(delta)) > 0.1) {
    //             p = 0.0;
    //         }

    //         z += sign(p) * q.count;
    //         merge_reservoir(invocation_id, &r, q, p);
    //     }
    // }

    // q_coords = min(size, max(vec2<i32>(0), coords - offset));
    // if (r.count < 60.0 && any(q_coords != coords)) {
    //     let q = load_reservoir(q_coords);
    //     if (dot(q.s.visible_normal, r.s.visible_normal) > 0.866 && distance(q.s.visible_position.w, r.s.visible_position.w) < 0.001) {
    //         let x2q = q.s.sample_position.xyz;
    //         let x1q = q.s.visible_position.xyz;
    //         let nq = q.s.sample_normal;
    //         let x1r = r.s.visible_position.xyz;

    //         let xqq = x1q - x2q;
    //         let xqr = x1r - x2q;

    //         let cos_phi_q = abs(dot(nq, normalize(xqq)));
    //         let cos_phi_r = abs(dot(nq, normalize(xqr)));

    //         let dqq = dot(xqq, xqq);
    //         let dqr = dot(xqr, xqr);

    //         let inv_jacobian = (cos_phi_q * dqr) / (cos_phi_r * dqq + 0.0001);
    //         let p = luminance(q.s.radiance) * inv_jacobian;
    //         merge_reservoir(invocation_id, &r, q, p);
    //     }
    // }

    // if (z > 0.0) {
    //     r.w = r.w_sum / (z * luminance(r.s.radiance));
    // }

    // storageBarrier();
    // store_reservoir(coords, r);

    textureStore(render_texture, coords, vec4<f32>(r.s.radiance * r.w, 1.0));
}