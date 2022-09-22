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
var render_texture: texture_storage_2d<rgba16float, write>;
@group(4) @binding(2)
var shadow_texture: texture_storage_2d<rgba16float, read_write>;
@group(4) @binding(3)
var shadow_cache_texture: texture_2d<f32>;
@group(4) @binding(4)
var shadow_cache_sampler: sampler;

let F32_EPSILON: f32 = 1.1920929E-7;
let F32_MAX: f32 = 3.402823466E+38;
let U32_MAX: u32 = 4294967295u;

let SOLAR_ANGLE: f32 = 0.5;

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

@compute @workgroup_size(8, 8, 1)
fn direct_cast(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let size = textureDimensions(render_texture);
    let uv = (vec2<f32>(invocation_id.xy) + 0.5) / vec2<f32>(size);

    let position = textureSampleLevel(position_texture, position_sampler, uv, 0.0);
    let normal = textureSampleLevel(normal_texture, normal_sampler, uv, 0.0).xyz;

    let coords = vec2<i32>(invocation_id.xy);
    if (position.w < 1.0) {
        textureStore(render_texture, coords, vec4<f32>(0.0));
        return;
    }

    var ray: Ray;
    ray.origin = view.world_position;
    ray.direction = normalize(position.xyz - ray.origin);
    ray.inv_direction = 1.0 / ray.direction;

    let hit = traverse_top(ray);
    if (hit.instance_index == U32_MAX) {
        return;
    }

    let instance = instance_buffer.data[hit.instance_index];
    let material = material_buffer.data[instance.material];

    var output_color = material.base_color;
    if (material.base_color_texture != U32_MAX) {
        let indices = primitive_buffer.data[hit.primitive_index].indices;
        let v0 = vertex_buffer.data[(instance.slice.vertex + indices[0])];
        let v1 = vertex_buffer.data[(instance.slice.vertex + indices[1])];
        let v2 = vertex_buffer.data[(instance.slice.vertex + indices[2])];
        let uv = v0.uv + hit.intersection.uv.x * (v1.uv - v0.uv) + hit.intersection.uv.y * (v2.uv - v0.uv);

        output_color = output_color * textureSampleLevel(textures[material.base_color_texture], samplers[material.base_color_texture], uv, 0.0);
    }

    textureStore(render_texture, coords, output_color);
}

// struct ShadowCache {
//     instance: u32,
//     normal: vec3<f32>,
//     position: vec3<f32>,
//     shadow: f32
// }

// var<workgroup> shared_shadow: array<array<vec4<f32>, 12>, 12>;

// fn decompose_shadow_cache(packed: vec4<u32>) -> ShadowCache {
//     var cache: ShadowCache;

//     cache.shadow = f32(packed.x) / f32(U32_MAX);
//     cache.normal = unpack4x8snorm(packed.y).xyz;

//     let xy = unpack2x16float(packed.z);
//     let z = unpack2x16float(packed.w);
//     cache.position = vec3<f32>(xy, z.x);
//     cache.instance = u32(z.y);

//     return cache;
// }

// fn compose_shadow_cache(cache: ShadowCache) -> vec4<u32> {
//     var packed: vec4<u32>;
//     packed.x = u32(cache.shadow * f32(U32_MAX));
//     packed.y = pack4x8snorm(vec4<f32>(cache.normal, 0.0));
//     packed.z = pack2x16float(cache.position.xy);
//     packed.w = pack2x16float(vec2<f32>(cache.position.z, f32(cache.instance) + 0.5));
//     return packed;
// }

@compute @workgroup_size(8, 8, 1)
fn direct_lit(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let hashed_frame_number = hash(frame.number);
    let rand = vec2<f32>(
        random_float(invocation_id.x << 16u ^ invocation_id.y + hashed_frame_number),
        random_float(invocation_id.y << 16u ^ invocation_id.x + hashed_frame_number)
    );
    let r = sqrt(rand.x);
    let theta = 2.0 * PI * rand.y;
    var disturb = vec3<f32>(
        r * SOLAR_ANGLE / PI * cos(theta),
        r * SOLAR_ANGLE / PI * sin(theta),
        0.0
    );
    disturb.z = sqrt(1.0 - dot(disturb.xy, disturb.xy));

    let size = textureDimensions(render_texture);
    let uv = (vec2<f32>(invocation_id.xy) + 0.5) / vec2<f32>(size);
    let coords = vec2<i32>(invocation_id.xy);

    let position = textureSampleLevel(position_texture, position_sampler, uv, 0.0);
    if (position.w < 0.5) {
        textureStore(render_texture, coords, vec4<f32>(0.0));
        return;
    }

    let normal = textureSampleLevel(normal_texture, normal_sampler, uv, 0.0).xyz;
    let instance_material = textureLoad(instance_material_texture, coords, 0);
    let velocity_uv = textureSampleLevel(velocity_uv_texture, velocity_uv_sampler, uv, 0.0);

    let material = material_buffer.data[instance_material.y];
    var output_color = material.base_color;
    var texture_id = material.base_color_texture;
    if (texture_id != U32_MAX) {
        output_color *= textureSampleLevel(textures[texture_id], samplers[texture_id], velocity_uv.zw, 0.0);
    }
    var emissive = material.emissive;
    texture_id = material.emissive_texture;
    if (texture_id != U32_MAX) {
        emissive *= textureSampleLevel(textures[texture_id], samplers[texture_id], velocity_uv.zw, 0.0);
    }
    var metallic = material.metallic;
    texture_id = material.metallic_roughness_texture;
    if (texture_id != U32_MAX) {
        metallic *= textureSampleLevel(textures[texture_id], samplers[texture_id], velocity_uv.zw, 0.0).r;
    }
    var occlusion = 1.0;
    texture_id = material.occlusion_texture;
    if (texture_id != U32_MAX) {
        occlusion = textureSampleLevel(textures[texture_id], samplers[texture_id], velocity_uv.zw, 0.0).r;
    }

    let roughness = perceptualRoughnessToRoughness(material.perceptual_roughness);

    let light = lights.directional_lights[0];
    var shadow = 0.0;

    var ray: Ray;
    ray.origin = position.xyz + light.direction_to_light * light.shadow_depth_bias + normal * light.shadow_normal_bias;
    ray.direction = light.direction_to_light;
    ray.direction = normalize(ray.direction + normal_basis(ray.direction) * disturb);
    ray.inv_direction = 1.0 / ray.direction;

    let hit = traverse_top(ray);
    if (hit.instance_index == U32_MAX) {
        shadow = 1.0;
    }

    // Spatio filtering
    // let shared_coords = vec2<i32>(local_id.xy) + 2;
    // shared_shadow[shared_coords.x][shared_coords.y] = shadow;
    // textureStore(shadow_texture, coords, shadow);
    // storageBarrier();

    // if (local_id.y < 4u) {
    //     // Top and bottom rows
    //     var dest_coords = vec2<i32>(local_id.xy);
    //     dest_coords.y = (dest_coords.y + 10) % 12;
    //     let src_coords = 8 * vec2<i32>(workgroup_id.xy) + dest_coords - 2;
    //     shared_shadow[dest_coords.x][dest_coords.y] = textureLoad(shadow_texture, src_coords);

    //     // Corners
    //     if (local_id.x < 4u) {
    //         dest_coords.x = (dest_coords.x + 10) % 12;
    //         let src_coords = 8 * vec2<i32>(workgroup_id.xy) + dest_coords - 2;
    //         shared_shadow[dest_coords.x][dest_coords.y] = textureLoad(shadow_texture, src_coords);
    //     }
    // } else {
    //     // Left and right cols
    //     let src_index = i32(local_id.y * 8u + local_id.x) - 32;
    //     var dest_coords = vec2<i32>(src_index % 4, src_index / 4);  // 4 rows, 8 cols
    //     dest_coords.x = (dest_coords.x + 10) % 12;
    //     let src_coords = 8 * vec2<i32>(workgroup_id.xy) + dest_coords - 2;
    //     shared_shadow[dest_coords.x][dest_coords.y] = textureLoad(shadow_texture, src_coords);
    // }
    // storageBarrier();

    // let kernel = compute_kernel();
    // for (var i = 0u; i < 25u; i = i + 1u) {
    //     let offset = vec2<i32>(kernel[i].xy);
    //     let sample_coords = shared_coords + offset;
    //     let shadow_sample = shared_shadow[sample_coords.x][sample_coords.y];

    //     let instance_delta = shadow_sample.x - shadow.x;
    //     var d2 = instance_delta * instance_delta;
    //     let instance_weight = min(exp(-d2 / 0.1), 1.0);
    // }

    // Temporal accumulation
    let cache_uv = uv - velocity_uv.xy;
    // let cache_coords = vec2<i32>(cache_uv * vec2<f32>(size));
    // var cache = decompose_shadow_cache(textureLoad(shadow_cache_texture, cache_coords));

    // var temporal_factor = 0.95;
    // let cache_miss = any(abs(cache_uv - 0.5) > vec2<f32>(0.5)) || (cache.instance != instance_material.x);
    // if (cache_miss) {
    //     temporal_factor = 0.05;
    // }

    var w_sum = 1.0E-4;
    var sum = 0.0;
    var temporal_factor = 0.95;
    let cache_miss = any(abs(cache_uv - 0.5) > vec2<f32>(0.5));
    if (!cache_miss) {
        for (var i = 0u; i < 25u; i += 1u) {
            let offset = vec2<f32>(frame.kernel[i].xy) / vec2<f32>(size);
            let cache = textureSampleLevel(shadow_cache_texture, shadow_cache_sampler, cache_uv + offset, 0.0);

            let dp = cache.xyz - position.xyz;
            var d2 = dot(dp, dp);
            let wp = min(exp(-d2 / 0.01), 1.0);

            let w = wp * frame.kernel[i].z;
            w_sum += w;
            sum += cache.w * w;
        }
        sum /= w_sum;
    }
    if (cache_miss || w_sum < 2.0E-4) {
        temporal_factor = 0.05;
        sum = 1.0;
    }

    shadow = mix(shadow, sum, temporal_factor);
    textureStore(shadow_texture, coords, vec4<f32>(position.xyz, shadow));

    // Shading
    // TODO: normal mapping
    let N = normal;
    let V = calculate_view(position, view.projection[3].w == 1.0);

    let NdotV = max(dot(N, V), 0.0001);

    let reflectance = material.reflectance;
    let F0 = 0.16 * reflectance * reflectance * (1.0 - metallic) + output_color.rgb * metallic;

    let diffuse_color = output_color.rgb * (1.0 - metallic);

    let R = reflect(-V, N);

    let light_contrib = directional_light(light, roughness, NdotV, N, V, R, F0, diffuse_color) * shadow;

    let diffuse_ambient = EnvBRDFApprox(diffuse_color, 1.0, NdotV);
    let specular_ambient = EnvBRDFApprox(F0, material.perceptual_roughness, NdotV);

    var color = light_contrib;
    color += (diffuse_ambient + specular_ambient) * lights.ambient_color.rgb * occlusion;
    color += emissive.rgb * output_color.a;
    output_color = vec4<f32>(color, output_color.a);
    textureStore(render_texture, coords, output_color);
}

@compute @workgroup_size(8, 8, 1)
fn indirect_lit(@builtin(global_invocation_id) invocation_id: vec3<u32>) {}