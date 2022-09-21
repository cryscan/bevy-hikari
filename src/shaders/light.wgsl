#import bevy_pbr::mesh_view_bindings
#import bevy_hikari::mesh_material_bindings

@group(1) @binding(0)
var depth_texture: texture_depth_2d;
@group(1) @binding(1)
var depth_sampler: sampler;
@group(1) @binding(2)
var normal_texture: texture_2d<f32>;
@group(1) @binding(3)
var normal_sampler: sampler;
@group(1) @binding(4)
var instance_material_texture: texture_2d<u32>;
@group(1) @binding(5)
var instance_material_sampler: sampler;
@group(1) @binding(6)
var velocity_uv_texture: texture_2d<f32>;
@group(1) @binding(7)
var velocity_uv_sampler: sampler;

@group(3) @binding(0)
var textures: binding_array<texture_2d<f32>>;
@group(3) @binding(1)
var samplers: binding_array<sampler>;

@group(4) @binding(0)
var render_texture: texture_storage_2d<rgba16float, write>;

let F32_EPSILON: f32 = 1.1920929E-7;
let F32_MAX: f32 = 3.402823466E+38;
let U32_MAX: u32 = 4294967295u;

let PI: f32 = 3.1415926;
let SOLAR_ANGLE: f32 = 0.1;

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

@compute @workgroup_size(8, 8, 1)
fn direct_cast(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let size = textureDimensions(render_texture);
    let uv = vec2<f32>(invocation_id.xy) / vec2<f32>(size);

    let depth: f32 = textureSampleLevel(depth_texture, depth_sampler, uv, 0.0);
    let normal = textureSampleLevel(normal_texture, normal_sampler, uv, 0.0).xyz;

    let location = vec2<i32>(invocation_id.xy);
    if (depth < F32_EPSILON) {
        textureStore(render_texture, location, vec4<f32>(0.0));
        return;
    }

    let ndc = vec4<f32>(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y, depth, 1.0);
    let position = view.inverse_view_proj * ndc;

    var ray: Ray;
    ray.origin = view.world_position;
    ray.direction = normalize(position.xyz / position.w - ray.origin);
    ray.inv_direction = 1.0 / ray.direction;

    let hit = traverse_top(ray);
    if (hit.instance_index == U32_MAX) {
        return;
    }

    let instance = instance_buffer.data[hit.instance_index];
    let material = material_buffer.data[instance.material];

    var color = material.base_color;
    if (material.base_color_texture != U32_MAX) {
        let indices = primitive_buffer.data[hit.primitive_index].indices;
        let v0 = vertex_buffer.data[(instance.slice.vertex + indices[0])];
        let v1 = vertex_buffer.data[(instance.slice.vertex + indices[1])];
        let v2 = vertex_buffer.data[(instance.slice.vertex + indices[2])];
        let uv = v0.uv + hit.intersection.uv.x * (v1.uv - v0.uv) + hit.intersection.uv.y * (v2.uv - v0.uv);

        color = color * textureSampleLevel(textures[material.base_color_texture], samplers[material.base_color_texture], uv, 0.0);
    }

    textureStore(render_texture, location, color);
}

@compute @workgroup_size(8, 8, 1)
fn direct_lit(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let rand = vec2<f32>(
        random_float(invocation_id.x << 16u ^ invocation_id.y),
        random_float(invocation_id.y << 16u ^ invocation_id.x)
    );
    let r = sqrt(rand.x);
    let theta = 2.0 * PI * rand.y;

    let size = textureDimensions(render_texture);
    let uv = vec2<f32>(invocation_id.xy) / vec2<f32>(size);
    let location = vec2<i32>(invocation_id.xy);

    let depth: f32 = textureSampleLevel(depth_texture, depth_sampler, uv, 0.0);
    if (depth < F32_EPSILON) {
        textureStore(render_texture, location, vec4<f32>(0.0));
        return;
    }

    let normal = textureSampleLevel(normal_texture, normal_sampler, uv, 0.0).xyz;
    let instance_material = textureLoad(instance_material_texture, location, 0);
    let velocity_uv = textureSampleLevel(velocity_uv_texture, velocity_uv_sampler, uv, 0.0);

    let ndc = vec4<f32>(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y, depth, 1.0);
    let position = view.inverse_view_proj * ndc;

    var intensity = vec3<f32>(0.0);

    let material = material_buffer.data[instance_material.y];
    var color = material.base_color;
    if (material.base_color_texture != U32_MAX) {
        color = color * textureSampleLevel(textures[material.base_color_texture], samplers[material.base_color_texture], velocity_uv.zw, 0.0);
    }

    for (var i = 0u; i < lights.n_directional_lights; i = i + 1u) {
        let light = lights.directional_lights[i];

        var disturb = vec3<f32>(
            r * SOLAR_ANGLE / PI * cos(theta),
            r * SOLAR_ANGLE / PI * sin(theta),
            0.0
        );
        disturb.z = sqrt(1.0 - dot(disturb.xy, disturb.xy));

        var ray: Ray;
        ray.origin = position.xyz / position.w + light.direction_to_light * light.shadow_depth_bias + normal * light.shadow_normal_bias;
        ray.direction = light.direction_to_light;
        ray.direction = normalize(ray.direction + normal_basis(ray.direction) * disturb);
        ray.inv_direction = 1.0 / ray.direction;

        let hit = traverse_top(ray);
        if (hit.intersection.distance > 1000.0) {
            intensity += light.color.xyz * max(dot(normal, ray.direction), 0.0);
        }
    }

    color = vec4<f32>(intensity, 1.0) * color;
    textureStore(render_texture, location, color);
}

@compute @workgroup_size(8, 8, 1)
fn indirect_lit(@builtin(global_invocation_id) invocation_id: vec3<u32>) {}