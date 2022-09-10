#import bevy_pbr::mesh_view_bindings
#import bevy_hikari::ray_tracing_bindings

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    inv_direction: vec3<f32>,
    signs: u32,
};

struct Aabb {
    min: vec3<f32>,
    max: vec3<f32>,
};

struct Intersection {
    distance: f32,
    uv: vec2<f32>,
};

struct Hit {
    hit: Intersection,
    primitive: Primitive,
};

fn intersects_aabb(ray: Ray, aabb: Aabb) -> bool {
    let t1 = (aabb.min - ray.origin) * ray.inv_direction;
    let t2 = (aabb.max - ray.origin) * ray.inv_direction;

    var t_min = min(t1.x, t2.x);
    var t_max = max(t1.x, t2.x);

    t_min = max(t_min, min(t1.y, t2.y));
    t_max = min(t_max, max(t1.y, t2.y));

    t_min = max(t_min, min(t1.z, t2.z));
    t_max = min(t_max, max(t1.z, t2.z));

    return t_max >= t_min && t_max >= 0.0;
}

fn intersects_triangle(ray: Ray, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> Intersection {
    var result: Intersection;
    result.distance = -1.0;

    let ab = b - a;
    let ac = c - a;

    let u_vec = cross(ray.direction, ac);
    let det = dot(ab, u_vec);
    if (det < 0.00001) {
        return result;
    }

    let inv_det = 1.0 / det;
    let ao = ray.origin - a;
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
    if (distance > 0.00001) {
        result.distance = distance;
    }

    return result;
}

fn traverse_top(ray: Ray) -> Hit {
    var hit: Hit;
    var index = 0u;

    for (; index < instance_node_buffer.count;) {
        let node = &instance_node_buffer.data[index];
        var aabb: Aabb;

        if ((*node).entry_index == 4294967295u) {
            let instance = &instance_buffer.data[(*node).primitive_index];
            aabb.min = (*instance).min;
            aabb.max = (*instance).max;

            if (intersects_aabb(ray, aabb)) {
                // Traverse bottom here.
            }

            index = (*node).exit_index;
        } else {
            aabb.min = (*node).min;
            aabb.max = (*node).max;

            if (intersects_aabb(ray, aabb)) {
                index = (*node).entry_index;
            } else {
                index = (*node).exit_index;
            }
        }
    }

    return hit;
}

@group(2) @binding(0)
var depth_texture: texture_depth_2d;
@group(2) @binding(1)
var depth_sampler: sampler;
@group(2) @binding(2)
var normal_velocity_texture: texture_2d<f32>;
@group(2) @binding(3)
var normal_velocity_sampler: sampler;

struct Input {
    @builtin(global_invocation_id) invocation_id: vec3<u32>;
    @builtin(num_workgroups) workgroups: vec3<u32>,
};

@compute @workgroup_size(8, 8, 1)
fn direct(in: Input) {
    let uv = vec2<f32>(in.invocation_id.xy) / vec2<f32>(in.workgroups.xy * 8u);
    let depth = textureSample(depth_texture, depth_sampler, uv);
    let normal_velocity = textureSample(normal_velocity_texture, normal_velocity_sampler, uv);

    var ray: Ray;
}