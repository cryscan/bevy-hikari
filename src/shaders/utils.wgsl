#define_import_path bevy_hikari::utils

fn is_nan(val: f32) -> bool {
    return !(val < 0.0 || 0.0 < val || val == 0.0);
}

fn any_is_nan_vec3(val: vec3<f32>) -> bool {
    return is_nan(val.x) || is_nan(val.y) || is_nan(val.z);
}

fn any_is_nan_vec4(val: vec4<f32>) -> bool {
    return is_nan(val.x) || is_nan(val.y) || is_nan(val.z) || is_nan(val.w);
}

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

fn clip_to_uv(clip: vec4<f32>) -> vec2<f32> {
    var uv = clip.xy / clip.w;
    uv = (uv + 1.0) * 0.5;
    uv.y = 1.0 - uv.y;
    return uv;
}

fn coords_to_uv(coords: vec2<i32>, size: vec2<i32>) -> vec2<f32> {
    return (vec2<f32>(coords) + 0.5) / vec2<f32>(size);
}

fn current_smaa_jitter(frame_number: u32) -> i32 {
    return select(1, 0, frame_number % 2u == 0u);
}

fn previous_smaa_jitter(frame_number: u32) -> i32 {
    return select(0, 1, frame_number % 2u == 0u);
}

fn jittered_deferred_uv(uv: vec2<f32>, scaled_size: vec2<i32>, frame_number: u32) -> vec2<f32> {
    let texel_size = 1.0 / vec2<f32>(scaled_size);
    return uv + select(0.25, -0.25, frame_number % 2u == 0u) * texel_size;
}

fn jittered_deferred_coords(uv: vec2<f32>, deferred_size: vec2<i32>, scaled_size: vec2<i32>, frame_number: u32) -> vec2<i32> {
    let deferred_uv = jittered_deferred_uv(uv, scaled_size, frame_number);
    return vec2<i32>(deferred_uv * vec2<f32>(deferred_size));
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

fn frame_jitter(index: u32, sequence: u32) -> vec2<f32> {
    let id = index % sequence + 7u;
    let delta = vec2<f32>(halton(2u, id), halton(3u, id));
    return delta;
}

// luminance coefficients from Rec. 709.
// https://en.wikipedia.org/wiki/Rec._709
fn luminance(v: vec3<f32>) -> f32 {
    return dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
}
