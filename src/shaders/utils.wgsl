#define_import_path bevy_hikari::utils

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

fn frame_jitter(index: u32) -> vec2<f32> {
    let id = index % 16u + 7u;
    let delta = vec2<f32>(halton(2u, id), halton(3u, id));
    return delta;
}

// YUV-RGB conversion routine from Hyper3D
fn encode_pal_yuv(rgb: vec3<f32>) -> vec3<f32> {
    let c = pow(rgb, vec3<f32>(2.2)); // gamma correction
    return vec3<f32>(
        dot(c, vec3<f32>(0.299, 0.587, 0.114)),
        dot(c, vec3<f32>(-0.14713, -0.28886, 0.436)),
        dot(c, vec3<f32>(0.615, -0.51499, -0.10001))
    );
}

fn decode_pal_yuv(yuv: vec3<f32>) -> vec3<f32> {
    let c = vec3<f32>(
        dot(yuv, vec3<f32>(1., 0., 1.13983)),
        dot(yuv, vec3<f32>(1., -0.39465, -0.58060)),
        dot(yuv, vec3<f32>(1., 2.03211, 0.))
    );
    return pow(c, vec3<f32>(1.0 / 2.2)); // gamma correction
}
