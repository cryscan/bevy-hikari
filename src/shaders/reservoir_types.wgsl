#define_import_path bevy_hikari::reservoir_types

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
    w: f32,
    w_sum: f32,
    w2_sum: f32,
};

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

fn empty_reservoir() -> Reservoir {
    var r: Reservoir;
    r.s = empty_sample();
    r.count = 0.0;
    r.w = 0.0;
    r.w_sum = 0.0;
    r.w2_sum = 0.0;
    return r;
}

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

    r.s.visible_position = packed.visible_position;
    r.s.sample_position = packed.sample_position;

    r.s.visible_normal = normalize(unpack4x8snorm(packed.visible_normal).xyz);
    r.s.sample_normal = normalize(unpack4x8snorm(packed.sample_normal).xyz);

    r.s.visible_instance = ((packed.visible_normal >> 24u) & 0xFFu) << 8u;
    r.s.visible_instance |= (packed.sample_normal >> 24u) & 0xFFu;

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
    packed.sample_position = r.s.sample_position;

    packed.visible_normal = pack4x8snorm(vec4<f32>(r.s.visible_normal, 0.0));
    packed.sample_normal = pack4x8snorm(vec4<f32>(r.s.sample_normal, 0.0));

    packed.visible_normal |= ((r.s.visible_instance >> 8u) & 0xFFu) << 24u;
    packed.sample_normal |= (r.s.visible_instance & 0xFFu) << 24u;

    return packed;
}

fn set_reservoir(r: ptr<function, Reservoir>, s: Sample, w_new: f32) {
    (*r).count = 1.0;
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
    if (rand < w_new / (*r).w_sum) {
        // trsh suggests that instead of substituting the sample in the reservoir,
        // merging the two with similar luminance works better.
        // (*r).s = s;

        let l1 = luminance(s.radiance.rgb);
        let l2 = luminance((*r).s.radiance.rgb);
        let ratio = l1 / max(l2, 0.0001);
        var radiance = s.radiance;

        if (ratio > 0.8 && ratio < 1.25) {
            radiance = mix((*r).s.radiance, s.radiance, 0.5);
        }

        (*r).s = s;
        (*r).s.radiance = radiance;
    }
}

fn merge_reservoir(r: ptr<function, Reservoir>, other: Reservoir, p: f32) {
    let count = (*r).count;
    update_reservoir(r, other.s, p * other.w * other.count);
    (*r).count = count + other.count;
}
