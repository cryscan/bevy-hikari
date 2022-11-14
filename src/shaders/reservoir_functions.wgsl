#define_import_path bevy_hikari::reservoir_functions

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
    r.lifetime = 0.0;
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
    (*r).lifetime += 1.0;

    let rand = fract(dot(s.random, vec4<f32>(1.0)));
    if rand < w_new / (*r).w_sum {
        // trsh suggests that instead of substituting the sample in the reservoir,
        // merging the two with similar luminance works better.
        // (*r).s = s;

        let l1 = luminance(s.radiance.rgb);
        let l2 = luminance((*r).s.radiance.rgb);
        let ratio = l1 / max(l2, 0.0001);
        var radiance = s.radiance;

        if ratio > 0.8 && ratio < 1.25 {
            radiance = mix((*r).s.radiance, s.radiance, 0.5);
        }

        (*r).s = s;
        (*r).s.radiance = radiance;
        (*r).lifetime = 0.0;
    }
}

fn merge_reservoir(r: ptr<function, Reservoir>, other: Reservoir, p: f32) {
    let count = (*r).count;
    update_reservoir(r, other.s, p * other.w * other.count);
    (*r).count = count + other.count;
}

fn load_previous_reservoir(uv: vec2<f32>, reservoir_size: vec2<i32>) -> Reservoir {
    var r = empty_reservoir();
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
    var r = empty_reservoir();
    if all(abs(uv - 0.5) < vec2<f32>(0.5)) {
        let coords = vec2<i32>(uv * vec2<f32>(reservoir_size));
        let index = coords.x + reservoir_size.x * coords.y;
        let packed = previous_spatial_reservoir_buffer.data[index];
        r = unpack_reservoir(packed);
    }
    return r;
}

fn store_previous_spatial_reservoir_uv(uv: vec2<f32>, reservoir_size: vec2<i32>, r: Reservoir) {
    if all(abs(uv - 0.5) < vec2<f32>(0.5)) {
        let coords = vec2<i32>(uv * vec2<f32>(reservoir_size));
        let index = coords.x + reservoir_size.x * coords.y;
        previous_spatial_reservoir_buffer.data[index] = pack_reservoir(r);
    }
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