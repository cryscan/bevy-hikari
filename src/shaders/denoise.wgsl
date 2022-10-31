#import bevy_hikari::mesh_view_bindings
#import bevy_hikari::deferred_bindings
#import bevy_hikari::utils

@group(2) @binding(0)
var nearest_sampler: sampler;
@group(2) @binding(1)
var linear_sampler: sampler;

@group(3) @binding(0)
var albedo_texture: texture_2d<f32>;
@group(3) @binding(1)
var internal_texture_0: texture_storage_2d<rgba16float, read_write>;
@group(3) @binding(2)
var internal_texture_1: texture_storage_2d<rgba16float, read_write>;

@group(4) @binding(0)
var previous_render_texture: texture_2d<f32>;
@group(4) @binding(1)
var render_texture: texture_2d<f32>;
@group(4) @binding(2)
var output_texture: texture_storage_2d<rgba16float, read_write>;

// Normal-weighting function (4.4.1)
fn normal_weight(n0: vec3<f32>, n1: vec3<f32>) -> f32 {
    let exponent = 64.0;
    return pow(max(0.0, dot(n0, n1)), exponent);
}

// Depth-weighting function (4.4.2)
fn depth_weight(d0: f32, d1: f32, gradient: vec2<f32>, offset: vec2<i32>) -> f32 {
    let eps = 0.01;
    return exp((-abs(d0 - d1)) / (abs(dot(gradient, vec2<f32>(offset))) + eps));
}

// Luminance-weighting function (4.4.3)
fn luminance_weight(l0: f32, l1: f32, variance: f32) -> f32 {
    let strictness = 4.0;
    let eps = 0.001;
    return exp((-abs(l0 - l1)) / (strictness * sqrt(variance) + eps));
}

fn instance_weight(i0: u32, i1: u32) -> f32 {
    return f32(i0 == i1);
}

@compute @workgroup_size(8, 8, 1)
fn denoise(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
}

// @compute @workgroup_size(8, 8, 1)
// fn denoise_atrous(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
//     let render_size = textureDimensions(render_texture);
//     let output_size = textureDimensions(denoised_texture_0);

//     let output_uv = (vec2<f32>(invocation_id.xy) + 0.5) / vec2<f32>(output_size);
//     let output_coords = vec2<i32>(invocation_id.xy);
//     let render_coords = vec2<i32>(output_uv * vec2<f32>(render_size));

//     let depth = textureLoad(position_texture, output_coords, 0).w;
//     let depth_gradient = textureLoad(depth_gradient_texture, output_coords, 0).xy;
//     let normal = normalize(textureLoad(normal_texture, output_coords, 0).xyz);
//     let instance = textureLoad(instance_material_texture, output_coords, 0).x;

//     let albedo = textureLoad(albedo_texture, output_coords);

//     var irradiance = textureLoad(render_texture, render_coords).rgb / max(albedo.rgb, vec3<f32>(0.01));
//     let lum = luminance(irradiance);

//     let r = load_reservoir(render_coords.x + output_size.x * render_coords.y);
//     let variance = 1.0 / clamp(r.w2_sum, 1.0, 4.0);

//     var irradiance_sum = vec3<f32>(0.0);
//     var w_sum = 0.0;

//     var w_ff = 1.0;

// #ifdef FIREFLY_FILTER
//     // 5x5 Firefly Filter
//     var lum_sum = 0.0;
//     var lum2_sum = 0.0;
//     var ff_count = 0.01;
//     for (var y = -2; y <= 2; y += 1) {
//         for (var x = -2; x <= 2; x += 1) {
//             if (x == 0 && y == 0) {
//                 continue;
//             }

//             let offset = vec2<i32>(x, y);
//             let sample_coords = output_coords + offset;
//             let render_sample_coords = render_coords + offset;
//             if (any(sample_coords < vec2<i32>(0)) || any(sample_coords >= output_size)) {
//                 continue;
//             }

//             let sample_albedo = textureLoad(albedo_texture, sample_coords).rgb;
//             irradiance = textureLoad(render_texture, render_sample_coords).rgb / max(sample_albedo, vec3<f32>(0.01));
//             let sample_luminance = luminance(irradiance);

//             lum_sum += sample_luminance;
//             lum2_sum += sample_luminance * sample_luminance;
//             ff_count += 1.0;
//         }
//     }

//     let lum_mean = lum_sum / ff_count;
//     let lum_var = lum2_sum / ff_count - lum_mean * lum_mean;
//     if (lum > lum_mean + 3.0 * sqrt(lum_var)) {
//         w_ff = 0.0;
//     }
// #endif

// #ifdef DENOISER_LEVEL_0
//     // Pass 0, stride 8
//     for (var y = -1; y <= 1; y += 1) {
//         for (var x = -1; x <= 1; x += 1) {
//             let offset = vec2<i32>(x, y);
//             let sample_coords = output_coords + offset * 8;
//             let render_sample_coords = render_coords + offset * 8;
//             if (any(sample_coords < vec2<i32>(0)) || any(sample_coords >= output_size)) {
//                 continue;
//             }

//             let sample_albedo = textureLoad(albedo_texture, sample_coords).rgb;
//             irradiance = textureLoad(render_texture, render_sample_coords).rgb / max(sample_albedo, vec3<f32>(0.01));

//             let sample_normal = textureLoad(normal_texture, sample_coords, 0).xyz;
//             let sample_depth = textureLoad(position_texture, sample_coords, 0).w;
//             let sample_instance = textureLoad(instance_material_texture, sample_coords, 0).x;
//             let sample_luminance = luminance(irradiance);

//             let w_normal = normal_weight(normal, sample_normal);
//             let w_depth = depth_weight(depth, sample_depth, depth_gradient, offset);
//             let w_instance = instance_weight(instance, sample_instance);
//             let w_luminance = luminance_weight(lum, sample_luminance, variance);

//             let w = saturate(w_normal * w_depth * w_instance * w_luminance) * frame.kernel[y + 1][x + 1];

//             irradiance_sum += irradiance * w * select(1.0, w_ff, (x == 0 && y == 0));
//             w_sum += w;
//         }
//     }

//     w_sum = max(w_sum, 0.0001);
//     textureStore(denoised_texture_0, output_coords, vec4<f32>(irradiance_sum / w_sum, w_sum));
// #endif

// #ifdef DENOISER_LEVEL_1
//     // Pass 1, stride 4
//     for (var y = -1; y <= 1; y += 1) {
//         for (var x = -1; x <= 1; x += 1) {
//             let offset = vec2<i32>(x, y);
//             let sample_coords = output_coords + offset * 4;
//             if (any(sample_coords < vec2<i32>(0)) || any(sample_coords >= output_size)) {
//                 continue;
//             }

//             irradiance = textureLoad(denoised_texture_0, sample_coords).rgb;
//             let sample_normal = textureLoad(normal_texture, sample_coords, 0).xyz;
//             let sample_depth = textureLoad(position_texture, sample_coords, 0).w;
//             let sample_instance = textureLoad(instance_material_texture, sample_coords, 0).x;
//             let sample_luminance = luminance(irradiance);

//             let w_normal = normal_weight(normal, sample_normal);
//             let w_depth = depth_weight(depth, sample_depth, depth_gradient, offset);
//             let w_instance = instance_weight(instance, sample_instance);
//             let w_luminance = luminance_weight(lum, sample_luminance, variance);

//             let w = saturate(w_normal * w_depth * w_instance * w_luminance) * frame.kernel[y + 1][x + 1];

//             irradiance_sum += irradiance * w;
//             w_sum += w;
//         }
//     }

//     w_sum = max(w_sum, 0.0001);
//     textureStore(denoised_texture_1, output_coords, vec4<f32>(irradiance_sum / w_sum, w_sum));
// #endif

// #ifdef DENOISER_LEVEL_2
//     // Pass 2, stride 2
//     for (var y = -1; y <= 1; y += 1) {
//         for (var x = -1; x <= 1; x += 1) {
//             let offset = vec2<i32>(x, y);
//             let sample_coords = output_coords + offset * 2;
//             if (any(sample_coords < vec2<i32>(0)) || any(sample_coords >= output_size)) {
//                 continue;
//             }

//             irradiance = textureLoad(denoised_texture_1, sample_coords).rgb;
//             let sample_normal = textureLoad(normal_texture, sample_coords, 0).xyz;
//             let sample_depth = textureLoad(position_texture, sample_coords, 0).w;
//             let sample_instance = textureLoad(instance_material_texture, sample_coords, 0).x;
//             let sample_luminance = luminance(irradiance);

//             let w_normal = normal_weight(normal, sample_normal);
//             let w_depth = depth_weight(depth, sample_depth, depth_gradient, offset);
//             let w_instance = instance_weight(instance, sample_instance);
//             let w_luminance = luminance_weight(lum, sample_luminance, variance);

//             let w = saturate(w_normal * w_depth * w_instance * w_luminance) * frame.kernel[y + 1][x + 1];

//             irradiance_sum += irradiance * w;
//             w_sum += w;
//         }
//     }

//     w_sum = max(w_sum, 0.0001);
//     textureStore(denoised_texture_2, output_coords, vec4<f32>(irradiance_sum / w_sum, w_sum));
// #endif

// #ifdef DENOISER_LEVEL_3
//     // Pass 3, stride 1
//     for (var y = -1; y <= 1; y += 1) {
//         for (var x = -1; x <= 1; x += 1) {
//             let offset = vec2<i32>(x, y);
//             let sample_coords = output_coords + offset * 1;
//             if (any(sample_coords < vec2<i32>(0)) || any(sample_coords >= output_size)) {
//                 continue;
//             }

//             irradiance = textureLoad(denoised_texture_2, sample_coords).rgb;
//             let sample_normal = textureLoad(normal_texture, sample_coords, 0).xyz;
//             let sample_depth = textureLoad(position_texture, sample_coords, 0).w;
//             let sample_instance = textureLoad(instance_material_texture, sample_coords, 0).x;
//             let sample_luminance = luminance(irradiance);

//             let w_normal = normal_weight(normal, sample_normal);
//             let w_depth = depth_weight(depth, sample_depth, depth_gradient, offset);
//             let w_instance = instance_weight(instance, sample_instance);
//             let w_luminance = luminance_weight(lum, sample_luminance, variance);

//             let w = saturate(w_normal * w_depth * w_instance * w_luminance) * frame.kernel[y + 1][x + 1];

//             irradiance_sum += irradiance * w;
//             w_sum += w;
//         }
//     }

//     w_sum = max(w_sum, 0.0001);

//     irradiance = irradiance_sum / w_sum;
//     let color = vec4<f32>(albedo.rgb * irradiance, albedo.a);
//     textureStore(denoised_texture_3, output_coords, color);
// #endif
// }
