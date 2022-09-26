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

    // Temporal accumulation
    let cache_uv = uv - velocity_uv.xy;

    var w_sum = 0.0001;
    var sum = 0.0;
    var temporal_factor = TEMPORAL_FACTOR_MIN_MAX.y;
    let cache_miss = any(abs(cache_uv - 0.5) > vec2<f32>(0.5));
    if (!cache_miss) {
        for (var i = 0u; i < 25u; i += 1u) {
            let offset = vec2<f32>(frame.kernel[i].xy) / vec2<f32>(size);
            let cache = textureSampleLevel(shadow_cache_texture, shadow_cache_sampler, cache_uv + offset, 0.0);

            let dp = cache.xyz - position.xyz;
            var d2 = dot(dp, dp);
            let wp = min(exp(-d2 / BILATERAL_POSITION_SIGMA), 1.0);

            let w = wp * frame.kernel[i].z;
            w_sum += w;
            sum += cache.w * w;
        }
        sum /= w_sum;
    }
    if (cache_miss || w_sum < 2.0E-4) {
        temporal_factor = TEMPORAL_FACTOR_MIN_MAX.x;
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