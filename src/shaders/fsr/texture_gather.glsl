AF4 fakeTextureGather(texture2D tex, sampler smp, vec2 p, int comp) {
	vec2 ps = (1.0 / vec2(textureSize(sampler2D(tex, smp), 0))) / 2.0;
	vec4 sp3 = texture(sampler2D(tex, smp), p + ps); // top-right
	vec4 sp1 = texture(sampler2D(tex, smp), p - ps); // bottom-left
	ps.y *= -1.0;
	vec4 sp2 = texture(sampler2D(tex, smp), p + ps); // bottom-right
	ps *= -1.0;
	vec4 sp4 = texture(sampler2D(tex, smp), p + ps); // top-left
	return AF4(sp1[comp], sp2[comp], sp3[comp], sp4[comp]);
}