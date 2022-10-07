#import bevy_pbr::mesh_view_types
#import bevy_pbr::mesh_types

struct PreviousView {
    view_proj: mat4x4<f32>,
    inverse_view_proj: mat4x4<f32>,
};

struct PreviousMesh {
    model: mat4x4<f32>,
    inverse_transpose_model: mat4x4<f32>,
};

struct InstanceIndex {
    instance: u32,
    material: u32
};

struct Frame {
    number: u32,
    kernel: array<vec3<f32>, 25>,
};

@group(0) @binding(0)
var<uniform> view: View;
@group(0) @binding(1)
var<uniform> previous_view: PreviousView;
@group(0) @binding(2)
var<uniform> frame: Frame;

@group(1) @binding(0)
var<uniform> mesh: Mesh;
@group(1) @binding(1)
var<uniform> previous_mesh: PreviousMesh;
@group(1) @binding(2)
var<uniform> instance_index: InstanceIndex;

#import bevy_pbr::mesh_functions

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) previous_world_position: vec4<f32>,
    @location(2) world_normal: vec3<f32>,
    @location(3) uv: vec2<f32>,
};

// https://en.wikipedia.org/wiki/Halton_sequence#Implementation_in_pseudocode
fn halton(base: u32, index: u32) -> f32
{
	var result = 0.;
	var f = 1.;
	for (var id = index; id > 0u; id /= base)
	{
		f = f / f32(base);
		result += f * f32(id % base);
	}
	return result;
}

fn frame_jitter() -> vec2<f32> {
    let index = frame.number % 64u;
    let delta = vec2<f32>(halton(2u, index), halton(3u, index)) - vec2<f32>(0.5);
    return delta;
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var model = mesh.model;
    let vertex_position = vec4<f32>(vertex.position, 1.0);

    var projection = view.projection;
    let jitter = frame_jitter();
    projection[3][0] += 16.0 * jitter.x / view.width;
    projection[3][1] += 16.0 * jitter.y / view.height;

    var out: VertexOutput;
    out.world_position = mesh_position_local_to_world(model, vertex_position);
    out.previous_world_position = mesh_position_local_to_world(previous_mesh.model, vertex_position);
    out.world_normal = mesh_normal_local_to_world(vertex.normal);
    out.clip_position = projection * view.inverse_view * out.world_position;
    out.uv = vertex.uv;

    return out;
}

struct FragmentOutput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) instance_material: vec2<u32>,
    @location(3) uv: vec2<f32>,
    @location(4) velocity: vec2<f32>,
};

fn clip_to_uv(clip: vec4<f32>) -> vec2<f32> {
    var uv = clip.xy / clip.w;
    uv = (uv + 1.0) * 0.5;
    uv.y = 1.0 - uv.y;
    return uv;
}

@fragment
fn fragment(in: VertexOutput) -> FragmentOutput {
    let clip_position = view.view_proj * in.world_position;
    let previous_clip_position = previous_view.view_proj * in.previous_world_position;
    let velocity = clip_to_uv(clip_position) - clip_to_uv(previous_clip_position);

    var out: FragmentOutput;
    out.position = in.world_position;
    out.normal = vec4<f32>(in.world_normal, 1.0);
    out.instance_material = vec2<u32>(instance_index.instance, instance_index.material);
    out.uv = in.uv;
    out.velocity = velocity * 10000.0;
    return out;
}
