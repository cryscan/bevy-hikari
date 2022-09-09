#define_import_path bevy_hikari::ray_tracing_bindings

#import bevy_hikari::ray_tracing_types

@group(1) @binding(0)
var<storage> vertex_buffer: Vertices;
@group(1) @binding(1)
var<storage> primitive_buffer: Primitives;
@group(1) @binding(2)
var<storage> asset_node_buffer: Nodes;
@group(1) @binding(3)
var<storage> instance_buffer: Instances;
@group(1) @binding(4)
var<storage> instance_node_buffer: Nodes;