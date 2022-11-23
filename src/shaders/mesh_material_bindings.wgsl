#define_import_path bevy_hikari::mesh_material_bindings

#import bevy_hikari::mesh_material_types

@group(2) @binding(0)
var<storage> vertex_buffer: Vertices;
@group(2) @binding(1)
var<storage> primitive_buffer: Primitives;
@group(2) @binding(2)
var<storage> asset_node_buffer: Nodes;
@group(2) @binding(3)
var<storage> alias_table_buffer: AliasTable;
@group(2) @binding(4)
var<storage> instance_buffer: Instances;
@group(2) @binding(5)
var<storage> instance_node_buffer: Nodes;
@group(2) @binding(6)
var<storage> material_buffer: Materials;
@group(2) @binding(7)
var<storage> emissive_node_buffer: Nodes;
@group(2) @binding(8)
var<storage> emissive_buffer: Emissives;
