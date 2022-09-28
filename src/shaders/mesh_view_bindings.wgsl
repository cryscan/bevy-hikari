#define_import_path bevy_hikari::mesh_view_bindings

#import bevy_pbr::mesh_view_types

@group(0) @binding(0)
var<uniform> view: View;
@group(0) @binding(1)
var<uniform> lights: Lights;