#define_import_path bevy_hikari::mesh_view_bindings

#import bevy_pbr::mesh_view_types
#import bevy_hikari::mesh_view_types

@group(0) @binding(0)
var<uniform> frame: Frame;
@group(0) @binding(1)
var<uniform> view: View;
@group(0) @binding(2)
var<uniform> previous_view: PreviousView;
@group(0) @binding(3)
var<uniform> lights: Lights;