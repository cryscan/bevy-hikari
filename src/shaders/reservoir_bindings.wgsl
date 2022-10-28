#define_import_path bevy_hikari::reservoir_bindings

#import bevy_hikari::reservoir_types

@group(6) @binding(0)
var<storage, read> previous_reservoir_buffer: Reservoirs;
@group(6) @binding(1)
var<storage, read_write> reservoir_buffer: Reservoirs;
@group(6) @binding(2)
var<storage, read> previous_spatial_reservoir_buffer: Reservoirs;
@group(6) @binding(3)
var<storage, read_write> spatial_reservoir_buffer: Reservoirs;

@group(7) @binding(0)
var<storage, read> direct_reservoir_cache: Reservoirs;
@group(7) @binding(1)
var<storage, read> emissive_reservoir_cache: Reservoirs;
