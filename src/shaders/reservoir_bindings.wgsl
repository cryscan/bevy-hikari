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

fn load_previous_reservoir(uv: vec2<f32>, reservoir_size: vec2<i32>) -> Reservoir {
    var r = empty_reservoir();
    if (all(abs(uv - 0.5) < vec2<f32>(0.5))) {
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
    if (all(abs(uv - 0.5) < vec2<f32>(0.5))) {
        let coords = vec2<i32>(uv * vec2<f32>(reservoir_size));
        let index = coords.x + reservoir_size.x * coords.y;
        let packed = previous_spatial_reservoir_buffer.data[index];
        r = unpack_reservoir(packed);
    }
    return r;
}

fn load_spatial_reservoir(index: i32) -> Reservoir {
    let packed = spatial_reservoir_buffer.data[index];
    return unpack_reservoir(packed);
}

fn store_spatial_reservoir(index: i32, r: Reservoir) {
    spatial_reservoir_buffer.data[index] = pack_reservoir(r);
}
