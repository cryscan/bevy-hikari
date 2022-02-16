struct Volume {
    min: vec3<f32>;
    max: vec3<f32>;
    projection: mat4x4<f32>;
};

struct List {
    data: array<u32, 4194304>;
    counter: atomic<u32>;
};

struct Node {
    children: u32;
};

struct Octree {
    nodes: array<Node, 524288>;
    levels: array<u32, 8>;
    node_counter: atomic<u32>;
    level_counter: atomic<u32>;
};

[[group(0), binding(0)]]
var<uniform> volume: Volume;

[[group(0), binding(1)]]
var<storage, read_write> fragments: List;

[[group(0), binding(2)]]
var<storage, read_write> octree: Octree;

[[group(0), binding(3)]]
var texture: texture_storage_1d<rgba8unorm, read_write>;

let node_mask: u32 = 0x7FFFFFFFu;

fn position_level_index(position: u32, level: u32) -> u32 {
    let mask: u32 = 1u << (7u - level);
    let x = u32(bool(position & (mask << 24u)));
    let y = u32(bool(position & (mask << 16u)));
    let z = u32(bool(position & (mask << 8u)));
    return x << 2u + y << 1u + z;
}

fn is_leaf_node(id: u32) -> bool {
    return (octree.nodes[id].children & node_mask) == 0u;
}

[[stage(compute), workgroup_size(1)]]
fn init() {
    atomicStore(&fragments.counter, 0u);

    // Pre-expand level 0
    atomicStore(&octree.node_counter, 9u);
    atomicStore(&octree.level_counter, 2u);

    octree.nodes[0].children = 1u;
    octree.levels[0] = 1u;

    for (var i = 1u; i < 9u; i = i + 1u) {
        octree.nodes[i].children = 0u;
    }
    octree.levels[1] = 9u;
}

// Called per fragment in the fragment list
[[stage(compute), workgroup_size(64)]]
fn mark_nodes([[builtin(global_invocation_id)]] invocation_id: vec3<u32>) {
    if (invocation_id.x >= atomicLoad(&fragments.counter)) { 
        return; 
    }
    let position = fragments.data[invocation_id.x];

    // Traverse the tree to the leaf
    var node_id: u32 = 0u;
    var level: u32;
    for (level = 0u; level < 8u; level = level + 1u) {
        // If this node is not expanded (pointer = 0)
        let node = &octree.nodes[node_id];
        let children = (*node).children & node_mask;
        if (children == 0u) {
            // Mark this node (set the highest bit to 1)
            (*node).children = ~node_mask;
            break;
        }
        node_id = children + position_level_index(position, level);
    }
}

// Called per octree node
[[stage(compute), workgroup_size(64)]]
fn expand_nodes([[builtin(global_invocation_id)]] invocation_id: vec3<u32>) {
    if (invocation_id.x >= atomicLoad(&octree.node_counter)) {
        return; 
    }

    storageBarrier();

    let node = &octree.nodes[invocation_id.x];

    // If this node has been marked, allocate 8 continuous child nodes
    if (((*node).children & ~node_mask) != 0u) {
        (*node).children = atomicAdd(&octree.node_counter, 8u);

        for (var i = 0u; i < 8u; i = i + 1u) {
            let child_id = (*node).children + i;
            octree.nodes[child_id].children = 0u;
        }
    }

    storageBarrier();

    // Mark the end of the current level
    if (invocation_id.x == 0u) {
        let level = atomicAdd(&octree.level_counter, 1u);
        octree.levels[level] = atomicLoad(&octree.node_counter);
    }
}

// The children of the final level nodes are not nodes, but texture units
[[stage(compute), workgroup_size(64)]]
fn expand_final_level_nodes([[builtin(global_invocation_id)]] invocation_id: vec3<u32>) {
    let node_count = atomicLoad(&octree.node_counter);
    if (invocation_id.x >= node_count) {
        return; 
    }

    storageBarrier();

    let node = &octree.nodes[invocation_id.x];
    if (((*node).children & ~node_mask) != 0u) {
        (*node).children = atomicAdd(&octree.node_counter, 8u);
    }

    // Recover previous node counter
    storageBarrier();

    if (invocation_id.x == 0u) {
        atomicStore(&octree.node_counter, node_count);
    }
}

[[stage(compute), workgroup_size(64)]]
fn build_mipmap([[builtin(global_invocation_id)]] invocation_id: vec3<u32>) {
    let node_id = invocation_id.x;
    if (node_id >= atomicLoad(&octree.node_counter)) {
        return; 
    }

    for (var level = 7u; level >= 1u; level = level - 1u) {
        if (node_id < octree.levels[level] && node_id > octree.levels[level - 1u]) {
            let node = &octree.nodes[node_id];

            var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            if (!is_leaf_node(node_id)) {
                for (var i = 0u; i < 8u; i = i + 1u) {
                    let child_id = (*node).children + i;
                    let child_color = textureLoad(texture, i32(child_id));
                    color = color + child_color;
                }
                color = color / 8.0;
            }
            textureStore(texture, i32(node_id), color);
        }

        storageBarrier();
    }

    // For level 0 only
    {
        var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        let node = &octree.nodes[0];
        for (var i = 0u; i < 8u; i = i + 1u) {
            let child_id = (*node).children + i;
            let child_color = textureLoad(texture, i32(child_id));
            color = color + child_color;
        }
        color = color / 8.0;
        textureStore(texture, i32(node_id), color);
    }
}