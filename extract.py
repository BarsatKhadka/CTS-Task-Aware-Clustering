import re
import numpy as np
import pandas as pd


def load_file_content(filepath):
    try:
        with open(filepath, "r") as f:
            return f.read()
    except FileNotFoundError:
        return None


def extract_die_area(def_text):
    for line in def_text.splitlines():
        if line.startswith("DIEAREA"):
            nums = re.findall(r'\d+', line)
            if len(nums) == 4:
                return int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])
    raise RuntimeError("DIEAREA not found in DEF file")


# ============================================================
# Cell Vocabulary (run once, reuse across all designs)
# ============================================================

GLOBAL_CELL_VOCAB = {}

def build_cell_vocabulary(lib_path="sky130_fd_sc_hd_tt_025C_1v80.lib"):
    global GLOBAL_CELL_VOCAB
    if GLOBAL_CELL_VOCAB:
        return GLOBAL_CELL_VOCAB
    
    print(f"Building cell vocabulary from {lib_path}...")
    lib_text = load_file_content(lib_path)
    if lib_text is None:
        raise FileNotFoundError(f"{lib_path} not found")
    
    cell_names = re.findall(r'cell\s*\("([^"]+)"\)', lib_text)
    GLOBAL_CELL_VOCAB = {name: i for i, name in enumerate(sorted(set(cell_names)))}
    print(f"Vocabulary: {len(GLOBAL_CELL_VOCAB)} unique cells")
    return GLOBAL_CELL_VOCAB


# ============================================================
# Liberty Data Cache
# ============================================================

_lib_content_cache = {}
_cell_data_cache = {}

def get_cell_data(cell_name, lib_path="sky130_fd_sc_hd_tt_025C_1v80.lib"):
    global _cell_data_cache, _lib_content_cache
    
    if cell_name in _cell_data_cache:
        return _cell_data_cache[cell_name]
    
    if "text" not in _lib_content_cache:
        _lib_content_cache["text"] = load_file_content(lib_path)
    
    lib_text = _lib_content_cache["text"]
    cell_marker = f'cell ("{cell_name}") {{'
    start_idx = lib_text.find(cell_marker)
    
    if start_idx == -1:
        _cell_data_cache[cell_name] = {"avg_cap": 0.0, "total_cap": 0.0, "area": 0.0}
        return _cell_data_cache[cell_name]
    
    end_idx = lib_text.find('cell ("', start_idx + len(cell_marker))
    if end_idx == -1:
        end_idx = len(lib_text)
    
    cell_block = lib_text[start_idx:end_idx]
    
    area_match = re.search(r'area\s*:\s*([0-9\.]+)\s*;', cell_block)
    cell_area = float(area_match.group(1)) if area_match else 0.0
    
    pin_blocks = cell_block.split('pin ("')[1:]
    input_caps = []
    for pin in pin_blocks:
        if 'direction : input' in pin or 'direction : "input"' in pin:
            cap_match = re.search(r'capacitance\s*:\s*([0-9\.]+)\s*;', pin)
            if cap_match:
                input_caps.append(float(cap_match.group(1)))
    
    total_cap = sum(input_caps) if input_caps else 0.0
    avg_cap = total_cap / len(input_caps) if input_caps else 0.0
    
    result = {"avg_cap": avg_cap, "total_cap": total_cap, "area": cell_area}
    _cell_data_cache[cell_name] = result
    return result


# ============================================================
# SAIF Activity Cache
# ============================================================

def build_activity_cache(saif_text):
    activity_cache = {}
    if not saif_text:
        return activity_cache
    
    start_pattern = re.compile(r'\(INSTANCE\s+([a-zA-Z0-9_]+)')
    tc_pattern = re.compile(r'\(TC\s+(\d+)\)')
    t1_pattern = re.compile(r'\(T1\s+(\d+)\)')
    t0_pattern = re.compile(r'\(T0\s+(\d+)\)')
    
    for match in start_pattern.finditer(saif_text):
        gate_name = match.group(1)
        start_index = match.start()
        current_index = start_index
        balance = 0
        
        while current_index < len(saif_text):
            char = saif_text[current_index]
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
            if balance == 0:
                break
            current_index += 1
        
        full_block = saif_text[start_index:current_index + 1]
        
        max_tc = 0
        sum_t1 = 0
        sum_t0 = 0
        sum_tc = 0
        non_zero_count = 0
        
        for line in full_block.splitlines():
            if "CLK" in line.upper() or "CLOCK" in line.upper():
                continue
            
            tc_m = tc_pattern.search(line)
            t1_m = t1_pattern.search(line)
            t0_m = t0_pattern.search(line)
            
            if tc_m:
                val = int(tc_m.group(1))
                if val > 0:
                    non_zero_count += 1
                if val > max_tc:
                    max_tc = val
                sum_tc += val
            if t1_m:
                sum_t1 += int(t1_m.group(1))
            if t0_m:
                sum_t0 += int(t0_m.group(1))
        
        total_time = sum_t1 + sum_t0
        activity_cache[gate_name] = {
            "max_toggle_count": np.log1p(max_tc),
            "signal_prob": (sum_t1 / total_time) if total_time > 0 else 0.0,
            "sum_toggle_count": np.log1p(sum_tc),
            "non_zero_count": non_zero_count
        }
    
    return activity_cache

def extract_timing_paths(timing_path_csv, node_to_idx):
    df = pd.read_csv(timing_path_csv)
    df = df[["launch_flop", "capture_flop"]].drop_duplicates()

    valid = df[
        df["launch_flop"].isin(node_to_idx) &
        df["capture_flop"].isin(node_to_idx)
    ]

    skip_edges = np.array([
        [node_to_idx[row.launch_flop], node_to_idx[row.capture_flop]]
        for row in valid.itertuples()
    ])

    dropped = len(df) - len(valid)
    print(f"Unique pairs: {len(valid)}, dropped: {dropped}")
    print(f"Skip edges shape: {skip_edges.shape}")

    return skip_edges


# ============================================================
# Main Extraction Function
# ============================================================

OUTPUT_PINS = {'X', 'Y', 'Q', 'Q_N'}

def extract_graph(def_path, saif_path, clock_port="clk"):
    """
    One-shot extraction of everything needed from a placement.
    Returns nodes, directed edges, undirected edges, and flip-flop indices.
    """
    def_text = load_file_content(def_path)
    saif_text = load_file_content(saif_path)
    
    if def_text is None:
        raise FileNotFoundError(f"DEF not found: {def_path}")
    
    build_cell_vocabulary()
    
    die_x_min, die_y_min, die_x_max, die_y_max = extract_die_area(def_text)
    
    # ---- 1. Identify flip-flops from clock net ----
    clock_pattern = rf'-\s+{re.escape(clock_port)}\s+\(\s+PIN\s+{re.escape(clock_port)}\s+\).*?;'
    clock_match = re.search(clock_pattern, def_text, re.DOTALL)
    
    if not clock_match:
        print(f"Warning: Clock net '{clock_port}' not found.")
        all_flops = set()
    else:
        all_flops = set(re.findall(r'\(\s+((?!PIN)\S+)\s+CLK\s+\)', clock_match.group(0)))
    
    print(f"Flip-flops: {len(all_flops)}")
    
    # ---- 2. Build activity cache ----
    activity_cache = build_activity_cache(saif_text)
    
    # ---- 3. Extract nodes from COMPONENTS ----
    block_match = re.search(r'COMPONENTS\s+\d+\s*;(.*?)END COMPONENTS', def_text, re.DOTALL)
    components_block = block_match.group(1) if block_match else ""
    
    line_pattern = re.compile(r'^\s*-\s+(\S+)\s+(\S+)(.*?);', re.MULTILINE)
    coord_pattern = re.compile(r'\(\s*(-?\d+)\s+(-?\d+)\s*\)')
    ignore_keywords = ["decap", "fill", "tap", "tapvpwrvgnd", "diode", "antenna"]
    
    nodes = []
    
    for match in line_pattern.finditer(components_block):
        inst = match.group(1)
        cell = match.group(2)
        props = match.group(3)
        
        if any(kw in cell.lower() for kw in ignore_keywords):
            continue
        
        coords = coord_pattern.search(props)
        x = int(coords.group(1)) if coords else 0
        y = int(coords.group(2)) if coords else 0
        
        drive_match = re.search(r'_(\d+)$', cell)
        drive_strength = int(drive_match.group(1)) if drive_match else 0
        
        cell_d = get_cell_data(cell)
        saif_d = activity_cache.get(inst, {
            "max_toggle_count": 0.0, "signal_prob": 0.0,
            "sum_toggle_count": 0.0, "non_zero_count": 0
        })
        
        nodes.append({
            "instance_name": inst,
            "cell_name": cell,
            "cell_type_id": GLOBAL_CELL_VOCAB.get(cell, 0),
            "x": x,
            "y": y,
            "cell_area": cell_d["area"],
            "avg_pin_cap": cell_d["avg_cap"],
            "total_pin_cap": cell_d["total_cap"],
            "dist_to_boundaries": [
                x - die_x_min, die_x_max - x,
                die_y_max - y, y - die_y_min
            ],
            "drive_strength": drive_strength,
            "is_sequential": 1 if inst in all_flops else 0,
            "is_buffer": 1 if ("buf" in cell.lower() or "inv" in cell.lower()) else 0,
            "toggle_count": saif_d["max_toggle_count"],
            "sum_toggle_count": saif_d["sum_toggle_count"],
            "signal_prob": saif_d["signal_prob"],
            "non_zero_count": saif_d["non_zero_count"],
        })
    
    # ---- 4. Build node index mapping ----
    node_to_idx = {n["instance_name"]: i for i, n in enumerate(nodes)}
    
    # ---- 5. Extract edges from NETS ----
    nets_match = re.search(r'NETS\s+\d+\s*;(.*?)END NETS', def_text, re.DOTALL)
    nets_block = nets_match.group(1) if nets_match else ""
    
    net_pattern = re.compile(r'^\s*-\s+\S+(.*?)\+ USE', re.MULTILINE | re.DOTALL)
    pin_pattern = re.compile(r'\(\s+(\S+)\s+(\S+)\s+\)')
    
    directed_set = set()
    
    for net_match in net_pattern.finditer(nets_block):
        connections = pin_pattern.findall(net_match.group(1))
        
        driver = None
        loads = []
        
        for inst, pin in connections:
            if inst == 'PIN' or inst not in node_to_idx:
                continue
            idx = node_to_idx[inst]
            if pin in OUTPUT_PINS:
                driver = idx
            else:
                loads.append(idx)
        
        if driver is not None:
            for load in loads:
                directed_set.add((driver, load))
    
    directed = np.array(list(directed_set))
    reverse = np.stack([directed[:, 1], directed[:, 0]], axis=1)
    undirected = np.concatenate([directed, reverse], axis=0)
    skip_edges = extract_timing_paths(timing_path_csv, node_to_idx)
    
    # ---- 6. Compute fan-in / fan-out ----
    n = len(nodes)
    fan_in = [0] * n
    fan_out = [0] * n
    
    for src, dst in directed:
        fan_out[src] += 1
        fan_in[dst] += 1
    
    for i in range(n):
        nodes[i]['fan_in'] = fan_in[i]
        nodes[i]['fan_out'] = fan_out[i]
    
    # ---- 7. Flip-flop indices ----
    flop_indices = [i for i, nd in enumerate(nodes) if nd['is_sequential'] == 1]
    
    # ---- Summary ----
    print(f"Nodes: {len(nodes)}")
    print(f"Directed edges: {len(directed)}")
    print(f"Undirected edges: {len(undirected)}")
    print(f"Flip-flops: {len(flop_indices)}")
    print(f"Fan-in  — max: {max(fan_in)}, avg: {sum(fan_in)/n:.1f}")
    print(f"Fan-out — max: {max(fan_out)}, avg: {sum(fan_out)/n:.1f}")
    
    return {
        'nodes': nodes,
        'node_to_idx': node_to_idx,
        'directed_edges': directed,
        'undirected_edges': undirected,
        'flop_indices': flop_indices,
        'skip_edges': skip_edges
    }


# ============================================================
# Run it
# ============================================================

# file_name = "aes_run_20260305_181833"
# def_path = f"./CTS-Bench/runs/{file_name}/11-openroad-detailedplacement/aes.def"
# saif_path = f"./CTS-Bench/runs/{file_name}/aes.saif"
# timing_path_csv = f"./CTS-Bench/runs/{file_name}/timing_paths.csv"

# graph = extract_graph(def_path, saif_path, clock_port="clk")

# nodes = graph['nodes']
# directed = graph['directed_edges']
# undirected = graph['undirected_edges']
# flops = graph['flop_indices']
# node_to_idx = graph['node_to_idx']

# print("=" * 80)
# print("EXTRACTION VERIFICATION")
# print("=" * 80)

# # Node counts
# print(f"\nTotal nodes: {len(nodes)}")
# print(f"Flip-flops: {len(flops)}")
# print(f"Buffers: {sum(1 for n in nodes if n['is_buffer'] == 1)}")
# print(f"Directed edges: {len(directed)}")
# print(f"Undirected edges: {len(undirected)}")

# # Feature check on first 10 nodes
# print(f"\n{'Instance':<20} {'Cell':<28} {'ID':<5} {'X':<8} {'Y':<8} {'Area':<8} {'AvgCap':<8} {'Drv':<4} {'Seq':<4} {'Buf':<4} {'FI':<5} {'FO':<5} {'TC':<8}")
# print("-" * 140)
# for n in nodes[:10]:
#     print(f"{n['instance_name']:<20} {n['cell_name']:<28} {n['cell_type_id']:<5} {n['x']:<8} {n['y']:<8} {n['cell_area']:<8.2f} {n['avg_pin_cap']:<8.4f} {n['drive_strength']:<4} {n['is_sequential']:<4} {n['is_buffer']:<4} {n['fan_in']:<5} {n['fan_out']:<5} {n['toggle_count']:<8.2f}")

# # Edge check
# idx_to_node = {i: n['instance_name'] for i, n in enumerate(nodes)}
# print(f"\nSample directed edges:")
# for src, dst in directed[:5]:
#     print(f"  {idx_to_node[src]} -> {idx_to_node[dst]}")

# # Fan-in/out distribution
# fan_ins = [n['fan_in'] for n in nodes]
# fan_outs = [n['fan_out'] for n in nodes]
# print(f"\nFan-in  — min: {min(fan_ins)}, max: {max(fan_ins)}, avg: {sum(fan_ins)/len(fan_ins):.1f}")
# print(f"Fan-out — min: {min(fan_outs)}, max: {max(fan_outs)}, avg: {sum(fan_outs)/len(fan_outs):.1f}")

# # Verify node_to_idx consistency
# print(f"\nnode_to_idx size: {len(node_to_idx)}")
# print(f"Matches node count: {len(node_to_idx) == len(nodes)}")

# # Verify all edge indices are valid
# max_idx = len(nodes) - 1
# edge_max = max(directed.max(), undirected.max())
# edge_min = min(directed.min(), undirected.min())
# print(f"Edge index range: [{edge_min}, {edge_max}], Valid range: [0, {max_idx}]")
# print(f"All edges valid: {edge_min >= 0 and edge_max <= max_idx}")