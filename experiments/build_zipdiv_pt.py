"""
Build zipdiv processed_graphs .pt files from DEF+SAIF+timing.

Features computed:
  X[N, 14] (subset of 18-dim, excludes cell_area, avg_pin_cap, total_pin_cap, drive_strength):
    0:x_norm   1:y_norm   2-5:dist_boundaries   6:log1p(toggle)
    7:log1p(sum_toggle)  8:signal_prob  9:non_zero
    10:is_sequential  11:is_buffer   12:log1p(fan_in)   13:log1p(fan_out)
    
  A_wire_csr: sparse wire adjacency from NETS section
  A_skip_csr: sparse timing-path adjacency from timing_paths.csv
  p_indices: 2-hop wire mask (build_X_hop_mask)
"""
import os, sys, re, pickle
import numpy as np
import torch
import scipy.sparse as sp
from collections import defaultdict, Counter
sys.path.insert(0, '/home/rain/CTS-Task-Aware-Clustering')

BASE = '/home/rain/CTS-Task-Aware-Clustering'
PLACEMENT_DIR = f'{BASE}/dataset_with_def/placement_files'
OUTPUT_DIR = f'{BASE}/processed_graphs_zipdiv'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLOCK_PORT_ZIPDIV = 'i_clk'

def build_X_hop_mask(n_nodes, edges_u, edges_v, hop_mask_len=2):
    """Build multi-hop wire mask from adjacency."""
    A = sp.csr_matrix((np.ones(len(edges_u), dtype=bool), (edges_u, edges_v)),
                      shape=(n_nodes, n_nodes))
    omega = A.copy(); temp = A.copy()
    for _ in range(2, hop_mask_len + 1):
        temp = temp.dot(A); omega = omega + temp
    omega.setdiag(0); omega.eliminate_zeros()
    coo = omega.tocoo()
    return coo.row, coo.col


def build_zipdiv_graph(placement_id, def_path, saif_path, tp_path):
    """Parse DEF+SAIF+timing → build PT-compatible dict."""
    print(f"  Processing {placement_id}...", flush=True)
    
    # --- Parse DEF ---
    with open(def_path) as f: def_text = f.read()
    
    units_m = re.search(r'UNITS DISTANCE MICRONS (\d+)', def_text)
    units = int(units_m.group(1)) if units_m else 1000
    
    die_m = re.search(r'DIEAREA\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)', def_text)
    x0, y0, x1, y1 = [float(v)/units for v in die_m.groups()]
    die_w, die_h = x1 - x0, y1 - y0
    
    # FF names from clock net
    clock_pat = rf'-\s+{re.escape(CLOCK_PORT_ZIPDIV)}\s+\(\s+PIN\s+{re.escape(CLOCK_PORT_ZIPDIV)}\s+\).*?;'
    cm = re.search(clock_pat, def_text, re.DOTALL)
    ff_names_clk = set(re.findall(r'\(\s+((?!PIN)\S+)\s+CLK\s+\)', cm.group(0))) if cm else set()
    
    # All cell positions and types
    comp_pat = re.compile(
        r'-\s+(\S+)\s+(sky130_fd_sc_hd__\S+)\s+\+\s+(?:PLACED|FIXED)\s+\(\s*([\d]+)\s+([\d]+)\s*\)')
    cell_info = {}  # inst → {type, x, y, norm_x, norm_y, is_ff, is_buf}
    for m in comp_pat.finditer(def_text):
        inst, ctype, cx, cy = m.group(1), m.group(2), float(m.group(3))/units, float(m.group(4))/units
        short = ctype.replace('sky130_fd_sc_hd__', '')
        is_ff  = short.startswith('df') or short.startswith('ff')
        is_buf = short.startswith('buf')
        cell_info[inst] = {
            'type': short, 'x': cx, 'y': cy,
            'norm_x': (cx - x0) / die_w, 'norm_y': (cy - y0) / die_h,
            'is_ff': is_ff, 'is_buf': is_buf,
            'toggle': 0.0, 'sum_toggle': 0.0, 'signal_prob': 0.0, 'non_zero': 0.0,
            'fan_in': [], 'fan_out': [],
        }
    
    # NETS section → wire edges + fan_in/fan_out
    nets_start = def_text.find('NETS')
    nets_end   = def_text.find('END NETS')
    nets_section = def_text[nets_start:nets_end] if nets_start >= 0 else ''
    conn_pat = re.compile(r'\(\s+(\S+)\s+(\S+)\s+\)')
    ff_gate_pat = re.compile(r'^_\d+_$')
    
    wire_edges_u, wire_edges_v = [], []  # inst_idx → inst_idx
    
    for net_block in nets_section.split(';'):
        conns = conn_pat.findall(net_block)
        net_ff_d, net_ff_q, net_logic = [], [], []
        for inst, pin in conns:
            if inst not in cell_info: continue
            if inst in ff_names_clk:
                if pin == 'D': net_ff_d.append(inst)
                elif pin == 'Q': net_ff_q.append(inst)
            elif ff_gate_pat.match(inst):
                net_logic.append(inst)
        for ff in net_ff_d: cell_info[ff]['fan_in'].extend(net_logic)
        for ff in net_ff_q: cell_info[ff]['fan_out'].extend(net_logic)
    
    # Build ordered list: FFs first, then logic
    ff_list   = [n for n in cell_info if n in ff_names_clk]
    logic_list = [n for n in cell_info if n not in ff_names_clk]
    all_nodes  = ff_list + logic_list
    node2idx   = {n: i for i, n in enumerate(all_nodes)}
    n_total    = len(all_nodes)
    n_ff       = len(ff_list)
    
    # Wire edges (from fan_in/fan_out)
    for ff in ff_list:
        u = node2idx[ff]
        for lg in cell_info[ff].get('fan_in', []):
            if lg in node2idx:
                v = node2idx[lg]
                wire_edges_u.extend([u, v]); wire_edges_v.extend([v, u])
        for lg in cell_info[ff].get('fan_out', []):
            if lg in node2idx:
                v = node2idx[lg]
                wire_edges_u.extend([u, v]); wire_edges_v.extend([v, u])
    
    # --- Parse SAIF ---
    try:
        with open(saif_path) as f: saif_text = f.read()
        inst_pat = re.compile(r'\(INSTANCE\s+(\S+)')
        tc_pat   = re.compile(r'\(TC\s+(\d+)\)')
        sp_pat   = re.compile(r'\(T1\s+(\d+)\)')  # time in high state
        
        for m in inst_pat.finditer(saif_text):
            inst = m.group(1)
            if inst not in cell_info: continue
            start = m.start(); bal = 0; cur = start
            while cur < len(saif_text):
                if saif_text[cur] == '(': bal += 1
                elif saif_text[cur] == ')': bal -= 1
                if bal == 0: break
                cur += 1
            block = saif_text[start:cur+1]
            max_tc = 0; t1_sum = 0; n_pins = 0
            for line in block.splitlines():
                if 'CLK' in line.upper() or 'CLOCK' in line.upper(): continue
                tm = tc_pat.search(line)
                sm = sp_pat.search(line)
                if tm:
                    val = int(tm.group(1)); max_tc = max(max_tc, val)
                    n_pins += 1
                if sm:
                    t1_sum += int(sm.group(1))
            cell_info[inst]['toggle'] = np.log1p(max_tc)
            cell_info[inst]['sum_toggle'] = float(max_tc)
            if n_pins > 0:
                cell_info[inst]['signal_prob'] = t1_sum / (n_pins * (max_tc + 1) + 1e-9)
                cell_info[inst]['non_zero'] = float(max_tc > 0)
    except Exception as e:
        print(f"    SAIF error: {e}")
    
    # --- Build X features [n_total, 14] ---
    rows = []
    for inst in all_nodes:
        ci = cell_info[inst]
        nx, ny = ci['norm_x'], ci['norm_y']
        dist = [nx, 1-nx, ny, 1-ny]  # dist to left, right, bottom, top boundary
        fan_in_n  = len(ci.get('fan_in', []))
        fan_out_n = len(ci.get('fan_out', []))
        row = [
            nx, ny,
            dist[0], dist[1], dist[2], dist[3],
            ci['toggle'],
            np.log1p(ci['sum_toggle']),
            ci['signal_prob'],
            ci['non_zero'],
            float(ci['is_ff']),
            float(ci['is_buf']),
            np.log1p(fan_in_n),
            np.log1p(fan_out_n),
        ]
        rows.append(row)
    
    X = torch.tensor(rows, dtype=torch.float32)
    
    # --- Skip adjacency from timing paths ---
    skip_u, skip_v = [], []
    try:
        import pandas as pd
        tp = pd.read_csv(tp_path)
        for _, r in tp.iterrows():
            lf, cf = r['launch_flop'], r['capture_flop']
            if lf in node2idx and cf in node2idx:
                u, v = node2idx[lf], node2idx[cf]
                skip_u.extend([u, v]); skip_v.extend([v, u])
    except Exception as e:
        print(f"    Timing error: {e}")
    
    # Deduplicate
    wire_edges = list(set(zip(wire_edges_u, wire_edges_v)))
    wu = [e[0] for e in wire_edges]; wv = [e[1] for e in wire_edges]
    skip_edges = list(set(zip(skip_u, skip_v)))
    su = [e[0] for e in skip_edges]; sv = [e[1] for e in skip_edges]
    
    # Sparse adjacency matrices
    if wu:
        A_wire = sp.csr_matrix((np.ones(len(wu)), (wu, wv)), shape=(n_total, n_total))
    else:
        A_wire = sp.csr_matrix((n_total, n_total))
    A_wire_t = torch.sparse_csr_tensor(
        torch.tensor(A_wire.indptr), torch.tensor(A_wire.indices),
        torch.ones(A_wire.nnz), size=(n_total, n_total))
    
    if su:
        A_skip = sp.csr_matrix((np.ones(len(su)), (su, sv)), shape=(n_total, n_total))
    else:
        A_skip = sp.csr_matrix((n_total, n_total))
    A_skip_t = torch.sparse_csr_tensor(
        torch.tensor(A_skip.indptr), torch.tensor(A_skip.indices),
        torch.ones(A_skip.nnz), size=(n_total, n_total))
    
    # 2-hop mask
    if wu:
        print(f"    Building 2-hop mask ({n_total} nodes, {len(wu)} wire edges)...", flush=True)
        p_rows, p_cols = build_X_hop_mask(n_total, np.array(wu), np.array(wv))
        p_indices = torch.tensor(np.stack([p_rows, p_cols]), dtype=torch.long)
    else:
        p_indices = torch.zeros((2, 0), dtype=torch.long)
    
    print(f"    n_nodes={n_total}, n_ff={n_ff}, wire_edges={len(wu)//2}, skip_edges={len(su)//2}", flush=True)
    print(f"    p_indices shape: {p_indices.shape}", flush=True)
    
    return {
        'name': 'zipdiv',
        'run_folder': placement_id,
        'num_nodes': n_total,
        'n_ff': n_ff,
        'X': X,
        'X_cell_ids': torch.zeros(n_total, dtype=torch.long),
        'A_wire_csr': A_wire_t,
        'A_skip_csr': A_skip_t,
        'p_indices': p_indices,
        'raw_areas': torch.ones(n_total, 1),
        'cts_runs': [],  # filled separately from CSV
    }


if __name__ == '__main__':
    import pandas as pd
    df_test = pd.read_csv(f'{BASE}/dataset_with_def/unified_manifest_normalized_test.csv')
    df_test = df_test.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
    
    print(f"Building .pt graphs for {df_test['placement_id'].nunique()} zipdiv placements...")
    
    for pid in df_test['placement_id'].unique():
        row = df_test[df_test['placement_id'] == pid].iloc[0]
        def_path  = row['def_path'].replace('../dataset_with_def/placement_files', PLACEMENT_DIR)
        saif_path = row['saif_path'].replace('../dataset_with_def/placement_files', PLACEMENT_DIR)
        tp_path   = os.path.join(os.path.dirname(def_path), 'timing_paths.csv')
        
        if not os.path.exists(def_path):
            print(f"  Missing DEF: {def_path}"); continue
        
        out_path = f'{OUTPUT_DIR}/{pid}.pt'
        if os.path.exists(out_path):
            print(f"  Already exists: {out_path}"); continue
        
        try:
            data = build_zipdiv_graph(pid, def_path, saif_path, tp_path)
            torch.save(data, out_path)
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  Error for {pid}: {e}")
    
    print("Done!")
