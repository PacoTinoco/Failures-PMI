"""
Router: Inventory Movements
Análisis de movimientos de inventario — parsing de Excel al vuelo (sin BD).
Detecta anomalías por usuario y por picos de cantidad.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from typing import Optional
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
import math
import json

router = APIRouter(prefix="/inventory", tags=["Inventory"])

SYSTEM_USERS = {"ADMIN", "System"}
HEADER_ROWS_TO_SKIP = 4  # 'PHILIP MORRIS - CONFIDENTIAL', 'Inventory Movements', 2 empty

COLUMN_MAP = {
    'Container': 'container',
    'Material Code': 'material_code',
    'Source W|H': 'source_wh',
    'Source  Loc': 'source_loc',
    'Dest W|H': 'dest_wh',
    'Dest Loc': 'dest_loc',
    'Old Qty': 'old_qty',
    'New Qty': 'new_qty',
    'UOM': 'uom',
    'SAP Batch': 'sap_batch',
    'Wip Order No': 'wip_order_no',
    'Order No': 'order_no',
    'Src Inventory Status': 'src_inv_status',
    'Dst Inventory Status': 'dst_inv_status',
    'Date of creation': 'date',
    'User': 'user',
    'Transaction Context': 'tx_context',
    'Qty Adjusted': 'qty_adjusted',
    'Qty Allocated': 'qty_allocated',
}


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy/pandas types."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        return super().default(obj)


def safe_val(v):
    """Convert pandas/numpy values to JSON-safe native Python types."""
    if v is None:
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        if np.isnan(v):
            return None
        return float(v)
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    return v


@router.post("/analyze")
async def analyze_inventory(
    file: UploadFile = File(...),
    anomaly_threshold: float = Query(50.0, description="Qty change threshold to flag as anomaly"),
):
    """Parse inventory movements Excel and return structured analysis."""
    contents = await file.read()

    try:
        df = pd.read_excel(BytesIO(contents), header=None, skiprows=HEADER_ROWS_TO_SKIP)
    except Exception as e:
        raise HTTPException(400, f"Error reading Excel: {str(e)}")

    # Row 0 after skip = headers
    raw_headers = df.iloc[0].tolist()
    df = df.iloc[1:].copy()
    df.columns = raw_headers

    # Rename columns
    rename = {}
    for orig, mapped in COLUMN_MAP.items():
        if orig in df.columns:
            rename[orig] = mapped
    df = df.rename(columns=rename)

    # Drop fully empty rows
    df = df.dropna(how='all')

    # Parse date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date')

    # Compute qty_diff
    df['qty_diff'] = pd.to_numeric(df.get('new_qty', 0), errors='coerce') - \
                     pd.to_numeric(df.get('old_qty', 0), errors='coerce')

    # Track original Excel row (1-indexed, +4 header rows + 1 column header row)
    df['excel_row'] = range(HEADER_ROWS_TO_SKIP + 2, HEADER_ROWS_TO_SKIP + 2 + len(df))

    # Detect anomalies
    df['is_anomaly_user'] = ~df['user'].isin(SYSTEM_USERS)
    df['is_anomaly_qty'] = df['qty_diff'].abs() > anomaly_threshold
    df['is_anomaly'] = df['is_anomaly_user'] | df['is_anomaly_qty']

    # Build movements list
    known_cols = list(COLUMN_MAP.values()) + ['qty_diff', 'excel_row', 'is_anomaly', 'is_anomaly_user', 'is_anomaly_qty']
    movements = []
    for _, row in df.iterrows():
        mov = {}
        for col in known_cols:
            if col in row.index:
                mov[col] = safe_val(row[col])
        # Include ALL original columns for "full row" display
        for orig_col in raw_headers:
            # Skip NaN / None / empty headers
            if orig_col is None or (isinstance(orig_col, float) and math.isnan(orig_col)):
                continue
            if not orig_col or orig_col in rename:
                continue
            key = str(orig_col).strip().lower().replace(' ', '_').replace('|', '')
            if key and key != 'nan' and key not in mov:
                val = row.get(orig_col)
                # If duplicate columns exist, row.get() may return a Series — take first value
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                mov[f"raw_{key}"] = safe_val(val)
        movements.append(mov)

    # ── Container summaries ──
    containers = {}
    for mov in movements:
        c = mov.get('container')
        if not c:
            continue
        if c not in containers:
            containers[c] = {
                'container': c,
                'movements': [],
                'total_moves': 0,
                'initial_qty': None,
                'final_qty': None,
                'anomaly_count': 0,
                'users': set(),
                'machines': set(),
                'sap_batches': set(),
            }
        containers[c]['movements'].append(mov)
        containers[c]['total_moves'] += 1
        if mov.get('is_anomaly'):
            containers[c]['anomaly_count'] += 1
        if mov.get('user'):
            containers[c]['users'].add(mov['user'])
        if mov.get('dest_loc'):
            containers[c]['machines'].add(mov['dest_loc'])
        if mov.get('sap_batch'):
            containers[c]['sap_batches'].add(mov['sap_batch'])

    container_summaries = []
    for c, info in containers.items():
        movs = info['movements']
        first_old = movs[0].get('old_qty') if movs else None
        last_new = movs[-1].get('new_qty') if movs else None
        net = round(last_new - first_old, 3) if first_old is not None and last_new is not None else None

        container_summaries.append({
            'container': c,
            'total_moves': info['total_moves'],
            'initial_qty': first_old,
            'final_qty': last_new,
            'net_change': net,
            'anomaly_count': info['anomaly_count'],
            'users': sorted(info['users']),
            'machines': sorted(info['machines']),
            'sap_batches': sorted(info['sap_batches']),
            'date_start': movs[0].get('date') if movs else None,
            'date_end': movs[-1].get('date') if movs else None,
        })

    # ── Filters metadata ──
    all_containers = sorted(set(m.get('container') for m in movements if m.get('container')))
    all_dest_locs = sorted(set(m.get('dest_loc') for m in movements if m.get('dest_loc')))
    all_users = sorted(set(m.get('user') for m in movements if m.get('user')))
    all_tx_contexts = sorted(set(m.get('tx_context') for m in movements if m.get('tx_context')))
    all_sap_batches = sorted(set(m.get('sap_batch') for m in movements if m.get('sap_batch')))

    # ── Global stats ──
    total_anomalies = sum(1 for m in movements if m.get('is_anomaly'))
    qty_diffs = [m['qty_diff'] for m in movements if m.get('qty_diff') is not None]

    result = {
        "total_movements": len(movements),
        "total_containers": len(all_containers),
        "total_anomalies": total_anomalies,
        "anomaly_threshold": anomaly_threshold,
        "avg_qty_diff": round(sum(qty_diffs) / len(qty_diffs), 3) if qty_diffs else 0,
        "movements": movements,
        "container_summaries": container_summaries,
        "filters": {
            "containers": all_containers,
            "dest_locs": all_dest_locs,
            "users": all_users,
            "tx_contexts": all_tx_contexts,
            "sap_batches": all_sap_batches,
        },
    }
    # Use custom JSON encoder to handle numpy/pandas types
    return JSONResponse(content=json.loads(json.dumps(result, cls=NumpyEncoder)))
