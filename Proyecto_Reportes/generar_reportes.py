#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════
  Generador de Reportes de Máquinas — Script Python
═══════════════════════════════════════════════════════════

Genera un reporte HTML ejecutivo a partir de archivos Excel
de fallas y cambios de marca de máquinas de producción.

USO:
  python generar_reporte.py --fallas mi_archivo_fallas.xlsx
  python generar_reporte.py --fallas fallas.xlsx --co cambios_marca.xlsx
  python generar_reporte.py --fallas fallas.xlsx --co co.xlsx --output mi_reporte.html

REQUISITOS:
  pip install pandas openpyxl
"""

import argparse, sys, math, re, warnings
from pathlib import Path
from datetime import datetime
from collections import Counter

warnings.filterwarnings('ignore')

try:
    import pandas as pd
except ImportError:
    print("❌ Se requiere pandas. Instala con: pip install pandas openpyxl")
    sys.exit(1)

# ══════════════════════════════════════════
# AUTO-DETECTION DE COLUMNAS
# ══════════════════════════════════════════
FALLAS_PATTERNS = {
    'falla': ['stop reason', 'falla', 'paro', 'reason', 'unplanned'],
    'stops': ['stops', 'paros', 'count', 'cantidad'],
    'downtime': ['downtime', 'tiempo muerto', 'down time'],
    'uptime_loss': ['uptime', 'loss', 'disponibilidad'],
    'mtbf': ['mtbf'],
    'mttr': ['mttr'],
    'rejects': ['reject', 'desperdicio', 'scrap'],
    'shift1': ['shift_1', 'turno 1', 'shift 1', 'turno_1'],
    'shift2': ['shift_2', 'turno 2', 'shift 2', 'turno_2'],
    'shift3': ['shift_3', 'turno 3', 'shift 3', 'turno_3'],
}

CO_PATTERNS = {
    'machine_col': ['lu', 'maquina', 'máquina', 'machine'],
    'marca_fin': ['finalizado', 'finished', 'terminad', 'nombre [finalizado'],
    'marca_ini': ['iniciado', 'started', 'nueva', 'nombre [iniciado'],
    'tiempo_obj': ['objetivo', 'target', 'planned'],
    'tiempo_real': ['real', 'actual'],
    'variacion': ['variación', 'variacion', 'deviation'],
    'stops_2h': ['stops next', 'paros next', '#stops'],
    'mtbf_2h': ['mtbf next', 'mtbf 2'],
    'razon': ['razón', 'razon', 'desviación', 'desviacion'],
    'coordinador': ['coordinador', 'lc', 'coordinator', 'lider'],
    'mes': ['mes', 'month'],
    'comentarios': ['comentario', 'comment', 'column1', 'recomendación', 'mejora'],
}

def auto_detect(patterns, columns):
    """Auto-detecta columnas basándose en patrones."""
    mapping = {}
    cols_lower = {c: c.lower().replace('\n', ' ') for c in columns}
    for key, pats in patterns.items():
        for pat in pats:
            for orig, low in cols_lower.items():
                if pat in low and key not in mapping:
                    mapping[key] = orig
                    break
            if key in mapping:
                break
    return mapping

def confirm_mapping(mapping, patterns, columns, file_type):
    """Muestra el mapeo detectado y permite al usuario confirmarlo o cambiarlo."""
    print(f"\n{'═'*50}")
    print(f"  Mapeo de columnas — {file_type}")
    print(f"{'═'*50}")
    print(f"\nColumnas encontradas en el archivo:")
    for i, c in enumerate(columns):
        print(f"  [{i}] {c}")
    
    print(f"\n🔗 Mapeo auto-detectado:")
    required = {'falla', 'stops', 'downtime'} if file_type == 'Fallas' else {'marca_ini', 'marca_fin', 'tiempo_obj', 'tiempo_real'}
    
    for key in patterns.keys():
        detected = mapping.get(key, '—')
        req = " *" if key in required else ""
        print(f"  {key}{req}: {detected}")
    
    print(f"\n  (* = obligatorio)")
    
    resp = input("\n¿El mapeo es correcto? [S/n] o escribe el campo a cambiar: ").strip()
    if resp.lower() in ['', 's', 'si', 'sí', 'y', 'yes']:
        return mapping
    
    # Manual override
    print("\nPara cada campo, escribe el número de columna o Enter para mantener:")
    for key in patterns.keys():
        current = mapping.get(key, '—')
        val = input(f"  {key} [{current}]: ").strip()
        if val:
            try:
                idx = int(val)
                if 0 <= idx < len(columns):
                    mapping[key] = columns[idx]
            except ValueError:
                if val in columns:
                    mapping[key] = val
    return mapping

# ══════════════════════════════════════════
# ANALYSIS ENGINE
# ══════════════════════════════════════════
def num(v):
    try:
        return float(v) if pd.notna(v) else 0
    except (ValueError, TypeError):
        return 0

def safe_avg(lst):
    lst = [x for x in lst if x and not math.isnan(x)]
    return sum(lst) / len(lst) if lst else 0

def safe_median(lst):
    lst = sorted([x for x in lst if x and not math.isnan(x)])
    if not lst: return 0
    m = len(lst) // 2
    return lst[m] if len(lst) % 2 else (lst[m-1] + lst[m]) / 2

def analyze_machine(name, fallas_df, co_df, f_map, c_map, color):
    """Analiza una máquina completa."""
    result = {'name': name, 'color': color, 'fallas': [], 'summary': {}, 'shifts': None, 'co': None}
    
    # === FALLAS ===
    fd = fallas_df
    total_stops = sum(num(r) for r in fd[f_map['stops']]) if 'stops' in f_map else 0
    total_down = sum(num(r) for r in fd[f_map['downtime']]) if 'downtime' in f_map else 0
    
    result['summary'] = {
        'totalStops': int(total_stops),
        'totalDown': round(total_down),
        'uptimeLoss': round(sum(num(r) for r in fd[f_map['uptime_loss']])) if 'uptime_loss' in f_map else None,
        'avgMtbf': round(safe_avg([num(r) for r in fd[f_map['mtbf']]])) if 'mtbf' in f_map else None,
        'avgMttr': round(safe_avg([num(r) for r in fd[f_map['mttr']]]), 1) if 'mttr' in f_map else None,
    }
    
    if all(k in f_map for k in ['shift1', 'shift2', 'shift3']):
        result['shifts'] = {
            't1': int(sum(num(r) for r in fd[f_map['shift1']])),
            't2': int(sum(num(r) for r in fd[f_map['shift2']])),
            't3': int(sum(num(r) for r in fd[f_map['shift3']])),
        }
    
    for _, row in fd.head(10).iterrows():
        f = {'falla': str(row.get(f_map.get('falla', ''), '—')),
             'stops': int(num(row.get(f_map.get('stops', ''), 0))),
             'downtime': round(num(row.get(f_map.get('downtime', ''), 0)), 1)}
        if 'uptime_loss' in f_map: f['uloss'] = round(num(row.get(f_map['uptime_loss'], 0)), 3)
        if 'mtbf' in f_map: f['mtbf'] = int(num(row.get(f_map['mtbf'], 0)))
        if 'mttr' in f_map: f['mttr'] = round(num(row.get(f_map['mttr'], 0)), 1)
        if 'rejects' in f_map: f['rejects'] = round(num(row.get(f_map['rejects'], 0)), 3)
        result['fallas'].append(f)
    
    # === CO ===
    if co_df is not None and len(co_df) > 0 and c_map:
        # Re-detect columns for this specific dataframe if needed
        local_cmap = {}
        for key, val in c_map.items():
            if val in co_df.columns:
                local_cmap[key] = val
            else:
                # Try to find equivalent column
                detected = auto_detect({key: CO_PATTERNS.get(key, [])}, list(co_df.columns))
                if key in detected:
                    local_cmap[key] = detected[key]
        
        if 'marca_ini' not in local_cmap or 'tiempo_real' not in local_cmap:
            result['co'] = None
            return result
        
        cmap = local_cmap  # Use local mapping for this machine
        co_df = co_df.copy()
        co_df['_real'] = pd.to_numeric(co_df[cmap['tiempo_real']], errors='coerce')
        co_df['_obj'] = pd.to_numeric(co_df[cmap['tiempo_obj']], errors='coerce') if 'tiempo_obj' in cmap else 0
        valid = co_df.dropna(subset=['_real']).copy()
        valid = valid[valid['_real'] > 0]
        if 'tiempo_obj' in cmap:
            valid = valid[valid['_obj'] > 0]
        
        co = {'total': len(co_df)}
        co['avgObj'] = round(valid['_obj'].mean(), 1) if '_obj' in valid and valid['_obj'].mean() > 0 else None
        co['avgReal'] = round(valid['_real'].mean(), 1)
        co['medianReal'] = round(valid['_real'].median(), 1)
        
        # Variación
        if 'variacion' in cmap:
            valid['_var'] = pd.to_numeric(valid[cmap['variacion']], errors='coerce')
        else:
            valid['_var'] = valid.apply(lambda r: (r['_real'] - r['_obj']) / r['_obj'] if r['_obj'] > 0 else 0, axis=1)
        
        on_time = (valid['_var'] <= 0).sum()
        co['pctOnTime'] = round(on_time / len(valid) * 100, 1) if len(valid) > 0 else 0
        
        if 'stops_2h' in cmap:
            vals = pd.to_numeric(co_df[cmap['stops_2h']], errors='coerce').dropna()
            co['avgStops2h'] = round(vals[vals > 0].mean(), 1) if len(vals[vals > 0]) > 0 else None
        if 'mtbf_2h' in cmap:
            vals = pd.to_numeric(co_df[cmap['mtbf_2h']], errors='coerce').dropna()
            co['avgMtbf2h'] = round(vals[vals > 0].mean(), 1) if len(vals[vals > 0]) > 0 else None
        
        # Marcas
        valid['_marca'] = valid[cmap['marca_ini']].astype(str).str.strip()
        valid_m = valid[~valid['_marca'].isin(['nan', 'None', '', 'null'])]
        brand_g = valid_m.groupby('_marca').agg(
            count=('_real', 'count'), avg_real=('_real', 'mean'),
            avg_var=('_var', 'mean'),
            avg_stops=('_real', 'count'),  # placeholder
            max_real=('_real', 'max')
        ).reset_index()
        
        if 'stops_2h' in cmap:
            valid_m_s = valid_m.copy()
            valid_m_s['_s2h'] = pd.to_numeric(valid_m_s[cmap['stops_2h']], errors='coerce')
            stops_by_brand = valid_m_s.groupby('_marca')['_s2h'].mean().reset_index()
            stops_by_brand.columns = ['_marca', 'real_avg_stops']
            brand_g = brand_g.merge(stops_by_brand, on='_marca', how='left')
            brand_g['avg_stops'] = brand_g['real_avg_stops'].round(1)
        else:
            brand_g['avg_stops'] = None
        
        brand_g = brand_g[brand_g['count'] >= 3]
        
        co['worstBrands'] = brand_g.nlargest(5, 'avg_var')[['_marca', 'count', 'avg_real', 'avg_var', 'avg_stops', 'max_real']].round(2).to_dict('records')
        co['bestBrands'] = brand_g.nsmallest(5, 'avg_var')[['_marca', 'count', 'avg_real', 'avg_var', 'avg_stops']].round(2).to_dict('records')
        
        # Transiciones
        if 'marca_fin' in cmap:
            valid['_fin'] = valid[cmap['marca_fin']].astype(str).str.strip()
            tv = valid[~valid['_fin'].str.lower().isin(['nan', 'none', '', 'arranque', 'null'])]
            tv = tv[~tv['_marca'].isin(['nan', 'None', '', 'null'])]
            tv['_trans'] = tv['_fin'] + ' → ' + tv['_marca']
            tg = tv.groupby('_trans').agg(count=('_real', 'count'), avg_real=('_real', 'mean'), avg_var=('_var', 'mean')).reset_index()
            
            if 'stops_2h' in cmap:
                tv_s = tv.copy()
                tv_s['_s2h'] = pd.to_numeric(tv_s[cmap['stops_2h']], errors='coerce')
                ts = tv_s.groupby('_trans')['_s2h'].mean().reset_index()
                ts.columns = ['_trans', 'avg_stops']
                tg = tg.merge(ts, on='_trans', how='left')
            else:
                tg['avg_stops'] = None
            
            tg = tg[tg['count'] >= 2]
            co['worstTrans'] = tg.nlargest(6, 'avg_var').round(2).to_dict('records')
            co['bestTrans'] = tg.nsmallest(6, 'avg_var').round(2).to_dict('records')
        
        # Coordinators - auto-detect the right column for this specific data
        coord_col = None
        if 'coordinador' in c_map and cmap['coordinador'] in co_df.columns:
            coord_col = cmap['coordinador']
        else:
            # Try to find a coordinator column in this specific dataframe
            for col in co_df.columns:
                cl = col.lower().replace('\n', ' ')
                if any(p in cl for p in ['coordinador', 'lc', 'coordinator', 'lider']):
                    coord_col = col
                    break
        
        if coord_col:
            valid['_lc'] = valid[coord_col].astype(str).str.strip()
            lc_v = valid[~valid['_lc'].str.lower().isin(['nan', 'none', '', 'null'])]
            lc_v = lc_v[~lc_v['_lc'].str.contains('/', na=False)]
            lcg = lc_v.groupby('_lc').agg(
                count=('_real', 'count'), avg_real=('_real', 'mean'), avg_var=('_var', 'mean'),
                pct_on=('_var', lambda x: round((x <= 0).sum() / len(x) * 100, 1))
            ).reset_index()
            
            if 'stops_2h' in cmap:
                lc_s = lc_v.copy()
                lc_s['_s2h'] = pd.to_numeric(lc_s[cmap['stops_2h']], errors='coerce')
                ls = lc_s.groupby('_lc')['_s2h'].mean().reset_index()
                ls.columns = ['_lc', 'avg_stops']
                lcg = lcg.merge(ls, on='_lc', how='left')
            else:
                lcg['avg_stops'] = None
            
            lcg = lcg[lcg['count'] >= 5].sort_values('avg_var')
            co['coordinators'] = lcg.round(2).to_dict('records')
        
        # Razones
        if 'razon' in cmap:
            razones = co_df[cmap['razon']].astype(str).str.strip()
            razones = razones[~razones.str.lower().isin(['nan', 'none', '', 'ok', 'null'])]
            razones = razones[razones.str.len() > 2]
            cnt = Counter(razones)
            co['razones'] = cnt.most_common(8)
        
        result['co'] = co
    
    return result

# ══════════════════════════════════════════
# HTML BUILDER (same template as web app)
# ══════════════════════════════════════════
def build_html(machines):
    """Genera el HTML completo del reporte."""
    ms = machines
    has_co = any(m['co'] for m in ms)
    has_shifts = any(m['shifts'] for m in ms)
    has_trans = any(m['co'] and m['co'].get('worstTrans') for m in ms)
    has_lc = any(m['co'] and m['co'].get('coordinators') for m in ms)
    has_raz = any(m['co'] and m['co'].get('razones') for m in ms)
    
    sec_num = [0]
    def sec():
        sec_num[0] += 1
        return f"0{sec_num[0]}" if sec_num[0] < 10 else str(sec_num[0])
    
    def heat(v):
        vp = round(v * 100)
        cls = 'heat-bad' if vp > 20 else ('heat-warn' if vp > 10 else 'heat-good')
        sign = '+' if vp > 0 else ''
        return f'<td class="num {cls}">{sign}{vp}%</td>'
    
    def on_heat(v):
        cls = 'heat-good' if v >= 50 else ('heat-warn' if v >= 35 else 'heat-bad')
        return f'<td class="num {cls}">{v}%</td>'
    
    def esc(s):
        return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Nav
    nav = '<a href="#overview">Overview</a><a href="#fallas">Fallas</a>'
    if has_shifts: nav += '<a href="#turnos">Turnos</a>'
    if has_co: nav += '<a href="#cambios">COs</a><a href="#marcas">Marcas</a>'
    if has_trans: nav += '<a href="#transiciones">Transiciones</a>'
    if has_lc: nav += '<a href="#coordinadores">Coordinadores</a>'
    nav += '<a href="#hallazgos">Hallazgos</a>'
    
    pills = ''.join(f'<div class="hero-machine"><span class="dot" style="background:{m["color"]};box-shadow:0 0 8px {m["color"]}"></span>{m["name"]}</div>' for m in ms)
    
    # Overview
    cards = ''
    for m in ms:
        rows = f'<div class="stat-row"><span class="stat-label">Total paros</span><span class="stat-val">{m["summary"]["totalStops"]:,}</span></div>'
        rows += f'<div class="stat-row"><span class="stat-label">Downtime total</span><span class="stat-val">{m["summary"]["totalDown"]:,} min</span></div>'
        if m['summary'].get('uptimeLoss'): rows += f'<div class="stat-row"><span class="stat-label">Uptime Loss</span><span class="stat-val">{m["summary"]["uptimeLoss"]}%</span></div>'
        if m['summary'].get('avgMtbf'): rows += f'<div class="stat-row"><span class="stat-label">MTBF prom.</span><span class="stat-val">{m["summary"]["avgMtbf"]} min</span></div>'
        if m['summary'].get('avgMttr'): rows += f'<div class="stat-row"><span class="stat-label">MTTR prom.</span><span class="stat-val">{m["summary"]["avgMttr"]} min</span></div>'
        if m['co']:
            rows += f'<div class="stat-row"><span class="stat-label">Total COs</span><span class="stat-val">{m["co"]["total"]}</span></div>'
            rows += f'<div class="stat-row"><span class="stat-label">% On-time CO</span><span class="stat-val">{m["co"]["pctOnTime"]}%</span></div>'
        cards += f'<div class="compare-card" style="--mc:{m["color"]}"><h3 style="color:{m["color"]}">{m["name"]}</h3>{rows}</div>'
    
    # Fallas
    f_tabs = ''.join(f'<div class="tab {"active" if i==0 else ""}" data-tab="fallas-{i}" style="--tc:{m["color"]}">{m["name"]}</div>' for i,m in enumerate(ms))
    f_contents = ''
    for i, m in enumerate(ms):
        has_u = any('uloss' in f for f in m['fallas'])
        has_b = any('mtbf' in f for f in m['fallas'])
        has_t = any('mttr' in f for f in m['fallas'])
        has_r = any('rejects' in f for f in m['fallas'])
        ths = '<th>#</th><th>Falla</th><th>Paros</th><th>Downtime</th>'
        if has_u: ths += '<th>Uptime Loss %</th>'
        if has_b: ths += '<th>MTBF</th>'
        if has_t: ths += '<th>MTTR</th>'
        if has_r: ths += '<th>Rejects %</th>'
        trs = ''
        for j, f in enumerate(m['fallas']):
            tr = f'<td>{j+1}</td><td>{esc(f["falla"])}</td><td class="num">{f["stops"]:,}</td><td class="num">{f["downtime"]:,.1f}</td>'
            if has_u: tr += f'<td class="num">{f.get("uloss","")}</td>'
            if has_b: tr += f'<td class="num">{f.get("mtbf","")}</td>'
            if has_t: tr += f'<td class="num">{f.get("mttr","")}</td>'
            if has_r: tr += f'<td class="num">{f.get("rejects","")}</td>'
            trs += f'<tr>{tr}</tr>'
        f_contents += f'<div class="tab-content {"active" if i==0 else ""}" id="fallas-{i}"><div class="table-wrap"><table><tr>{ths}</tr>{trs}</table></div></div>'
    
    # Shifts
    shifts_html = ''
    if has_shifts:
        sc = ''
        for m in ms:
            if not m['shifts']: continue
            mx = max(m['shifts']['t1'], m['shifts']['t2'], m['shifts']['t3'])
            bars = ''
            for label, key in [('Turno 1','t1'),('Turno 2','t2'),('Turno 3','t3')]:
                v = m['shifts'][key]
                w = round(v/mx*100) if mx > 0 else 0
                op = ';opacity:0.7' if key == 't3' else ''
                bars += f'<div class="sbi"><span class="sl">{label}</span><div class="sb"><div class="sf" style="width:{w}%;background:{m["color"]}{op}"></div></div><span class="sv">{v:,}</span></div>'
            sc += f'<div class="shift-card"><h4 style="color:{m["color"]}">{m["name"]}</h4>{bars}</div>'
        shifts_html = f'<section id="turnos"><div class="container"><div class="section-number">{sec()} — Turnos</div><h2 class="section-title">Distribución de paros por turno</h2><div class="shift-grid">{sc}</div></div></section>'
    
    # CO
    co_html = ''
    if has_co:
        kpis = ''
        donuts = ''
        for m in ms:
            if not m['co']: continue
            kpis += f'<div class="kpi-card" style="--mc:{m["color"]}"><div class="kpi-label">{m["name"]} — T. prom. CO</div><div class="kpi-value">{m["co"]["avgReal"] or "—"} <span style="font-size:.9rem;color:var(--text-dim)">min</span></div><div class="kpi-sub">Objetivo: {m["co"].get("avgObj") or "—"} · Mediana: {m["co"].get("medianReal") or "—"} min</div></div>'
            if m['co'].get('avgStops2h'):
                kpis += f'<div class="kpi-card" style="--mc:{m["color"]}"><div class="kpi-label">{m["name"]} — Paros 2h</div><div class="kpi-value">{m["co"]["avgStops2h"]}</div><div class="kpi-sub">MTBF post-CO: {m["co"].get("avgMtbf2h") or "—"} min</div></div>'
            pct = m['co']['pctOnTime']
            donuts += f'<div class="donut-item"><svg viewBox="0 0 36 36"><path d="M18 2.0845a15.9155 15.9155 0 010 31.831 15.9155 15.9155 0 010-31.831" fill="none" stroke="var(--surface2)" stroke-width="3"/><path d="M18 2.0845a15.9155 15.9155 0 010 31.831 15.9155 15.9155 0 010-31.831" fill="none" stroke="{m["color"]}" stroke-width="3" stroke-dasharray="{pct},100" stroke-linecap="round"/><text x="18" y="18.5" text-anchor="middle" font-family="JetBrains Mono" font-size="6.5" font-weight="700" fill="#fff">{pct}%</text></svg><div class="donut-label" style="color:{m["color"]}">{m["name"]}</div></div>'
        co_html = f'<section id="cambios"><div class="container"><div class="section-number">{sec()} — Cambios de Marca</div><h2 class="section-title">Eficiencia en changeovers</h2><div class="kpi-grid">{kpis}</div><h3 style="text-align:center;color:#fff;font-size:1.1rem">% COs a tiempo</h3><div class="donut-row">{donuts}</div></div></section>'
    
    # Brands
    brands_html = ''
    if has_co:
        co_ms = [m for m in ms if m['co'] and m['co'].get('worstBrands')]
        if co_ms:
            b_tabs = ''.join(f'<div class="tab {"active" if i==0 else ""}" data-tab="marca-{i}" style="--tc:{m["color"]}">{m["name"]}</div>' for i,m in enumerate(co_ms))
            b_cont = ''
            for i, m in enumerate(co_ms):
                def brand_rows(arr, show_max=False):
                    r = ''
                    for b in arr:
                        name_key = '_marca' if '_marca' in b else 'Nombre [Iniciado]' if 'Nombre [Iniciado]' in b else list(b.keys())[0]
                        r += f'<tr><td>{esc(b[name_key])}</td><td class="num">{b["count"]}</td><td class="num">{round(b["avg_real"],1)} min</td>{heat(b["avg_var"])}<td class="num">{round(b["avg_stops"],1) if b.get("avg_stops") else "—"}</td></tr>'
                    return r
                bc = f'<div class="tab-content {"active" if i==0 else ""}" id="marca-{i}">'
                bc += f'<h3 style="color:var(--red);font-size:1rem;margin-bottom:1rem">🔴 Marcas más problemáticas</h3>'
                bc += f'<div class="table-wrap"><table><tr><th>Marca</th><th>#COs</th><th>T.Real</th><th>Var.</th><th>Paros 2h</th></tr>{brand_rows(m["co"]["worstBrands"])}</table></div>'
                bc += f'<h3 style="color:var(--accent3);font-size:1rem;margin:1.5rem 0 1rem">🟢 Mejores marcas</h3>'
                bc += f'<div class="table-wrap"><table><tr><th>Marca</th><th>#COs</th><th>T.Real</th><th>Var.</th><th>Paros 2h</th></tr>{brand_rows(m["co"]["bestBrands"])}</table></div></div>'
                b_cont += bc
            brands_html = f'<section id="marcas"><div class="container"><div class="section-number">{sec()} — Marcas</div><h2 class="section-title">¿Qué marcas causan más problemas?</h2><p class="section-desc">Marcas con ≥3 COs. Variación: (+) excede objetivo, (−) más rápido.</p><div class="tabs" data-group="marcas">{b_tabs}</div>{b_cont}</div></section>'
    
    # Transitions
    trans_html = ''
    if has_trans:
        t_ms = [m for m in ms if m['co'] and m['co'].get('worstTrans')]
        t_tabs = ''.join(f'<div class="tab {"active" if i==0 else ""}" data-tab="trans-{i}" style="--tc:{m["color"]}">{m["name"]}</div>' for i,m in enumerate(t_ms))
        t_cont = ''
        for i, m in enumerate(t_ms):
            def trans_rows(arr):
                r = ''
                for t in arr:
                    r += f'<tr><td style="font-size:.78rem">{esc(t["_trans"])}</td><td class="num">{t["count"]}</td><td class="num">{round(t["avg_real"],1)} min</td>{heat(t["avg_var"])}<td class="num">{round(t["avg_stops"],1) if t.get("avg_stops") and not pd.isna(t["avg_stops"]) else "—"}</td></tr>'
                return r
            tc = f'<div class="tab-content {"active" if i==0 else ""}" id="trans-{i}">'
            tc += f'<h3 style="color:var(--red);font-size:1rem;margin-bottom:1rem">⚠️ Peores transiciones</h3><div class="table-wrap"><table><tr><th>Transición</th><th>#</th><th>T.Real</th><th>Var.</th><th>Paros 2h</th></tr>{trans_rows(m["co"]["worstTrans"])}</table></div>'
            tc += f'<h3 style="color:var(--accent3);font-size:1rem;margin:1.5rem 0 1rem">✅ Mejores</h3><div class="table-wrap"><table><tr><th>Transición</th><th>#</th><th>T.Real</th><th>Var.</th><th>Paros 2h</th></tr>{trans_rows(m["co"]["bestTrans"])}</table></div></div>'
            t_cont += tc
        trans_html = f'<section id="transiciones"><div class="container"><div class="section-number">{sec()} — Transiciones</div><h2 class="section-title">Cambios marca→marca más difíciles</h2><div class="tabs" data-group="trans">{t_tabs}</div>{t_cont}</div></section>'
    
    # LC
    lc_html = ''
    if has_lc:
        lc_ms = [m for m in ms if m['co'] and m['co'].get('coordinators')]
        lc_tabs = ''.join(f'<div class="tab {"active" if i==0 else ""}" data-tab="lc-{i}" style="--tc:{m["color"]}">{m["name"]}</div>' for i,m in enumerate(lc_ms))
        lc_cont = ''
        for i, m in enumerate(lc_ms):
            rows = ''
            for c in m['co']['coordinators']:
                name = c.get('_lc', c.get('Coordinador', ''))
                rows += f'<tr><td style="font-weight:600">{esc(name)}</td><td class="num">{c["count"]}</td><td class="num">{round(c["avg_real"],1)} min</td>{heat(c["avg_var"])}{on_heat(c["pct_on"])}<td class="num">{round(c["avg_stops"],1) if c.get("avg_stops") and not pd.isna(c["avg_stops"]) else "—"}</td></tr>'
            lc_cont += f'<div class="tab-content {"active" if i==0 else ""}" id="lc-{i}"><div class="table-wrap"><table><tr><th>Coordinador</th><th>#COs</th><th>T.Real</th><th>Var.</th><th>% On-Time</th><th>Paros 2h</th></tr>{rows}</table></div></div>'
        lc_html = f'<section id="coordinadores"><div class="container"><div class="section-number">{sec()} — Coordinadores</div><h2 class="section-title">¿Quién ejecuta mejor los cambios?</h2><div class="tabs" data-group="lc">{lc_tabs}</div>{lc_cont}</div></section>'
    
    # Razones
    raz_html = ''
    if has_raz:
        r_ms = [m for m in ms if m['co'] and m['co'].get('razones')]
        r_tabs = ''.join(f'<div class="tab {"active" if i==0 else ""}" data-tab="raz-{i}" style="--tc:{m["color"]}">{m["name"]}</div>' for i,m in enumerate(r_ms))
        r_cont = ''
        for i, m in enumerate(r_ms):
            items = ''.join(f'<li class="razon-item"><span class="razon-count">{cnt}</span><span class="razon-text">{esc(txt[:150])}</span></li>' for txt, cnt in m['co']['razones'])
            r_cont += f'<div class="tab-content {"active" if i==0 else ""}" id="raz-{i}"><ul class="razon-list">{items}</ul></div>'
        raz_html = f'<div style="margin-top:4rem"><div class="section-number">{sec()} — Razones de Desviación</div><h2 class="section-title">¿Por qué se excede el tiempo?</h2><div class="tabs" data-group="raz">{r_tabs}</div>{r_cont}</div>'
    
    # Conclusions
    conc = ''
    for m in ms:
        f_str = ''
        if m['summary'].get('avgMtbf'): f_str += f'<p style="margin-bottom:.5rem">MTBF: <strong style="color:#fff">{m["summary"]["avgMtbf"]} min</strong></p>'
        if m['co']: f_str += f'<p style="margin-bottom:.5rem">CO on-time: <strong style="color:{"var(--accent3)" if m["co"]["pctOnTime"]>=50 else "var(--red)"}">{m["co"]["pctOnTime"]}%</strong></p>'
        if m['fallas']: f_str += f'<p>Falla #1: <strong style="color:#fff">{esc(m["fallas"][0]["falla"])}</strong> ({m["fallas"][0]["stops"]:,} paros)</p>'
        conc += f'<div class="compare-card" style="border-left:3px solid {m["color"]}"><h3 style="color:{m["color"]};font-size:1rem">{m["name"]}</h3><div style="font-size:.85rem;color:var(--text-dim);line-height:1.7;margin-top:.75rem">{f_str}</div></div>'
    
    date_str = datetime.now().strftime('%d de %B de %Y')
    
    # Assemble
    return f'''<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Reporte — Análisis de Máquinas</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{{--bg:#0B0F1A;--surface:#111827;--surface2:#1A2235;--border:#1E293B;--text:#E2E8F0;--text-dim:#94A3B8;--text-muted:#64748B;--accent:#F59E0B;--accent3:#10B981;--red:#EF4444;--orange:#F97316}}
*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text);line-height:1.6;-webkit-font-smoothing:antialiased}}
.hero{{position:relative;min-height:100vh;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;padding:4rem 2rem;background:radial-gradient(ellipse 80% 60% at 50% 40%,rgba(245,158,11,0.08) 0%,transparent 60%),var(--bg)}}
.hero::before{{content:'';position:absolute;inset:0;background:url("data:image/svg+xml,%3Csvg width='60' height='60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.015'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/svg%3E")}}
.hero-label{{font-family:'JetBrains Mono',monospace;font-size:.75rem;letter-spacing:.2em;text-transform:uppercase;color:var(--accent);margin-bottom:1.5rem;position:relative;animation:fadeUp .8s ease both}}
.hero h1{{font-family:'Instrument Serif',serif;font-size:clamp(2.8rem,7vw,5.5rem);font-weight:400;line-height:1.1;color:#fff;margin-bottom:1rem;position:relative;animation:fadeUp .8s ease .1s both}}
.hero h1 em{{font-style:italic;background:linear-gradient(135deg,var(--accent),var(--orange));-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.hero-sub{{font-size:1.15rem;color:var(--text-dim);max-width:600px;position:relative;animation:fadeUp .8s ease .2s both}}
.hero-machines{{display:flex;gap:1.5rem;margin-top:3rem;position:relative;animation:fadeUp .8s ease .3s both;flex-wrap:wrap;justify-content:center}}
.hero-machine{{display:flex;align-items:center;gap:.5rem;padding:.6rem 1.2rem;border-radius:100px;background:var(--surface);border:1px solid var(--border);font-weight:600;font-size:.9rem}}
.dot{{width:8px;height:8px;border-radius:50%}}
.scroll-ind{{position:absolute;bottom:2rem;animation:bounce 2s infinite;color:var(--text-muted);font-size:1.5rem}}
@keyframes bounce{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(8px)}}}}@keyframes fadeUp{{from{{opacity:0;transform:translateY(20px)}}to{{opacity:1;transform:translateY(0)}}}}
.nav{{position:fixed;top:0;left:0;right:0;z-index:100;background:rgba(11,15,26,0.9);backdrop-filter:blur(12px);border-bottom:1px solid var(--border);padding:0 2rem;display:flex;align-items:center;height:56px;transform:translateY(-100%);transition:transform .3s}}
.nav.visible{{transform:translateY(0)}}.nav-brand{{font-family:'Instrument Serif',serif;font-size:1.15rem;color:#fff;margin-right:auto}}
.nav-links{{display:flex;gap:.25rem;flex-wrap:wrap}}.nav-links a{{color:var(--text-dim);text-decoration:none;font-size:.78rem;padding:.4rem .7rem;border-radius:6px;transition:all .2s}}.nav-links a:hover{{color:#fff;background:var(--surface)}}
.container{{max-width:1280px;margin:0 auto;padding:0 2rem}}section{{padding:5rem 0;border-top:1px solid var(--border)}}
.section-number{{font-family:'JetBrains Mono',monospace;font-size:.7rem;letter-spacing:.2em;text-transform:uppercase;color:var(--accent);margin-bottom:.5rem}}
.section-title{{font-family:'Instrument Serif',serif;font-size:clamp(1.8rem,4vw,2.8rem);font-weight:400;color:#fff;line-height:1.2;margin-bottom:.75rem}}
.section-desc{{font-size:1rem;color:var(--text-dim);max-width:700px;margin-bottom:2rem}}
.compare-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1.5rem;margin-bottom:2rem}}
.compare-card{{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1.5rem;position:relative}}
.compare-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:14px 14px 0 0;background:var(--mc,var(--accent))}}
.stat-row{{display:flex;justify-content:space-between;padding:.45rem 0;border-bottom:1px solid var(--border);font-size:.85rem}}.stat-row:last-child{{border-bottom:none}}
.stat-label{{color:var(--text-dim)}}.stat-val{{font-family:'JetBrains Mono',monospace;font-weight:600;color:#fff}}
.table-wrap{{overflow-x:auto;border-radius:12px;border:1px solid var(--border);background:var(--surface);margin-bottom:1.5rem}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th{{background:var(--surface2);color:var(--text-dim);font-weight:600;text-transform:uppercase;letter-spacing:.04em;font-size:.7rem;padding:.9rem 1rem;text-align:left;white-space:nowrap;border-bottom:1px solid var(--border)}}
td{{padding:.75rem 1rem;border-bottom:1px solid var(--border);color:var(--text)}}tr:last-child td{{border-bottom:none}}tr:hover td{{background:rgba(255,255,255,0.02)}}
td.num{{font-family:'JetBrains Mono',monospace;font-size:.82rem}}
.tabs{{display:flex;gap:.5rem;margin-bottom:1.5rem;flex-wrap:wrap}}
.tab{{padding:.55rem 1.2rem;border-radius:8px;font-size:.85rem;font-weight:600;cursor:pointer;border:1px solid var(--border);background:var(--surface);color:var(--text-dim);transition:all .2s;user-select:none}}
.tab:hover{{color:var(--text)}}.tab.active{{background:rgba(245,158,11,0.12);border-color:var(--tc,var(--accent));color:var(--tc,var(--accent))}}
.tab-content{{display:none;animation:fadeUp .4s ease}}.tab-content.active{{display:block}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;margin-bottom:2rem}}
.kpi-card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.25rem;position:relative;overflow:hidden}}
.kpi-card::after{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--mc,var(--accent))}}
.kpi-label{{font-size:.75rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:.5rem}}
.kpi-value{{font-family:'JetBrains Mono',monospace;font-size:1.8rem;font-weight:700;color:#fff}}.kpi-sub{{font-size:.8rem;color:var(--text-dim);margin-top:.25rem}}
.shift-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1.5rem;margin-bottom:2rem}}
.shift-card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.25rem}}.shift-card h4{{font-size:.9rem;font-weight:600;margin-bottom:1rem}}
.sbi{{display:flex;align-items:center;gap:.5rem;margin-bottom:.5rem}}.sbi .sl{{flex:0 0 55px;font-size:.75rem;color:var(--text-dim)}}
.sbi .sb{{flex:1;height:20px;background:var(--surface2);border-radius:4px;overflow:hidden}}.sbi .sf{{height:100%;border-radius:4px}}
.sbi .sv{{flex:0 0 50px;font-family:'JetBrains Mono',monospace;font-size:.75rem;color:var(--text-dim);text-align:right}}
.donut-row{{display:flex;justify-content:center;gap:3rem;flex-wrap:wrap;margin:2rem 0}}.donut-item{{text-align:center}}.donut-item svg{{width:130px;height:130px}}.donut-label{{margin-top:.5rem;font-size:.9rem;font-weight:600}}
.heat-good{{color:var(--accent3)}}.heat-warn{{color:var(--orange)}}.heat-bad{{color:var(--red)}}
.razon-list{{list-style:none;padding:0}}.razon-item{{display:flex;gap:.75rem;align-items:flex-start;padding:.6rem 0;border-bottom:1px solid var(--border);font-size:.85rem}}.razon-item:last-child{{border-bottom:none}}
.razon-count{{font-family:'JetBrains Mono',monospace;font-size:.75rem;background:var(--surface2);padding:.2rem .5rem;border-radius:4px;color:var(--accent);flex-shrink:0;min-width:30px;text-align:center}}
.razon-text{{color:var(--text-dim)}}
.footer{{padding:3rem 0;border-top:1px solid var(--border);text-align:center;color:var(--text-muted);font-size:.8rem}}
@media(max-width:900px){{.compare-grid,.shift-grid{{grid-template-columns:1fr}}}}
</style></head><body>
<nav class="nav" id="nav"><div class="nav-brand">Reporte</div><div class="nav-links">{nav}</div></nav>
<header class="hero"><div class="hero-label">Reporte de Análisis</div><h1>Análisis de <em>Máquinas</em></h1><p class="hero-sub">Diagnóstico integral de fallas, cambios de marca y rendimiento operativo</p><div class="hero-machines">{pills}</div><div class="scroll-ind">↓</div></header>
<section id="overview"><div class="container"><div class="section-number">{sec()} — Panorama General</div><h2 class="section-title">Las máquinas en números</h2><div class="compare-grid">{cards}</div></div></section>
<section id="fallas"><div class="container"><div class="section-number">{sec()} — Principales Fallas</div><h2 class="section-title">Top paros no planeados</h2><div class="tabs" data-group="fallas">{f_tabs}</div>{f_contents}</div></section>
{shifts_html}{co_html}{brands_html}{trans_html}{lc_html}
<section id="hallazgos"><div class="container">{raz_html}<div style="margin-top:4rem"><div class="section-number">{sec()} — Resumen</div><h2 class="section-title">Hallazgos clave</h2></div><div class="compare-grid" style="margin-top:2rem">{conc}</div></div></section>
<footer class="footer"><p>Reporte generado · {date_str}</p></footer>
<script>
document.querySelectorAll('.tab').forEach(tab=>{{tab.addEventListener('click',()=>{{const g=tab.parentElement;g.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));tab.classList.add('active');let p=g.parentElement;p.querySelectorAll(':scope>.tab-content').forEach(tc=>tc.classList.remove('active'));document.getElementById(tab.dataset.tab).classList.add('active')}})}}); const nav=document.getElementById('nav');window.addEventListener('scroll',()=>{{if(window.scrollY>400)nav.classList.add('visible');else nav.classList.remove('visible')}});document.querySelectorAll('a[href^="#"]').forEach(a=>{{a.addEventListener('click',e=>{{e.preventDefault();const el=document.querySelector(a.getAttribute('href'));if(el)el.scrollIntoView({{behavior:'smooth',block:'start'}})}})}}); </script></body></html>'''


# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Generador de Reportes de Máquinas')
    parser.add_argument('--fallas', required=True, help='Archivo Excel de fallas (cada pestaña = una máquina)')
    parser.add_argument('--co', help='Archivo Excel de cambios de marca (opcional)')
    parser.add_argument('--output', default='reporte_maquinas.html', help='Nombre del archivo de salida')
    parser.add_argument('--auto', action='store_true', help='Auto-detectar columnas sin preguntar')
    args = parser.parse_args()
    
    colors = ['#F59E0B', '#3B82F6', '#10B981', '#EC4899', '#A78BFA', '#06B6D4', '#EF4444', '#F97316']
    
    # Load fallas
    print(f"\n📊 Cargando archivo de fallas: {args.fallas}")
    xls_f = pd.ExcelFile(args.fallas)
    print(f"   Pestañas encontradas: {xls_f.sheet_names}")
    
    fallas_data = {}
    for s in xls_f.sheet_names:
        fallas_data[s] = pd.read_excel(xls_f, sheet_name=s)
    
    # Auto-detect fallas columns
    first_sheet = xls_f.sheet_names[0]
    f_cols = list(fallas_data[first_sheet].columns)
    f_map = auto_detect(FALLAS_PATTERNS, f_cols)
    
    if not args.auto:
        f_map = confirm_mapping(f_map, FALLAS_PATTERNS, f_cols, 'Fallas')
    else:
        print(f"\n🔗 Mapeo auto-detectado (Fallas): {f_map}")
    
    # Validate
    if 'falla' not in f_map or 'stops' not in f_map or 'downtime' not in f_map:
        print("❌ No se pudieron detectar las columnas obligatorias (falla, stops, downtime)")
        sys.exit(1)
    
    # Load CO
    co_data = {}
    c_map = {}
    if args.co:
        print(f"\n🔄 Cargando archivo de cambios de marca: {args.co}")
        xls_c = pd.ExcelFile(args.co)
        print(f"   Pestañas: {xls_c.sheet_names}")
        for s in xls_c.sheet_names:
            co_data[s] = pd.read_excel(xls_c, sheet_name=s)
        
        first_co = xls_c.sheet_names[0]
        c_cols = list(co_data[first_co].columns)
        c_map = auto_detect(CO_PATTERNS, c_cols)
        
        if not args.auto:
            c_map = confirm_mapping(c_map, CO_PATTERNS, c_cols, 'Cambios de Marca')
        else:
            print(f"🔗 Mapeo auto-detectado (CO): {c_map}")
    
    # Build machines
    print(f"\n🏭 Construyendo análisis...")
    machines_result = []
    
    for i, sheet in enumerate(xls_f.sheet_names):
        name = sheet.replace('-PF', '').replace('_PF', '').strip()
        print(f"   Analizando {name}...")
        
        # Find matching CO data
        machine_co = None
        if co_data:
            # Try exact match
            for cs, cd in co_data.items():
                if name.lower() in cs.lower():
                    machine_co = cd
                    break
            
            # Try filtering by machine column
            if machine_co is None and 'machine_col' in c_map:
                for cs, cd in co_data.items():
                    col = c_map['machine_col']
                    if col in cd.columns:
                        filtered = cd[cd[col].astype(str).str.strip().str.lower() == name.lower()]
                        if len(filtered) > 0:
                            machine_co = filtered
                            break
        
        if machine_co is not None:
            print(f"     ↳ CO data: {len(machine_co)} registros")
        
        m = analyze_machine(name, fallas_data[sheet], machine_co, f_map, c_map, colors[i % len(colors)])
        machines_result.append(m)
    
    # Generate HTML
    print(f"\n📝 Generando reporte HTML...")
    html = build_html(machines_result)
    
    output_path = Path(args.output)
    output_path.write_text(html, encoding='utf-8')
    print(f"\n✅ Reporte generado: {output_path.absolute()}")
    print(f"   Tamaño: {len(html):,} caracteres")
    print(f"   Ábrelo en tu navegador para verlo.\n")


if __name__ == '__main__':
    main()
