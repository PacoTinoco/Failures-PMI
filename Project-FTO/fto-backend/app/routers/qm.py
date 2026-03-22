"""
Router para QM (Qualification Management).
- Subir calendario anual (referencia fija)
- Subir data semanal actualizada
- Análisis: cumplimiento, atrasados, avances, próximos cambios
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Optional
from datetime import date, datetime
from collections import defaultdict
from io import BytesIO
import json

from app.services.supabase_client import get_supabase_admin

router = APIRouter(prefix="/qm", tags=["QM — Qualification Management"])

MONTH_COLS = ['feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep']
MONTH_MAP = {1: 'feb', 2: 'feb', 3: 'mar', 4: 'abr', 5: 'may', 6: 'jun', 7: 'jul', 8: 'ago', 9: 'sep', 10: 'sep', 11: 'sep', 12: 'sep'}


def get_current_month_col() -> str:
    """Retorna la columna del mes actual para comparar."""
    return MONTH_MAP.get(datetime.now().month, 'mar')


# ══════════════════════════════════════════════════════════════
# CALENDARIO
# ══════════════════════════════════════════════════════════════

@router.post("/calendario/upload")
async def upload_calendario(
    file: UploadFile = File(...),
    cedula_id: str = Query(..., description="ID de la cédula")
):
    """Sube el calendario QM anual. Reemplaza el calendario existente."""
    import pandas as pd

    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos Excel (.xlsx)")

    content = await file.read()
    df = pd.read_excel(BytesIO(content))

    required = {'Employee', 'Competency', 'Current', 'Role'}
    if not required.issubset(set(df.columns)):
        raise HTTPException(status_code=400, detail=f"Faltan columnas: {required - set(df.columns)}")

    # Detectar columna Target (puede tener trailing space)
    target_col = None
    for col in df.columns:
        if col.strip().lower() == 'target':
            target_col = col
            break
    if not target_col:
        raise HTTPException(status_code=400, detail="Falta columna 'Target'")

    # Detectar columnas de meses
    month_mapping = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl in MONTH_COLS:
            month_mapping[cl] = col

    sb = get_supabase_admin()

    # Borrar calendario anterior de esta cédula
    sb.table("qm_calendario").delete().eq("cedula_id", cedula_id).execute()

    # Insertar nuevo calendario
    records = []
    for _, row in df.iterrows():
        emp = str(row['Employee']).strip() if pd.notna(row['Employee']) else None
        comp = str(row['Competency']).strip() if pd.notna(row['Competency']) else None
        if not emp or not comp:
            continue

        rec = {
            "cedula_id": cedula_id,
            "employee": emp,
            "competency": comp,
            "role": str(row['Role']).strip() if pd.notna(row.get('Role')) else None,
            "current_base": int(row['Current']) if pd.notna(row['Current']) else 0,
            "target": int(row[target_col]) if pd.notna(row[target_col]) else 0,
        }
        for month in MONTH_COLS:
            if month in month_mapping:
                val = row[month_mapping[month]]
                rec[month] = int(val) if pd.notna(val) else None
            else:
                rec[month] = None

        records.append(rec)

    # Deduplicar por (employee, competency) — quedarse con el último
    seen = {}
    for rec in records:
        key = (rec["employee"], rec["competency"])
        seen[key] = rec
    records = list(seen.values())

    # Insertar en lotes de 100 con upsert
    inserted = 0
    for i in range(0, len(records), 100):
        batch = records[i:i+100]
        sb.table("qm_calendario").upsert(batch, on_conflict="cedula_id,employee,competency").execute()
        inserted += len(batch)

    employees = sorted(set(r["employee"] for r in records))
    competencies = sorted(set(r["competency"] for r in records))

    return {
        "success": True,
        "message": f"Calendario cargado: {inserted} registros ({len(employees)} empleados, {len(competencies)} competencias)",
        "employees": len(employees),
        "competencies": len(competencies),
        "total_records": inserted,
    }


@router.get("/calendario")
async def get_calendario(
    cedula_id: str = Query(...),
    employee: Optional[str] = Query(None)
):
    """Obtiene el calendario QM."""
    sb = get_supabase_admin()
    query = sb.table("qm_calendario").select("*").eq("cedula_id", cedula_id)
    if employee:
        query = query.eq("employee", employee)
    result = query.order("employee").order("competency").execute()
    return {"data": result.data or [], "count": len(result.data or [])}


@router.put("/calendario/{record_id}")
async def update_calendario_entry(record_id: str, body: dict):
    """Edita una entrada del calendario (target, meses, etc.)."""
    sb = get_supabase_admin()
    allowed = {"target", "current_base", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep"}
    update_data = {k: v for k, v in body.items() if k in allowed}
    if not update_data:
        raise HTTPException(status_code=400, detail="No hay campos válidos para actualizar")
    update_data["updated_at"] = datetime.now().isoformat()
    result = sb.table("qm_calendario").update(update_data).eq("id", record_id).execute()
    return {"success": True, "data": result.data[0] if result.data else None}


# ══════════════════════════════════════════════════════════════
# DATA SEMANAL
# ══════════════════════════════════════════════════════════════

@router.post("/data/upload")
async def upload_data_semanal(
    file: UploadFile = File(...),
    cedula_id: str = Query(...),
    semana: date = Query(..., description="Fecha del lunes de la semana (YYYY-MM-DD)")
):
    """Sube data semanal QM. Reemplaza datos de la misma semana."""
    import pandas as pd

    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos Excel (.xlsx)")

    content = await file.read()
    df = pd.read_excel(BytesIO(content))

    required = {'Employee', 'Competency', 'Current'}
    if not required.issubset(set(df.columns)):
        raise HTTPException(status_code=400, detail=f"Faltan columnas: {required - set(df.columns)}")

    target_col = None
    for col in df.columns:
        if col.strip().lower() == 'target':
            target_col = col
            break
    if not target_col:
        raise HTTPException(status_code=400, detail="Falta columna 'Target'")

    # Cargar competencias del calendario para filtrar
    sb = get_supabase_admin()
    cal_result = sb.table("qm_calendario").select("employee, competency").eq("cedula_id", cedula_id).execute()
    cal_pairs = set()
    for r in (cal_result.data or []):
        cal_pairs.add((r["employee"], r["competency"]))

    # Borrar data anterior de esta semana/cédula
    sb.table("qm_data_semanal").delete().eq("cedula_id", cedula_id).eq("semana", semana.isoformat()).execute()

    records = []
    skipped = 0
    for _, row in df.iterrows():
        emp = str(row['Employee']).strip() if pd.notna(row['Employee']) else None
        comp = str(row['Competency']).strip() if pd.notna(row['Competency']) else None
        if not emp or not comp:
            continue

        # Solo incluir competencias que estén en el calendario
        if (emp, comp) not in cal_pairs:
            skipped += 1
            continue

        on_target = 0
        if 'OnTarget' in df.columns and pd.notna(row.get('OnTarget')):
            on_target = int(row['OnTarget'])

        role_col = None
        for c in df.columns:
            if c == 'Role' or c == 'Role.1':
                role_col = c

        records.append({
            "cedula_id": cedula_id,
            "semana": semana.isoformat(),
            "employee": emp,
            "competency": comp,
            "role": str(row[role_col]).strip() if role_col and pd.notna(row.get(role_col)) else None,
            "current_level": int(row['Current']) if pd.notna(row['Current']) else 0,
            "target": int(row[target_col]) if pd.notna(row[target_col]) else 0,
            "on_target": on_target,
        })

    # Deduplicar por (employee, competency)
    seen = {}
    for rec in records:
        key = (rec["employee"], rec["competency"])
        seen[key] = rec
    records = list(seen.values())

    inserted = 0
    for i in range(0, len(records), 100):
        batch = records[i:i+100]
        sb.table("qm_data_semanal").upsert(batch, on_conflict="cedula_id,semana,employee,competency").execute()
        inserted += len(batch)

    return {
        "success": True,
        "message": f"Data semana {semana} cargada: {inserted} registros guardados, {skipped} competencias fuera del calendario ignoradas.",
        "total_records": inserted,
        "skipped": skipped,
        "semana": semana.isoformat(),
    }


@router.get("/data/semanas")
async def get_semanas_disponibles(cedula_id: str = Query(...)):
    """Lista las semanas con data cargada, con conteo de registros por semana."""
    sb = get_supabase_admin()
    result = sb.table("qm_data_semanal") \
        .select("semana") \
        .eq("cedula_id", cedula_id) \
        .execute()
    # Contar registros por semana
    counts = {}
    for r in (result.data or []):
        s = r["semana"]
        counts[s] = counts.get(s, 0) + 1
    semanas = sorted(counts.keys(), reverse=True)
    return {"data": [{"semana": s, "registros": counts[s]} for s in semanas]}


@router.delete("/data/semana")
async def delete_semana(
    cedula_id: str = Query(...),
    semana: date = Query(..., description="Semana a eliminar (YYYY-MM-DD)")
):
    """Elimina toda la data de una semana específica."""
    sb = get_supabase_admin()
    result = sb.table("qm_data_semanal") \
        .delete() \
        .eq("cedula_id", cedula_id) \
        .eq("semana", semana.isoformat()) \
        .execute()
    deleted = len(result.data or [])
    return {
        "success": True,
        "message": f"Semana {semana} eliminada ({deleted} registros borrados).",
        "deleted": deleted,
        "semana": semana.isoformat(),
    }


# ══════════════════════════════════════════════════════════════
# ANÁLISIS
# ══════════════════════════════════════════════════════════════

@router.get("/analisis")
async def get_analisis(
    cedula_id: str = Query(...),
    semana: date = Query(..., description="Semana a analizar")
):
    """
    Análisis completo QM: compara data semanal vs calendario.
    Retorna: cumplimiento por empleado, atrasados, sobresalientes,
    próximos cambios, etc.
    """
    sb = get_supabase_admin()

    # Cargar calendario
    cal_result = sb.table("qm_calendario").select("*").eq("cedula_id", cedula_id).execute()
    calendario = cal_result.data or []
    if not calendario:
        raise HTTPException(status_code=404, detail="No hay calendario cargado. Sube el calendario primero.")

    # Cargar data de la semana solicitada
    data_result = sb.table("qm_data_semanal").select("*") \
        .eq("cedula_id", cedula_id).eq("semana", semana.isoformat()).execute()
    data_semanal = data_result.data or []
    if not data_semanal:
        raise HTTPException(status_code=404, detail=f"No hay data para la semana {semana}.")

    # Cargar data de la semana anterior (si existe) para comparar avances
    from datetime import timedelta
    semana_anterior = semana - timedelta(days=7)
    prev_result = sb.table("qm_data_semanal").select("*") \
        .eq("cedula_id", cedula_id).eq("semana", semana_anterior.isoformat()).execute()
    data_prev = prev_result.data or []

    # Indexar data anterior: (employee, competency) → current_level
    prev_map = {}
    for r in data_prev:
        prev_map[(r["employee"], r["competency"])] = r["current_level"]

    # Indexar calendario: (employee, competency) → record
    cal_map = {}
    for r in calendario:
        cal_map[(r["employee"], r["competency"])] = r

    # Mes actual para comparar con pronóstico
    month_col = get_current_month_col()

    # ── Análisis por empleado ──
    employee_stats = defaultdict(lambda: {
        "total": 0, "on_target": 0, "below_target": 0, "above_target": 0,
        "below_schedule": 0, "on_schedule": 0, "above_schedule": 0,
        "advanced": 0, "regressed": 0, "unchanged": 0,
        "role": "", "details": []
    })

    competency_stats = defaultdict(lambda: {
        "total": 0, "on_target": 0, "below_target": 0, "above_target": 0
    })

    for d in data_semanal:
        emp = d["employee"]
        comp = d["competency"]
        current = d["current_level"]
        cal = cal_map.get((emp, comp))
        if not cal:
            continue

        target = cal["target"]
        forecast = cal.get(month_col) or target
        prev_level = prev_map.get((emp, comp))

        stats = employee_stats[emp]
        stats["role"] = d.get("role") or cal.get("role") or ""
        stats["total"] += 1

        # vs Target final
        if current >= target:
            stats["on_target"] += 1
            status = "on_target"
        else:
            stats["below_target"] += 1
            status = "below_target"

        # vs Forecast del mes (calendario)
        if current >= forecast:
            stats["on_schedule"] += 1
            schedule_status = "on_schedule"
        elif current > forecast:
            stats["above_schedule"] += 1
            schedule_status = "above_schedule"
        else:
            stats["below_schedule"] += 1
            schedule_status = "below_schedule"

        # Avance vs semana anterior
        advance_status = "sin_dato_previo"
        if prev_level is not None:
            if current > prev_level:
                stats["advanced"] += 1
                advance_status = "avanzó"
            elif current < prev_level:
                stats["regressed"] += 1
                advance_status = "retrocedió"
            else:
                stats["unchanged"] += 1
                advance_status = "sin_cambio"

        # Competency stats
        cs = competency_stats[comp]
        cs["total"] += 1
        if current >= target:
            cs["on_target"] += 1
        else:
            cs["below_target"] += 1

        stats["details"].append({
            "competency": comp,
            "current": current,
            "target": target,
            "forecast": forecast,
            "prev_level": prev_level,
            "status": status,
            "schedule_status": schedule_status,
            "advance": advance_status,
        })

    # ── Construir respuesta ──

    # Resumen por empleado
    employees_summary = []
    for emp in sorted(employee_stats.keys()):
        s = employee_stats[emp]
        total = s["total"]
        compliance = round(s["on_target"] / total * 100, 1) if total > 0 else 0
        schedule_compliance = round(s["on_schedule"] / total * 100, 1) if total > 0 else 0
        employees_summary.append({
            "employee": emp,
            "role": s["role"],
            "total_competencies": total,
            "on_target": s["on_target"],
            "below_target": s["below_target"],
            "above_target": s["above_target"],
            "compliance_pct": compliance,
            "on_schedule": s["on_schedule"],
            "below_schedule": s["below_schedule"],
            "schedule_compliance_pct": schedule_compliance,
            "advanced_this_week": s["advanced"],
            "regressed_this_week": s["regressed"],
            "details": s["details"],
        })

    # Competencias con más gente atrasada
    comp_summary = []
    for comp in sorted(competency_stats.keys()):
        cs = competency_stats[comp]
        comp_summary.append({
            "competency": comp,
            "total": cs["total"],
            "on_target": cs["on_target"],
            "below_target": cs["below_target"],
            "compliance_pct": round(cs["on_target"] / cs["total"] * 100, 1) if cs["total"] > 0 else 0,
        })
    comp_summary.sort(key=lambda x: x["compliance_pct"])

    # ── Empleados con datos en el calendario pero SIN data esta semana ──
    cal_employees = set(r["employee"] for r in calendario)
    data_employees = set(d["employee"] for d in data_semanal)
    missing_employees = sorted(cal_employees - data_employees)

    # ── Empleados/competencias NO proyectados a llegar al target este año ──
    # Un registro "no proyectado" es aquel donde el último mes del calendario (sep)
    # tiene un valor menor al target final, por lo que el plan no contempla llegar al nivel.
    not_projected = []
    last_month = MONTH_COLS[-1]  # "sep"
    for r in calendario:
        target_val = r.get("target") or 0
        year_end_val = r.get(last_month)
        if year_end_val is not None and year_end_val < target_val:
            not_projected.append({
                "employee": r["employee"],
                "competency": r["competency"],
                "year_end_forecast": year_end_val,
                "target": target_val,
                "gap": target_val - year_end_val,
            })
    # Agrupar por empleado para resumen
    not_proj_by_emp = {}
    for np in not_projected:
        emp = np["employee"]
        if emp not in not_proj_by_emp:
            not_proj_by_emp[emp] = {"employee": emp, "count": 0, "competencies": []}
        not_proj_by_emp[emp]["count"] += 1
        not_proj_by_emp[emp]["competencies"].append({
            "competency": np["competency"],
            "year_end_forecast": np["year_end_forecast"],
            "target": np["target"],
            "gap": np["gap"],
        })
    not_projected_summary = sorted(not_proj_by_emp.values(), key=lambda x: -x["count"])

    # Próximos cambios esperados en el calendario
    upcoming_changes = []
    month_idx = MONTH_COLS.index(month_col) if month_col in MONTH_COLS else 0
    for i in range(month_idx, min(month_idx + 3, len(MONTH_COLS) - 1)):
        curr_m = MONTH_COLS[i]
        next_m = MONTH_COLS[i + 1]
        changes_this_month = []
        for r in calendario:
            curr_val = r.get(curr_m)
            next_val = r.get(next_m)
            if curr_val is not None and next_val is not None and next_val > curr_val:
                changes_this_month.append({
                    "employee": r["employee"],
                    "competency": r["competency"],
                    "from_level": curr_val,
                    "to_level": next_val,
                })
        if changes_this_month:
            upcoming_changes.append({
                "transition": f"{curr_m.capitalize()} → {next_m.capitalize()}",
                "count": len(changes_this_month),
                "changes": changes_this_month,
            })

    # Totales globales
    total_all      = sum(s["total_competencies"] for s in employees_summary)
    total_on       = sum(s["on_target"]          for s in employees_summary)
    total_on_sched = sum(s["on_schedule"]        for s in employees_summary)
    total_adv      = sum(s["advanced_this_week"] for s in employees_summary)

    return {
        "semana": semana.isoformat(),
        "month_reference": month_col,
        "has_previous_week": len(data_prev) > 0,
        "global": {
            "total_entries": total_all,
            "total_employees": len(employees_summary),
            # Cumplimiento Target: ¿ya llegó al nivel final?
            "on_target": total_on,
            "compliance_pct": round(total_on / total_all * 100, 1) if total_all > 0 else 0,
            # Cumplimiento Año: ¿va al ritmo del pronóstico de este mes?
            "on_schedule": total_on_sched,
            "schedule_compliance_pct": round(total_on_sched / total_all * 100, 1) if total_all > 0 else 0,
            "advanced_this_week": total_adv,
        },
        "employees": employees_summary,
        "competencies": comp_summary,
        "upcoming_changes": upcoming_changes,
        "missing_employees": missing_employees,
        "not_projected": not_projected_summary,
        "not_projected_count": len(not_projected),
    }
