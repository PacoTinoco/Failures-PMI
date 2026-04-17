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

    # Normalize column aliases: "Personal Number" → "Employee"
    rename_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ('personal number', 'personal_number', 'personalnumber', 'personnel number'):
            rename_map[c] = 'Employee'
    if rename_map:
        df = df.rename(columns=rename_map)

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

    # Filtrar filas de Total y vacías
    df = df[df['Employee'].notna()]
    df = df[~df['Employee'].astype(str).str.strip().str.lower().isin(['total', 'nan', ''])]
    df = df.dropna(how='all')

    # Insertar nuevo calendario
    records = []
    for _, row in df.iterrows():
        emp = str(row['Employee']).strip() if pd.notna(row['Employee']) else None
        comp = str(row['Competency']).strip() if pd.notna(row['Competency']) else None
        if not emp or not comp or emp.lower() == 'total':
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


@router.post("/calendario/entry")
async def create_calendario_entry(body: dict):
    """Crea una nueva entrada en el calendario QM."""
    sb = get_supabase_admin()
    required = {"cedula_id", "employee", "competency"}
    if not required.issubset(set(body.keys())):
        raise HTTPException(status_code=400, detail=f"Faltan campos requeridos: {required - set(body.keys())}")

    allowed = {"cedula_id", "employee", "competency", "role", "target", "current_base",
               "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep"}
    rec = {k: v for k, v in body.items() if k in allowed}
    rec.setdefault("target", 0)
    rec.setdefault("current_base", 0)

    result = sb.table("qm_calendario").upsert(
        rec, on_conflict="cedula_id,employee,competency"
    ).execute()
    return {"success": True, "data": result.data[0] if result.data else None}


@router.delete("/calendario/{record_id}")
async def delete_calendario_entry(record_id: str):
    """Elimina una entrada del calendario QM."""
    sb = get_supabase_admin()
    result = sb.table("qm_calendario").delete().eq("id", record_id).execute()
    return {"success": True, "deleted": len(result.data or [])}


# ══════════════════════════════════════════════════════════════
# DATA SEMANAL
# ══════════════════════════════════════════════════════════════

@router.post("/data/upload")
async def upload_data_preview(
    file: UploadFile = File(...),
    cedula_id: str = Query(...),
    semana: date = Query(..., description="Fecha del lunes de la semana (YYYY-MM-DD)")
):
    """
    Parsea el Excel y compara con datos existentes. NO guarda en DB.
    Devuelve vista previa con cambios detectados y los records listos para guardar.
    El usuario debe llamar a POST /qm/data/save para persistir.
    """
    import pandas as pd

    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos Excel (.xlsx)")

    content = await file.read()
    df = pd.read_excel(BytesIO(content))

    # Normalize column aliases: "Personal Number" → "Employee"
    rename_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ('personal number', 'personal_number', 'personalnumber', 'personnel number'):
            rename_map[c] = 'Employee'
    if rename_map:
        df = df.rename(columns=rename_map)

    # Filtrar filas de Total y vacías
    df = df[df['Employee'].notna()]
    df = df[~df['Employee'].astype(str).str.strip().str.lower().isin(['total', 'nan', ''])]
    df = df.dropna(how='all')

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

    sb = get_supabase_admin()

    # Cargar competencias del calendario para filtrar
    cal_result = sb.table("qm_calendario").select("employee, competency").eq("cedula_id", cedula_id).execute()
    cal_pairs = set()
    for r in (cal_result.data or []):
        cal_pairs.add((r["employee"], r["competency"]))

    # Cargar data anterior de esta semana para detectar cambios
    prev_result = sb.table("qm_data_semanal").select("employee, competency, current_level") \
        .eq("cedula_id", cedula_id).eq("semana", semana.isoformat()).execute()
    prev_map = {}
    for r in (prev_result.data or []):
        prev_map[(r["employee"], r["competency"])] = r["current_level"]

    # Parsear registros del Excel
    records = []
    skipped = 0
    role_col = None
    for c in df.columns:
        if c == 'Role' or c == 'Role.1':
            role_col = c

    for _, row in df.iterrows():
        emp = str(row['Employee']).strip() if pd.notna(row['Employee']) else None
        comp = str(row['Competency']).strip() if pd.notna(row['Competency']) else None
        if not emp or not comp or emp.lower() == 'total':
            continue
        if (emp, comp) not in cal_pairs:
            skipped += 1
            continue

        on_target = 0
        if 'OnTarget' in df.columns and pd.notna(row.get('OnTarget')):
            on_target = int(row['OnTarget'])

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

    # Deduplicar
    seen = {}
    for rec in records:
        seen[(rec["employee"], rec["competency"])] = rec
    records = list(seen.values())

    # Detectar cambios
    new_count = 0
    changed_count = 0
    unchanged_count = 0
    changes_detail = []

    for rec in records:
        old = prev_map.get((rec["employee"], rec["competency"]))
        if old is None:
            new_count += 1
        elif old != rec["current_level"]:
            changed_count += 1
            changes_detail.append({
                "employee": rec["employee"],
                "competency": rec["competency"],
                "old_level": old,
                "new_level": rec["current_level"],
            })
        else:
            unchanged_count += 1

    return {
        "success": True,
        "preview": True,
        "semana": semana.isoformat(),
        "total_records": len(records),
        "skipped": skipped,
        "new_records": new_count,
        "changed_records": changed_count,
        "unchanged_records": unchanged_count,
        "changes_detail": changes_detail,
        "is_first_upload": len(prev_map) == 0,
        "already_saved": len(prev_map) > 0,
        "records": records,   # devolver records para que el frontend los envíe al save
    }


@router.post("/data/save")
async def save_data_semanal(body: dict):
    """
    Guarda los records parseados en DB. Recibe:
      { cedula_id, semana, records: [...], changed_records, new_records, changes_detail }
    Usa DELETE + INSERT con verificación robusta.
    """
    cedula_id = body.get("cedula_id")
    semana_str = body.get("semana")
    records = body.get("records", [])
    changed_records = body.get("changed_records", 0)
    new_records = body.get("new_records", 0)
    changes_detail = body.get("changes_detail", [])

    if not cedula_id or not semana_str or not records:
        raise HTTPException(status_code=400, detail="Faltan campos: cedula_id, semana, records")

    sb = get_supabase_admin()

    # Asegurar que cada record tiene cedula_id y semana correctos
    clean_records = []
    for rec in records:
        clean_records.append({
            "cedula_id": cedula_id,
            "semana": semana_str,
            "employee": rec.get("employee", ""),
            "competency": rec.get("competency", ""),
            "role": rec.get("role"),
            "current_level": rec.get("current_level", 0),
            "target": rec.get("target", 0),
            "on_target": rec.get("on_target", 0),
        })

    # 1. Borrar datos existentes de la semana
    try:
        del_result = sb.table("qm_data_semanal").delete() \
            .eq("cedula_id", cedula_id) \
            .eq("semana", semana_str) \
            .execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al borrar datos existentes: {str(e)}")

    # 2. Insertar en lotes de 50 (lotes más pequeños = más confiable)
    total_inserted = 0
    insert_errors = []
    for i in range(0, len(clean_records), 50):
        batch = clean_records[i:i+50]
        try:
            result = sb.table("qm_data_semanal").insert(batch).execute()
            if result.data:
                total_inserted += len(result.data)
            else:
                insert_errors.append(f"Lote {i//50+1}: sin datos en respuesta")
        except Exception as e:
            insert_errors.append(f"Lote {i//50+1}: {str(e)}")

    # 3. Verificar: hacer SELECT real y contar filas
    try:
        verify = sb.table("qm_data_semanal") \
            .select("id") \
            .eq("cedula_id", cedula_id) \
            .eq("semana", semana_str) \
            .limit(1000) \
            .execute()
        verified_count = len(verify.data) if verify.data else 0
    except Exception:
        verified_count = total_inserted

    # Si verificación retornó menos de lo esperado pero hay más, hacer otro SELECT
    if verified_count < len(clean_records) and verified_count > 0:
        try:
            verify2 = sb.table("qm_data_semanal") \
                .select("id") \
                .eq("cedula_id", cedula_id) \
                .eq("semana", semana_str) \
                .execute()
            verified_count = len(verify2.data) if verify2.data else verified_count
        except Exception:
            pass

    if verified_count == 0 and len(clean_records) > 0:
        error_detail = "; ".join(insert_errors) if insert_errors else "Los datos no se encontraron en la base de datos después de insertar."
        raise HTTPException(status_code=500, detail=f"Error al guardar: {error_detail}")

    # 4. Registrar en historial
    try:
        sb.table("qm_upload_log").insert({
            "cedula_id": cedula_id,
            "semana": semana_str,
            "total_records": verified_count,
            "new_records": new_records,
            "changed_records": changed_records,
            "unchanged_records": len(clean_records) - new_records - changed_records,
            "changes_detail": json.dumps(changes_detail),
        }).execute()
    except Exception:
        pass

    return {
        "success": True,
        "message": f"{verified_count} registros guardados correctamente en la semana {semana_str}.",
        "total_records": verified_count,
        "semana": semana_str,
        "insert_errors": insert_errors if insert_errors else None,
    }


@router.get("/data/verificar")
async def verificar_data(
    cedula_id: str = Query(...),
    semana: date = Query(...)
):
    """Verifica cuántos registros hay en DB para una semana."""
    sb = get_supabase_admin()
    result = sb.table("qm_data_semanal").select("semana", count="exact") \
        .eq("cedula_id", cedula_id) \
        .eq("semana", semana.isoformat()) \
        .execute()
    count = result.count or 0
    return {
        "semana": semana.isoformat(),
        "count": count,
        "saved": count > 0,
    }


@router.get("/data/semanas")
async def get_semanas_disponibles(cedula_id: str = Query(...)):
    """
    Lista las semanas con data cargada, con conteo de registros por semana.
    Combina datos de qm_data_semanal (data real) + qm_upload_log (historial).
    Esto asegura que incluso si por alguna razón la data principal se perdió,
    el historial sigue visible.
    """
    sb = get_supabase_admin()

    # Fuente 1: data real en qm_data_semanal
    result = sb.table("qm_data_semanal") \
        .select("semana") \
        .eq("cedula_id", cedula_id) \
        .execute()
    counts = {}
    for r in (result.data or []):
        s = r["semana"]
        counts[s] = counts.get(s, 0) + 1

    # Fuente 2: historial de uploads (puede tener semanas donde data ya fue borrada)
    try:
        log_result = sb.table("qm_upload_log") \
            .select("semana, total_records, uploaded_at") \
            .eq("cedula_id", cedula_id) \
            .order("uploaded_at", desc=True) \
            .execute()
        # Agrupar uploads por semana
        uploads_by_semana = {}
        for log in (log_result.data or []):
            s = log["semana"]
            if s not in uploads_by_semana:
                uploads_by_semana[s] = []
            uploads_by_semana[s].append({
                "uploaded_at": log["uploaded_at"],
                "total_records": log["total_records"],
            })
    except Exception:
        uploads_by_semana = {}

    # Combinar: todas las semanas que tengan datos O historial
    all_semanas = set(counts.keys()) | set(uploads_by_semana.keys())
    semanas = sorted(all_semanas, reverse=True)

    data = []
    for s in semanas:
        entry = {
            "semana": s,
            "registros": counts.get(s, 0),
            "has_data": s in counts,
            "upload_count": len(uploads_by_semana.get(s, [])),
            "last_upload": uploads_by_semana[s][0]["uploaded_at"] if s in uploads_by_semana else None,
        }
        data.append(entry)

    return {"data": data}


@router.delete("/data/semana")
async def delete_semana(
    cedula_id: str = Query(...),
    semana: date = Query(..., description="Semana a eliminar (YYYY-MM-DD)")
):
    """Elimina toda la data de una semana específica + su historial de uploads."""
    sb = get_supabase_admin()

    # 1. Borrar data real
    result = sb.table("qm_data_semanal") \
        .delete() \
        .eq("cedula_id", cedula_id) \
        .eq("semana", semana.isoformat()) \
        .execute()
    deleted_data = len(result.data or [])

    # 2. Borrar historial de uploads de esa semana
    deleted_logs = 0
    try:
        log_result = sb.table("qm_upload_log") \
            .delete() \
            .eq("cedula_id", cedula_id) \
            .eq("semana", semana.isoformat()) \
            .execute()
        deleted_logs = len(log_result.data or [])
    except Exception:
        pass

    return {
        "success": True,
        "message": f"Semana {semana} eliminada ({deleted_data} registros + {deleted_logs} logs borrados).",
        "deleted": deleted_data,
        "deleted_logs": deleted_logs,
        "semana": semana.isoformat(),
    }


# ══════════════════════════════════════════════════════════════
# HISTORIAL DE UPLOADS
# ══════════════════════════════════════════════════════════════

@router.get("/data/uploads")
async def get_upload_history(
    cedula_id: str = Query(...),
    semana: Optional[date] = Query(None, description="Filtrar por semana específica")
):
    """Historial detallado de uploads de data semanal."""
    sb = get_supabase_admin()
    try:
        query = sb.table("qm_upload_log").select("*").eq("cedula_id", cedula_id)
        if semana:
            query = query.eq("semana", semana.isoformat())
        result = query.order("uploaded_at", desc=True).limit(100).execute()
        uploads = result.data or []
        # Parsear changes_detail de string JSON a lista
        for u in uploads:
            if isinstance(u.get("changes_detail"), str):
                try:
                    u["changes_detail"] = json.loads(u["changes_detail"])
                except Exception:
                    u["changes_detail"] = []
        return {"data": uploads}
    except Exception:
        return {"data": []}


@router.delete("/data/upload/{log_id}")
async def delete_upload_log(
    log_id: int,
    cedula_id: str = Query(..., description="Cédula del usuario (para verificar ownership)")
):
    """Elimina un entry específico del historial de uploads."""
    sb = get_supabase_admin()
    # Verificar que el log pertenece a esta cédula antes de borrar
    check = sb.table("qm_upload_log").select("id, cedula_id") \
        .eq("id", log_id).eq("cedula_id", cedula_id).execute()
    if not check.data:
        raise HTTPException(status_code=404, detail="Historial no encontrado o no tienes permiso para eliminarlo.")
    sb.table("qm_upload_log").delete().eq("id", log_id).execute()
    return {"success": True, "message": "Entrada de historial eliminada."}


# ══════════════════════════════════════════════════════════════
# SYNC QM → DASHBOARD (registros_semanales.qm_on_target)
# ══════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    """Normaliza: lowercase, sin espacios extra."""
    if not text:
        return ""
    return " ".join(text.lower().strip().split())


def _match_qm_employee_to_operador(qm_name: str, operadores: list, aliases: list) -> str | None:
    """
    Match robusto: nombre QM → operador_id.
    Estrategias (en orden de prioridad):
    1. Match exacto por operador.nombre
    2. Match por alias nombre_qbos (normalizado)
    3. Match por alias nombre_bd (normalizado)
    4. Containment (nombre QM contiene nombre operador o viceversa)
    5. Name parts overlap (≥2 palabras en común)
    """
    qm_lower = _normalize(qm_name)
    qm_parts = set(qm_lower.split())

    # Build lookup maps
    op_name_to_id = {}
    for op in operadores:
        op_name_to_id[_normalize(op["nombre"])] = op["id"]

    alias_qbos_to_id = {}
    alias_bd_to_id = {}
    for a in aliases:
        if a.get("nombre_qbos"):
            alias_qbos_to_id[_normalize(a["nombre_qbos"])] = a["persona_id"]
        if a.get("nombre_bd"):
            alias_bd_to_id[_normalize(a["nombre_bd"])] = a["persona_id"]

    # 1° Match exacto por nombre operador
    if qm_lower in op_name_to_id:
        return op_name_to_id[qm_lower]

    # 2° Match exacto por alias nombre_qbos
    if qm_lower in alias_qbos_to_id:
        return alias_qbos_to_id[qm_lower]

    # 3° Match exacto por alias nombre_bd
    if qm_lower in alias_bd_to_id:
        return alias_bd_to_id[qm_lower]

    # 4° Containment: nombre QM contiene nombre operador o viceversa
    for op_nombre, oid in op_name_to_id.items():
        if op_nombre in qm_lower or qm_lower in op_nombre:
            return oid

    for alias_nombre, pid in alias_qbos_to_id.items():
        if alias_nombre in qm_lower or qm_lower in alias_nombre:
            return pid

    for alias_nombre, pid in alias_bd_to_id.items():
        if alias_nombre in qm_lower or qm_lower in alias_nombre:
            return pid

    # 5° Name parts overlap: ≥2 palabras en común con operador nombre
    best_match = None
    best_score = 0
    for op_nombre, oid in op_name_to_id.items():
        op_parts = set(op_nombre.split())
        common = qm_parts & op_parts
        if len(common) >= 2 and len(common) > best_score:
            best_score = len(common)
            best_match = oid

    if best_match:
        return best_match

    # 5b° Name parts overlap con aliases
    for a in aliases:
        for field in ["nombre_qbos", "nombre_bd"]:
            alias_name = a.get(field)
            if not alias_name:
                continue
            alias_parts = set(_normalize(alias_name).split())
            common = qm_parts & alias_parts
            if len(common) >= 2 and len(common) > best_score:
                best_score = len(common)
                best_match = a["persona_id"]

    return best_match


@router.post("/sync-dashboard")
async def sync_qm_to_dashboard(
    cedula_id: str = Query(...),
    semana: date = Query(..., description="Semana a sincronizar")
):
    """
    Calcula el % de cumplimiento QM por empleado y lo escribe
    en registros_semanales.qm_on_target para cada operador.
    Usa sistema completo de aliases + fuzzy matching para el match.
    """
    sb = get_supabase_admin()

    # 1. Cargar calendario QM
    cal_result = sb.table("qm_calendario").select("*").eq("cedula_id", cedula_id).execute()
    calendario = cal_result.data or []
    if not calendario:
        raise HTTPException(status_code=404, detail="No hay calendario QM cargado.")

    # 2. Cargar data semanal
    data_result = sb.table("qm_data_semanal").select("*") \
        .eq("cedula_id", cedula_id).eq("semana", semana.isoformat()).execute()
    data_semanal = data_result.data or []
    if not data_semanal:
        raise HTTPException(status_code=404, detail=f"No hay data QM para la semana {semana}.")

    # Indexar calendario
    cal_map = {}
    for r in calendario:
        cal_map[(r["employee"], r["competency"])] = r

    # 3. Calcular compliance_pct por empleado QM
    emp_stats = defaultdict(lambda: {"total": 0, "on_target": 0})
    for d in data_semanal:
        cal = cal_map.get((d["employee"], d["competency"]))
        if not cal:
            continue
        target = cal["target"]
        current = d["current_level"]
        emp_stats[d["employee"]]["total"] += 1
        if current >= target:
            emp_stats[d["employee"]]["on_target"] += 1

    qm_by_employee = {}
    for emp, s in emp_stats.items():
        pct = round(s["on_target"] / s["total"] * 100, 1) if s["total"] > 0 else 0
        qm_by_employee[emp] = pct

    # 4. Cargar operadores de esta cédula (activos e inactivos, para cubrir más nombres)
    ops_result = sb.table("operadores").select("id, nombre").eq("cedula_id", cedula_id).execute()
    operadores = ops_result.data or []

    # También cargar LCs y LS (pueden estar en el QM)
    lcs_result = sb.table("line_coordinators").select("id, nombre").eq("cedula_id", cedula_id).execute()
    lcs = lcs_result.data or []

    try:
        ls_result = sb.table("linea_estructura").select("id, nombre").eq("cedula_id", cedula_id).execute()
        ls_list = ls_result.data or []
    except Exception:
        ls_list = []

    # Combinar todas las personas para matching
    all_personas = operadores + lcs + ls_list

    # 5. Cargar TODOS los aliases de esta cédula
    all_ids = [p["id"] for p in all_personas]
    aliases = []
    if all_ids:
        try:
            alias_result = sb.table("operador_aliases") \
                .select("persona_id, persona_tipo, nombre_bd, nombre_qbos, email_bos, email_dh") \
                .in_("persona_id", all_ids) \
                .execute()
            aliases = alias_result.data or []
        except Exception:
            aliases = []

    # 6. Match QM employee → persona_id usando matching robusto
    matched = 0
    not_matched = []
    updates = []
    match_details = []

    # Set de IDs de operadores (solo operadores van al dashboard)
    op_ids = set(op["id"] for op in operadores)

    for qm_emp, pct in qm_by_employee.items():
        persona_id = _match_qm_employee_to_operador(qm_emp, all_personas, aliases)

        if persona_id:
            matched += 1
            # Solo escribir en dashboard si es operador
            if persona_id in op_ids:
                updates.append({"operador_id": persona_id, "qm_on_target": pct, "qm_name": qm_emp})
            match_details.append({"qm": qm_emp, "matched_id": persona_id, "is_operador": persona_id in op_ids})
        else:
            not_matched.append(qm_emp)

    # 7. Escribir en registros_semanales
    written = 0
    write_errors = []
    for upd in updates:
        try:
            sb.table("registros_semanales").upsert({
                "operador_id": upd["operador_id"],
                "cedula_id": cedula_id,
                "semana": semana.isoformat(),
                "qm_on_target": upd["qm_on_target"],
            }, on_conflict="operador_id,semana").execute()
            written += 1
        except Exception as e:
            write_errors.append(f"{upd.get('qm_name', '?')}: {str(e)}")

    return {
        "success": True,
        "message": f"QM → Dashboard sincronizado: {written}/{len(qm_by_employee)} operadores actualizados.",
        "matched": matched,
        "written": written,
        "not_matched": not_matched,
        "total_qm_employees": len(qm_by_employee),
        "write_errors": write_errors if write_errors else None,
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
        "total": 0, "on_target": 0, "below_target": 0, "above_target": 0,
        "below_employees": []
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
        # Si ya alcanzó el target final, NO es atrasado (aunque forecast sea mayor)
        if current >= target:
            stats["on_schedule"] += 1
            schedule_status = "on_schedule"
        elif current >= forecast:
            stats["on_schedule"] += 1
            schedule_status = "on_schedule"
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
            cs["below_employees"].append({
                "employee": emp,
                "current": current,
                "target": target,
                "forecast": forecast,
                "gap": target - current,
            })

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
            "below_employees": sorted(cs["below_employees"], key=lambda x: -x["gap"]),
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

    # ── Cambios pendientes por mes (incluye atrasados de meses pasados) ──
    # Para cada mes en el calendario, si el nivel esperado > nivel del mes anterior
    # y el current_level del empleado aún no alcanza ese nivel → pendiente.
    # Si el mes ya pasó → atrasado.
    upcoming_changes = []
    month_idx = MONTH_COLS.index(month_col) if month_col in MONTH_COLS else 0

    # Indexar current_level por (employee, competency) desde data semanal
    current_map = {}
    for d in data_semanal:
        current_map[(d["employee"], d["competency"])] = d.get("current_level", 0)

    # Recorrer desde el primer mes hasta el final
    for i in range(1, len(MONTH_COLS)):
        prev_m = MONTH_COLS[i - 1]
        dest_m = MONTH_COLS[i]
        dest_m_idx = i  # posición del mes destino

        changes_this_month = []
        for r in calendario:
            prev_val = r.get(prev_m)
            dest_val = r.get(dest_m)
            if prev_val is not None and dest_val is not None and dest_val > prev_val:
                # Hay un cambio programado para este mes
                key = (r["employee"], r["competency"])
                current = current_map.get(key, 0)
                # Solo incluir si aún no se cumplió (current < nivel esperado)
                if current < dest_val:
                    is_overdue = dest_m_idx <= month_idx  # el mes ya pasó o es el actual
                    changes_this_month.append({
                        "employee": r["employee"],
                        "competency": r["competency"],
                        "from_level": prev_val,
                        "to_level": dest_val,
                        "current_level": current,
                        "overdue": is_overdue,
                    })
        if changes_this_month:
            is_past = MONTH_COLS.index(dest_m) < month_idx
            label = dest_m.capitalize()
            if is_past:
                label = f"{dest_m.capitalize()} (atrasado)"
            upcoming_changes.append({
                "month": dest_m.capitalize(),
                "transition": label,
                "count": len(changes_this_month),
                "changes": changes_this_month,
                "is_overdue": is_past,
            })

    # ── Enriquecer detalles por empleado con due_month (mes vencido) ──
    # Mapear (employee, competency) → mes más antiguo vencido
    overdue_map = {}  # (employee, competency) → month label
    for uc in upcoming_changes:
        if uc.get("is_overdue"):
            for ch in uc["changes"]:
                if ch.get("overdue"):
                    key = (ch["employee"], ch["competency"])
                    if key not in overdue_map:
                        overdue_map[key] = uc["month"]  # primer mes vencido

    for es in employees_summary:
        for det in es["details"]:
            key = (es["employee"], det["competency"])
            det["due_month"] = overdue_map.get(key)  # None si no hay atraso

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
