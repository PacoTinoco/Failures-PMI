"""
Router para procesamiento de archivos BOS y QBOS.
- BOS: CSV con columnas USER (email), BOS_CREATED (conteo). Match por email.
- QBOS: Excel con columnas Personnel Name (nombre), Frecuency (conteo). Match flexible por nombre.
Ambos guardan su valor en registros_semanales (columnas bos / qbos).
Soporte de deduplicación: si se re-sube en la misma semana, detecta nuevos/actualizados/sin cambio.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Body
from typing import Optional
from datetime import date
from io import StringIO, BytesIO
from collections import defaultdict
import csv

from app.services.supabase_client import get_supabase_admin

router = APIRouter(prefix="/registros", tags=["BOS / QBOS"])


def normalize_name(name: str) -> str:
    """Normaliza nombre para match flexible: lowercase, sin espacios extra."""
    return " ".join(name.lower().strip().split())


def email_to_name_parts(email: str) -> set:
    """Extrae partes del nombre desde un email (ej. 'Gabrielle.Urbina@pmi.com' → {'gabrielle', 'urbina'})."""
    if not email or '@' not in email:
        return set()
    local = email.split('@')[0].lower()
    # Separar por puntos y también por mayúsculas (ej. JulioCesar → julio, cesar)
    import re
    parts = set()
    for segment in local.split('.'):
        # Split CamelCase
        words = re.findall(r'[a-záéíóúñü]+', segment.lower())
        parts.update(words)
    return parts


def flexible_name_match(file_name: str, db_operators: list) -> list:
    """
    Match flexible con 3 estrategias:
    1. Nombre exacto (score 100)
    2. Match por email: nombre del archivo vs partes del email (score 95)
    3. Partes comunes del nombre: ≥2 partes coinciden (score proporcional)
    """
    fn = normalize_name(file_name)
    fn_parts = set(fn.split())
    matches = []

    for op in db_operators:
        dn = normalize_name(op["nombre"])
        best_score = 0

        # 1. Exact name match
        if fn == dn:
            best_score = 100
        else:
            # 2. Email-based match: comparar partes del nombre del archivo con partes del email
            email_parts = email_to_name_parts(op.get("email", ""))
            if email_parts and fn_parts:
                common_email = fn_parts & email_parts
                if len(common_email) >= 2:
                    best_score = 95
                elif len(common_email) == 1 and len(fn_parts) <= 2:
                    # Solo un match pero el nombre tiene pocas partes
                    best_score = 70

            # 3. Name parts match
            dn_parts = set(dn.split())
            common_name = fn_parts & dn_parts
            if len(common_name) >= 2:
                name_score = int(len(common_name) / max(len(fn_parts), len(dn_parts)) * 100)
                # Boost: si coinciden ≥2 partes, mínimo 60
                name_score = max(name_score, 60)
                best_score = max(best_score, name_score)

            # 4. One contains the other
            if fn in dn or dn in fn:
                best_score = max(best_score, 85)

        if best_score > 0:
            matches.append((op, best_score))

    matches.sort(key=lambda x: -x[1])
    return matches


# ══════════════════════════════════════════════════════════════
# BOS — Upload CSV
# ══════════════════════════════════════════════════════════════

@router.post("/bos/upload")
async def upload_bos(
    file: UploadFile = File(...),
    cedula_id: str = Query(...),
    semana: date = Query(..., description="Lunes de la semana (YYYY-MM-DD)")
):
    """Procesa CSV de BOS. Match por email (USER) → bos = BOS_CREATED."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos CSV para BOS")

    content = await file.read()
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(StringIO(text))
    rows = list(reader)

    if not rows:
        raise HTTPException(status_code=400, detail="El archivo CSV está vacío")

    # Verificar columnas
    required = {'USER', 'BOS_CREATED'}
    actual = set(rows[0].keys())
    missing = required - actual
    if missing:
        raise HTTPException(status_code=400, detail=f"Faltan columnas: {missing}")

    # Cargar operadores con email
    sb = get_supabase_admin()
    ops_result = sb.table("operadores").select("id, nombre, email").eq("cedula_id", cedula_id).execute()
    operadores = ops_result.data or []

    # Map email → operador_id
    email_map = {}
    for op in operadores:
        if op.get("email"):
            email_map[op["email"].lower().strip()] = op["id"]

    # Cargar registros existentes para esta semana
    existing = sb.table("registros_semanales") \
        .select("id, operador_id, bos") \
        .eq("cedula_id", cedula_id) \
        .eq("semana", semana.isoformat()) \
        .execute()
    existing_map = {}
    for r in (existing.data or []):
        existing_map[r["operador_id"]] = r

    # Procesar CSV
    results = []
    unmatched = []
    new_count = 0
    updated_count = 0
    unchanged_count = 0

    for row in rows:
        email = row.get("USER", "").lower().strip()
        try:
            bos_val = int(float(row.get("BOS_CREATED", 0)))
        except (ValueError, TypeError):
            bos_val = 0

        operador_id = email_map.get(email)
        if not operador_id:
            unmatched.append(email)
            continue

        # Determinar acción
        existing_rec = existing_map.get(operador_id)
        if existing_rec:
            old_val = existing_rec.get("bos") or 0
            if old_val != bos_val:
                action = "actualizado"
                updated_count += 1
            else:
                action = "sin_cambio"
                unchanged_count += 1
        else:
            action = "nuevo"
            new_count += 1

        # Buscar nombre del operador
        op_name = next((op["nombre"] for op in operadores if op["id"] == operador_id), email)

        results.append({
            "operador_id": operador_id,
            "nombre": op_name,
            "email": email,
            "bos": bos_val,
            "action": action,
        })

    return {
        "success": True,
        "semana": semana.isoformat(),
        "total_rows": len(rows),
        "matched": len(results),
        "unmatched": unmatched,
        "new": new_count,
        "updated": updated_count,
        "unchanged": unchanged_count,
        "results": results,
    }


@router.post("/bos/save")
async def save_bos(
    cedula_id: str = Query(...),
    semana: date = Query(...),
    body: list = Body(...)
):
    """Guarda los resultados de BOS en registros_semanales."""
    sb = get_supabase_admin()
    saved = 0
    for item in body:
        operador_id = item.get("operador_id")
        bos_val = item.get("bos", 0)
        if not operador_id:
            continue

        # Upsert en registros_semanales
        # Captura usa 'bos_num' como campo, también calcular bos_eng (target=3)
        bos_eng = min(100, round(bos_val / 3 * 100, 1)) if bos_val else 0
        rec = {
            "operador_id": operador_id,
            "cedula_id": cedula_id,
            "semana": semana.isoformat(),
            "bos_num": bos_val,
            "bos_eng": bos_eng,
        }
        sb.table("registros_semanales") \
            .upsert(rec, on_conflict="operador_id,semana") \
            .execute()
        saved += 1

    return {"success": True, "message": f"BOS guardado: {saved} registros para semana {semana}", "saved": saved}


# ══════════════════════════════════════════════════════════════
# QBOS — Upload Excel
# ══════════════════════════════════════════════════════════════

@router.post("/qbos/upload")
async def upload_qbos(
    file: UploadFile = File(...),
    cedula_id: str = Query(...),
    semana: date = Query(..., description="Lunes de la semana (YYYY-MM-DD)")
):
    """Procesa Excel de QBOS. Match flexible por nombre → qbos = Frecuency."""
    import pandas as pd

    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos Excel para QBOS")

    content = await file.read()
    df = pd.read_excel(BytesIO(content))

    if 'Personnel Name' not in df.columns:
        raise HTTPException(status_code=400, detail="Falta columna 'Personnel Name'")
    if 'Frecuency' not in df.columns:
        raise HTTPException(status_code=400, detail="Falta columna 'Frecuency'")

    # Cargar operadores
    sb = get_supabase_admin()
    ops_result = sb.table("operadores").select("id, nombre, email").eq("cedula_id", cedula_id).execute()
    operadores = ops_result.data or []

    # Cargar registros existentes
    existing = sb.table("registros_semanales") \
        .select("id, operador_id, qbos") \
        .eq("cedula_id", cedula_id) \
        .eq("semana", semana.isoformat()) \
        .execute()
    existing_map = {}
    for r in (existing.data or []):
        existing_map[r["operador_id"]] = r

    results = []
    unmatched = []
    new_count = 0
    updated_count = 0
    unchanged_count = 0

    for _, row in df.iterrows():
        name = str(row.get("Personnel Name", "")).strip()
        if not name or name == "nan":
            continue

        # Skip rows that look like filter metadata
        if name.startswith("Applied filters") or name.startswith("AUTHOR") or name.startswith("Department") or name.startswith("DATE"):
            continue

        freq = row.get("Frecuency")
        qbos_val = int(freq) if pd.notna(freq) else 0

        # Flexible name match
        matches = flexible_name_match(name, operadores)

        if not matches:
            unmatched.append({"name": name, "qbos": qbos_val, "suggestions": []})
            continue

        best_match, best_score = matches[0]

        if best_score >= 60:
            operador_id = best_match["id"]
            op_name = best_match["nombre"]
        else:
            # Ambiguous match — include suggestions for user
            unmatched.append({
                "name": name,
                "qbos": qbos_val,
                "suggestions": [{"id": m[0]["id"], "nombre": m[0]["nombre"], "score": m[1]} for m in matches[:3]]
            })
            continue

        # Determinar acción
        existing_rec = existing_map.get(operador_id)
        if existing_rec:
            old_val = existing_rec.get("qbos") or 0
            if old_val != qbos_val:
                action = "actualizado"
                updated_count += 1
            else:
                action = "sin_cambio"
                unchanged_count += 1
        else:
            action = "nuevo"
            new_count += 1

        results.append({
            "operador_id": operador_id,
            "nombre": op_name,
            "file_name": name,
            "qbos": qbos_val,
            "match_score": best_score,
            "action": action,
        })

    return {
        "success": True,
        "semana": semana.isoformat(),
        "total_rows": len(df),
        "matched": len(results),
        "unmatched": unmatched,
        "new": new_count,
        "updated": updated_count,
        "unchanged": unchanged_count,
        "results": results,
    }


@router.post("/qbos/save")
async def save_qbos(
    cedula_id: str = Query(...),
    semana: date = Query(...),
    body: list = Body(...)
):
    """Guarda los resultados de QBOS en registros_semanales."""
    sb = get_supabase_admin()
    saved = 0
    for item in body:
        operador_id = item.get("operador_id")
        qbos_val = item.get("qbos", 0)
        if not operador_id:
            continue

        # Captura usa 'qbos_num' como campo, también calcular qbos_eng (target=1)
        qbos_eng = min(100, round(qbos_val / 1 * 100, 1)) if qbos_val else 0
        rec = {
            "operador_id": operador_id,
            "cedula_id": cedula_id,
            "semana": semana.isoformat(),
            "qbos_num": qbos_val,
            "qbos_eng": qbos_eng,
        }
        sb.table("registros_semanales") \
            .upsert(rec, on_conflict="operador_id,semana") \
            .execute()
        saved += 1

    return {"success": True, "message": f"QBOS guardado: {saved} registros para semana {semana}", "saved": saved}
