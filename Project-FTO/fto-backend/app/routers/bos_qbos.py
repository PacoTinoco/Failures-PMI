"""
Router para procesamiento de archivos BOS, QBOS y mapeo de aliases.

Flujo de match (prioridad):
1. Buscar en operador_aliases (match exacto, 100% confiable)
2. Fallback a fuzzy matching (email parts, name parts)

Endpoints:
- POST /registros/aliases/upload  → Sube Excel de mapeo y puebla operador_aliases
- POST /registros/bos/upload      → Procesa CSV de BOS (match por email)
- POST /registros/bos/save        → Guarda resultados en registros_semanales
- POST /registros/qbos/upload     → Procesa Excel de QBOS (match por nombre)
- POST /registros/qbos/save       → Guarda resultados en registros_semanales
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Body
from typing import Optional
from datetime import date
from io import StringIO, BytesIO
from collections import defaultdict
import csv
import re

from app.services.supabase_client import get_supabase_admin

router = APIRouter(prefix="/registros", tags=["BOS / QBOS"])


# ══════════════════════════════════════════════════════════════
# Helpers de normalización y matching
# ══════════════════════════════════════════════════════════════

def normalize(text: str) -> str:
    """Normaliza: lowercase, sin espacios extra, sin acentos comunes."""
    if not text:
        return ""
    return " ".join(text.lower().strip().split())


def email_to_name_parts(email: str) -> set:
    """Extrae partes del nombre desde un email.
    Ej: 'JulioCesar.FariasVillalvazo@pmi.com' → {'julio', 'cesar', 'farias', 'villalvazo'}
    """
    if not email or '@' not in email:
        return set()
    local = email.split('@')[0].lower()
    parts = set()
    for segment in local.split('.'):
        words = re.findall(r'[a-záéíóúñü]+', segment.lower())
        parts.update(words)
    return parts


def fuzzy_name_match(file_name: str, db_operators: list) -> list:
    """
    Fallback: match flexible cuando no hay alias.
    Estrategias (en orden de confianza):
    1. Nombre exacto (score 100)
    2. Containment (score 85)
    3. Email parts match (score 70-95)
    4. Name parts overlap (score 60+)
    """
    fn = normalize(file_name)
    fn_parts = set(fn.split())
    matches = []

    for op in db_operators:
        dn = normalize(op["nombre"])
        best_score = 0

        # 1. Exact
        if fn == dn:
            best_score = 100
        else:
            # 2. Containment
            if fn in dn or dn in fn:
                best_score = max(best_score, 85)

            # 3. Email-based
            email_parts = email_to_name_parts(op.get("email", ""))
            if email_parts and fn_parts:
                common = fn_parts & email_parts
                if len(common) >= 2:
                    best_score = max(best_score, 95)
                elif len(common) == 1 and len(fn_parts) <= 2:
                    best_score = max(best_score, 70)

            # 4. Name parts
            dn_parts = set(dn.split())
            common_name = fn_parts & dn_parts
            if len(common_name) >= 2:
                score = max(int(len(common_name) / max(len(fn_parts), len(dn_parts)) * 100), 60)
                best_score = max(best_score, score)

        if best_score > 0:
            matches.append((op, best_score))

    matches.sort(key=lambda x: -x[1])
    return matches


# ══════════════════════════════════════════════════════════════
# ALIASES — Subir Excel de mapeo
# ══════════════════════════════════════════════════════════════

@router.post("/aliases/upload")
async def upload_aliases(
    file: UploadFile = File(...),
    cedula_id: str = Query(...)
):
    """
    Sube el Excel de mapeo de empleados y puebla/actualiza operador_aliases.
    Columnas esperadas: nombre_en_bd, email_en_bos_csv, nombre_en_qbos_excel, email_dh
    (operador_id del Excel se ignora — se busca por nombre_en_bd contra la BD)
    """
    import pandas as pd

    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos Excel")

    content = await file.read()
    df = pd.read_excel(BytesIO(content))

    # Validar columnas mínimas
    if 'nombre_en_bd' not in df.columns:
        raise HTTPException(status_code=400, detail="Falta columna 'nombre_en_bd'")

    sb = get_supabase_admin()

    # Cargar TODOS los empleados de esta cédula: operadores + LCs + LS
    ops = sb.table("operadores").select("id, nombre").eq("cedula_id", cedula_id).eq("activo", True).execute()
    lcs = sb.table("line_coordinators").select("id, nombre").eq("cedula_id", cedula_id).eq("activo", True).execute()
    lss = sb.table("linea_estructura").select("id, nombre").eq("cedula_id", cedula_id).eq("activo", True).execute()

    # Map nombre normalizado → { id, nombre, tipo }
    nombre_map = {}
    for op in (ops.data or []):
        nombre_map[normalize(op["nombre"])] = {**op, "tipo": "operador"}
    for lc in (lcs.data or []):
        nombre_map[normalize(lc["nombre"])] = {**lc, "tipo": "lc"}
    for ls in (lss.data or []):
        nombre_map[normalize(ls["nombre"])] = {**ls, "tipo": "ls"}

    matched = 0
    skipped = []
    aliases_to_upsert = []

    for _, row in df.iterrows():
        nombre_bd = str(row.get("nombre_en_bd", "")).strip()
        if not nombre_bd or nombre_bd == "nan":
            continue

        # Buscar en TODAS las tablas por nombre normalizado
        persona = nombre_map.get(normalize(nombre_bd))
        if not persona:
            skipped.append(nombre_bd)
            continue

        email_bos = str(row.get("email_en_bos_csv", "")).strip() if pd.notna(row.get("email_en_bos_csv")) else None
        nombre_qbos = str(row.get("nombre_en_qbos_excel", "")).strip() if pd.notna(row.get("nombre_en_qbos_excel")) else None
        email_dh = str(row.get("email_dh", "")).strip() if pd.notna(row.get("email_dh")) else None

        aliases_to_upsert.append({
            "persona_id": persona["id"],
            "persona_tipo": persona["tipo"],
            "nombre_bd": persona["nombre"],
            "email_bos": email_bos,
            "nombre_qbos": nombre_qbos,
            "email_dh": email_dh,
        })
        matched += 1

    # Upsert all aliases
    if aliases_to_upsert:
        sb.table("operador_aliases") \
            .upsert(aliases_to_upsert, on_conflict="persona_id") \
            .execute()

    return {
        "success": True,
        "matched": matched,
        "skipped": skipped,
        "message": f"{matched} aliases actualizados. {len(skipped)} nombres no encontrados en BD.",
    }


def _load_aliases(sb, cedula_id: str) -> dict:
    """
    Carga aliases para TODOS los empleados de una cédula (operadores, LCs, LS).
    Retorna dict con mapas por tipo de alias.
    Cada mapa: clave → { persona_id, persona_tipo }
    """
    # Obtener IDs de todas las tablas de esta cédula
    all_ids = []
    for table in ["operadores", "line_coordinators", "linea_estructura"]:
        res = sb.table(table).select("id").eq("cedula_id", cedula_id).eq("activo", True).execute()
        all_ids.extend([r["id"] for r in (res.data or [])])

    if not all_ids:
        return {"email_bos": {}, "nombre_qbos": {}, "email_dh": {}}

    # Cargar aliases
    aliases = sb.table("operador_aliases") \
        .select("persona_id, persona_tipo, email_bos, nombre_qbos, email_dh") \
        .in_("persona_id", all_ids) \
        .execute()

    email_bos_map = {}    # email_bos (lower) → { persona_id, persona_tipo }
    nombre_qbos_map = {}  # nombre_qbos (normalized) → { persona_id, persona_tipo }
    email_dh_map = {}     # email_dh (lower) → { persona_id, persona_tipo }

    for a in (aliases.data or []):
        info = {"persona_id": a["persona_id"], "persona_tipo": a.get("persona_tipo", "operador")}
        if a.get("email_bos"):
            email_bos_map[a["email_bos"].lower().strip()] = info
        if a.get("nombre_qbos"):
            nombre_qbos_map[normalize(a["nombre_qbos"])] = info
        if a.get("email_dh"):
            email_dh_map[a["email_dh"].lower().strip()] = info

    return {
        "email_bos": email_bos_map,
        "nombre_qbos": nombre_qbos_map,
        "email_dh": email_dh_map,
    }


# ══════════════════════════════════════════════════════════════
# BOS — Upload CSV
# ══════════════════════════════════════════════════════════════

@router.post("/bos/upload")
async def upload_bos(
    file: UploadFile = File(...),
    cedula_id: str = Query(...),
    semana: date = Query(..., description="Lunes de la semana (YYYY-MM-DD)")
):
    """Procesa CSV de BOS. Match por email: primero aliases, luego operadores.email."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos CSV para BOS")

    content = await file.read()
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(StringIO(text))
    rows = list(reader)

    if not rows:
        raise HTTPException(status_code=400, detail="El archivo CSV está vacío")

    required = {'USER', 'BOS_CREATED'}
    actual = set(rows[0].keys())
    missing = required - actual
    if missing:
        raise HTTPException(status_code=400, detail=f"Faltan columnas: {missing}")

    sb = get_supabase_admin()

    # 1. Cargar aliases (match primario) — incluye operadores, LCs y LS
    aliases = _load_aliases(sb, cedula_id)
    alias_email_map = aliases["email_bos"]  # email → { persona_id, persona_tipo }

    # 2. Cargar operadores (fallback por email principal)
    ops_result = sb.table("operadores").select("id, nombre, email").eq("cedula_id", cedula_id).eq("activo", True).execute()
    operadores = ops_result.data or []

    fallback_email_map = {}
    for op in operadores:
        if op.get("email"):
            fallback_email_map[op["email"].lower().strip()] = {"persona_id": op["id"], "persona_tipo": "operador"}

    # 3. Map de id → nombre para TODAS las personas
    id_to_name = {}
    id_to_tipo = {}
    for op in operadores:
        id_to_name[op["id"]] = op["nombre"]
        id_to_tipo[op["id"]] = "operador"
    for table, tipo in [("line_coordinators", "lc"), ("linea_estructura", "ls")]:
        res = sb.table(table).select("id, nombre").eq("cedula_id", cedula_id).eq("activo", True).execute()
        for r in (res.data or []):
            id_to_name[r["id"]] = r["nombre"]
            id_to_tipo[r["id"]] = tipo

    # 4. Registros existentes para dedup (solo operadores)
    existing = sb.table("registros_semanales") \
        .select("id, operador_id, bos_num") \
        .eq("cedula_id", cedula_id) \
        .eq("semana", semana.isoformat()) \
        .execute()
    existing_map = {r["operador_id"]: r for r in (existing.data or [])}

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

        # Match: 1° alias, 2° fallback email
        match_info = alias_email_map.get(email) or fallback_email_map.get(email)

        if not match_info:
            unmatched.append(email)
            continue

        persona_id = match_info["persona_id"]
        persona_tipo = match_info["persona_tipo"]
        op_name = id_to_name.get(persona_id, email)

        # Dedup solo aplica a operadores (que tienen registros_semanales)
        if persona_tipo == "operador":
            existing_rec = existing_map.get(persona_id)
            if existing_rec:
                old_val = existing_rec.get("bos_num") or 0
                action = "actualizado" if old_val != bos_val else "sin_cambio"
                if action == "actualizado":
                    updated_count += 1
                else:
                    unchanged_count += 1
            else:
                action = "nuevo"
                new_count += 1
        else:
            action = "info"  # LC/LS: solo informativo, no se guarda en registros_semanales

        results.append({
            "operador_id": persona_id,
            "nombre": op_name,
            "email": email,
            "bos": bos_val,
            "persona_tipo": persona_tipo,
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
    """Guarda los resultados de BOS en registros_semanales (bos_num + bos_eng). Solo operadores."""
    sb = get_supabase_admin()
    saved = 0
    for item in body:
        operador_id = item.get("operador_id")
        bos_val = item.get("bos", 0)
        # Solo guardar operadores (LC/LS son informativos)
        if not operador_id or item.get("persona_tipo") in ("lc", "ls"):
            continue

        # bos_eng = min(100, bos_num / target * 100), target BOS = 3
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
    """Procesa Excel de QBOS. Match por nombre: primero aliases, luego fuzzy."""
    import pandas as pd

    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos Excel para QBOS")

    content = await file.read()
    df = pd.read_excel(BytesIO(content))

    if 'Personnel Name' not in df.columns:
        raise HTTPException(status_code=400, detail="Falta columna 'Personnel Name'")
    if 'Frecuency' not in df.columns:
        raise HTTPException(status_code=400, detail="Falta columna 'Frecuency'")

    sb = get_supabase_admin()

    # 1. Cargar aliases (match primario) — incluye operadores, LCs y LS
    aliases = _load_aliases(sb, cedula_id)
    alias_nombre_map = aliases["nombre_qbos"]  # nombre → { persona_id, persona_tipo }

    # 2. Cargar operadores (fallback fuzzy)
    ops_result = sb.table("operadores").select("id, nombre, email").eq("cedula_id", cedula_id).eq("activo", True).execute()
    operadores = ops_result.data or []

    # 3. Map id → nombre para TODAS las personas
    id_to_name = {}
    for op in operadores:
        id_to_name[op["id"]] = op["nombre"]
    for table in ["line_coordinators", "linea_estructura"]:
        res = sb.table(table).select("id, nombre").eq("cedula_id", cedula_id).eq("activo", True).execute()
        for r in (res.data or []):
            id_to_name[r["id"]] = r["nombre"]

    # 4. Registros existentes para dedup (solo operadores)
    existing = sb.table("registros_semanales") \
        .select("id, operador_id, qbos_num") \
        .eq("cedula_id", cedula_id) \
        .eq("semana", semana.isoformat()) \
        .execute()
    existing_map = {r["operador_id"]: r for r in (existing.data or [])}

    results = []
    unmatched = []
    new_count = 0
    updated_count = 0
    unchanged_count = 0

    for _, row in df.iterrows():
        name = str(row.get("Personnel Name", "")).strip()
        if not name or name == "nan":
            continue

        # Skip metadata rows
        if any(name.startswith(prefix) for prefix in ("Applied filters", "AUTHOR", "Department", "DATE")):
            continue

        freq = row.get("Frecuency")
        qbos_val = int(freq) if pd.notna(freq) else 0

        # Match: 1° alias exacto, 2° fuzzy (solo operadores)
        alias_info = alias_nombre_map.get(normalize(name))

        if alias_info:
            # Alias match — confianza 100%
            persona_id = alias_info["persona_id"]
            persona_tipo = alias_info["persona_tipo"]
            op_name = id_to_name.get(persona_id, name)
            match_score = 100
        else:
            # Fuzzy fallback (solo contra operadores)
            matches = fuzzy_name_match(name, operadores)
            if matches and matches[0][1] >= 60:
                best_match, match_score = matches[0]
                persona_id = best_match["id"]
                persona_tipo = "operador"
                op_name = best_match["nombre"]
            else:
                suggestions = [{"id": m[0]["id"], "nombre": m[0]["nombre"], "score": m[1]} for m in (matches or [])[:3]]
                unmatched.append({"name": name, "qbos": qbos_val, "suggestions": suggestions})
                continue

        # Dedup solo aplica a operadores
        if persona_tipo == "operador":
            existing_rec = existing_map.get(persona_id)
            if existing_rec:
                old_val = existing_rec.get("qbos_num") or 0
                action = "actualizado" if old_val != qbos_val else "sin_cambio"
                if action == "actualizado":
                    updated_count += 1
                else:
                    unchanged_count += 1
            else:
                action = "nuevo"
                new_count += 1
        else:
            action = "info"  # LC/LS: solo informativo

        results.append({
            "operador_id": persona_id,
            "nombre": op_name,
            "file_name": name,
            "qbos": qbos_val,
            "match_score": match_score,
            "persona_tipo": persona_tipo,
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
    """Guarda los resultados de QBOS en registros_semanales (qbos_num + qbos_eng). Solo operadores."""
    sb = get_supabase_admin()
    saved = 0
    for item in body:
        operador_id = item.get("operador_id")
        qbos_val = item.get("qbos", 0)
        # Solo guardar operadores (LC/LS son informativos)
        if not operador_id or item.get("persona_tipo") in ("lc", "ls"):
            continue

        # qbos_eng = min(100, qbos_num / target * 100), target QBOS = 1
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
