"""
Router para procesamiento de archivos DH (Defect Handling).
Recibe un CSV exportado del sistema de defectos, hace match de emails
con operadores registrados y cuenta DH Encontrados / DH Reparados
por operador y semana.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from io import StringIO
from datetime import datetime, date, timedelta
from collections import defaultdict
import csv

from app.services.supabase_client import get_supabase_admin

router = APIRouter(prefix="/registros/dh", tags=["DH — Defect Handling"])


def get_week_monday(d: date) -> str:
    """Retorna el lunes de la semana ISO de una fecha dada (formato YYYY-MM-DD)."""
    monday = d - timedelta(days=d.weekday())
    return monday.isoformat()


def parse_date(date_str: str) -> date | None:
    """Parsea fechas en formato DD/MM/YYYY HH:MM o similar."""
    if not date_str or not date_str.strip():
        return None
    date_str = date_str.strip()
    for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


@router.post("/upload")
async def upload_dh_csv(
    file: UploadFile = File(...),
    cedula_id: str = Query(..., description="ID de la cédula")
):
    """
    Procesa un CSV de defectos DH.
    - Deduplica por columna NUMBER (ID único del defecto)
    - Excluye registros con STATUS = DELETED
    - Hace match de REPORTED BY / CLOSED BY (emails) con operadores
    - Cuenta DH Encontrados y DH Reparados por operador y semana
    - Actualiza solo los campos DH sin sobrescribir otros indicadores
    - Muestra claramente qué registros son nuevos vs actualizados
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos CSV")

    # Leer y decodificar CSV
    content = await file.read()
    try:
        text = content.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = content.decode('latin-1')

    reader = csv.DictReader(StringIO(text))
    rows = list(reader)

    if not rows:
        raise HTTPException(status_code=400, detail="El archivo CSV está vacío")

    # Validar columnas requeridas
    required_cols = {'STATUS', 'REPORTED BY', 'REPORTED AT', 'CLOSED BY', 'CLOSED AT'}
    csv_cols = set(rows[0].keys())
    missing = required_cols - csv_cols
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Faltan columnas requeridas en el CSV: {', '.join(missing)}"
        )

    # ── Deduplicar por NUMBER (ID único del defecto en el sistema DH) ──────────
    # Si el CSV contiene el mismo defecto dos veces (ej. exportación duplicada),
    # lo contamos una sola vez usando su NUMBER como clave única.
    seen_numbers_enc = set()   # para REPORTED (encontrados)
    seen_numbers_rep = set()   # para CLOSED (reparados)

    stats = {
        "total_rows": len(rows),
        "excluded_deleted": 0,
        "excluded_duplicate": 0,
        "matched_encontrados": 0,
        "matched_reparados": 0,
        "unmatched_encontrados": 0,
        "unmatched_reparados": 0,
    }

    # Cargar operadores con emails
    sb = get_supabase_admin()
    ops_result = sb.table("operadores") \
        .select("id, nombre, email") \
        .eq("cedula_id", cedula_id) \
        .eq("activo", True) \
        .execute()

    operadores = ops_result.data or []
    email_map = {}
    for op in operadores:
        if op.get("email"):
            email_map[op["email"].lower().strip()] = op["id"]

    # ── Procesar filas ──────────────────────────────────────────────────────────
    # conteos[( operador_id, semana )] = { encontrados: int, reparados: int }
    conteos = defaultdict(lambda: {"encontrados": 0, "reparados": 0})
    unmatched_emails = set()

    for row in rows:
        status = (row.get("STATUS") or "").strip().upper()
        defect_number = (row.get("NUMBER") or "").strip()

        # Excluir DELETED
        if status == "DELETED":
            stats["excluded_deleted"] += 1
            continue

        # DH Encontrados: REPORTED BY + REPORTED AT
        reported_email = (row.get("REPORTED BY") or "").strip().lower()
        reported_date  = parse_date(row.get("REPORTED AT", ""))

        if reported_email and reported_date:
            # Deduplicar: mismo NUMBER ya procesado como "encontrado"
            enc_key = defect_number or f"{reported_email}_{reported_date}"
            if enc_key in seen_numbers_enc:
                stats["excluded_duplicate"] += 1
            else:
                seen_numbers_enc.add(enc_key)
                op_id = email_map.get(reported_email)
                if op_id:
                    week = get_week_monday(reported_date)
                    conteos[(op_id, week)]["encontrados"] += 1
                    stats["matched_encontrados"] += 1
                else:
                    unmatched_emails.add(reported_email)
                    stats["unmatched_encontrados"] += 1

        # DH Reparados: CLOSED BY + CLOSED AT
        closed_email = (row.get("CLOSED BY") or "").strip().lower()
        closed_date  = parse_date(row.get("CLOSED AT", ""))

        if closed_email and closed_date:
            # Deduplicar: mismo NUMBER ya procesado como "reparado"
            rep_key = f"rep_{defect_number}" if defect_number else f"{closed_email}_{closed_date}"
            if rep_key in seen_numbers_rep:
                stats["excluded_duplicate"] += 1
            else:
                seen_numbers_rep.add(rep_key)
                op_id = email_map.get(closed_email)
                if op_id:
                    week = get_week_monday(closed_date)
                    conteos[(op_id, week)]["reparados"] += 1
                    stats["matched_reparados"] += 1
                else:
                    unmatched_emails.add(closed_email)
                    stats["unmatched_reparados"] += 1

    if not conteos:
        return {
            "success": True,
            "message": "No se encontraron coincidencias de emails con operadores registrados.",
            "stats": stats,
            "unmatched_emails": sorted(unmatched_emails),
            "new_count": 0,
            "updated_count": 0,
            "unchanged_count": 0,
            "details": []
        }

    # ── Guardar en BD (solo campos DH, sin tocar otros indicadores) ─────────────
    op_names = {op["id"]: op["nombre"] for op in operadores}
    details = []
    new_count = 0
    updated_count = 0
    unchanged_count = 0

    for (op_id, week), counts in sorted(conteos.items(), key=lambda x: (x[0][1], x[0][0])):
        existing = sb.table("registros_semanales") \
            .select("id, dh_encontrados, dh_reparados") \
            .eq("operador_id", op_id) \
            .eq("semana", week) \
            .execute()

        new_enc = counts["encontrados"]
        new_rep = counts["reparados"]

        if existing.data:
            prev_enc = existing.data[0].get("dh_encontrados") or 0
            prev_rep = existing.data[0].get("dh_reparados") or 0

            if prev_enc == new_enc and prev_rep == new_rep:
                # Sin cambios — no tocar la BD
                action = "sin_cambio"
                unchanged_count += 1
            else:
                # Valores diferentes — actualizar solo campos DH
                sb.table("registros_semanales") \
                    .update({
                        "dh_encontrados": new_enc,
                        "dh_reparados": new_rep
                    }) \
                    .eq("id", existing.data[0]["id"]) \
                    .execute()
                action = "actualizado"
                updated_count += 1
        else:
            # Registro nuevo
            sb.table("registros_semanales") \
                .insert({
                    "operador_id": op_id,
                    "semana": week,
                    "cedula_id": cedula_id,
                    "dh_encontrados": new_enc,
                    "dh_reparados": new_rep,
                }) \
                .execute()
            action = "nuevo"
            new_count += 1

        details.append({
            "operador": op_names.get(op_id, "Desconocido"),
            "semana": week,
            "dh_encontrados": new_enc,
            "dh_reparados": new_rep,
            "accion": action,  # "nuevo" | "actualizado" | "sin_cambio"
        })

    total_saved = new_count + updated_count
    message = (
        f"CSV procesado: {stats['total_rows']} filas totales, "
        f"{stats['excluded_deleted']} eliminadas (DELETED), "
        f"{stats['excluded_duplicate']} duplicadas ignoradas. "
        f"Resultado: {new_count} nuevo(s), {updated_count} actualizado(s), {unchanged_count} sin cambio."
    )

    return {
        "success": True,
        "message": message,
        "stats": stats,
        "unmatched_emails": sorted(unmatched_emails),
        "new_count": new_count,
        "updated_count": updated_count,
        "unchanged_count": unchanged_count,
        "details": details,
    }


@router.get("/preview-operators")
async def preview_operator_emails(cedula_id: str = Query(...)):
    """Devuelve la lista de operadores con sus emails configurados."""
    sb = get_supabase_admin()
    result = sb.table("operadores") \
        .select("id, nombre, email, line_coordinators(nombre)") \
        .eq("cedula_id", cedula_id) \
        .eq("activo", True) \
        .order("nombre") \
        .execute()

    return {
        "data": result.data or [],
        "total": len(result.data or []),
        "with_email": sum(1 for op in (result.data or []) if op.get("email")),
    }
