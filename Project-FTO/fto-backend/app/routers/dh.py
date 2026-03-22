"""
Router para procesamiento de archivos DH (Defect Handling).
Recibe un CSV exportado del sistema de defectos, hace match de emails
con operadores registrados y cuenta:
  - DH Encontrados [#] (REPORTED BY)
  - DH Reparados [#] (CLOSED BY)
  - Curva Aut [%] = min(100, (Reparados / Encontrados) × 100)
  - Contram [%] = (defectos cerrados por ese operador CON contramedida válida / total reparados) × 100
                  (NA, N/A y vacío no cuentan como contramedida válida)
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
    Calcula por operador y semana:
      - dh_encontrados: # defectos reportados (REPORTED BY)
      - dh_reparados: # defectos cerrados (CLOSED BY + CLOSED AT)
      - curva_autonomia: min(100, (reparados / encontrados) × 100)
      - contramedidas_defectos: (defectos con DEFECT COUNTERMEASURES / encontrados) × 100
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

    has_countermeasures_col = 'DEFECT COUNTERMEASURES' in csv_cols

    # Deduplicar por NUMBER
    seen_numbers_enc = set()
    seen_numbers_rep = set()

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
    # conteos[(op_id, semana)] = { encontrados, reparados, reparados_con_cm, reparados_total_cm }
    # Contramedidas se mide sobre los defectos que CERRÓ el operador:
    #   con_contramedida = cuántos de los que cerró tienen CM válida (no vacía, no NA/N/A)
    #   reparados_con_cm_base = total reparados por ese operador (denominador de Contram)
    conteos = defaultdict(lambda: {
        "encontrados": 0,
        "reparados": 0,
        "con_contramedida": 0,   # reparados con CM válida
    })
    unmatched_emails = set()

    # Valores de DEFECT COUNTERMEASURES que NO cuentan como contramedida
    NA_VALUES = {"na", "n/a", "n/a.", "n.a", "n.a.", "-", ""}

    for row in rows:
        status = (row.get("STATUS") or "").strip().upper()
        defect_number = (row.get("NUMBER") or "").strip()

        if status == "DELETED":
            stats["excluded_deleted"] += 1
            continue

        # ── DH Encontrados: REPORTED BY + REPORTED AT ──
        reported_email = (row.get("REPORTED BY") or "").strip().lower()
        reported_date  = parse_date(row.get("REPORTED AT", ""))

        if reported_email and reported_date:
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

        # ── DH Reparados: CLOSED BY + CLOSED AT ──
        closed_email = (row.get("CLOSED BY") or "").strip().lower()
        closed_date  = parse_date(row.get("CLOSED AT", ""))

        if closed_email and closed_date:
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

                    # Contramedida: se evalúa sobre quien CERRÓ el defecto
                    # Solo cuenta si tiene contenido real (no NA / N/A / vacío)
                    if has_countermeasures_col:
                        cm = (row.get("DEFECT COUNTERMEASURES") or "").strip().lower()
                        if cm not in NA_VALUES:
                            conteos[(op_id, week)]["con_contramedida"] += 1
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

    # ── Guardar en BD ───────────────────────────────────────────────────────────
    op_names = {op["id"]: op["nombre"] for op in operadores}
    details = []
    new_count = 0
    updated_count = 0
    unchanged_count = 0

    for (op_id, week), counts in sorted(conteos.items(), key=lambda x: (x[0][1], x[0][0])):
        new_enc = counts["encontrados"]
        new_rep = counts["reparados"]
        con_cm  = counts["con_contramedida"]

        # Curva Autonomía = min(100, (reparados / encontrados) × 100)
        if new_enc > 0:
            curva_aut = min(100.0, round((new_rep / new_enc) * 100, 1))
        else:
            curva_aut = 0.0

        # Contramedidas = (defectos reparados CON contramedida válida / total reparados) × 100
        # Denominador: reparados (quien cerró el defecto es responsable de la contramedida)
        if new_rep > 0:
            contram = round((con_cm / new_rep) * 100, 1)
        else:
            contram = 0.0

        # Campos a guardar
        dh_fields = {
            "dh_encontrados": new_enc,
            "dh_reparados": new_rep,
            "curva_autonomia": curva_aut,
            "contramedidas_defectos": contram,
        }

        existing = sb.table("registros_semanales") \
            .select("id, dh_encontrados, dh_reparados, curva_autonomia, contramedidas_defectos") \
            .eq("operador_id", op_id) \
            .eq("semana", week) \
            .execute()

        if existing.data:
            prev = existing.data[0]
            prev_enc = prev.get("dh_encontrados") or 0
            prev_rep = prev.get("dh_reparados") or 0
            prev_curva = prev.get("curva_autonomia") or 0
            prev_contram = prev.get("contramedidas_defectos") or 0

            # Comparar todos los campos DH
            same = (
                prev_enc == new_enc
                and prev_rep == new_rep
                and round(float(prev_curva), 1) == curva_aut
                and round(float(prev_contram), 1) == contram
            )

            if same:
                action = "sin_cambio"
                unchanged_count += 1
            else:
                sb.table("registros_semanales") \
                    .update(dh_fields) \
                    .eq("id", prev["id"]) \
                    .execute()
                action = "actualizado"
                updated_count += 1
        else:
            sb.table("registros_semanales") \
                .insert({
                    "operador_id": op_id,
                    "semana": week,
                    "cedula_id": cedula_id,
                    **dh_fields,
                }) \
                .execute()
            action = "nuevo"
            new_count += 1

        details.append({
            "operador": op_names.get(op_id, "Desconocido"),
            "semana": week,
            "dh_encontrados": new_enc,
            "dh_reparados": new_rep,
            "curva_autonomia": curva_aut,
            "contramedidas_defectos": contram,
            "accion": action,
        })

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
