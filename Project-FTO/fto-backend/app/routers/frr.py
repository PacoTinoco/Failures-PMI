"""
Router para FRR (Filter Reject Rate) y calendario de turnos ROL.

Flujo:
1. Subir ROL Excel (calendario anual de turnos/máquinas) → parsea fechas y guarda en rol_calendario
2. Subir FRR Excel (semanal) → cruza con rol_calendario para calcular FRR por operador
3. Guardar → escribe frr en registros_semanales

Mapeos:
- Turno: S1↔D (día), S2↔T (tarde), S3↔N (noche)
- Máquina: KDF07 → Filter 07, KDF08 → Filter 08, etc.
- KDF07/KDF09 → promedio de Filter 07 + Filter 09
- AUX, LC, Tecnico → se excluyen del cálculo FRR
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Body
from typing import Optional
from datetime import date, datetime, timedelta
from io import BytesIO
from collections import defaultdict
import re

from app.services.supabase_client import get_supabase_admin

router = APIRouter(prefix="/frr", tags=["FRR"])

# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

SHIFT_MAP_FRR_TO_ROL = {"S1": "D", "S2": "T", "S3": "N"}
SHIFT_MAP_ROL_TO_FRR = {"D": "S1", "T": "S2", "N": "S3"}

MONTH_MAP_ES = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12,
}


def parse_date_header(header: str, base_year: int = 2026) -> Optional[date]:
    """
    Parsea headers de fecha del Excel ROL.
    Formatos: '10-Feb', '09-mar', '1-Mar', etc.
    Retorna date o None si no es fecha.
    """
    if not header or not isinstance(header, str):
        return None

    header = header.strip()

    # Skip non-date columns
    if header.lower() in ('num. personal', 'nombre', 'lc', 'kdf', 'tecnología', 'tecnologia'):
        return None

    # Try format: day-month (e.g., "10-Feb", "09-mar", "1-Mar")
    match = re.match(r'^(\d{1,2})[- ]([A-Za-záéíóúü]+)$', header)
    if match:
        day = int(match.group(1))
        month_str = match.group(2).lower()[:3]
        month = MONTH_MAP_ES.get(month_str)
        if month is None:
            # Try English month abbreviations
            eng_months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                          "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
            month = eng_months.get(month_str)
        if month:
            # Handle year rollover (Jan = next year if calendar starts in Feb)
            year = base_year + 1 if month == 1 else base_year
            try:
                return date(year, month, day)
            except ValueError:
                return None
    return None


def kdf_to_filter_numbers(kdf: str) -> list:
    """
    Convierte KDF a números de máquina Filter.
    'KDF07' → [7], 'KDF07/KDF09' → [7, 9], 'KDF17' → [17]
    'AUX', 'LC', 'Tecnico' → []
    """
    if not kdf:
        return []
    kdf_upper = kdf.upper().strip()
    if kdf_upper in ('AUX', 'LC', 'TECNICO', 'TÉCNICO'):
        return []

    numbers = []
    for part in kdf_upper.split('/'):
        match = re.search(r'KDF(\d+)', part)
        if match:
            numbers.append(int(match.group(1)))
    return numbers


def filter_line_to_number(line_name: str) -> Optional[int]:
    """
    Extrae número de máquina de LINE_NAME del FRR.
    'Filter 07-MS14' → 7, 'Filter 17-MS14' → 17
    """
    if not line_name:
        return None
    match = re.search(r'Filter\s+(\d+)', line_name)
    if match:
        return int(match.group(1))
    return None


def parse_frr_date(cal_shift: str) -> tuple:
    """
    Parsea CAL_SHIFT del FRR.
    'S1 17-03-2026' → ('S1', date(2026, 3, 17))
    """
    if not cal_shift:
        return None, None
    parts = cal_shift.strip().split(' ', 1)
    if len(parts) != 2:
        return None, None
    shift = parts[0]  # S1, S2, S3
    date_str = parts[1]
    try:
        d = datetime.strptime(date_str, "%d-%m-%Y").date()
        return shift, d
    except ValueError:
        return None, None


def normalize_name(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.lower().strip().split())


# ══════════════════════════════════════════════════════════════
# ROL Calendar — Upload & CRUD
# ══════════════════════════════════════════════════════════════

@router.post("/rol/upload")
async def upload_rol(
    file: UploadFile = File(...),
    cedula_id: str = Query(...),
    base_year: int = Query(2026, description="Año base del calendario")
):
    """
    Sube Excel de ROL (calendario de turnos anual).
    Parsea fechas de los headers, match nombres con aliases/operadores,
    y guarda en rol_calendario. No sobreescribe overrides manuales.
    """
    import pandas as pd

    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos Excel")

    content = await file.read()
    wb_data = pd.read_excel(BytesIO(content), header=None)

    # Row 0 = headers, rows 1+ = data
    headers = [str(c) if c is not None else "" for c in wb_data.iloc[0].tolist()]

    # Parse date columns
    date_columns = {}  # col_index → date
    for i, h in enumerate(headers):
        if i < 4:  # Skip Num.Personal, Nombre, LC, KDF
            continue
        d = parse_date_header(h, base_year)
        if d:
            date_columns[i] = d

    if not date_columns:
        raise HTTPException(status_code=400, detail="No se encontraron fechas válidas en los headers")

    sb = get_supabase_admin()

    # Load aliases for matching
    from app.routers.bos_qbos import _load_aliases, normalize
    aliases = _load_aliases(sb, cedula_id)

    # Load all people for name-based matching
    all_people = {}  # normalized name → {id, tipo}
    for table, tipo in [("operadores", "operador"), ("line_coordinators", "lc"), ("linea_estructura", "ls")]:
        res = sb.table(table).select("id, nombre").eq("cedula_id", cedula_id).eq("activo", True).execute()
        for r in (res.data or []):
            all_people[normalize(r["nombre"])] = {"id": r["id"], "tipo": tipo, "nombre": r["nombre"]}

    # Get existing overrides to preserve them
    all_dates = list(date_columns.values())
    min_date = min(all_dates).isoformat()
    max_date = max(all_dates).isoformat()

    overrides = sb.table("rol_calendario") \
        .select("operador_id, fecha") \
        .eq("cedula_id", cedula_id) \
        .eq("es_override", True) \
        .gte("fecha", min_date) \
        .lte("fecha", max_date) \
        .execute()
    override_set = set()
    for o in (overrides.data or []):
        override_set.add((o["operador_id"], o["fecha"]))

    records_to_upsert = []
    matched_count = 0
    skipped_names = []

    for row_idx in range(1, len(wb_data)):
        row = wb_data.iloc[row_idx].tolist()
        nombre = str(row[1]).strip() if pd.notna(row[1]) else ""
        kdf = str(row[3]).strip() if pd.notna(row[3]) else ""

        if not nombre or nombre == "nan":
            continue

        # Match person: first try alias nombre_qbos (same name format), then direct name
        norm_name = normalize(nombre)
        person = all_people.get(norm_name)

        if not person:
            # Try aliases
            for alias_name, info in aliases.get("nombre_qbos", {}).items():
                if normalize_name(alias_name) == norm_name:
                    person = {"id": info["persona_id"], "tipo": info["persona_tipo"], "nombre": nombre}
                    break

        if not person:
            skipped_names.append(nombre)
            continue

        matched_count += 1

        for col_idx, fecha in date_columns.items():
            # Skip if there's a manual override for this person+date
            if (person["id"], fecha.isoformat()) in override_set:
                continue

            turno_val = row[col_idx] if col_idx < len(row) else None
            turno = str(turno_val).strip().upper() if pd.notna(turno_val) else None

            # Validate turno value
            if turno and turno not in ('D', 'T', 'N'):
                turno = None  # Not a valid shift, treat as day off

            records_to_upsert.append({
                "operador_id": person["id"],
                "cedula_id": cedula_id,
                "nombre": person["nombre"],
                "kdf": kdf,
                "fecha": fecha.isoformat(),
                "turno": turno,
                "es_override": False,
            })

    # Batch upsert (in chunks to avoid payload limits)
    saved = 0
    chunk_size = 500
    for i in range(0, len(records_to_upsert), chunk_size):
        chunk = records_to_upsert[i:i + chunk_size]
        sb.table("rol_calendario") \
            .upsert(chunk, on_conflict="operador_id,fecha") \
            .execute()
        saved += len(chunk)

    return {
        "success": True,
        "matched_operators": matched_count,
        "skipped_names": skipped_names,
        "total_records": saved,
        "date_range": f"{min(all_dates).isoformat()} a {max(all_dates).isoformat()}",
        "dates_found": len(date_columns),
    }


@router.get("/rol/semana")
async def get_rol_semana(
    cedula_id: str = Query(...),
    semana: date = Query(..., description="Lunes de la semana"),
):
    """Obtiene el calendario de turnos para una semana (lun-dom)."""
    sb = get_supabase_admin()
    fin_semana = semana + timedelta(days=6)

    result = sb.table("rol_calendario") \
        .select("*") \
        .eq("cedula_id", cedula_id) \
        .gte("fecha", semana.isoformat()) \
        .lte("fecha", fin_semana.isoformat()) \
        .order("nombre") \
        .order("fecha") \
        .execute()

    return {"data": result.data or [], "semana": semana.isoformat()}


@router.put("/rol/override")
async def override_turno(
    cedula_id: str = Query(...),
    operador_id: str = Query(...),
    fecha: date = Query(...),
    turno: Optional[str] = Query(None, description="D, T, N, o null para descanso"),
    kdf: Optional[str] = Query(None, description="Máquina, ej: KDF07"),
):
    """Modifica manualmente un turno/máquina para un día específico."""
    sb = get_supabase_admin()

    # Get existing record to preserve kdf if not changing it
    existing = sb.table("rol_calendario") \
        .select("*") \
        .eq("operador_id", operador_id) \
        .eq("fecha", fecha.isoformat()) \
        .execute()

    if existing.data:
        update = {"turno": turno, "es_override": True}
        if kdf is not None:
            update["kdf"] = kdf
        sb.table("rol_calendario") \
            .update(update) \
            .eq("operador_id", operador_id) \
            .eq("fecha", fecha.isoformat()) \
            .execute()
    else:
        # Need nombre
        ops = sb.table("operadores").select("nombre").eq("id", operador_id).execute()
        nombre = ops.data[0]["nombre"] if ops.data else "Desconocido"
        sb.table("rol_calendario").insert({
            "operador_id": operador_id,
            "cedula_id": cedula_id,
            "nombre": nombre,
            "kdf": kdf or "",
            "fecha": fecha.isoformat(),
            "turno": turno,
            "es_override": True,
        }).execute()

    return {"success": True, "message": f"Turno actualizado para {fecha}"}


# ══════════════════════════════════════════════════════════════
# FRR — Upload & Process
# ══════════════════════════════════════════════════════════════

@router.post("/upload")
async def upload_frr(
    file: UploadFile = File(...),
    cedula_id: str = Query(...),
    semana: date = Query(..., description="Lunes de la semana"),
):
    """
    Sube Excel semanal de FRR. Cruza con rol_calendario para calcular
    el FRR promedio por operador en esa semana.
    """
    import pandas as pd

    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos Excel")

    content = await file.read()
    df = pd.read_excel(BytesIO(content))

    required = {'CAL_SHIFT', 'Reject Rate', 'LINE_NAME'}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise HTTPException(status_code=400, detail=f"Faltan columnas: {missing}")

    sb = get_supabase_admin()

    # 1. Parse FRR data into: (date, shift_letter, machine_num) → reject_rate
    frr_data = {}  # (date_iso, turno_letter, machine_num) → [reject_rates]
    for _, row in df.iterrows():
        shift_code, frr_date = parse_frr_date(str(row.get("CAL_SHIFT", "")))
        if not shift_code or not frr_date:
            continue

        turno_letter = SHIFT_MAP_FRR_TO_ROL.get(shift_code)
        if not turno_letter:
            continue

        machine_num = filter_line_to_number(str(row.get("LINE_NAME", "")))
        if machine_num is None:
            continue

        reject_rate = row.get("Reject Rate")
        if pd.isna(reject_rate):
            reject_rate = 0.0
        else:
            reject_rate = float(reject_rate)

        key = (frr_date.isoformat(), turno_letter, machine_num)
        if key not in frr_data:
            frr_data[key] = []
        frr_data[key].append(reject_rate)

    # Average per (date, shift, machine) if multiple entries
    frr_avg = {}
    for key, rates in frr_data.items():
        frr_avg[key] = sum(rates) / len(rates)

    # 2. Load ROL calendar for this week
    fin_semana = semana + timedelta(days=6)
    rol_result = sb.table("rol_calendario") \
        .select("operador_id, nombre, kdf, fecha, turno") \
        .eq("cedula_id", cedula_id) \
        .gte("fecha", semana.isoformat()) \
        .lte("fecha", fin_semana.isoformat()) \
        .execute()

    if not rol_result.data:
        raise HTTPException(
            status_code=400,
            detail="No hay calendario de turnos cargado para esta semana. Sube primero el ROL."
        )

    # 3. Group ROL by operador
    operador_rol = defaultdict(list)  # operador_id → [{fecha, turno, kdf}]
    operador_names = {}
    operador_kdfs = {}

    for r in rol_result.data:
        oid = r["operador_id"]
        operador_rol[oid].append(r)
        operador_names[oid] = r["nombre"]
        operador_kdfs[oid] = r["kdf"]

    # 4. For each operator, calculate weekly FRR
    results = []
    no_data = []

    # Load existing for dedup
    existing = sb.table("registros_semanales") \
        .select("id, operador_id, frr") \
        .eq("cedula_id", cedula_id) \
        .eq("semana", semana.isoformat()) \
        .execute()
    existing_map = {r["operador_id"]: r for r in (existing.data or [])}

    for oid, days in operador_rol.items():
        kdf = operador_kdfs.get(oid, "")
        machine_nums = kdf_to_filter_numbers(kdf)

        # Skip non-machine personnel
        if not machine_nums:
            continue

        # Collect all FRR values for this operator across the week
        frr_values = []
        matched_days = []

        for day in days:
            turno = day.get("turno")
            if not turno:
                continue  # Day off

            fecha_str = day["fecha"]
            for mnum in machine_nums:
                key = (fecha_str, turno, mnum)
                if key in frr_avg:
                    frr_values.append(frr_avg[key])
                    matched_days.append(f"{fecha_str} {turno} F{mnum:02d}")

        if not frr_values:
            no_data.append({"nombre": operador_names[oid], "kdf": kdf, "reason": "Sin datos FRR para su turno/máquina"})
            continue

        # Weekly average FRR (as percentage, multiply by 100)
        avg_frr = round(sum(frr_values) / len(frr_values) * 100, 2)

        # Dedup
        existing_rec = existing_map.get(oid)
        if existing_rec:
            old_val = existing_rec.get("frr") or 0
            action = "actualizado" if abs(old_val - avg_frr) > 0.001 else "sin_cambio"
        else:
            action = "nuevo"

        results.append({
            "operador_id": oid,
            "nombre": operador_names[oid],
            "kdf": kdf,
            "frr": avg_frr,
            "num_registros": len(frr_values),
            "detalle": matched_days[:5],  # First 5 for display
            "action": action,
        })

    # Sort by name
    results.sort(key=lambda x: x["nombre"])

    # Count actions
    new_count = sum(1 for r in results if r["action"] == "nuevo")
    updated_count = sum(1 for r in results if r["action"] == "actualizado")

    return {
        "success": True,
        "semana": semana.isoformat(),
        "matched": len(results),
        "no_data": no_data,
        "new": new_count,
        "updated": updated_count,
        "frr_records_parsed": len(frr_avg),
        "results": results,
    }


@router.post("/save")
async def save_frr(
    cedula_id: str = Query(...),
    semana: date = Query(...),
    body: list = Body(...)
):
    """Guarda FRR en registros_semanales."""
    sb = get_supabase_admin()
    saved = 0
    for item in body:
        operador_id = item.get("operador_id")
        frr_val = item.get("frr")
        if not operador_id or frr_val is None:
            continue

        rec = {
            "operador_id": operador_id,
            "cedula_id": cedula_id,
            "semana": semana.isoformat(),
            "frr": round(float(frr_val), 2),
        }
        sb.table("registros_semanales") \
            .upsert(rec, on_conflict="operador_id,semana") \
            .execute()
        saved += 1

    return {"success": True, "message": f"FRR guardado: {saved} registros", "saved": saved}
