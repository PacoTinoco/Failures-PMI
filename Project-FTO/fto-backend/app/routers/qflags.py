"""
Router: Q Flags (ComitDB)
Procesa el Excel exportado desde ComitDB para llenar automáticamente la columna
'QFlags [#]' (campo qflags_num en registros_semanales).

Flujo:
1. Usuario sube el Excel de ComitDB (columnas: ID, Status, Created, [username], Workcenter, ...)
2. Por cada fila:
   - 'Created' → fecha → se calcula el lunes de la semana (semana ISO)
   - '[username]' (ej. 'PMI\\jbernard4') → se limpia el prefijo y se hace match contra
     operador_aliases.username_comitdb, o fallback fuzzy contra operadores (nombre/email)
   - 'Workcenter' (ej. 'FI17') → se extrae número para referencia/KDF (sólo validación)
3. Se agregan los conteos por (operador_id, semana) y se hace upsert en registros_semanales.qflags_num
"""

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Body
from app.services.supabase_client import get_supabase_admin
from app.routers.bos_qbos import normalize, fuzzy_name_match
from io import BytesIO
from datetime import date, datetime, timedelta
from collections import defaultdict
import re
import pandas as pd

router = APIRouter(prefix="/qflags", tags=["Q Flags"])


# ══════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════

def monday_of(d: date) -> date:
    """Regresa el lunes de la semana a la que pertenece la fecha dada."""
    return d - timedelta(days=d.weekday())


def clean_username(raw: str) -> str:
    """Limpia 'PMI\\jbernard4' → 'jbernard4'. Lowercase."""
    if not raw:
        return ""
    s = str(raw).strip()
    # Remove domain prefix (PMI\, PMINTL\, etc.)
    if "\\" in s:
        s = s.split("\\")[-1]
    return s.lower().strip()


def strip_trailing_digits(s: str) -> str:
    return re.sub(r"\d+$", "", s)


def parse_date_flexible(val) -> date | None:
    """Acepta datetime, string ISO, string 'DD/MM/YYYY' o 'MM/DD/YYYY'."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    s = str(val).strip()
    if not s:
        return None
    # Try pandas first (handles many formats)
    try:
        return pd.to_datetime(s, dayfirst=False, errors="coerce").date()
    except Exception:
        pass
    try:
        return pd.to_datetime(s, dayfirst=True, errors="coerce").date()
    except Exception:
        return None


def username_matches_operator(username: str, op_nombre: str, op_email: str | None) -> int:
    """
    Devuelve score heurístico 0-100 de cuán probable es que un username de ComitDB
    (ej. 'jbernard4') corresponda a un operador.
    Estrategia: primer letra del primer nombre + apellido parcial.
    """
    if not username or not op_nombre:
        return 0
    u = username.lower()
    u_clean = strip_trailing_digits(u)  # 'jbernard4' → 'jbernard'
    if not u_clean:
        return 0

    # Check email local part match first
    if op_email:
        local = op_email.lower().split("@")[0]
        # 'jahzzeluriel.bernardo' vs 'jbernard'
        local_joined = re.sub(r"[^a-z]", "", local)
        if u_clean in local_joined or local_joined.startswith(u_clean):
            return 90

    # Parse operator name parts
    parts = [p for p in re.split(r"\s+", normalize(op_nombre)) if p]
    if not parts:
        return 0

    # Assume MX convention: "Bernardo Rodriguez Jahzzel Uriel" → apellidos first, then nombres
    # 'j' + 'bernard' → nombre starts with 'j' (Jahzzel) and apellido starts with 'bernard' (Bernardo)
    first_char = u_clean[0]
    rest = u_clean[1:]
    if not rest:
        return 0

    for i, word in enumerate(parts):
        if word.startswith(rest) or rest.startswith(word[: min(len(word), len(rest))]):
            # Check if any other word starts with first_char (the first name initial)
            for j, w2 in enumerate(parts):
                if j == i:
                    continue
                if w2.startswith(first_char):
                    return 85
    # Weaker: just a single part prefix match
    for word in parts:
        if word.startswith(u_clean):
            return 70
    return 0


def _load_aliases_qflags(sb, cedula_id: str) -> dict:
    """Carga mapa username_comitdb → persona_id."""
    all_ids = []
    for table in ["operadores", "line_coordinators", "linea_estructura"]:
        res = sb.table(table).select("id").eq("cedula_id", cedula_id).eq("activo", True).execute()
        all_ids.extend([r["id"] for r in (res.data or [])])
    if not all_ids:
        return {}
    aliases = sb.table("operador_aliases") \
        .select("persona_id, persona_tipo, username_comitdb") \
        .in_("persona_id", all_ids) \
        .execute()
    m = {}
    for a in (aliases.data or []):
        u = a.get("username_comitdb")
        if u:
            m[u.lower().strip()] = {
                "persona_id": a["persona_id"],
                "persona_tipo": a.get("persona_tipo", "operador"),
            }
    return m


# ══════════════════════════════════════════════════════
# UPLOAD — Parsea el Excel y regresa el preview agregado
# ══════════════════════════════════════════════════════

@router.post("/upload")
async def upload_qflags(
    file: UploadFile = File(...),
    cedula_id: str = Query(...),
):
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(400, "Solo se aceptan archivos Excel")

    content = await file.read()
    try:
        df = pd.read_excel(BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"No se pudo leer el Excel: {e}")

    # Normaliza nombres de columnas
    cols_map = {c: str(c).strip().lower() for c in df.columns}
    df = df.rename(columns=cols_map)

    # Busca columnas requeridas con flexibilidad
    def find_col(*candidates):
        for c in df.columns:
            for cand in candidates:
                if cand in str(c):
                    return c
        return None

    created_col = find_col("created")
    user_col = find_col("[username]", "username", "user ")
    wc_col = find_col("workcenter", "work center")

    if not created_col or not user_col:
        raise HTTPException(
            400,
            f"Faltan columnas. Encontradas: {list(df.columns)}. Se requieren 'Created' y '[username]'."
        )

    # ── Filtrar solo filas de "Single caps" / "Double caps" ──
    # Busca en columnas type, category, subcategory o cualquiera que contenga esos valores
    caps_col = find_col("type", "category", "subcategory")
    caps_filter_applied = False
    if caps_col:
        # Verificar si la columna realmente tiene valores de caps
        vals_lower = df[caps_col].astype(str).str.strip().str.lower().unique()
        has_caps = any("single caps" in v or "double caps" in v for v in vals_lower)
        if has_caps:
            df = df[df[caps_col].astype(str).str.strip().str.lower().isin(["single caps", "double caps"])]
            caps_filter_applied = True

    # Si no encontró en la primera columna, buscar en todas las columnas por si acaso
    if not caps_filter_applied:
        for col in df.columns:
            vals_lower = df[col].astype(str).str.strip().str.lower().unique()
            has_caps = any("single caps" in v or "double caps" in v for v in vals_lower)
            if has_caps:
                df = df[df[col].astype(str).str.strip().str.lower().isin(["single caps", "double caps"])]
                caps_filter_applied = True
                caps_col = col
                break

    sb = get_supabase_admin()

    # Carga operadores y aliases
    ops_res = sb.table("operadores") \
        .select("id, nombre, email, maquina") \
        .eq("cedula_id", cedula_id) \
        .eq("activo", True) \
        .execute()
    operadores = ops_res.data or []
    id_to_name = {op["id"]: op["nombre"] for op in operadores}
    id_to_maquina = {op["id"]: op.get("maquina") for op in operadores}

    alias_map = _load_aliases_qflags(sb, cedula_id)

    # Agrupa conteos por (persona_id, semana)
    counts = defaultdict(int)
    unmatched_users = defaultdict(int)
    total_rows = 0
    skipped_rows = 0

    for _, row in df.iterrows():
        total_rows += 1
        raw_user = row.get(user_col)
        raw_date = row.get(created_col)
        username = clean_username(raw_user)
        d = parse_date_flexible(raw_date)
        if not username or not d:
            skipped_rows += 1
            continue

        # Match: alias → fuzzy heurístico
        info = alias_map.get(username)
        if not info:
            # Intenta heurístico
            best_score = 0
            best_op = None
            for op in operadores:
                score = username_matches_operator(username, op["nombre"], op.get("email"))
                if score > best_score:
                    best_score = score
                    best_op = op
            if best_op and best_score >= 70:
                info = {"persona_id": best_op["id"], "persona_tipo": "operador"}
            else:
                unmatched_users[username] += 1
                continue

        if info["persona_tipo"] != "operador":
            continue  # QFlags solo aplica a operadores

        sem = monday_of(d)
        counts[(info["persona_id"], sem.isoformat())] += 1

    # Formato de salida: lista de { operador_id, nombre, semana, qflags, maquina }
    results = []
    for (op_id, sem_iso), n in counts.items():
        results.append({
            "operador_id": op_id,
            "nombre": id_to_name.get(op_id, "?"),
            "maquina": id_to_maquina.get(op_id),
            "semana": sem_iso,
            "qflags": n,
        })
    results.sort(key=lambda r: (r["semana"], r["nombre"]))

    return {
        "success": True,
        "total_rows": total_rows,
        "skipped_rows": skipped_rows,
        "caps_filter": caps_col if caps_filter_applied else None,
        "rows_after_filter": len(df) if caps_filter_applied else total_rows,
        "matched_count": sum(counts.values()),
        "unique_operators": len({k[0] for k in counts.keys()}),
        "unique_weeks": sorted({k[1] for k in counts.keys()}),
        "unmatched_users": [{"username": u, "count": c} for u, c in sorted(unmatched_users.items(), key=lambda x: -x[1])],
        "results": results,
    }


# ══════════════════════════════════════════════════════
# SAVE — Upsert de qflags_num en registros_semanales
# ══════════════════════════════════════════════════════

@router.post("/save")
async def save_qflags(
    cedula_id: str = Query(...),
    body: list = Body(...),
):
    """Upsertea cada { operador_id, semana, qflags } en registros_semanales.qflags_num."""
    sb = get_supabase_admin()
    saved = 0
    for item in body:
        op_id = item.get("operador_id")
        sem = item.get("semana")
        q = item.get("qflags")
        if not op_id or not sem:
            continue
        rec = {
            "operador_id": op_id,
            "cedula_id": cedula_id,
            "semana": sem,
            "qflags_num": int(q or 0),
        }
        sb.table("registros_semanales") \
            .upsert(rec, on_conflict="operador_id,semana") \
            .execute()
        saved += 1
    return {"success": True, "saved": saved, "message": f"Q Flags guardadas: {saved} registros"}
