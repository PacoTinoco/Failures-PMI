"""
Router: Weekly DOS/DCS
Gestión de indicadores semanales con gráficas rojo/verde,
targets configurables y captura de valores.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from app.services.supabase_client import get_supabase_admin

router = APIRouter(prefix="/weekly", tags=["Weekly"])


# ══════════════════════════════════════════════════════
# Schemas
# ══════════════════════════════════════════════════════

class CategoryCreate(BaseModel):
    cedula_id: str
    name: str
    display_order: int = 0

class CategoryUpdate(BaseModel):
    name: Optional[str] = None
    display_order: Optional[int] = None

class IndicatorCreate(BaseModel):
    category_id: str
    cedula_id: str
    name: str
    subtitle: Optional[str] = None
    direction: str = "lower_better"   # 'higher_better' | 'lower_better'
    unit: str = "#"
    display_order: int = 0

class IndicatorUpdate(BaseModel):
    name: Optional[str] = None
    subtitle: Optional[str] = None
    direction: Optional[str] = None
    unit: Optional[str] = None
    display_order: Optional[int] = None
    category_id: Optional[str] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    band_size: Optional[float] = None  # For middle_better: half-width of green band

class TargetUpsert(BaseModel):
    indicator_id: str
    cedula_id: str
    year: int
    quarter: int
    week_number: int
    target_value: float

class TargetBulk(BaseModel):
    targets: List[TargetUpsert]

class ValueUpsert(BaseModel):
    indicator_id: str
    cedula_id: str
    year: int
    quarter: int
    week_number: int
    actual_value: float
    auto_source: Optional[str] = None

class ValueBulk(BaseModel):
    values: List[ValueUpsert]

class ReorderItem(BaseModel):
    id: str
    display_order: int

class ReorderRequest(BaseModel):
    items: List[ReorderItem]


# ══════════════════════════════════════════════════════
# CATEGORIES — CRUD
# ══════════════════════════════════════════════════════

@router.get("/categories")
async def list_categories(cedula_id: str = Query(...)):
    sb = get_supabase_admin()
    result = sb.table("weekly_categories") \
        .select("*") \
        .eq("cedula_id", cedula_id) \
        .order("display_order") \
        .execute()
    return {"data": result.data}


@router.post("/categories")
async def create_category(body: CategoryCreate):
    sb = get_supabase_admin()
    result = sb.table("weekly_categories").insert({
        "cedula_id": body.cedula_id,
        "name": body.name,
        "display_order": body.display_order,
    }).execute()
    return {"data": result.data[0] if result.data else None}


@router.put("/categories/{cat_id}")
async def update_category(cat_id: str, body: CategoryUpdate):
    sb = get_supabase_admin()
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if not updates:
        raise HTTPException(400, "Nada que actualizar")
    result = sb.table("weekly_categories").update(updates).eq("id", cat_id).execute()
    return {"data": result.data[0] if result.data else None}


@router.delete("/categories/{cat_id}")
async def delete_category(cat_id: str):
    sb = get_supabase_admin()
    sb.table("weekly_categories").delete().eq("id", cat_id).execute()
    return {"ok": True}


@router.post("/categories/reorder")
async def reorder_categories(body: ReorderRequest):
    sb = get_supabase_admin()
    for item in body.items:
        sb.table("weekly_categories").update({"display_order": item.display_order}).eq("id", item.id).execute()
    return {"ok": True}


# ══════════════════════════════════════════════════════
# INDICATORS — CRUD
# ══════════════════════════════════════════════════════

@router.get("/indicators")
async def list_indicators(cedula_id: str = Query(...), category_id: Optional[str] = Query(None)):
    sb = get_supabase_admin()
    q = sb.table("weekly_indicators").select("*").eq("cedula_id", cedula_id)
    if category_id:
        q = q.eq("category_id", category_id)
    result = q.order("display_order").execute()
    return {"data": result.data}


@router.post("/indicators")
async def create_indicator(body: IndicatorCreate):
    sb = get_supabase_admin()
    result = sb.table("weekly_indicators").insert({
        "category_id": body.category_id,
        "cedula_id": body.cedula_id,
        "name": body.name,
        "subtitle": body.subtitle,
        "direction": body.direction,
        "unit": body.unit,
        "display_order": body.display_order,
    }).execute()
    return {"data": result.data[0] if result.data else None}


@router.post("/indicators/bulk")
async def create_indicators_bulk(indicators: List[IndicatorCreate]):
    sb = get_supabase_admin()
    records = [{
        "category_id": ind.category_id,
        "cedula_id": ind.cedula_id,
        "name": ind.name,
        "subtitle": ind.subtitle,
        "direction": ind.direction,
        "unit": ind.unit,
        "display_order": ind.display_order,
    } for ind in indicators]
    result = sb.table("weekly_indicators").insert(records).execute()
    return {"data": result.data, "count": len(result.data)}


@router.put("/indicators/{ind_id}")
async def update_indicator(ind_id: str, body: IndicatorUpdate):
    sb = get_supabase_admin()
    # Use __fields_set__ so fields explicitly sent as null/None are included
    updates = {k: getattr(body, k) for k in body.__fields_set__}
    if not updates:
        raise HTTPException(400, "Nada que actualizar")
    result = sb.table("weekly_indicators").update(updates).eq("id", ind_id).execute()
    return {"data": result.data[0] if result.data else None}


@router.delete("/indicators/{ind_id}")
async def delete_indicator(ind_id: str):
    sb = get_supabase_admin()
    sb.table("weekly_indicators").delete().eq("id", ind_id).execute()
    return {"ok": True}


@router.post("/indicators/reorder")
async def reorder_indicators(body: ReorderRequest):
    sb = get_supabase_admin()
    for item in body.items:
        sb.table("weekly_indicators").update({"display_order": item.display_order}).eq("id", item.id).execute()
    return {"ok": True}


# ══════════════════════════════════════════════════════
# TARGETS — Upsert y consulta
# ══════════════════════════════════════════════════════

@router.get("/targets")
async def list_targets(
    cedula_id: str = Query(...),
    year: int = Query(...),
    quarter: int = Query(...),
    indicator_id: Optional[str] = Query(None),
):
    sb = get_supabase_admin()
    q = sb.table("weekly_targets") \
        .select("*") \
        .eq("cedula_id", cedula_id) \
        .eq("year", year) \
        .eq("quarter", quarter)
    if indicator_id:
        q = q.eq("indicator_id", indicator_id)
    result = q.order("week_number").execute()
    return {"data": result.data}


@router.post("/targets")
async def upsert_targets(body: TargetBulk):
    sb = get_supabase_admin()
    records = [{
        "indicator_id": t.indicator_id,
        "cedula_id": t.cedula_id,
        "year": t.year,
        "quarter": t.quarter,
        "week_number": t.week_number,
        "target_value": t.target_value,
    } for t in body.targets]
    result = sb.table("weekly_targets").upsert(
        records,
        on_conflict="indicator_id,year,quarter,week_number"
    ).execute()
    return {"data": result.data, "count": len(result.data)}


# Endpoint rápido: poner mismo target en todas las semanas de un trimestre
@router.post("/targets/fill")
async def fill_targets(
    indicator_id: str = Query(...),
    cedula_id: str = Query(...),
    year: int = Query(...),
    quarter: int = Query(...),
    target_value: float = Query(...),
    weeks: int = Query(15),
):
    sb = get_supabase_admin()
    records = [{
        "indicator_id": indicator_id,
        "cedula_id": cedula_id,
        "year": year,
        "quarter": quarter,
        "week_number": w,
        "target_value": target_value,
    } for w in range(1, weeks + 1)]
    result = sb.table("weekly_targets").upsert(
        records,
        on_conflict="indicator_id,year,quarter,week_number"
    ).execute()
    return {"count": len(result.data)}


# ══════════════════════════════════════════════════════
# VALUES — Upsert y consulta
# ══════════════════════════════════════════════════════

@router.get("/values")
async def list_values(
    cedula_id: str = Query(...),
    year: int = Query(...),
    quarter: int = Query(...),
    indicator_id: Optional[str] = Query(None),
):
    sb = get_supabase_admin()
    q = sb.table("weekly_values") \
        .select("*") \
        .eq("cedula_id", cedula_id) \
        .eq("year", year) \
        .eq("quarter", quarter)
    if indicator_id:
        q = q.eq("indicator_id", indicator_id)
    result = q.order("week_number").execute()
    return {"data": result.data}


@router.post("/values")
async def upsert_values(body: ValueBulk):
    sb = get_supabase_admin()
    records = [{
        "indicator_id": v.indicator_id,
        "cedula_id": v.cedula_id,
        "year": v.year,
        "quarter": v.quarter,
        "week_number": v.week_number,
        "actual_value": v.actual_value,
        "auto_source": v.auto_source,
    } for v in body.values]
    result = sb.table("weekly_values").upsert(
        records,
        on_conflict="indicator_id,year,quarter,week_number"
    ).execute()
    return {"data": result.data, "count": len(result.data)}


# ══════════════════════════════════════════════════════
# CHART DATA — Todo junto para una categoría
# ══════════════════════════════════════════════════════

@router.get("/chart-data")
async def get_chart_data(
    cedula_id: str = Query(...),
    year: int = Query(...),
    quarter: int = Query(...),
    category_id: Optional[str] = Query(None),
):
    """Retorna indicadores + targets + valores para renderizar las gráficas."""
    sb = get_supabase_admin()

    # 1. Indicadores
    q = sb.table("weekly_indicators").select("*").eq("cedula_id", cedula_id)
    if category_id:
        q = q.eq("category_id", category_id)
    indicators = q.order("display_order").execute().data

    if not indicators:
        return {"indicators": [], "targets": {}, "values": {}}

    ind_ids = [i["id"] for i in indicators]

    # 2. Targets para esos indicadores en ese trimestre
    targets_raw = sb.table("weekly_targets") \
        .select("indicator_id, week_number, target_value") \
        .eq("cedula_id", cedula_id) \
        .eq("year", year) \
        .eq("quarter", quarter) \
        .in_("indicator_id", ind_ids) \
        .order("week_number") \
        .execute().data

    # Agrupar por indicator_id
    targets = {}
    for t in targets_raw:
        iid = t["indicator_id"]
        if iid not in targets:
            targets[iid] = {}
        targets[iid][t["week_number"]] = t["target_value"]

    # 3. Valores
    values_raw = sb.table("weekly_values") \
        .select("indicator_id, week_number, actual_value, auto_source") \
        .eq("cedula_id", cedula_id) \
        .eq("year", year) \
        .eq("quarter", quarter) \
        .in_("indicator_id", ind_ids) \
        .order("week_number") \
        .execute().data

    values = {}
    for v in values_raw:
        iid = v["indicator_id"]
        if iid not in values:
            values[iid] = {}
        values[iid][v["week_number"]] = {
            "value": v["actual_value"],
            "auto": v["auto_source"],
        }

    return {
        "indicators": indicators,
        "targets": targets,
        "values": values,
    }


# ══════════════════════════════════════════════════════
# SEED — Carga inicial de categorías e indicadores
# ══════════════════════════════════════════════════════

@router.post("/seed")
async def seed_weekly(cedula_id: str = Query(...)):
    """Crea las 10 categorías y ~70 indicadores iniciales para una cédula."""
    sb = get_supabase_admin()

    # Verificar que no exista ya
    existing = sb.table("weekly_categories").select("id").eq("cedula_id", cedula_id).limit(1).execute()
    if existing.data:
        raise HTTPException(400, "Ya existen categorías Weekly para esta cédula. Borra primero si quieres re-seed.")

    MACHINES_6 = ["KDF 10", "KDF 17", "KDF 7", "KDF 9", "KDF 8", "KDF 11"]

    # Definición de categorías e indicadores
    SEED_DATA = [
        {
            "name": "Incident Elimination DMS",
            "indicators": [
                {"name": "# HSE Incidents", "direction": "lower_better", "unit": "#"},
                {"name": "IE - BOS Engagement", "direction": "higher_better", "unit": "%"},
                {"name": "IE - DMS Health Check", "direction": "higher_better", "unit": "%"},
                {"name": "5S - DMS Health Check", "direction": "higher_better", "unit": "%"},
            ]
        },
        {
            "name": "Quality Incident Elimination DMS",
            "indicators": [
                *[{"name": "QIE # Quality Incidents", "subtitle": m, "direction": "lower_better", "unit": "#"} for m in MACHINES_6],
                *[{"name": "QIE # Quality Flags", "subtitle": m, "direction": "lower_better", "unit": "#"} for m in MACHINES_6],
                {"name": "QIE - QBOS Engagement", "direction": "higher_better", "unit": "%"},
                {"name": "QIE - DMS Health Check", "direction": "higher_better", "unit": "%"},
            ]
        },
        {
            "name": "Data Integrity Caps",
            "indicators": [
                {"name": "Data Integrity", "direction": "higher_better", "unit": "%"},
            ]
        },
        {
            "name": "Attainment",
            "indicators": [
                {"name": "Attainment", "subtitle": "KDF 7, 9, 10", "direction": "higher_better", "unit": "%"},
                {"name": "Attainment", "subtitle": "KDF 17", "direction": "higher_better", "unit": "%"},
                {"name": "Attainment", "subtitle": "KDF 8", "direction": "higher_better", "unit": "%"},
                {"name": "Attainment", "subtitle": "KDF 11", "direction": "higher_better", "unit": "%"},
            ]
        },
        {
            "name": "Process Reliability",
            "indicators": [
                *[{"name": "Process Reliability", "subtitle": m, "direction": "higher_better", "unit": "%"} for m in MACHINES_6],
            ]
        },
        {
            "name": "Unplanned Downtime",
            "indicators": [
                *[{"name": "Unplanned Downtime", "subtitle": m, "direction": "lower_better", "unit": "%"} for m in MACHINES_6],
            ]
        },
        {
            "name": "Planned Downtime",
            "indicators": [
                *[{"name": "Planned Downtime", "subtitle": m, "direction": "lower_better", "unit": "%"} for m in MACHINES_6],
            ]
        },
        {
            "name": "Finish Product Quality DMS",
            "indicators": [
                *[{"name": "FRR", "subtitle": m, "direction": "lower_better", "unit": "%"} for m in MACHINES_6],
                *[{"name": "PSI Score", "subtitle": m, "direction": "higher_better", "unit": "#"} for m in MACHINES_6],
            ]
        },
        {
            "name": "Centerline DMS",
            "indicators": [
                *[{"name": "MTBF", "subtitle": m, "direction": "higher_better", "unit": "min"} for m in MACHINES_6],
                {"name": "CL - % Completion", "direction": "higher_better", "unit": "%"},
                {"name": "CL Eliminated of Key", "direction": "higher_better", "unit": "#"},
                {"name": "CL - DMS Health Check", "direction": "higher_better", "unit": "%"},
            ]
        },
        {
            "name": "Clean-Inspect-Lubricate DMS",
            "indicators": [
                *[{"name": "Unplanned Stops", "subtitle": m, "direction": "lower_better", "unit": "#"} for m in MACHINES_6],
                {"name": "CIL - % Completion", "direction": "higher_better", "unit": "%"},
                {"name": "CIL - DMS Health Check", "direction": "higher_better", "unit": "%"},
            ]
        },
    ]

    total_indicators = 0

    for cat_order, cat_data in enumerate(SEED_DATA):
        # Crear categoría
        cat_result = sb.table("weekly_categories").insert({
            "cedula_id": cedula_id,
            "name": cat_data["name"],
            "display_order": cat_order,
        }).execute()
        cat_id = cat_result.data[0]["id"]

        # Crear indicadores
        ind_records = []
        for ind_order, ind in enumerate(cat_data["indicators"]):
            ind_records.append({
                "category_id": cat_id,
                "cedula_id": cedula_id,
                "name": ind["name"],
                "subtitle": ind.get("subtitle"),
                "direction": ind["direction"],
                "unit": ind["unit"],
                "display_order": ind_order,
            })

        if ind_records:
            sb.table("weekly_indicators").insert(ind_records).execute()
            total_indicators += len(ind_records)

    return {
        "message": f"Seed completado: 10 categorías, {total_indicators} indicadores",
        "categories": len(SEED_DATA),
        "indicators": total_indicators,
    }


@router.post("/seed-extras")
async def seed_extras(cedula_id: str = Query(...)):
    """Agrega las categorías e indicadores nuevos que faltan, sin tocar los existentes."""
    sb = get_supabase_admin()

    MACHINES_6 = ["KDF 10", "KDF 17", "KDF 7", "KDF 9", "KDF 8", "KDF 11"]
    MACHINES_4_GROUPS = [
        {"subtitle": "KDF 7, 9, 10", "direction_prefix": ""},
        {"subtitle": "KDF 17",       "direction_prefix": ""},
        {"subtitle": "KDF 8",        "direction_prefix": ""},
        {"subtitle": "KDF 11",       "direction_prefix": ""},
    ]

    EXTRA_CATEGORIES = [
        {
            "name": "Defect Handling DMS",
            "indicators": [
                {"name": "DH - # Defects Found", "subtitle": "KDF 7, 9, 10", "direction": "higher_better", "unit": "#"},
                {"name": "DH - # Defects Fixed", "subtitle": "KDF 7, 9, 10", "direction": "higher_better", "unit": "#"},
                {"name": "DH - # Defects Found", "subtitle": "KDF 17", "direction": "higher_better", "unit": "#"},
                {"name": "DH - # Defects Fixed", "subtitle": "KDF 17", "direction": "higher_better", "unit": "#"},
                {"name": "DH - # Defects Found", "subtitle": "KDF 8", "direction": "higher_better", "unit": "#"},
                {"name": "DH - # Defects Fixed", "subtitle": "KDF 8", "direction": "higher_better", "unit": "#"},
                {"name": "DH - # Defects Found", "subtitle": "KDF 11", "direction": "higher_better", "unit": "#"},
                {"name": "DH - # Defects Fixed", "subtitle": "KDF 11", "direction": "higher_better", "unit": "#"},
                {"name": "DH - Health Check", "direction": "higher_better", "unit": "%"},
                {"name": "# Breakdowns", "direction": "lower_better", "unit": "#"},
            ]
        },
        {
            "name": "Breakdown Elimination",
            "indicators": [
                *[{"name": "# Repeated Breakdowns", "subtitle": m, "direction": "lower_better", "unit": "#"} for m in MACHINES_6],
                *[{"name": "BDE - % BD w/countermeasures", "subtitle": m, "direction": "higher_better", "unit": "%"} for m in MACHINES_6],
                *[{"name": "BDE - % BD qualified for IDA", "subtitle": m, "direction": "higher_better", "unit": "%"} for m in MACHINES_6],
                {"name": "BD Elimination - Health Check", "direction": "higher_better", "unit": "%"},
            ]
        },
        {
            "name": "IPS",
            "indicators": [
                {"name": "# Process Failures", "subtitle": "KDF 7, 9, 10", "direction": "lower_better", "unit": "#"},
                {"name": "# Process Failures", "subtitle": "KDF 17", "direction": "lower_better", "unit": "#"},
                {"name": "# Process Failures", "subtitle": "KDF 8", "direction": "lower_better", "unit": "#"},
                {"name": "# Process Failures", "subtitle": "KDF 11", "direction": "lower_better", "unit": "#"},
                {"name": "# IPS Still Open", "subtitle": "KDF 7, 9, 10", "direction": "lower_better", "unit": "#"},
                {"name": "# IPS Still Open", "subtitle": "KDF 17", "direction": "lower_better", "unit": "#"},
                {"name": "# IPS Still Open", "subtitle": "KDF 8", "direction": "lower_better", "unit": "#"},
                {"name": "# IPS Still Open", "subtitle": "KDF 11", "direction": "lower_better", "unit": "#"},
                {"name": "IPS - SWP Health Check", "direction": "higher_better", "unit": "%"},
            ]
        },
        {
            "name": "Maint. Planning & Scheduling DMS",
            "indicators": [
                *[{"name": "% PM's Executed vs Planned", "subtitle": m, "direction": "higher_better", "unit": "%"} for m in MACHINES_6],
                *[{"name": "Backlog", "subtitle": m, "direction": "lower_better", "unit": "#"} for m in MACHINES_6],
                {"name": "Maintenance Effectiveness", "direction": "higher_better", "unit": "%"},
                {"name": "# PM's Created / Modified", "direction": "higher_better", "unit": "#"},
                {"name": "MP&S - DMS Health Check", "direction": "higher_better", "unit": "%"},
            ]
        },
        {
            "name": "Changeover DMS",
            "indicators": [
                {"name": "ChangeOver Time Variance", "subtitle": "KDF 7, 9, 10", "direction": "lower_better", "unit": "min"},
                {"name": "CO - Waste 2 Hour", "subtitle": "KDF 7, 9, 10", "direction": "lower_better", "unit": "#"},
                {"name": "ChangeOver Time Variance", "subtitle": "KDF 17", "direction": "lower_better", "unit": "min"},
                {"name": "CO - Waste 2 Hour", "subtitle": "KDF 17", "direction": "lower_better", "unit": "#"},
                {"name": "ChangeOver Time Variance", "subtitle": "KDF 8", "direction": "lower_better", "unit": "min"},
                {"name": "CO - Waste 2 Hour", "subtitle": "KDF 8", "direction": "lower_better", "unit": "#"},
                {"name": "ChangeOver Time Variance", "subtitle": "KDF 11", "direction": "lower_better", "unit": "min"},
                {"name": "CO - Waste 2 Hour", "subtitle": "KDF 11", "direction": "lower_better", "unit": "#"},
                {"name": "CO DMS - Health Check", "direction": "higher_better", "unit": "%"},
            ]
        },
        {
            "name": "Reaplications",
            "indicators": [
                {"name": "# Sub-Themes Generated", "subtitle": "KDF 7, 9, 10", "direction": "higher_better", "unit": "#"},
                {"name": "% Sub-Themes Reapplied / Finalized", "subtitle": "KDF 7, 9, 10", "direction": "higher_better", "unit": "%"},
                {"name": "# Sub-Themes Generated", "subtitle": "KDF 17", "direction": "higher_better", "unit": "#"},
                {"name": "% Sub-Themes Reapplied / Finalized", "subtitle": "KDF 17", "direction": "higher_better", "unit": "%"},
                {"name": "# Sub-Themes Generated", "subtitle": "KDF 8", "direction": "higher_better", "unit": "#"},
                {"name": "% Sub-Themes Reapplied / Finalized", "subtitle": "KDF 8", "direction": "higher_better", "unit": "%"},
                {"name": "# Sub-Themes Generated", "subtitle": "KDF 11", "direction": "higher_better", "unit": "#"},
                {"name": "% Sub-Themes Reapplied / Finalized", "subtitle": "KDF 11", "direction": "higher_better", "unit": "%"},
                {"name": "RM - DMS Health Check", "direction": "higher_better", "unit": "%"},
            ]
        },
        {
            "name": "UPTIME",
            "indicators": [
                *[{"name": "UPTIME", "subtitle": m, "direction": "higher_better", "unit": "%"} for m in MACHINES_6],
            ]
        },
    ]

    # Get existing categories to find the max display_order
    existing_cats = sb.table("weekly_categories").select("name, display_order") \
        .eq("cedula_id", cedula_id).order("display_order", desc=True).execute()
    existing_names = {c["name"] for c in (existing_cats.data or [])}
    start_order = (existing_cats.data[0]["display_order"] + 1) if existing_cats.data else 0

    created_cats = 0
    created_inds = 0

    for i, cat_data in enumerate(EXTRA_CATEGORIES):
        if cat_data["name"] in existing_names:
            continue  # Skip already existing categories

        cat_result = sb.table("weekly_categories").insert({
            "cedula_id": cedula_id,
            "name": cat_data["name"],
            "display_order": start_order + i,
        }).execute()
        cat_id = cat_result.data[0]["id"]
        created_cats += 1

        ind_records = []
        for ind_order, ind in enumerate(cat_data["indicators"]):
            ind_records.append({
                "category_id": cat_id,
                "cedula_id": cedula_id,
                "name": ind["name"],
                "subtitle": ind.get("subtitle"),
                "direction": ind["direction"],
                "unit": ind["unit"],
                "display_order": ind_order,
            })

        if ind_records:
            sb.table("weekly_indicators").insert(ind_records).execute()
            created_inds += len(ind_records)

    return {
        "message": f"Extras: {created_cats} categorías nuevas, {created_inds} indicadores nuevos agregados",
        "new_categories": created_cats,
        "new_indicators": created_inds,
        "skipped": len(EXTRA_CATEGORIES) - created_cats,
    }
