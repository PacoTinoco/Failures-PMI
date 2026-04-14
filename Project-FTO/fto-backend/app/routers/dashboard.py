from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import date
from app.services.supabase_client import get_supabase_admin

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get("/resumen-lc")
async def resumen_por_lc(
    cedula_id: str,
    semana: Optional[date] = Query(None),
    lc_id: Optional[str] = Query(None)):
    """Obtiene el resumen (promedios) por Line Coordinator usando la vista SQL."""
    sb = get_supabase_admin()
    query = sb.table("resumen_lc").select("*").eq("cedula_id", cedula_id)

    if semana:
        query = query.eq("semana", semana.isoformat())
    if lc_id:
        query = query.eq("lc_id", lc_id)

    query = query.order("semana", desc=True)
    result = query.execute()

    return {"data": result.data}

@router.get("/semanas")
async def listar_semanas(cedula_id: str):
    """Lista todas las semanas que tienen registros para una cédula."""
    sb = get_supabase_admin()
    result = sb.table("registros_semanales").select(
        "semana"
    ).eq("cedula_id", cedula_id).order("semana", desc=True).execute()

    semanas = sorted(set(r["semana"] for r in result.data), reverse=True) if result.data else []
    return {"semanas": semanas}


@router.get("/indicadores")
async def obtener_indicadores():
    """Retorna la configuración de todos los indicadores con sus targets."""
    sb = get_supabase_admin()
    result = sb.table("indicadores_config").select("*").order("orden").execute()
    return {"data": result.data}


@router.get("/operadores-semana")
async def operadores_por_semana(
    cedula_id: str,
    semana: date,
    lc_id: Optional[str] = Query(None)):
    """Obtiene todos los registros de operadores para una semana específica,
    con info del operador y su LC. Ideal para la vista tipo spreadsheet."""
    sb = get_supabase_admin()

    # Obtener operadores de la cédula
    op_query = sb.table("operadores").select(
        "id, nombre, turno, maquina, lc_id, line_coordinators(nombre, grupo)"
    ).eq("cedula_id", cedula_id).eq("activo", True)

    if lc_id:
        op_query = op_query.eq("lc_id", lc_id)

    operadores = op_query.order("nombre").execute()

    # Obtener registros de esa semana
    reg_query = sb.table("registros_semanales").select("*").eq(
        "cedula_id", cedula_id
    ).eq("semana", semana.isoformat())
    registros = reg_query.execute()

    # Mapear registros por operador_id
    reg_map = {r["operador_id"]: r for r in registros.data} if registros.data else {}

    # Combinar: cada operador con su registro (o vacío)
    combined = []
    for op in operadores.data:
        entry = {
            "operador": op,
            "registro": reg_map.get(op["id"], None)
        }
        combined.append(entry)

    return {"semana": semana.isoformat(), "data": combined}


@router.get("/tendencia")
async def tendencia_indicador(
    cedula_id: str,
    campo: str,
    operador_id: Optional[str] = Query(None),
    lc_id: Optional[str] = Query(None),
    semanas: int = Query(12, description="Número de semanas hacia atrás")):
    """Retorna la evolución de un indicador específico a lo largo del tiempo.
    Útil para gráficas de tendencia."""
    campos_validos = [
        "bos_num", "bos_eng", "qbos_num", "qbos_eng", "qflags_num", "qi_pnc_num",
        "dh_encontrados", "dh_reparados", "curva_autonomia", "contramedidas_defectos",
        "ips_num", "frr", "dim_waste", "sobrepeso", "eventos_laika", "casos_estudio", "qm_on_target"
    ]
    if campo not in campos_validos:
        raise HTTPException(status_code=400, detail=f"Campo inválido. Opciones: {campos_validos}")

    sb = get_supabase_admin()
    query = sb.table("registros_semanales").select(
        f"semana, operador_id, {campo}, operadores(nombre, line_coordinators(nombre))"
    ).eq("cedula_id", cedula_id).eq("estatus", "Activo")

    if operador_id:
        query = query.eq("operador_id", operador_id)
    if lc_id:
        ops = sb.table("operadores").select("id").eq("lc_id", lc_id).execute()
        op_ids = [o["id"] for o in ops.data] if ops.data else []
        if op_ids:
            query = query.in_("operador_id", op_ids)

    query = query.order("semana", desc=True).limit(semanas * 20)
    result = query.execute()

    return {"campo": campo, "data": result.data}
