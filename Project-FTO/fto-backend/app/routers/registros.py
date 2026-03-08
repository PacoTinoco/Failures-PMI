from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional
from datetime import date
from app.models.schemas import RegistroSemanalCreate, RegistroSemanalUpdate
from app.services.supabase_client import get_supabase_admin
from app.services.auth import get_current_user

router = APIRouter(prefix="/registros", tags=["Registros Semanales"])


@router.get("/")
async def listar_registros(
    semana: Optional[date] = Query(None, description="Filtrar por semana (YYYY-MM-DD)"),
    operador_id: Optional[str] = Query(None, description="Filtrar por operador"),
    cedula_id: Optional[str] = Query(None, description="Filtrar por cédula"),
    lc_id: Optional[str] = Query(None, description="Filtrar por Line Coordinator"),
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
    user=Depends(get_current_user)
):
    """Lista registros semanales con filtros opcionales."""
    sb = get_supabase_admin()
    query = sb.table("registros_semanales").select(
        "*, operadores(nombre, turno, maquina, lc_id, line_coordinators(nombre, grupo))"
    )

    if semana:
        query = query.eq("semana", semana.isoformat())
    if operador_id:
        query = query.eq("operador_id", operador_id)
    if cedula_id:
        query = query.eq("cedula_id", cedula_id)
    if lc_id:
        # Filtrar por LC requiere join con operadores
        ops = sb.table("operadores").select("id").eq("lc_id", lc_id).execute()
        op_ids = [o["id"] for o in ops.data] if ops.data else []
        if op_ids:
            query = query.in_("operador_id", op_ids)
        else:
            return {"data": [], "count": 0}

    query = query.order("semana", desc=True).range(offset, offset + limit - 1)
    result = query.execute()

    return {"data": result.data, "count": len(result.data)}


@router.post("/")
async def crear_registro(registro: RegistroSemanalCreate, user=Depends(get_current_user)):
    """Crea un nuevo registro semanal para un operador."""
    sb = get_supabase_admin()

    # Obtener cedula_id del operador
    op = sb.table("operadores").select("cedula_id").eq("id", registro.operador_id).single().execute()
    if not op.data:
        raise HTTPException(status_code=404, detail="Operador no encontrado")

    data = registro.model_dump(exclude_none=True)
    data["cedula_id"] = op.data["cedula_id"]

    # Buscar usuario para created_by
    usr = sb.table("usuarios").select("id").eq("email", user.email).maybe_single().execute()
    if usr.data:
        data["created_by"] = usr.data["id"]

    try:
        result = sb.table("registros_semanales").insert(data).execute()
        return {"success": True, "data": result.data[0] if result.data else None}
    except Exception as e:
        if "unique" in str(e).lower() or "duplicate" in str(e).lower():
            raise HTTPException(
                status_code=409,
                detail="Ya existe un registro para este operador en esta semana. Usa PUT para actualizar."
            )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def crear_registros_batch(registros: list[RegistroSemanalCreate], user=Depends(get_current_user)):
    """Crea múltiples registros en una sola operación (para captura rápida)."""
    sb = get_supabase_admin()
    usr = sb.table("usuarios").select("id").eq("email", user.email).maybe_single().execute()
    created_by = usr.data["id"] if usr.data else None

    records = []
    for reg in registros:
        op = sb.table("operadores").select("cedula_id").eq("id", reg.operador_id).single().execute()
        if not op.data:
            continue
        data = reg.model_dump(exclude_none=True)
        data["cedula_id"] = op.data["cedula_id"]
        if created_by:
            data["created_by"] = created_by
        records.append(data)

    if not records:
        raise HTTPException(status_code=400, detail="No se encontraron operadores válidos")

    try:
        result = sb.table("registros_semanales").upsert(records, on_conflict="operador_id,semana").execute()
        return {"success": True, "count": len(result.data), "data": result.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{registro_id}")
async def actualizar_registro(registro_id: str, update: RegistroSemanalUpdate, user=Depends(get_current_user)):
    """Actualiza un registro semanal existente."""
    sb = get_supabase_admin()
    data = update.model_dump(exclude_none=True)
    if not data:
        raise HTTPException(status_code=400, detail="No hay campos para actualizar")

    try:
        result = sb.table("registros_semanales").update(data).eq("id", registro_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Registro no encontrado")
        return {"success": True, "data": result.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{registro_id}")
async def eliminar_registro(registro_id: str, user=Depends(get_current_user)):
    """Elimina un registro semanal."""
    sb = get_supabase_admin()
    try:
        result = sb.table("registros_semanales").delete().eq("id", registro_id).execute()
        return {"success": True, "message": "Registro eliminado"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
