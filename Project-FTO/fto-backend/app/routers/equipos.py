from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import (
    OperadorCreate, OperadorUpdate,
    LCCreate, LCUpdate,
    CedulaCreate, CedulaUpdate,
    UsuarioCreate
)
from app.services.supabase_client import get_supabase_admin
from app.services.auth import get_current_user

router = APIRouter(prefix="/equipos", tags=["Gestión de Equipos"])


# ============================================================
# CÉDULAS
# ============================================================

@router.get("/cedulas")
async def listar_cedulas(user=Depends(get_current_user)):
    sb = get_supabase_admin()
    result = sb.table("cedulas").select("*").eq("activa", True).order("nombre").execute()
    return {"data": result.data}


@router.get("/cedulas/{cedula_id}")
async def detalle_cedula(cedula_id: str, user=Depends(get_current_user)):
    sb = get_supabase_admin()
    cedula = sb.table("cedulas").select("*").eq("id", cedula_id).single().execute()
    lcs = sb.table("line_coordinators").select("*").eq("cedula_id", cedula_id).eq("activo", True).execute()
    ops = sb.table("operadores").select(
        "*, line_coordinators(nombre, grupo)"
    ).eq("cedula_id", cedula_id).eq("activo", True).order("nombre").execute()

    return {
        "cedula": cedula.data,
        "line_coordinators": lcs.data,
        "operadores": ops.data,
        "total_lc": len(lcs.data) if lcs.data else 0,
        "total_operadores": len(ops.data) if ops.data else 0
    }


@router.post("/cedulas")
async def crear_cedula(cedula: CedulaCreate, user=Depends(get_current_user)):
    sb = get_supabase_admin()
    try:
        result = sb.table("cedulas").insert(
            cedula.model_dump(exclude_none=True)
        ).execute()
        return {"success": True, "data": result.data[0] if result.data else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/cedulas/{cedula_id}")
async def actualizar_cedula(cedula_id: str, cedula: CedulaUpdate, user=Depends(get_current_user)):
    sb = get_supabase_admin()
    result = sb.table("cedulas").update(
        cedula.model_dump(exclude_none=True)
    ).eq("id", cedula_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Cédula no encontrada")
    return {"success": True, "data": result.data[0]}


@router.delete("/cedulas/{cedula_id}")
async def eliminar_cedula(cedula_id: str, user=Depends(get_current_user)):
    """Soft delete — marca como inactiva."""
    sb = get_supabase_admin()
    result = sb.table("cedulas").update({"activa": False}).eq("id", cedula_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Cédula no encontrada")
    return {"success": True, "message": "Cédula desactivada"}


# ============================================================
# LINE COORDINATORS
# ============================================================

@router.get("/lc")
async def listar_lcs(cedula_id: str, user=Depends(get_current_user)):
    sb = get_supabase_admin()
    result = sb.table("line_coordinators").select("*").eq(
        "cedula_id", cedula_id
    ).eq("activo", True).order("nombre").execute()
    return {"data": result.data}


@router.post("/lc")
async def crear_lc(lc: LCCreate, user=Depends(get_current_user)):
    sb = get_supabase_admin()
    try:
        result = sb.table("line_coordinators").insert(lc.model_dump(exclude_none=True)).execute()
        return {"success": True, "data": result.data[0] if result.data else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/lc/{lc_id}")
async def actualizar_lc(lc_id: str, lc: LCUpdate, user=Depends(get_current_user)):
    sb = get_supabase_admin()
    result = sb.table("line_coordinators").update(
        lc.model_dump(exclude_none=True)
    ).eq("id", lc_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="LC no encontrado")
    return {"success": True, "data": result.data[0]}


@router.delete("/lc/{lc_id}")
async def eliminar_lc(lc_id: str, user=Depends(get_current_user)):
    """Soft delete — marca como inactivo."""
    sb = get_supabase_admin()
    result = sb.table("line_coordinators").update({"activo": False}).eq("id", lc_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="LC no encontrado")
    return {"success": True, "message": "LC desactivado"}


# ============================================================
# OPERADORES
# ============================================================

@router.get("/operadores")
async def listar_operadores(cedula_id: str, lc_id: str = None, user=Depends(get_current_user)):
    sb = get_supabase_admin()
    query = sb.table("operadores").select(
        "*, line_coordinators(nombre, grupo)"
    ).eq("cedula_id", cedula_id).eq("activo", True)

    if lc_id:
        query = query.eq("lc_id", lc_id)

    result = query.order("nombre").execute()
    return {"data": result.data}


@router.post("/operadores")
async def crear_operador(op: OperadorCreate, user=Depends(get_current_user)):
    sb = get_supabase_admin()
    try:
        result = sb.table("operadores").insert(op.model_dump(exclude_none=True)).execute()
        return {"success": True, "data": result.data[0] if result.data else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/operadores/{op_id}")
async def actualizar_operador(op_id: str, op: OperadorUpdate, user=Depends(get_current_user)):
    sb = get_supabase_admin()
    result = sb.table("operadores").update(
        op.model_dump(exclude_none=True)
    ).eq("id", op_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Operador no encontrado")
    return {"success": True, "data": result.data[0]}


@router.delete("/operadores/{op_id}")
async def desactivar_operador(op_id: str, user=Depends(get_current_user)):
    """Soft delete — marca como inactivo."""
    sb = get_supabase_admin()
    result = sb.table("operadores").update({"activo": False}).eq("id", op_id).execute()
    return {"success": True, "message": "Operador desactivado"}


# ============================================================
# USUARIOS DE LA PLATAFORMA
# ============================================================

@router.get("/usuarios")
async def listar_usuarios(cedula_id: str = None, user=Depends(get_current_user)):
    sb = get_supabase_admin()
    query = sb.table("usuarios").select("*, cedulas(nombre)").eq("activo", True)
    if cedula_id:
        query = query.eq("cedula_id", cedula_id)
    result = query.order("nombre").execute()
    return {"data": result.data}


@router.post("/usuarios")
async def crear_usuario(usuario: UsuarioCreate, user=Depends(get_current_user)):
    sb = get_supabase_admin()
    try:
        result = sb.table("usuarios").insert(usuario.model_dump()).execute()
        return {"success": True, "data": result.data[0] if result.data else None}
    except Exception as e:
        if "unique" in str(e).lower():
            raise HTTPException(status_code=409, detail="Ya existe un usuario con ese email")
        raise HTTPException(status_code=500, detail=str(e))
