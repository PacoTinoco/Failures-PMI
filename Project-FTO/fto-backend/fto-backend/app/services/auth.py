from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.services.supabase_client import get_supabase

security = HTTPBearer()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verifica el JWT de Supabase y retorna el usuario."""
    token = credentials.credentials
    try:
        sb = get_supabase()
        user_response = sb.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Token inválido")
        return user_response.user
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"No autorizado: {str(e)}")


async def require_role(allowed_roles: list[str]):
    """Factory para verificar que el usuario tenga un rol permitido."""
    async def checker(user=Depends(get_current_user)):
        from app.services.supabase_client import get_supabase_admin
        sb = get_supabase_admin()
        result = sb.table("usuarios").select("rol, cedula_id").eq("email", user.email).single().execute()
        if not result.data:
            raise HTTPException(status_code=403, detail="Usuario no registrado en la plataforma")
        if result.data["rol"] not in allowed_roles:
            raise HTTPException(status_code=403, detail="No tienes permisos para esta acción")
        return {**result.data, "email": user.email, "auth_id": user.id}
    return checker
