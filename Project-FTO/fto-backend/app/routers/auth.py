from fastapi import APIRouter, HTTPException
from app.models.schemas import MagicLinkRequest, TokenVerifyRequest
from app.services.supabase_client import get_supabase, get_supabase_admin
from app.config import get_settings

router = APIRouter(prefix="/auth", tags=["Autenticación"])


@router.post("/magic-link")
async def send_magic_link(request: MagicLinkRequest):
    """Envía un Magic Link al correo corporativo @PMINTL.NET."""
    settings = get_settings()

    # Validar dominio
    if not request.email.endswith(f"@{settings.allowed_email_domain}"):
        raise HTTPException(
            status_code=400,
            detail=f"Solo se permiten correos @{settings.allowed_email_domain}"
        )

    try:
        sb = get_supabase()
        response = sb.auth.sign_in_with_otp({
            "email": request.email,
            "options": {
                "email_redirect_to": f"{settings.frontend_url}/auth/callback"
            }
        })
        return {
            "success": True,
            "message": f"Magic Link enviado a {request.email}. Revisa tu bandeja de entrada."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enviando Magic Link: {str(e)}")


@router.post("/verify")
async def verify_token(request: TokenVerifyRequest):
    """Verifica un token JWT y retorna info del usuario."""
    try:
        sb = get_supabase()
        user_response = sb.auth.get_user(request.access_token)
        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Token inválido")

        user = user_response.user
        # Buscar info adicional en tabla usuarios
        sb_admin = get_supabase_admin()
        profile = sb_admin.table("usuarios").select(
            "*, cedulas(nombre)"
        ).eq("email", user.email).maybe_single().execute()

        return {
            "authenticated": True,
            "email": user.email,
            "auth_id": user.id,
            "profile": profile.data if profile.data else None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token inválido: {str(e)}")


@router.post("/logout")
async def logout():
    """Cierra la sesión."""
    try:
        sb = get_supabase()
        sb.auth.sign_out()
        return {"success": True, "message": "Sesión cerrada"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
