from supabase import create_client, Client
from app.config import get_settings

_client: Client | None = None
_admin_client: Client | None = None


def get_supabase() -> Client:
    """Cliente con anon key — para operaciones del usuario autenticado."""
    global _client
    if _client is None:
        s = get_settings()
        _client = create_client(s.supabase_url, s.supabase_anon_key)
    return _client


def get_supabase_admin() -> Client:
    """Cliente con service_role key — para operaciones admin (crear usuarios, etc)."""
    global _admin_client
    if _admin_client is None:
        s = get_settings()
        _admin_client = create_client(s.supabase_url, s.supabase_service_key)
    return _admin_client
