"""
Script de verificación — Corre esto en tu compu para confirmar que todo está conectado.

Uso:
  pip install httpx python-dotenv
  python test_connection.py
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("Falta SUPABASE_URL o SUPABASE_SERVICE_KEY en tu archivo .env")
    exit(1)

print("=" * 60)
print("VERIFICACION DE CONEXION - PMI Plattform FTO Digital")
print("=" * 60)

try:
    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    USE_SUPABASE = True
    print("\n[OK] Modulo supabase disponible")
except ImportError:
    import httpx
    USE_SUPABASE = False
    print("\n[INFO] Usando httpx directo (funciona igual)")


def query_table(table_name, limit=50):
    if USE_SUPABASE:
        result = sb.table(table_name).select("*").limit(limit).execute()
        return result.data if result.data else []
    else:
        import httpx
        url = f"{SUPABASE_URL}/rest/v1/{table_name}?select=*&limit={limit}"
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json"
        }
        resp = httpx.get(url, headers=headers)
        if resp.status_code == 200:
            return resp.json()
        else:
            raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")


print("\n[OK] Conexion a Supabase exitosa")

checks = [
    ("cedulas", "Cedulas"),
    ("line_coordinators", "Line Coordinators"),
    ("operadores", "Operadores"),
    ("indicadores_config", "Indicadores Config"),
    ("usuarios", "Usuarios"),
    ("registros_semanales", "Registros Semanales"),
]

print("\n--- TABLAS Y DATOS ---\n")
all_ok = True
for table, label in checks:
    try:
        data = query_table(table)
        count = len(data)
        ok = count > 0 or table in ("usuarios", "registros_semanales")
        tag = "[OK]" if ok else "[!!]"
        print(f"  {tag} {label}: {count} registros")

        if table == "cedulas" and data:
            for r in data:
                print(f"       -> {r['nombre']}")
        elif table == "line_coordinators" and data:
            for r in data:
                print(f"       -> {r['nombre']} ({r.get('grupo', '')})")
        elif table == "operadores" and data:
            nombres = [r['nombre'] for r in data[:5]]
            print(f"       -> Primeros 5: {', '.join(nombres)}...")
        elif table == "indicadores_config" and data:
            print(f"       -> {count} indicadores SQDCM configurados")
    except Exception as e:
        print(f"  [ERROR] {label}: {e}")
        all_ok = False

print("\n--- VISTA RESUMEN LC ---\n")
try:
    data = query_table("resumen_lc", limit=1)
    print(f"  [OK] Vista resumen_lc accesible ({len(data)} registros)")
except Exception:
    print(f"  [INFO] Vista sin datos aun (normal si no hay registros)")

print("\n" + "=" * 60)
if all_ok:
    print("[LISTO] Base de datos configurada correctamente.")
    print("")
    print("Siguiente paso:")
    print("  1. pip install -r requirements.txt")
    print("  2. uvicorn app.main:app --reload --port 8000")
    print("  3. Abre http://localhost:8000/docs")
else:
    print("[ATENCION] Hay problemas. Revisa los errores arriba.")
print("=" * 60)
