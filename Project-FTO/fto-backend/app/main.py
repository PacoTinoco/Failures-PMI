import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.config import get_settings
from app.routers import registros, dashboard, equipos, dh, qm, bos_qbos

settings = get_settings()

app = FastAPI(
    title="PMI Plattform — FTO Digital",
    description="API para la gestión de indicadores FTO (SQDCM) de Philip Morris International",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS — permitir requests desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_url,
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(registros.router)
app.include_router(dashboard.router)
app.include_router(equipos.router)
app.include_router(dh.router)
app.include_router(qm.router)
app.include_router(bos_qbos.router)


@app.get("/health", tags=["Health"])
async def health():
    from app.services.supabase_client import get_supabase_admin
    try:
        sb = get_supabase_admin()
        result = sb.table("cedulas").select("id").limit(1).execute()
        db_ok = True
    except Exception:
        db_ok = False

    return {
        "status": "healthy" if db_ok else "degraded",
        "database": "connected" if db_ok else "error",
        "environment": settings.environment
    }


# ── Servir frontend compilado (dist/) ───────────────────
# El build de React/Vite genera una carpeta "dist" con index.html y assets.
# FastAPI sirve esos archivos estáticos para que todo esté en una sola URL.
DIST_DIR = Path(__file__).resolve().parent.parent / "dist"

if DIST_DIR.exists():
    # Servir assets (JS, CSS, imágenes)
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")

    # Cualquier ruta que NO sea /api, /docs, /health, etc. → devuelve index.html
    # Esto permite que React Router maneje las rutas del frontend
    @app.get("/{full_path:path}")
    async def serve_frontend(request: Request, full_path: str):
        # Si el archivo existe en dist (favicon, etc.), servirlo directo
        file_path = DIST_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        # Si no, devolver index.html para que React Router maneje la ruta
        return FileResponse(DIST_DIR / "index.html")
else:
    # Si no hay dist/, mostrar la API info normal
    @app.get("/", tags=["Health"])
    async def root():
        return {
            "app": "PMI Plattform — FTO Digital",
            "status": "running",
            "version": "1.0.0",
            "docs": "/docs"
        }
