from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.routers import auth, registros, dashboard, equipos

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
app.include_router(auth.router)
app.include_router(registros.router)
app.include_router(dashboard.router)
app.include_router(equipos.router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "app": "PMI Plattform — FTO Digital",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }


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
