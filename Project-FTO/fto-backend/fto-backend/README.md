# PMI Plattform — FTO Digital API

Backend FastAPI para la plataforma de gestión de indicadores FTO (SQDCM).

## Estructura del proyecto

```
fto-backend/
├── app/
│   ├── main.py              ← Entry point de FastAPI
│   ├── config.py             ← Configuración (variables de entorno)
│   ├── routers/
│   │   ├── auth.py           ← Login con Magic Link
│   │   ├── registros.py      ← CRUD registros semanales SQDCM
│   │   ├── dashboard.py      ← Resúmenes, tendencias, indicadores
│   │   └── equipos.py        ← Gestión de cédulas, LC, operadores
│   ├── models/
│   │   └── schemas.py        ← Validación de datos (Pydantic)
│   └── services/
│       ├── supabase_client.py ← Conexión a Supabase
│       └── auth.py           ← Verificación de JWT
├── requirements.txt
├── render.yaml               ← Config para deploy en Render
├── Procfile                  ← Alternativa para Render
├── .env.example              ← Template de variables de entorno
└── .gitignore
```

## Setup local

```bash
# 1. Clonar y entrar al proyecto
cd fto-backend

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales de Supabase

# 5. Correr el servidor
uvicorn app.main:app --reload --port 8000

# 6. Abrir docs interactivos
# http://localhost:8000/docs
```

## Deploy en Render

1. Subir este proyecto a GitHub
2. En Render → New Web Service → conectar el repo
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Agregar variables de entorno (Environment):
   - `SUPABASE_URL`
   - `SUPABASE_ANON_KEY`
   - `SUPABASE_SERVICE_KEY`
   - `ALLOWED_EMAIL_DOMAIN` = pmintl.net
   - `FRONTEND_URL` = (tu URL de Render)
   - `ENVIRONMENT` = production

## Endpoints principales

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/auth/magic-link` | Envía Magic Link al correo |
| POST | `/auth/verify` | Verifica token JWT |
| GET | `/registros/` | Lista registros con filtros |
| POST | `/registros/` | Crea registro semanal |
| POST | `/registros/batch` | Crea múltiples registros |
| PUT | `/registros/{id}` | Actualiza registro |
| GET | `/dashboard/resumen-lc` | Promedios por LC |
| GET | `/dashboard/operadores-semana` | Vista spreadsheet |
| GET | `/dashboard/tendencia` | Evolución de indicador |
| GET | `/equipos/cedulas` | Lista cédulas |
| GET | `/equipos/lc` | Lista Line Coordinators |
| GET | `/equipos/operadores` | Lista operadores |
