# FTO Dashboard - Philip Morris International

Dashboard de indicadores SQDCM para el seguimiento semanal de operadores en la línea de Filtros Blancos.

## Tecnologías

- **Frontend**: Next.js 14, TypeScript, Tailwind CSS, Recharts
- **Backend**: Next.js API Routes
- **Base de datos**: PostgreSQL (Prisma ORM)
- **Autenticación**: NextAuth.js
- **Deploy**: Render

## Inicio Rápido (Desarrollo Local)

### Prerrequisitos
- Node.js 18+
- PostgreSQL (o usar Docker)

### Instalación

```bash
# Clonar e instalar dependencias
cd fto-dashboard
npm install

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tu DATABASE_URL y NEXTAUTH_SECRET

# Crear la base de datos y tablas
npx prisma db push

# Cargar datos iniciales (operadores, KPIs, datos de ejemplo)
npx prisma db seed

# Iniciar servidor de desarrollo
npm run dev
```

Abrir http://localhost:3000

### Credenciales de prueba
- **Admin**: admin@fto.com / admin123
- **Coordinador**: rvargas@fto.com / coord123

## Deploy en Render

### Opción 1: Blueprint (Recomendado)

1. Sube el proyecto a un repositorio de GitHub
2. Ve a [Render Dashboard](https://dashboard.render.com)
3. Click en "New" → "Blueprint"
4. Conecta tu repositorio
5. Render detectará el `render.yaml` y creará automáticamente:
   - Web Service (Next.js)
   - PostgreSQL Database
6. Agrega las variables de entorno en el grupo `fto-env`:
   - `NEXTAUTH_SECRET`: genera uno con `openssl rand -base64 32`
   - `NEXTAUTH_URL`: la URL de tu servicio en Render (ej: https://fto-dashboard.onrender.com)

### Opción 2: Manual

1. Crea un PostgreSQL en Render (Free tier)
2. Crea un Web Service:
   - Build: `npm install && npx prisma generate && npx prisma db push && npm run build`
   - Start: `npm start`
   - Variables: DATABASE_URL (de tu PostgreSQL), NEXTAUTH_SECRET, NEXTAUTH_URL

### Después del deploy
Ejecuta el seed para cargar datos iniciales:
```bash
# Desde tu máquina local, con DATABASE_URL apuntando a Render
npx prisma db seed
```

## Estructura del Proyecto

```
fto-dashboard/
├── prisma/
│   ├── schema.prisma          # Esquema de BD
│   └── seed.ts                # Datos iniciales
├── src/
│   ├── app/
│   │   ├── api/               # API endpoints
│   │   ├── dashboard/         # Dashboard principal
│   │   ├── upload/            # Carga de Excel
│   │   ├── capture/           # Captura manual
│   │   ├── history/           # Histórico
│   │   ├── admin/             # Gestión (admin)
│   │   └── login/             # Login
│   ├── components/
│   │   ├── charts/            # Gráficas (Recharts)
│   │   ├── forms/             # Formularios
│   │   ├── layout/            # Sidebar, Header
│   │   └── ui/                # KpiCard, etc.
│   ├── lib/
│   │   ├── auth.ts            # Config NextAuth
│   │   ├── prisma.ts          # Prisma client
│   │   └── excel-parser.ts    # Parser de Excel
│   └── types/                 # TypeScript types
├── render.yaml                # Config Render
└── package.json
```

## Indicadores SQDCM

| Categoría | Color | Indicadores |
|-----------|-------|-------------|
| **S** Sustentabilidad | Verde | BOS, BOS ENG |
| **Q** Calidad | Azul | QBOS, QBOS ENG, QFlags, QI/PNC |
| **D** Desempeño | Ámbar | DH encontrados, DH reparados, Curva Autonomía, Contramedidas |
| **C** Costo | Rojo | IPS, FRR, DIM WASTE, Sobrepeso, Eventos LAIKA |
| **M** Moral | Morado | Casos de estudio, QM On Target |

## Roles

- **Admin**: Acceso total, gestión de usuarios y máquinas
- **Coordinador**: Ve y edita datos de su grupo de operadores
- **Operador**: Ve sus propios datos y puede capturar manualmente

## Funcionalidades

- Dashboard interactivo con gráficas de tendencia
- Semáforo visual (verde/amarillo/rojo) vs objetivos
- Carga masiva de datos vía Excel
- Captura manual de datos semanales
- Histórico con filtros y exportación a CSV
- Comparativo entre operadores
- Multi-máquina escalable
