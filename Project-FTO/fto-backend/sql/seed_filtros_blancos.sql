-- ═══════════════════════════════════════════════════════════════
-- Seed: Cédula "Filtros Blancos"
-- ═══════════════════════════════════════════════════════════════
-- Crea la cédula, sus 3 Line Coordinators, los 13 operadores y
-- los 5 registros de linea_estructura (1 Line Lead placeholder,
-- 3 mecánicos reales y 1 Process Lead placeholder).
--
-- KDFs que usa Filtros Blancos (referencia — asignación operador↔máquina
-- se hace después desde la UI de Administración):
--   • KDF1
--   • KDF3y6  (KDF 3 y 6 van juntas)
--   • KDF4
--   • KDF18
--
-- Idempotente: usa NOT EXISTS en cada INSERT, así puedes correrlo
-- múltiples veces sin duplicar registros.
-- ═══════════════════════════════════════════════════════════════

-- ─── 1. Cédula ──────────────────────────────────────────────────
INSERT INTO cedulas (nombre)
SELECT 'Filtros Blancos'
WHERE NOT EXISTS (SELECT 1 FROM cedulas WHERE nombre = 'Filtros Blancos');

-- ─── 2. Line Coordinators ───────────────────────────────────────
-- (Nuevos registros; son personas distintas a las de Cápsula)
WITH ced AS (SELECT id FROM cedulas WHERE nombre = 'Filtros Blancos' LIMIT 1)
INSERT INTO line_coordinators (nombre, cedula_id)
SELECT v.nombre, (SELECT id FROM ced)
FROM (VALUES
  ('Ricardo Vargas'),
  ('César Lara'),
  ('Miguel Vidrio')
) AS v(nombre)
WHERE NOT EXISTS (
  SELECT 1 FROM line_coordinators lc
  WHERE lc.nombre = v.nombre
    AND lc.cedula_id = (SELECT id FROM ced)
);

-- ─── 3. Línea estructura (Line Lead, Maintenance, Process Lead) ─
WITH ced AS (SELECT id FROM cedulas WHERE nombre = 'Filtros Blancos' LIMIT 1)
INSERT INTO linea_estructura (nombre, cedula_id, rol)
SELECT v.nombre, (SELECT id FROM ced), v.rol
FROM (VALUES
  ('Line Lead — Filtros Blancos',    'Line Lead'),
  ('Gilberto Magaña',                'Maintenance Lead'),
  ('Gilberto Alvarez',               'Maintenance Lead'),
  ('Felipe García',                  'Maintenance Lead'),
  ('Process Lead — Filtros Blancos', 'Process Lead')
) AS v(nombre, rol)
WHERE NOT EXISTS (
  SELECT 1 FROM linea_estructura le
  WHERE le.nombre = v.nombre
    AND le.cedula_id = (SELECT id FROM ced)
);

-- ─── 4. Operadores ──────────────────────────────────────────────
-- 13 operadores: máquina y turno quedan NULL (se asignan después).
-- Agrupados por su LC correspondiente.

-- 4a. Operadores de Ricardo Vargas
WITH
  ced AS (SELECT id FROM cedulas WHERE nombre = 'Filtros Blancos' LIMIT 1),
  lc  AS (
    SELECT id FROM line_coordinators
    WHERE nombre = 'Ricardo Vargas'
      AND cedula_id = (SELECT id FROM ced)
    LIMIT 1
  )
INSERT INTO operadores (nombre, lc_id, cedula_id, activo)
SELECT v.nombre, (SELECT id FROM lc), (SELECT id FROM ced), true
FROM (VALUES
  ('Cristian Quintanilla'),
  ('Diana Magaña'),
  ('Luis Ramírez'),
  ('Miriam Ramírez'),
  ('Erick Macias')
) AS v(nombre)
WHERE NOT EXISTS (
  SELECT 1 FROM operadores o
  WHERE o.nombre = v.nombre
    AND o.cedula_id = (SELECT id FROM ced)
);

-- 4b. Operadores de César Lara
WITH
  ced AS (SELECT id FROM cedulas WHERE nombre = 'Filtros Blancos' LIMIT 1),
  lc  AS (
    SELECT id FROM line_coordinators
    WHERE nombre = 'César Lara'
      AND cedula_id = (SELECT id FROM ced)
    LIMIT 1
  )
INSERT INTO operadores (nombre, lc_id, cedula_id, activo)
SELECT v.nombre, (SELECT id FROM lc), (SELECT id FROM ced), true
FROM (VALUES
  ('Alexis Cruz'),
  ('Ramiro Mendez'),
  ('Victoria Karman'),
  ('Manuel Martinez')
) AS v(nombre)
WHERE NOT EXISTS (
  SELECT 1 FROM operadores o
  WHERE o.nombre = v.nombre
    AND o.cedula_id = (SELECT id FROM ced)
);

-- 4c. Operadores de Miguel Vidrio
WITH
  ced AS (SELECT id FROM cedulas WHERE nombre = 'Filtros Blancos' LIMIT 1),
  lc  AS (
    SELECT id FROM line_coordinators
    WHERE nombre = 'Miguel Vidrio'
      AND cedula_id = (SELECT id FROM ced)
    LIMIT 1
  )
INSERT INTO operadores (nombre, lc_id, cedula_id, activo)
SELECT v.nombre, (SELECT id FROM lc), (SELECT id FROM ced), true
FROM (VALUES
  ('Jonathan Mares'),
  ('Roberto Mora'),
  ('Violeta Navarro'),
  ('Luis Vazquez')
) AS v(nombre)
WHERE NOT EXISTS (
  SELECT 1 FROM operadores o
  WHERE o.nombre = v.nombre
    AND o.cedula_id = (SELECT id FROM ced)
);

-- ─── 5. Verificación (opcional, solo imprime conteos) ───────────
-- Descomenta si quieres ver el resumen al terminar:
-- SELECT
--   (SELECT COUNT(*) FROM cedulas            WHERE nombre = 'Filtros Blancos')                                                     AS cedula,
--   (SELECT COUNT(*) FROM line_coordinators  WHERE cedula_id = (SELECT id FROM cedulas WHERE nombre = 'Filtros Blancos'))           AS line_coordinators,
--   (SELECT COUNT(*) FROM linea_estructura   WHERE cedula_id = (SELECT id FROM cedulas WHERE nombre = 'Filtros Blancos'))           AS linea_estructura,
--   (SELECT COUNT(*) FROM operadores         WHERE cedula_id = (SELECT id FROM cedulas WHERE nombre = 'Filtros Blancos'));
