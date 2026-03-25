-- ═══════════════════════════════════════════════════════════════
-- Tabla operador_aliases: Mapeo de variaciones de nombre/email
-- por sistema fuente (BOS CSV, QBOS Excel, DH, etc.)
-- Soporta operadores, LCs y miembros de Línea de Estructura.
-- ═══════════════════════════════════════════════════════════════

-- Si ya existe la tabla anterior (con FK a operadores), eliminarla
DROP TABLE IF EXISTS operador_aliases;

CREATE TABLE operador_aliases (
  id serial PRIMARY KEY,
  persona_id uuid NOT NULL,          -- ID del operador, LC o LS (sin FK para flexibilidad)
  persona_tipo text NOT NULL DEFAULT 'operador',  -- 'operador', 'lc', 'ls'
  nombre_bd text,                     -- Nombre tal cual está en la BD
  email_bos text,                     -- Email que aparece en el CSV de BOS
  nombre_qbos text,                   -- Nombre que aparece en el Excel de QBOS
  email_dh text,                      -- Email que aparece en reportes de DH
  created_at timestamptz DEFAULT now(),
  UNIQUE(persona_id)
);

-- Índices para búsquedas rápidas por alias
CREATE INDEX IF NOT EXISTS idx_aliases_email_bos ON operador_aliases (lower(email_bos));
CREATE INDEX IF NOT EXISTS idx_aliases_nombre_qbos ON operador_aliases (lower(nombre_qbos));
CREATE INDEX IF NOT EXISTS idx_aliases_email_dh ON operador_aliases (lower(email_dh));
CREATE INDEX IF NOT EXISTS idx_aliases_persona_tipo ON operador_aliases (persona_tipo);

-- ═══════════════════════════════════════════════════════════════
-- Actualizar emails principales en tabla operadores
-- ═══════════════════════════════════════════════════════════════

UPDATE operadores SET email = 'Eduardo.Vela@pmi.com' WHERE nombre = 'Vela Torres Juan Eduardo' AND (email IS NULL OR email != 'Eduardo.Vela@pmi.com');
UPDATE operadores SET email = 'JoseRaul.Rodriguez@pmi.com' WHERE nombre = 'Rodriguez Cabrera Jose Raul' AND (email IS NULL OR email != 'JoseRaul.Rodriguez@pmi.com');
UPDATE operadores SET email = 'JahzzelUriel.Bernardo@pmi.com' WHERE nombre = 'Bernardo Rodriguez Jahzzel Uriel' AND (email IS NULL OR email != 'JahzzelUriel.Bernardo@pmi.com');
UPDATE operadores SET email = 'EdithSoledad.Preciado@pmi.com' WHERE nombre = 'Preciado Velasco Edith Soledad' AND (email IS NULL OR email != 'EdithSoledad.Preciado@pmi.com');
UPDATE operadores SET email = 'JulioCesar.Martinez@pmi.com' WHERE nombre = 'Martinez Robles Julio Cesar' AND (email IS NULL OR email != 'JulioCesar.Martinez@pmi.com');
UPDATE operadores SET email = 'Atxel.Landin@pmi.com' WHERE nombre = 'Landin Tule Israel Atxel' AND (email IS NULL OR email != 'Atxel.Landin@pmi.com');
UPDATE operadores SET email = 'Victor.Zuniga@pmi.com' WHERE nombre = 'Sanchez Zuniga Victor Manuel' AND (email IS NULL OR email != 'Victor.Zuniga@pmi.com');
