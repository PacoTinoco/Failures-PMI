-- ═══════════════════════════════════════════════════════════════
-- Tabla linea_estructura (LS): Personal de estructura que no es
-- operador ni LC pero forma parte de la cédula.
-- Ej: Process Lead, Line Lead, Maintenance Lead
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS linea_estructura (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  nombre text NOT NULL,
  cedula_id uuid REFERENCES cedulas(id) ON DELETE CASCADE,
  rol text,                       -- 'Process Lead', 'Line Lead', 'Maintenance Lead', etc.
  email text,
  turno text,
  activo boolean DEFAULT true,
  created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ls_cedula ON linea_estructura (cedula_id);
