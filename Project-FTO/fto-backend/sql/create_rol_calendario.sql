-- ═══════════════════════════════════════════════════════════════
-- Tabla rol_calendario: Calendario de turnos/máquinas por operador
-- Cada fila = un operador en una fecha con su turno y máquina
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS rol_calendario (
  id serial PRIMARY KEY,
  operador_id uuid NOT NULL,          -- Puede ser operador, LC o LS
  cedula_id uuid NOT NULL,
  nombre text NOT NULL,               -- Nombre para referencia rápida
  kdf text,                           -- KDF07, KDF08, KDF07/KDF09, AUX, LC, Tecnico
  fecha date NOT NULL,
  turno text,                         -- D=Día(S1), T=Tarde(S2), N=Noche(S3), null=descanso
  es_override boolean DEFAULT false,  -- true si fue editado manualmente
  created_at timestamptz DEFAULT now(),
  UNIQUE(operador_id, fecha)
);

CREATE INDEX IF NOT EXISTS idx_rol_cedula_fecha ON rol_calendario (cedula_id, fecha);
CREATE INDEX IF NOT EXISTS idx_rol_operador_fecha ON rol_calendario (operador_id, fecha);
