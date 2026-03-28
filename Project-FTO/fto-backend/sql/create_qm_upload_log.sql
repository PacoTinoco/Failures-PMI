-- Tabla para registrar historial de uploads de data semanal QM
-- Permite trackear cuántas veces se actualizó la data en la semana,
-- cuántos cambios hubo en cada upload, y qué competencias cambiaron.

CREATE TABLE IF NOT EXISTS qm_upload_log (
  id serial PRIMARY KEY,
  cedula_id uuid NOT NULL REFERENCES cedulas(id) ON DELETE CASCADE,
  semana date NOT NULL,
  uploaded_at timestamptz DEFAULT now(),
  total_records integer DEFAULT 0,
  new_records integer DEFAULT 0,
  changed_records integer DEFAULT 0,
  unchanged_records integer DEFAULT 0,
  changes_detail jsonb DEFAULT '[]'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_qm_upload_log_cedula_semana
  ON qm_upload_log (cedula_id, semana);
