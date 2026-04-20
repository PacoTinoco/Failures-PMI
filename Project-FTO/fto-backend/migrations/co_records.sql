-- CO Records table for Changeover Analysis
-- Run this in Supabase SQL Editor

CREATE TABLE IF NOT EXISTS co_records (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  maquina text NOT NULL,
  fecha date,
  semana text,
  operador text NOT NULL,
  marca_termina text,
  marca_nueva text,
  formatos_completados text,
  analisis_cf numeric,
  razon_desviacion text,
  recomendacion text,
  tiempo_objetivo numeric,
  tiempo_real numeric,
  runtime_next_2h numeric,
  stops_next_2h integer,
  mtbf numeric,
  variacion_co numeric,
  desperdicio_hora2 numeric,
  created_at timestamptz DEFAULT now()
);

-- Index for common queries
CREATE INDEX IF NOT EXISTS idx_co_records_maquina ON co_records(maquina);
CREATE INDEX IF NOT EXISTS idx_co_records_operador ON co_records(operador);
CREATE INDEX IF NOT EXISTS idx_co_records_semana ON co_records(semana);
CREATE INDEX IF NOT EXISTS idx_co_records_fecha ON co_records(fecha);

-- Enable RLS (service_role key bypasses)
ALTER TABLE co_records ENABLE ROW LEVEL SECURITY;
