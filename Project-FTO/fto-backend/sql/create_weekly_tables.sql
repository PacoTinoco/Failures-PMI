-- ══════════════════════════════════════════════════════
-- WEEKLY DOS/DCS — Tablas para el módulo Weekly
-- ══════════════════════════════════════════════════════

-- 1. Categorías (DMS agrupadores)
CREATE TABLE IF NOT EXISTS weekly_categories (
  id          uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  cedula_id   uuid NOT NULL REFERENCES cedulas(id) ON DELETE CASCADE,
  name        text NOT NULL,
  display_order integer NOT NULL DEFAULT 0,
  created_at  timestamptz DEFAULT now()
);

-- 2. Indicadores individuales
CREATE TABLE IF NOT EXISTS weekly_indicators (
  id            uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  category_id   uuid NOT NULL REFERENCES weekly_categories(id) ON DELETE CASCADE,
  cedula_id     uuid NOT NULL REFERENCES cedulas(id) ON DELETE CASCADE,
  name          text NOT NULL,                    -- ej: "QIE # Quality Incidents"
  subtitle      text,                              -- ej: "KDF 10" o null si es global
  direction     text NOT NULL DEFAULT 'lower_better', -- 'higher_better' | 'lower_better'
  unit          text NOT NULL DEFAULT '#',         -- '%', '#', 'min', etc.
  display_order integer NOT NULL DEFAULT 0,
  created_at    timestamptz DEFAULT now()
);

-- 3. Targets por indicador/semana (la línea rojo/verde)
CREATE TABLE IF NOT EXISTS weekly_targets (
  id            uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  indicator_id  uuid NOT NULL REFERENCES weekly_indicators(id) ON DELETE CASCADE,
  cedula_id     uuid NOT NULL REFERENCES cedulas(id) ON DELETE CASCADE,
  year          integer NOT NULL,
  quarter       integer NOT NULL,                  -- 1,2,3,4
  week_number   integer NOT NULL,                  -- 1-15 dentro del trimestre
  target_value  numeric NOT NULL,
  UNIQUE(indicator_id, year, quarter, week_number)
);

-- 4. Valores reales capturados
CREATE TABLE IF NOT EXISTS weekly_values (
  id            uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  indicator_id  uuid NOT NULL REFERENCES weekly_indicators(id) ON DELETE CASCADE,
  cedula_id     uuid NOT NULL REFERENCES cedulas(id) ON DELETE CASCADE,
  year          integer NOT NULL,
  quarter       integer NOT NULL,
  week_number   integer NOT NULL,
  actual_value  numeric NOT NULL,
  auto_source   text,                              -- null = manual, 'frr', 'bos', etc.
  updated_at    timestamptz DEFAULT now(),
  UNIQUE(indicator_id, year, quarter, week_number)
);

-- Índices para consultas rápidas
CREATE INDEX IF NOT EXISTS idx_weekly_indicators_cedula ON weekly_indicators(cedula_id);
CREATE INDEX IF NOT EXISTS idx_weekly_indicators_category ON weekly_indicators(category_id);
CREATE INDEX IF NOT EXISTS idx_weekly_targets_indicator ON weekly_targets(indicator_id, year, quarter);
CREATE INDEX IF NOT EXISTS idx_weekly_values_indicator ON weekly_values(indicator_id, year, quarter);
CREATE INDEX IF NOT EXISTS idx_weekly_categories_cedula ON weekly_categories(cedula_id);

-- Habilitar RLS pero permitir todo (sin auth por ahora)
ALTER TABLE weekly_categories ENABLE ROW LEVEL SECURITY;
ALTER TABLE weekly_indicators ENABLE ROW LEVEL SECURITY;
ALTER TABLE weekly_targets ENABLE ROW LEVEL SECURITY;
ALTER TABLE weekly_values ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all on weekly_categories" ON weekly_categories FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all on weekly_indicators" ON weekly_indicators FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all on weekly_targets" ON weekly_targets FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all on weekly_values" ON weekly_values FOR ALL USING (true) WITH CHECK (true);
