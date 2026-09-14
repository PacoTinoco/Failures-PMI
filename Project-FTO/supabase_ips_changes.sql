-- ═══════════════════════════════════════════════════════════════
-- IPS Platform Changes — Run in Supabase SQL Editor
-- ═══════════════════════════════════════════════════════════════

-- 1. Add new columns to ips_records
ALTER TABLE ips_records
  ADD COLUMN IF NOT EXISTS turno_responsable TEXT,
  ADD COLUMN IF NOT EXISTS sig_accion TEXT;

-- 2. Migrate existing ubicacion data to turno_responsable (optional)
-- UPDATE ips_records SET turno_responsable = ubicacion WHERE ubicacion IS NOT NULL;

-- 3. Create weekly tracking table
CREATE TABLE IF NOT EXISTS ips_weekly_tracking (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  ips_id UUID NOT NULL REFERENCES ips_records(id) ON DELETE CASCADE,
  week_start DATE NOT NULL,  -- Always a Monday
  lunes TEXT CHECK (lunes IN ('1', '0', 'NA')),
  martes TEXT CHECK (martes IN ('1', '0', 'NA')),
  miercoles TEXT CHECK (miercoles IN ('1', '0', 'NA')),
  jueves TEXT CHECK (jueves IN ('1', '0', 'NA')),
  viernes TEXT CHECK (viernes IN ('1', '0', 'NA')),
  sabado TEXT CHECK (sabado IN ('1', '0', 'NA')),
  domingo TEXT CHECK (domingo IN ('1', '0', 'NA')),
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(ips_id, week_start)
);

-- 4. Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_ips_weekly_tracking_week
  ON ips_weekly_tracking(week_start);
CREATE INDEX IF NOT EXISTS idx_ips_weekly_tracking_ips
  ON ips_weekly_tracking(ips_id);

-- 5. Enable RLS (match existing pattern)
ALTER TABLE ips_weekly_tracking ENABLE ROW LEVEL SECURITY;

-- Allow all operations via service role (same as other tables)
CREATE POLICY "Allow all for service role" ON ips_weekly_tracking
  FOR ALL USING (true) WITH CHECK (true);
