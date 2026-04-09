-- IPS Tables for FTO Platform
-- Run in Supabase SQL Editor

-- Main IPS records
CREATE TABLE IF NOT EXISTS ips_records (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  cedula_id uuid REFERENCES cedulas(id) ON DELETE CASCADE,
  kdf integer NOT NULL,
  titulo text NOT NULL,
  fecha date,
  ubicacion text,  -- HUB, PIS, etc.
  participants text[] DEFAULT '{}',  -- Array of participant names
  section_6w2h boolean DEFAULT false,
  section_bbc boolean DEFAULT false,
  section_5w boolean DEFAULT false,
  section_res boolean DEFAULT false,
  status text DEFAULT 'Open',  -- Open, Closed, Cancelled, Merged, BCC, 6W2H, Ascended, Missing
  priority text,  -- High, Medium, Low
  notes text,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Countermeasures linked to IPS records
CREATE TABLE IF NOT EXISTS ips_countermeasures (
  id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  ips_id uuid REFERENCES ips_records(id) ON DELETE CASCADE,
  descripcion text NOT NULL,
  owner text,
  status text DEFAULT 'Pending',  -- Pending, Done, Cancelled, On going
  priority text,
  due_date date,
  notes text,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_ips_cedula ON ips_records(cedula_id);
CREATE INDEX IF NOT EXISTS idx_ips_kdf ON ips_records(kdf);
CREATE INDEX IF NOT EXISTS idx_ips_status ON ips_records(status);
CREATE INDEX IF NOT EXISTS idx_ips_cm_ips ON ips_countermeasures(ips_id);
CREATE INDEX IF NOT EXISTS idx_ips_cm_status ON ips_countermeasures(status);
