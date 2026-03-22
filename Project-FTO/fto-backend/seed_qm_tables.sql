-- ============================================================
-- TABLAS PARA QM (Qualification Management)
-- ============================================================

-- Calendario QM: referencia fija anual por empleado + competencia
CREATE TABLE IF NOT EXISTS qm_calendario (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    cedula_id uuid REFERENCES cedulas(id),
    employee varchar(200) NOT NULL,
    competency varchar(300) NOT NULL,
    role varchar(200),
    current_base int NOT NULL DEFAULT 0,
    target int NOT NULL DEFAULT 0,
    feb int,
    mar int,
    abr int,
    may int,
    jun int,
    jul int,
    ago int,
    sep int,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    UNIQUE(cedula_id, employee, competency)
);

-- Data semanal QM: se sube cada semana con los datos actualizados
CREATE TABLE IF NOT EXISTS qm_data_semanal (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    cedula_id uuid REFERENCES cedulas(id),
    semana date NOT NULL,
    employee varchar(200) NOT NULL,
    competency varchar(300) NOT NULL,
    role varchar(200),
    current_level int NOT NULL DEFAULT 0,
    target int NOT NULL DEFAULT 0,
    on_target int DEFAULT 0,
    created_at timestamptz DEFAULT now(),
    UNIQUE(cedula_id, semana, employee, competency)
);

-- Índices para consultas frecuentes
CREATE INDEX IF NOT EXISTS idx_qm_calendario_cedula ON qm_calendario(cedula_id);
CREATE INDEX IF NOT EXISTS idx_qm_calendario_employee ON qm_calendario(employee);
CREATE INDEX IF NOT EXISTS idx_qm_data_cedula_semana ON qm_data_semanal(cedula_id, semana);
CREATE INDEX IF NOT EXISTS idx_qm_data_employee ON qm_data_semanal(employee);
