-- =============================================================
-- SEED: Filtros Cápsula — Operadores y LCs reales
-- La cédula "Filtros Cápsula" ya existe con este ID, no se toca.
-- =============================================================

-- Limpiar operadores y LCs viejos de esta cédula
DELETE FROM operadores WHERE cedula_id = 'a0000000-0000-0000-0000-000000000001';
DELETE FROM line_coordinators WHERE cedula_id = 'a0000000-0000-0000-0000-000000000001';

-- LINE COORDINATORS (4)
INSERT INTO line_coordinators (id, nombre, cedula_id, turno, grupo, activo) VALUES
  ('a0000001-0000-0000-0000-000000000001', 'Hernandez Martinez Mayra Grisel', 'a0000000-0000-0000-0000-000000000001', NULL, 'Grupo Mayra', true),
  ('a0000001-0000-0000-0000-000000000002', 'Aguilar Jimenez Pedro', 'a0000000-0000-0000-0000-000000000001', NULL, 'Grupo Pedro', true),
  ('a0000001-0000-0000-0000-000000000003', 'Baltazar Hernandez Andres', 'a0000000-0000-0000-0000-000000000001', NULL, 'Grupo Andres', true),
  ('a0000001-0000-0000-0000-000000000004', 'Mecanicos', 'a0000000-0000-0000-0000-000000000001', NULL, 'Mecanicos', true);

-- OPERADORES — LC Mayra (6)
INSERT INTO operadores (nombre, cedula_id, lc_id, turno, maquina, activo) VALUES
  ('Gonzalez Nunez Milton Misael',   'a0000000-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000001', NULL, NULL, true),
  ('Ramirez Alvarado Fernando',      'a0000000-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000001', NULL, NULL, true),
  ('Vela Torres Juan Eduardo',       'a0000000-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000001', NULL, NULL, true),
  ('Rodriguez Cabrera Jose Raul',    'a0000000-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000001', NULL, NULL, true),
  ('Solorzano De la Cruz Nicolas',   'a0000000-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000001', NULL, NULL, true),
  ('Najera Ivan',                    'a0000000-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000001', NULL, NULL, true);

-- OPERADORES — LC Pedro (6)
INSERT INTO operadores (nombre, cedula_id, lc_id, turno, maquina, activo) VALUES
  ('Urbina Pelayo Gabrielle',                'a0000000-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000002', NULL, NULL, true),
  ('Mejia Plascencia Leonardo Rafael',       'a0000000-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000002', NULL, NULL, true),
  ('Flores Diaz Silvia',                     'a0000000-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000002', NULL, NULL, true),
  ('Roblero Cruz Marcos Daniel',             'a0000000-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000002', NULL, NULL, true),
  ('Sanchez Zuniga Victor Manuel',           'a0000000-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000002', NULL, NULL, true),
  ('Dominguez Macias Guadalupe Monserrat',   'a0000000-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000002', NULL, NULL, true);

-- OPERADORES — LC Andres (6)
INSERT INTO operadores (nombre, cedula_id, lc_id, turno, maquina, activo) VALUES
  ('Bautista Martinez Marco Antonio',    'a0000000-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000003', NULL, NULL, true),
  ('Guirao Morales Beatriz',             'a0000000-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000003', NULL, NULL, true),
  ('Landin Tule Israel Atxel',           'a0000000-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000003', NULL, NULL, true),
  ('Preciado Velasco Edith Soledad',     'a0000000-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000003', NULL, NULL, true),
  ('Bernardo Rodriguez Jahzzel Uriel',   'a0000000-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000003', NULL, NULL, true),
  ('Martinez Robles Julio Cesar',        'a0000000-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000003', NULL, NULL, true);

-- MECÁNICOS (3)
INSERT INTO operadores (nombre, cedula_id, lc_id, turno, maquina, activo) VALUES
  ('Omar Barajas',          'a0000000-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000004', NULL, NULL, true),
  ('Farias Julio Cesar',    'a0000000-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000004', NULL, NULL, true),
  ('Ocegueda Omar',         'a0000000-0000-0000-0000-000000000001', 'a0000001-0000-0000-0000-000000000004', NULL, NULL, true);

-- Verificar resultado
SELECT 'LCs:' as tipo, count(*) as total FROM line_coordinators WHERE cedula_id = 'a0000000-0000-0000-0000-000000000001' AND activo = true
UNION ALL
SELECT 'Operadores:', count(*) FROM operadores WHERE cedula_id = 'a0000000-0000-0000-0000-000000000001' AND activo = true;
-- Esperado: LCs = 4, Operadores = 21
