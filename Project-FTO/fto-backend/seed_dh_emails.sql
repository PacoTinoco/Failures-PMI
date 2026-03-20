-- ============================================================
-- PASO 1: Agregar columna 'email' a la tabla operadores
-- ============================================================
ALTER TABLE operadores ADD COLUMN IF NOT EXISTS email varchar(255);

-- ============================================================
-- PASO 2: Poblar emails de los operadores existentes
-- ============================================================

-- LC Mayra
UPDATE operadores SET email = 'Milton.Gonzalez@pmi.com'   WHERE nombre = 'Gonzalez Nunez Milton Misael';
UPDATE operadores SET email = 'Fernando.Ramirez@pmi.com'   WHERE nombre = 'Ramirez Alvarado Fernando';
UPDATE operadores SET email = 'Juan.Vela@pmi.com'          WHERE nombre = 'Vela Torres Juan Eduardo';
UPDATE operadores SET email = 'Jose.Rodriguez@pmi.com'     WHERE nombre = 'Rodriguez Cabrera Jose Raul';
UPDATE operadores SET email = 'Nicolas.Solorzano@pmi.com'  WHERE nombre = 'Solorzano De la Cruz Nicolas';
UPDATE operadores SET email = 'Ivan.Najera@pmi.com'        WHERE nombre = 'Najera Ivan';

-- LC Pedro
UPDATE operadores SET email = 'Gabrielle.Urbina@pmi.com'   WHERE nombre = 'Urbina Pelayo Gabrielle';
UPDATE operadores SET email = 'Leonardo.Mejia@pmi.com'     WHERE nombre = 'Mejia Plascencia Leonardo Rafael';
UPDATE operadores SET email = 'Silvia.Flores@pmi.com'      WHERE nombre = 'Flores Diaz Silvia';
UPDATE operadores SET email = 'Marcos.Roblero@pmi.com'     WHERE nombre = 'Roblero Cruz Marcos Daniel';
UPDATE operadores SET email = 'Victor.Sanchez@pmi.com'     WHERE nombre = 'Sanchez Zuniga Victor Manuel';
UPDATE operadores SET email = 'Guadalupe.Dominguez@pmi.com' WHERE nombre = 'Dominguez Macias Guadalupe Monserrat';

-- LC Andres
UPDATE operadores SET email = 'Marco.Bautista@pmi.com'     WHERE nombre = 'Bautista Martinez Marco Antonio';
UPDATE operadores SET email = 'Beatriz.Guirao@pmi.com'     WHERE nombre = 'Guirao Morales Beatriz';
UPDATE operadores SET email = 'Israel.Landin@pmi.com'      WHERE nombre = 'Landin Tule Israel Atxel';
UPDATE operadores SET email = 'Edith.Preciado@pmi.com'     WHERE nombre = 'Preciado Velasco Edith Soledad';
UPDATE operadores SET email = 'Jahzzel.Bernardo@pmi.com'   WHERE nombre = 'Bernardo Rodriguez Jahzzel Uriel';
UPDATE operadores SET email = 'Julio.Martinez@pmi.com'     WHERE nombre = 'Martinez Robles Julio Cesar';

-- Mecánicos
UPDATE operadores SET email = 'Omar.Barajas@pmi.com'       WHERE nombre = 'Omar Barajas';
UPDATE operadores SET email = 'JulioCesar.FariasVillalvazo@pmi.com' WHERE nombre = 'Farias Julio Cesar';
UPDATE operadores SET email = 'Omar.Ocegueda@pmi.com'      WHERE nombre = 'Ocegueda Omar';

-- ============================================================
-- PASO 3: Crear LC "Maintenance Lead"
-- ============================================================
INSERT INTO line_coordinators (id, nombre, cedula_id, activo)
VALUES (
  'a0000001-0000-0000-0000-000000000005',
  'Maintenance Lead',
  'a0000000-0000-0000-0000-000000000001',
  true
)
ON CONFLICT (id) DO NOTHING;

-- ============================================================
-- PASO 4: Agregar nuevos operadores
-- ============================================================

-- Rafael Salazar → LC Pedro (a0000001-0000-0000-0000-000000000002)
INSERT INTO operadores (id, nombre, email, lc_id, cedula_id, activo)
VALUES (
  'b0000001-0000-0000-0000-000000000001',
  'Salazar Rafael',
  'Rafael.Salazar@pmi.com',
  'a0000001-0000-0000-0000-000000000002',
  'a0000000-0000-0000-0000-000000000001',
  true
)
ON CONFLICT (id) DO NOTHING;

-- Saul Enriquez → LC Maintenance Lead (a0000001-0000-0000-0000-000000000005)
INSERT INTO operadores (id, nombre, email, lc_id, cedula_id, activo)
VALUES (
  'b0000001-0000-0000-0000-000000000002',
  'Enriquez Saul',
  'Saul.Enriquez@pmi.com',
  'a0000001-0000-0000-0000-000000000005',
  'a0000000-0000-0000-0000-000000000001',
  true
)
ON CONFLICT (id) DO NOTHING;
