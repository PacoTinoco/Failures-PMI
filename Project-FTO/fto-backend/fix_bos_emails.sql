-- ============================================================
-- FIX 1: Corregir emails que difieren entre BD y BOS CSV
-- El CSV de BOS usa el email real de login
-- ============================================================

UPDATE operadores SET email = 'Eduardo.Vela@pmi.com'          WHERE nombre = 'Vela Torres Juan Eduardo';
UPDATE operadores SET email = 'JoseRaul.Rodriguez@pmi.com'    WHERE nombre = 'Rodriguez Cabrera Jose Raul';
UPDATE operadores SET email = 'EdithSoledad.Preciado@pmi.com' WHERE nombre = 'Preciado Velasco Edith Soledad';
UPDATE operadores SET email = 'JahzzelUriel.Bernardo@pmi.com' WHERE nombre = 'Bernardo Rodriguez Jahzzel Uriel';
UPDATE operadores SET email = 'JulioCesar.Martinez@pmi.com'   WHERE nombre = 'Martinez Robles Julio Cesar';
UPDATE operadores SET email = 'Atxel.Landin@pmi.com'          WHERE nombre = 'Landin Tule Israel Atxel';
UPDATE operadores SET email = 'Victor.Zuniga@pmi.com'         WHERE nombre = 'Sanchez Zuniga Victor Manuel';

-- ============================================================
-- FIX 2: Agregar email a LCs que ya están como operadores
-- (pueden no existir como operadores — los UPDATE no afectan si no encuentran match)
-- ============================================================

-- Estos son LCs que también hacen BOS. Si ya existen como operadores, se actualiza su email.
-- Si NO existen, el UPDATE simplemente no hace nada (0 rows affected).
UPDATE operadores SET email = 'Andres.Baltazar@pmi.com'       WHERE lower(nombre) LIKE '%baltazar%andres%' OR lower(nombre) LIKE '%andres%baltazar%';
UPDATE operadores SET email = 'Pedro.Aguilar@pmi.com'         WHERE lower(nombre) LIKE '%aguilar%pedro%' OR lower(nombre) LIKE '%pedro%aguilar%';
UPDATE operadores SET email = 'Mayra.Hernandez@pmi.com'       WHERE lower(nombre) LIKE '%hernandez%mayra%' OR lower(nombre) LIKE '%mayra%hernandez%';
UPDATE operadores SET email = 'Bernardette.Olvera@pmi.com'    WHERE lower(nombre) LIKE '%bernardette%olvera%' OR lower(nombre) LIKE '%olvera%bernardette%';
UPDATE operadores SET email = 'Oscar.Pellegrini@pmi.com'      WHERE lower(nombre) LIKE '%oscar%pellegrini%' OR lower(nombre) LIKE '%pellegrini%oscar%';

-- ============================================================
-- FIX 3: Si las 5 personas de arriba NO existen como operadores,
-- crearlos. Necesitan un LC asignado.
-- Descomenta y ajusta si es necesario:
-- ============================================================

-- Obtener IDs de LCs para asignar
-- SELECT id, nombre FROM line_coordinators WHERE cedula_id = 'a0000000-0000-0000-0000-000000000001';

-- INSERT INTO operadores (id, nombre, email, lc_id, cedula_id, turno, maquina, activo) VALUES
-- (gen_random_uuid(), 'Baltazar Hernandez Andres', 'Andres.Baltazar@pmi.com', 'LC_ID_AQUI', 'a0000000-0000-0000-0000-000000000001', 'A', NULL, true),
-- (gen_random_uuid(), 'Aguilar Jimenez Pedro', 'Pedro.Aguilar@pmi.com', 'LC_ID_AQUI', 'a0000000-0000-0000-0000-000000000001', 'A', NULL, true),
-- (gen_random_uuid(), 'Hernandez Martinez Mayra Grisel', 'Mayra.Hernandez@pmi.com', 'LC_ID_AQUI', 'a0000000-0000-0000-0000-000000000001', 'A', NULL, true),
-- (gen_random_uuid(), 'Bernardette Olvera', 'Bernardette.Olvera@pmi.com', 'LC_ID_AQUI', 'a0000000-0000-0000-0000-000000000001', 'A', NULL, true),
-- (gen_random_uuid(), 'Oscar Pellegrini', 'Oscar.Pellegrini@pmi.com', 'LC_ID_AQUI', 'a0000000-0000-0000-0000-000000000001', 'A', NULL, true);
