-- Add username_comitdb column to operador_aliases for matching ComitDB QFlags exports.
-- Example: 'jbernard4' → maps to operator "Bernardo Rodriguez Jahzzel Uriel"
ALTER TABLE operador_aliases
  ADD COLUMN IF NOT EXISTS username_comitdb text;

CREATE INDEX IF NOT EXISTS idx_aliases_username_comitdb
  ON operador_aliases (lower(username_comitdb));
