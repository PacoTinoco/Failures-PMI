-- Add is_na column to weekly_values to support "Not Applicable" entries.
-- When is_na = true, actual_value is stored as 0 (placeholder) but UI/charts
-- should treat the cell as NA.
ALTER TABLE weekly_values
  ADD COLUMN IF NOT EXISTS is_na boolean NOT NULL DEFAULT false;
