-- Migration: add y_min, y_max, and band_size to weekly_indicators
-- Run this in Supabase SQL Editor

ALTER TABLE weekly_indicators
  ADD COLUMN IF NOT EXISTS y_min numeric,
  ADD COLUMN IF NOT EXISTS y_max numeric,
  ADD COLUMN IF NOT EXISTS band_size numeric;
