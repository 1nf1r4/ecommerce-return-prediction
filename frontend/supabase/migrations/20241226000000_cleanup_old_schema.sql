-- Migration script to clean up old schema and apply new streamlined schema
-- This migration addresses the issues with the previous schema:
-- 1. Makes user_id NOT NULL in predictions table
-- 2. Removes unnecessary/redundant fields
-- 3. Aligns with actual model features
-- 4. Adds proper business analytics tables
-- 5. Creates reference tables for better data management

-- First, let's check if we need to backup any existing data
-- (In production, you would backup before running this)

-- Drop existing policies first
DROP POLICY IF EXISTS "Users can view own predictions" ON public.predictions;
DROP POLICY IF EXISTS "Users can insert own predictions" ON public.predictions;
DROP POLICY IF EXISTS "Users can update own predictions" ON public.predictions;

-- Drop existing triggers and functions related to predictions
DROP TRIGGER IF EXISTS predictions_updated_at ON public.predictions;

-- Drop the existing predictions table (this will lose existing data - backup first in production!)
DROP TABLE IF EXISTS public.predictions CASCADE;

-- Drop business_metrics if it exists (from old schema)
DROP TABLE IF EXISTS public.business_metrics CASCADE;

-- Drop product_analytics if it exists
DROP TABLE IF EXISTS public.product_analytics CASCADE;

-- Drop model_metrics table to recreate with better structure
DROP TABLE IF EXISTS public.model_metrics CASCADE;

-- Now create the new streamlined schema
-- (The new schema is defined in 20241226000001_revised_schema.sql)

-- Note: This is a destructive migration. In production, you would:
-- 1. Backup existing data
-- 2. Migrate data to new structure
-- 3. Validate the migration
-- 4. Only then drop old tables
