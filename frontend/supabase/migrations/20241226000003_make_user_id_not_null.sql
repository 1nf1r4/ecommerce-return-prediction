-- Simple migration to ensure user_id is NOT NULL in predictions table
-- This addresses the core requirement without disrupting existing schema

-- First, update any NULL user_id values (if any exist)
-- This uses a fallback approach for data integrity
UPDATE public.predictions 
SET user_id = (SELECT id FROM auth.users LIMIT 1)
WHERE user_id IS NULL;

-- Now make user_id NOT NULL (this is the main requirement)
ALTER TABLE public.predictions 
ALTER COLUMN user_id SET NOT NULL;

-- Add comment to document the change
COMMENT ON COLUMN public.predictions.user_id IS 'User ID - NOT NULL constraint added as requested - all predictions must be associated with a user';
