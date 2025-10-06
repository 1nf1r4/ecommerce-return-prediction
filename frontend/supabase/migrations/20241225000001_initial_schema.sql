-- Enable RLS (Row Level Security)
-- ALTER DATABASE postgres SET "app.jwt_secret" TO 'your-jwt-secret';

-- Create custom types (handle existing types gracefully)
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'prediction_status') THEN
        CREATE TYPE prediction_status AS ENUM ('pending', 'completed', 'failed');
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role') THEN
        CREATE TYPE user_role AS ENUM ('user', 'admin');
    END IF;
END $$;

-- Create profiles table to extend auth.users (handle existing table)
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    avatar_url VARCHAR(255),
    role user_role DEFAULT 'user',
    company_name VARCHAR(255),
    industry VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on profiles (handle existing)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies 
        WHERE schemaname = 'public' 
        AND tablename = 'profiles' 
        AND policyname = 'Users can view own profile'
    ) THEN
        ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
        
        -- Create policies for profiles
        CREATE POLICY "Users can view own profile" ON public.profiles
            FOR SELECT USING (auth.uid() = id);

        CREATE POLICY "Users can update own profile" ON public.profiles
            FOR UPDATE USING (auth.uid() = id);

        CREATE POLICY "Users can insert own profile" ON public.profiles
            FOR INSERT WITH CHECK (auth.uid() = id);
    END IF;
END $$;

-- Create predictions table (handle existing table)
CREATE TABLE IF NOT EXISTS public.predictions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,  -- Nullable for anonymous predictions
    
    -- Order identification
    order_id VARCHAR(255),
    
    -- Input features
    product_category INTEGER,
    product_price DECIMAL(10,2),
    order_quantity INTEGER,
    return_reason INTEGER,
    user_age INTEGER,
    user_gender INTEGER,
    payment_method INTEGER,
    shipping_method INTEGER,
    discount_applied DECIMAL(5,2),
    total_order_value DECIMAL(10,2),
    order_year INTEGER,
    order_month INTEGER,
    order_weekday INTEGER,
    user_location_num INTEGER,
    
    -- Human-readable fields for analytics
    risk_level VARCHAR(20),
    category_name VARCHAR(100),
    customer_gender VARCHAR(20),
    customer_location VARCHAR(50),
    customer_age INTEGER,
    payment_method_name VARCHAR(50),
    shipping_method_name VARCHAR(50),
    price DECIMAL(10,2),
    quantity INTEGER,
    
    -- Prediction results
    predicted_return_probability DECIMAL(5,4),
    predicted_return_flag BOOLEAN,
    confidence_score DECIMAL(5,4),
    model_version VARCHAR(50) DEFAULT 'v1.0',
    recommendations JSONB,  -- Business recommendations as JSON array
    
    -- Metadata
    status prediction_status DEFAULT 'pending',
    error_message TEXT,
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add comments (only if table was created)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'predictions' AND table_schema = 'public') THEN
        -- Add comment explaining nullable user_id
        COMMENT ON COLUMN public.predictions.user_id IS 'User ID - nullable for anonymous/system predictions';
        COMMENT ON TABLE public.predictions IS 'Stores all return predictions with input features and results';
    END IF;
END $$;

-- Enable RLS on predictions (handle existing)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies 
        WHERE schemaname = 'public' 
        AND tablename = 'predictions' 
        AND policyname = 'Users can view own predictions'
    ) THEN
        ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;
        
        -- Create policies for predictions
        CREATE POLICY "Users can view own predictions" ON public.predictions
            FOR SELECT USING (auth.uid() = user_id);

        CREATE POLICY "Users can insert own predictions" ON public.predictions
            FOR INSERT WITH CHECK (auth.uid() = user_id);

        CREATE POLICY "Users can update own predictions" ON public.predictions
            FOR UPDATE USING (auth.uid() = user_id);
    END IF;
END $$;

-- Create datasets table for bulk uploads (handle existing)
CREATE TABLE IF NOT EXISTS public.datasets (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    file_name VARCHAR(255),
    file_size INTEGER,
    row_count INTEGER,
    column_count INTEGER,
    upload_status VARCHAR(50) DEFAULT 'uploaded',
    processing_status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on datasets (handle existing)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies 
        WHERE schemaname = 'public' 
        AND tablename = 'datasets' 
        AND policyname = 'Users can view own datasets'
    ) THEN
        ALTER TABLE public.datasets ENABLE ROW LEVEL SECURITY;
        
        -- Create policies for datasets
        CREATE POLICY "Users can view own datasets" ON public.datasets
            FOR SELECT USING (auth.uid() = user_id);

        CREATE POLICY "Users can insert own datasets" ON public.datasets
            FOR INSERT WITH CHECK (auth.uid() = user_id);

        CREATE POLICY "Users can update own datasets" ON public.datasets
            FOR UPDATE USING (auth.uid() = user_id);

        CREATE POLICY "Users can delete own datasets" ON public.datasets
            FOR DELETE USING (auth.uid() = user_id);
    END IF;
END $$;

-- Create model_metrics table to track model performance (handle existing)
CREATE TABLE IF NOT EXISTS public.model_metrics (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL UNIQUE,
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    auc_score DECIMAL(5,4),
    training_date TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance (handle existing)
CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON public.predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON public.predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_status ON public.predictions(status);
CREATE INDEX IF NOT EXISTS idx_predictions_risk_level ON public.predictions(risk_level);
CREATE INDEX IF NOT EXISTS idx_predictions_order_id ON public.predictions(order_id);
CREATE INDEX IF NOT EXISTS idx_datasets_user_id ON public.datasets(user_id);
CREATE INDEX IF NOT EXISTS idx_profiles_email ON public.profiles(email);

-- Create function to automatically create profile on user signup (handle existing)
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email, full_name)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'full_name', NEW.email)
    );
    RETURN NEW;
EXCEPTION
    WHEN others THEN
        -- Profile may already exist, continue
        RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create trigger to automatically create profile (handle existing)
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Create function to update updated_at timestamps (handle existing)
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at (handle existing)
DROP TRIGGER IF EXISTS profiles_updated_at ON public.profiles;
CREATE TRIGGER profiles_updated_at
    BEFORE UPDATE ON public.profiles
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

DROP TRIGGER IF EXISTS predictions_updated_at ON public.predictions;
CREATE TRIGGER predictions_updated_at
    BEFORE UPDATE ON public.predictions
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

DROP TRIGGER IF EXISTS datasets_updated_at ON public.datasets;
CREATE TRIGGER datasets_updated_at
    BEFORE UPDATE ON public.datasets
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

-- Insert sample model metrics (handle existing)
INSERT INTO public.model_metrics (
    model_version, 
    accuracy, 
    precision_score, 
    recall, 
    f1_score, 
    auc_score, 
    training_date, 
    is_active
) VALUES (
    'v1.0', 
    0.8523, 
    0.8234, 
    0.8654, 
    0.8441, 
    0.9123, 
    NOW(), 
    true
) ON CONFLICT (model_version) DO UPDATE SET
    accuracy = EXCLUDED.accuracy,
    precision_score = EXCLUDED.precision_score,
    recall = EXCLUDED.recall,
    f1_score = EXCLUDED.f1_score,
    auc_score = EXCLUDED.auc_score,
    training_date = EXCLUDED.training_date,
    is_active = EXCLUDED.is_active;

-- Create storage bucket for file uploads (handle existing)
INSERT INTO storage.buckets (id, name, public) VALUES ('datasets', 'datasets', false)
ON CONFLICT (id) DO NOTHING;

-- Create policy for dataset files (handle existing)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies 
        WHERE schemaname = 'storage' 
        AND tablename = 'objects' 
        AND policyname = 'Users can upload own dataset files'
    ) THEN
        CREATE POLICY "Users can upload own dataset files" ON storage.objects
            FOR INSERT WITH CHECK (
                bucket_id = 'datasets' AND 
                auth.uid()::text = (storage.foldername(name))[1]
            );

        CREATE POLICY "Users can view own dataset files" ON storage.objects
            FOR SELECT USING (
                bucket_id = 'datasets' AND 
                auth.uid()::text = (storage.foldername(name))[1]
            );

        CREATE POLICY "Users can delete own dataset files" ON storage.objects
            FOR DELETE USING (
                bucket_id = 'datasets' AND 
                auth.uid()::text = (storage.foldername(name))[1]
            );
    END IF;
END $$;
