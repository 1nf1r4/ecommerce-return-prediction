-- Enable RLS (Row Level Security)
-- ALTER DATABASE postgres SET "app.jwt_secret" TO 'your-jwt-secret';

-- Create custom types
CREATE TYPE prediction_status AS ENUM ('pending', 'completed', 'failed');
CREATE TYPE user_role AS ENUM ('user', 'admin');

-- Create profiles table to extend auth.users
CREATE TABLE public.profiles (
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

-- Enable RLS on profiles
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Create policy for profiles
CREATE POLICY "Users can view own profile" ON public.profiles
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON public.profiles
    FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile" ON public.profiles
    FOR INSERT WITH CHECK (auth.uid() = id);

-- Create predictions table
CREATE TABLE public.predictions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    
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
    
    -- Prediction results
    predicted_return_probability DECIMAL(5,4),
    predicted_return_flag BOOLEAN,
    confidence_score DECIMAL(5,4),
    model_version VARCHAR(50) DEFAULT 'v1.0',
    
    -- Metadata
    status prediction_status DEFAULT 'pending',
    error_message TEXT,
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on predictions
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;

-- Create policies for predictions
CREATE POLICY "Users can view own predictions" ON public.predictions
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own predictions" ON public.predictions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own predictions" ON public.predictions
    FOR UPDATE USING (auth.uid() = user_id);

-- Create datasets table for bulk uploads
CREATE TABLE public.datasets (
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

-- Enable RLS on datasets
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

-- Create model_metrics table to track model performance
CREATE TABLE public.model_metrics (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    auc_score DECIMAL(5,4),
    training_date TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_predictions_user_id ON public.predictions(user_id);
CREATE INDEX idx_predictions_created_at ON public.predictions(created_at);
CREATE INDEX idx_predictions_status ON public.predictions(status);
CREATE INDEX idx_datasets_user_id ON public.datasets(user_id);
CREATE INDEX idx_profiles_email ON public.profiles(email);

-- Create function to automatically create profile on user signup
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
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create trigger to automatically create profile
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Create function to update updated_at timestamps
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at
CREATE TRIGGER profiles_updated_at
    BEFORE UPDATE ON public.profiles
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

CREATE TRIGGER predictions_updated_at
    BEFORE UPDATE ON public.predictions
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

CREATE TRIGGER datasets_updated_at
    BEFORE UPDATE ON public.datasets
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

-- Insert sample model metrics
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
);

-- Create storage bucket for file uploads
INSERT INTO storage.buckets (id, name, public) VALUES ('datasets', 'datasets', false);

-- Create policy for dataset files
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
