# Environment Variables for Vercel Deployment

## Required Environment Variables

You need to set these in your Vercel project dashboard:

### Supabase Configuration

- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_SERVICE_KEY`: Your Supabase service role key

### Database Configuration

- `DATABASE_URL`: Your PostgreSQL database connection string

### API Configuration

- `API_HOST`: 0.0.0.0 (for Vercel)
- `API_PORT`: 8000
- `CORS_ORIGINS`: https://\*.vercel.app (will be set automatically)

### Optional Frontend Environment Variables

- `VITE_API_URL`: Will default to /api in production, http://localhost:8000 in development

## How to set environment variables in Vercel:

1. Go to your Vercel project dashboard
2. Click on "Settings" tab
3. Click on "Environment Variables" in the sidebar
4. Add each variable with its value
5. Make sure to set the environment (Production, Preview, Development)

## Example values (replace with your actual values):

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
DATABASE_URL=postgresql://user:password@host:port/database
```
