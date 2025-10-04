# Vercel Deployment Guide

This guide will help you deploy your E-commerce Return Prediction application to Vercel.

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Node.js**: Ensure you have Node.js 18+ installed
3. **Git Repository**: Your code should be in a Git repository (GitHub, GitLab, etc.)

## Project Structure

Your project is now configured for Vercel deployment with:

- **Frontend**: React/Vite application in `/frontend`
- **Backend**: FastAPI serverless functions in `/api`
- **Configuration**: `vercel.json` for deployment settings

## Deployment Steps

### 1. Push to Git Repository

Make sure all your changes are committed and pushed:

```bash
git add .
git commit -m "Configure for Vercel deployment"
git push origin main
```

### 2. Connect to Vercel

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click "New Project"
3. Import your Git repository
4. Select your project

### 3. Configure Project Settings

Vercel should automatically detect this as a full-stack application. If not:

1. **Framework Preset**: Select "Other" or "Vite"
2. **Root Directory**: Leave as root (.)
3. **Build Command**: `cd frontend && npm run build`
4. **Output Directory**: `frontend/dist`
5. **Install Command**: `cd frontend && npm install`

### 4. Set Environment Variables

In your Vercel project dashboard:

1. Go to **Settings** → **Environment Variables**
2. Add the following variables for **Production**, **Preview**, and **Development**:

```
SUPABASE_URL=https://xhyyahaspqvepixivzjf.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhoeXlhaGFzcHF2ZXBpeGl2empmIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1ODc5MDk2MywiZXhwIjoyMDc0MzY2OTYzfQ.e_UrkqNTi8fESQg9SOhHdPyPpUtDplbvRbMk7EH2tLI
DATABASE_URL=postgresql://postgres.xhyyahaspqvepixivzjf:uEDavZTzTYq#TP2@aws-1-ap-southeast-1.pooler.supabase.com:5432/postgres
API_HOST=0.0.0.0
API_PORT=8000
```

### 5. Deploy

1. Click **Deploy** to start the deployment
2. Vercel will:
   - Install frontend dependencies
   - Build the React application
   - Set up serverless functions for the API
   - Deploy everything

### 6. Test Your Deployment

After deployment:

1. **Frontend**: Visit your Vercel domain (e.g., `your-app.vercel.app`)
2. **API Health Check**: Test `your-app.vercel.app/api/health`
3. **API Documentation**: Visit `your-app.vercel.app/api/docs`

## Vercel CLI Deployment (Alternative)

You can also deploy using the Vercel CLI:

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy (first time)
vercel

# Deploy to production
vercel --prod
```

## File Structure After Configuration

```
ecommerce-return-prediction/
├── api/
│   └── index.py                 # Serverless function entry point
├── frontend/
│   ├── src/
│   ├── dist/                    # Build output (generated)
│   ├── package.json
│   └── vite.config.js
├── services/                    # Your FastAPI application
│   ├── main.py
│   ├── api/
│   ├── agents/
│   └── ...
├── vercel.json                  # Vercel configuration
├── requirements.txt             # Python dependencies
├── build.sh                     # Build script
└── VERCEL_ENV_SETUP.md         # Environment setup guide
```

## API Endpoints

Your API will be available at:

- **Base URL**: `https://your-app.vercel.app/api`
- **Health Check**: `/api/health`
- **Documentation**: `/api/docs`
- **Single Prediction**: `/api/predict/single`
- **Batch Processing**: `/api/predict/batch/upload`

## Troubleshooting

### Common Issues:

1. **Build Failures**: Check the build logs in Vercel dashboard
2. **API Errors**: Ensure all environment variables are set correctly
3. **CORS Issues**: CORS is configured for `*.vercel.app` domains
4. **Missing Dependencies**: Check `requirements.txt` and `package.json`

### Debugging:

1. **Function Logs**: Check the Functions tab in Vercel dashboard
2. **Build Logs**: Review build output for errors
3. **Environment Variables**: Verify all required variables are set

## Updating Your Deployment

To update your deployment:

1. Push changes to your Git repository
2. Vercel will automatically trigger a new deployment
3. Check the deployment status in your Vercel dashboard

## Custom Domain (Optional)

To use a custom domain:

1. Go to **Settings** → **Domains** in your Vercel project
2. Add your custom domain
3. Configure DNS settings as instructed
4. Update CORS settings if needed

## Performance Optimization

For better performance:

1. **Caching**: Vercel automatically handles static asset caching
2. **CDN**: Your app is served from Vercel's global CDN
3. **Serverless Functions**: API endpoints scale automatically
4. **Environment**: Use environment variables for configuration

## Security Considerations

1. **Environment Variables**: Never commit sensitive data to Git
2. **API Keys**: Use Vercel's environment variables for secrets
3. **CORS**: Configured for your domain only
4. **HTTPS**: Vercel provides HTTPS by default

Your application is now ready for production deployment on Vercel!
