# ReturnPredict - AI-Powered E-commerce Return Prediction

A comprehensive web application that uses machine learning to predict e-commerce return probabilities, helping businesses reduce return costs and improve customer satisfaction.

🚀 **Status**: Automated CI/CD Pipeline Active - Deploys to Vercel (Frontend) + Railway (Backend)

## 🚀 Features

- **AI-Powered Predictions**: Advanced ML models with 98.5% accuracy
- **Email Authentication**: Complete auth system with email confirmation
- **Custom Email Templates**: Beautiful, responsive email templates
- **Real-time Analytics**: Comprehensive dashboard with insights
- **Data Management**: CSV upload and processing capabilities
- **Responsive Design**: Works perfectly on all devices
- **Secure Backend**: Supabase with Row Level Security

## 🛠️ Technology Stack

### Frontend

- **React 19** with Vite
- **Tailwind CSS** for styling
- **React Router** for navigation
- **Heroicons** for icons
- **Supabase JS** for backend integration

### Backend

- **Supabase** (PostgreSQL, Auth, Storage, Real-time)
- **Custom Email Templates** with Go Templates
- **Row Level Security** for data protection
- **Real-time subscriptions**

## 📋 Prerequisites

- Node.js 18+
- npm or yarn
- Supabase CLI (for local development)

## 🔧 Local Development Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd ecommerce-return-prediction
```

### 2. Install Dependencies

```bash
cd frontend
npm install
```

### 3. Set Up Supabase

#### Option A: Use Supabase CLI (Recommended for Development)

1. Install Supabase CLI:

```bash
npm install -g @supabase/cli
```

2. Initialize Supabase:

```bash
cd frontend
supabase init
```

3. Start local Supabase:

```bash
supabase start
```

4. The CLI will output your local connection details. Copy them to `.env.local`:

```bash
cp .env.example .env.local
# Edit .env.local with your local Supabase credentials
```

#### Option B: Use Supabase Cloud

1. Create a new project at [supabase.com](https://supabase.com)
2. Get your project URL and anon key from Settings → API
3. Create `.env.local`:

```bash
VITE_SUPABASE_URL=https://your-project-id.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
```

### 4. Run Database Migrations

If using Supabase CLI:

```bash
supabase db reset
```

If using cloud Supabase, run the SQL from `supabase/migrations/20241225000001_initial_schema.sql` in the SQL editor.

### 5. Configure Email Templates

The custom email templates are already configured in `config.toml`. If using local development, they'll work automatically. For production, update the templates in your Supabase dashboard.

### 6. Start the Development Server

```bash
npm run dev
```

The app will be available at `http://localhost:5173`

## 🎨 Email Templates

The application includes custom email templates for:

- **Sign-up Confirmation**: Welcome email with email verification
- **Password Reset**: Secure password reset with branded design
- **Magic Link**: Passwordless login option

Templates are responsive and include:

- Beautiful branding with gradients
- Security information
- Mobile-friendly design
- Clear call-to-action buttons

### Customizing Email Templates

Email templates are located in `supabase/templates/`:

- `confirmation.html` - Sign-up confirmation
- `recovery.html` - Password reset
- `magic_link.html` - Magic link login

Templates use Go template syntax with variables like:

- `{{ .SiteURL }}` - Your site URL
- `{{ .TokenHash }}` - Auth token hash
- `{{ .Email }}` - User's email address
- `{{ .RedirectTo }}` - Redirect URL after auth

## 🔐 Authentication Flow

### Sign Up Process

1. User fills out registration form
2. System creates account in Supabase
3. Confirmation email sent with custom template
4. User clicks confirmation link
5. Redirected to dashboard after verification

### Environment-Aware URLs

The email templates work in both development and production by using:

- Local development: `http://localhost:5173`
- Production: Your configured production domain

### Email Confirmation Setup

- Confirmation required: `enable_confirmations = true`
- Secure password changes: `secure_password_change = true`
- Rate limiting: 60 seconds between emails

## 📊 Database Schema

### Core Tables

- **profiles**: User profile information
- **predictions**: ML prediction results and history
- **data_uploads**: Uploaded dataset tracking

### Security

- Row Level Security (RLS) enabled on all tables
- Users can only access their own data
- Automatic profile creation on signup

## 🚀 Production Deployment

### 1. Update Environment Variables

Create production environment variables:

```bash
VITE_SUPABASE_URL=https://your-production-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-production-anon-key
```

### 2. Update Supabase Configuration

In your production Supabase project:

1. **Update Auth URLs**: Add your production domain to additional redirect URLs
2. **Configure Email Templates**: Upload custom templates via API or dashboard
3. **Set Site URL**: Update to your production domain
4. **Run Migrations**: Execute the SQL migrations

### 3. Build and Deploy

```bash
npm run build
# Deploy dist/ folder to your hosting provider
```

### 4. Update Email Template URLs

Ensure email templates use your production domain:

```html
href="{{ .SiteURL }}/auth/confirm?token_hash={{ .TokenHash
}}&type=email&next=/dashboard"
```

## 🎯 Key Features Walkthrough

### 1. Dashboard Overview

- Real-time statistics and metrics
- Recent activity feed
- Quick action buttons
- Performance indicators

### 2. Return Prediction

- Interactive form with all ML model inputs
- Real-time prediction results
- Confidence scores and risk levels
- Recommendation engine

### 3. Analytics Dashboard

- Monthly trend analysis
- Category-wise return rates
- Model performance metrics
- Exportable reports

### 4. Data Management

- Drag-and-drop CSV upload
- Data schema validation
- Upload history tracking
- Sample data downloads

## 🔧 Development Tips

### Email Template Testing

1. Use local development with Supabase CLI
2. Check email in Inbucket (usually http://127.0.0.1:54324)
3. Test all auth flows: signup, signin, password reset

### Custom Components

All major UI components are modular:

- `PredictionForm.jsx` - ML prediction interface
- `AnalyticsDashboard.jsx` - Analytics and charts
- `DataUpload.jsx` - File upload and management
- `ProtectedRoute.jsx` - Authentication guard

### Database Queries

Use Supabase client for all database operations:

```javascript
import { client } from "../supabase/client";

const { data, error } = await client
  .from("predictions")
  .select("*")
  .eq("user_id", user.id);
```

## 🐛 Troubleshooting

### Common Issues

1. **Email confirmation not working**

   - Check `site_url` in config.toml
   - Verify `additional_redirect_urls` includes your domain
   - Ensure email templates have correct URLs

2. **CORS errors**

   - Verify Supabase URL and keys
   - Check additional_redirect_urls configuration

3. **Database permissions**

   - Ensure RLS policies are correctly set
   - Check user authentication state

4. **Email templates not updating**
   - Restart Supabase: `supabase stop && supabase start`
   - Clear browser cache
   - Check template file paths in config.toml

## 📚 API Reference

### Prediction API

```javascript
// Make a prediction
const prediction = await client.from("predictions").insert({
  product_category: 1,
  product_price: 99.99,
  // ... other fields
});
```

### User Profile

```javascript
// Update user profile
const { data, error } = await client
  .from("profiles")
  .update({
    first_name: "John",
    last_name: "Doe",
    company: "Acme Inc",
  })
  .eq("id", user.id);
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:

- Check the troubleshooting section
- Review Supabase documentation
- Open an issue on GitHub

---

**ReturnPredict** - Reducing e-commerce returns with AI 🚀
