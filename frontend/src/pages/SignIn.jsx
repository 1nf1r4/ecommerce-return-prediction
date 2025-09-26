import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";
import {
  EyeIcon,
  EyeSlashIcon,
  ShoppingBagIcon,
  ChartBarIcon,
  BoltIcon,
  CpuChipIcon,
} from "@heroicons/react/24/outline";

export default function SignIn() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [message, setMessage] = useState(null);
  const [useMagicLink, setUseMagicLink] = useState(false);

  const { signIn, signInWithMagicLink } = useAuth();
  const navigate = useNavigate();

  const handleSignIn = async (e) => {
    e.preventDefault();
    setError(null);
    setMessage(null);
    setLoading(true);

    try {
      if (useMagicLink) {
        const { error } = await signInWithMagicLink(email);
        if (error) {
          setError(error.message);
        } else {
          setMessage("Check your email for a magic link to sign in!");
        }
      } else {
        const { data, error } = await signIn(email, password);
        if (error) {
          setError(error.message);
        } else if (data.user) {
          navigate("/dashboard");
        }
      }
    } catch {
      setError("An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex">
      {/* Left Panel - Branding */}
      <div className="hidden lg:flex lg:w-1/2 bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 relative overflow-hidden">
        {/* Background decoration */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-8 -left-8 w-48 h-48 bg-gradient-to-br from-indigo-200/40 to-purple-200/40 rounded-full blur-3xl"></div>
          <div className="absolute -bottom-8 -right-8 w-64 h-64 bg-gradient-to-br from-purple-200/40 to-pink-200/40 rounded-full blur-3xl"></div>
          <div className="absolute top-1/2 left-1/4 w-32 h-32 bg-gradient-to-br from-blue-200/30 to-indigo-200/30 rounded-full blur-2xl"></div>
        </div>
        
        <div className="relative z-10 flex flex-col justify-center items-center p-12 w-full">
          <div className="flex flex-col items-center w-full max-w-lg mx-auto">
            <div className="mb-8 p-4 bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg">
              <ShoppingBagIcon className="h-16 w-16 text-indigo-600" />
            </div>
            <h1 className="text-5xl font-bold mb-6 text-center bg-gradient-to-r from-indigo-700 to-purple-700 bg-clip-text text-transparent">
              ReturnPredict
            </h1>
            <p className="text-xl text-center leading-relaxed text-gray-700 font-medium mb-12">
              AI-Powered E-commerce Return Prediction Platform
            </p>
            
            <div className="grid grid-cols-1 gap-6 w-full max-w-sm">
            <div className="flex items-center space-x-4 bg-white/70 backdrop-blur-sm rounded-xl p-4 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
              <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
                <CpuChipIcon className="h-6 w-6 text-white" />
              </div>
              <span className="font-semibold text-gray-800">Advanced ML Models</span>
            </div>
            <div className="flex items-center space-x-4 bg-white/70 backdrop-blur-sm rounded-xl p-4 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
                <ChartBarIcon className="h-6 w-6 text-white" />
              </div>
              <span className="font-semibold text-gray-800">Real-time Analytics</span>
            </div>
            <div className="flex items-center space-x-4 bg-white/70 backdrop-blur-sm rounded-xl p-4 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
              <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl flex items-center justify-center shadow-lg">
                <BoltIcon className="h-6 w-6 text-white" />
              </div>
              <span className="font-semibold text-gray-800">Instant Predictions</span>
            </div>
          </div>
        </div>
      </div>
      </div>

      {/* Right Panel - Sign In Form */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-8 bg-white">
        <div className="max-w-md w-full">
          <div className="text-center mb-8">
            <div className="lg:hidden flex justify-center mb-6">
              <div className="p-3 bg-indigo-100 rounded-2xl">
                <ShoppingBagIcon className="h-10 w-10 text-indigo-600" />
              </div>
            </div>
            <h2 className="text-3xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent mb-3">
              Welcome back
            </h2>
            <p className="text-gray-600 text-lg">
              Sign in to your ReturnPredict account
            </p>
          </div>

          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg">
              {error}
            </div>
          )}

          {message && (
            <div className="mb-6 p-4 bg-green-50 border border-green-200 text-green-700 rounded-lg">
              {message}
            </div>
          )}

          <form onSubmit={handleSignIn} className="space-y-6">
            <div>
              <label
                htmlFor="email"
                className="block text-sm font-medium text-gray-700 mb-2"
              >
                Email address
              </label>
              <input
                id="email"
                name="email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-3 py-3 border border-gray-300 rounded-lg shadow-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                placeholder="Enter your email"
              />
            </div>

            {!useMagicLink && (
              <div>
                <label
                  htmlFor="password"
                  className="block text-sm font-medium text-gray-700 mb-2"
                >
                  Password
                </label>
                <div className="relative">
                  <input
                    id="password"
                    name="password"
                    type={showPassword ? "text" : "password"}
                    autoComplete="current-password"
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full px-3 py-3 pr-10 border border-gray-300 rounded-lg shadow-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    placeholder="Enter your password"
                  />
                  <button
                    type="button"
                    className="absolute inset-y-0 right-0 pr-3 flex items-center"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? (
                      <EyeSlashIcon className="h-5 w-5 text-gray-400" />
                    ) : (
                      <EyeIcon className="h-5 w-5 text-gray-400" />
                    )}
                  </button>
                </div>
              </div>
            )}

            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <input
                  id="magic-link"
                  name="magic-link"
                  type="checkbox"
                  checked={useMagicLink}
                  onChange={(e) => setUseMagicLink(e.target.checked)}
                  className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                />
                <label
                  htmlFor="magic-link"
                  className="ml-2 block text-sm text-gray-700"
                >
                  Use magic link instead
                </label>
              </div>

              {!useMagicLink && (
                <Link
                  to="/forgot-password"
                  className="text-sm text-indigo-600 hover:text-indigo-500"
                >
                  Forgot password?
                </Link>
              )}
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full flex justify-center py-4 px-6 border border-transparent rounded-xl shadow-lg text-base font-semibold text-white bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-[1.02] hover:shadow-xl"
            >
              {loading ? (
                <div className="flex items-center">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  {useMagicLink ? "Sending magic link..." : "Signing in..."}
                </div>
              ) : useMagicLink ? (
                "Send magic link"
              ) : (
                "Sign in"
              )}
            </button>
          </form>

          <div className="mt-8 text-center">
            <p className="text-sm text-gray-600">
              Don't have an account?{" "}
              <Link
                to="/signup"
                className="font-medium text-indigo-600 hover:text-indigo-500"
              >
                Sign up for free
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
