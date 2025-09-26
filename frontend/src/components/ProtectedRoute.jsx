import { Navigate } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";
import { EnvelopeIcon } from "@heroicons/react/24/outline";

export default function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/signin" replace />;
  }

  // Check if email is confirmed
  if (!user.email_confirmed_at) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
        <div className="max-w-md w-full text-center">
          <div className="bg-white rounded-lg shadow-lg p-8">
            <div className="flex justify-center mb-6">
              <EnvelopeIcon className="h-16 w-16 text-indigo-600" />
            </div>
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              Please confirm your email
            </h2>
            <p className="text-gray-600 mb-6">
              We've sent a confirmation link to <strong>{user.email}</strong>.
              Please click the link in the email to access your dashboard.
            </p>
            <div className="text-sm text-gray-500">
              <p>Didn't receive the email? Check your spam folder.</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return children;
}
