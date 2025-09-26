import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { client } from "../../supabase/client";

export default function AuthCallback() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const handleAuthCallback = async () => {
      try {
        const token_hash = searchParams.get("token_hash");
        const type = searchParams.get("type");
        const next = searchParams.get("next") || "/dashboard";

        if (token_hash && type) {
          // Handle email confirmation with token
          console.log("Confirming email with token:", { token_hash, type });

          const { data, error } = await client.auth.verifyOtp({
            token_hash,
            type,
          });

          if (error) {
            console.error("Email confirmation error:", error);
            setError(
              error.message ||
                "Failed to confirm email. The link may have expired."
            );
            return;
          }

          if (data?.user) {
            console.log("Email confirmed successfully!", data.user);
            // Redirect after confirmation
            setTimeout(() => {
              navigate(next, { replace: true });
            }, 1000);
            return;
          }
        }

        // Fallback: check for existing session
        const { data, error } = await client.auth.getSession();

        if (error) {
          console.error("Auth callback error:", error);
          setError(error.message);
          return;
        }

        if (data.session) {
          const type = searchParams.get("type");

          if (type === "recovery") {
            // Redirect to password reset page
            navigate("/reset-password", { replace: true });
          } else {
            // Regular sign in or email confirmation
            navigate("/dashboard", { replace: true });
          }
        } else {
          // No session, redirect to sign in
          navigate("/signin", { replace: true });
        }
      } catch (err) {
        console.error("Unexpected error:", err);
        setError("An unexpected error occurred");
      } finally {
        setLoading(false);
      }
    };

    handleAuthCallback();
  }, [navigate, searchParams]);

  if (loading) {
    const token_hash = searchParams.get("token_hash");
    const confirmingEmail = token_hash && searchParams.get("type");

    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
          <p className="mt-4 text-gray-600">
            {confirmingEmail
              ? "Confirming your email address..."
              : "Confirming your authentication..."}
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="text-red-600 text-xl mb-4">Authentication Error</div>
          <p className="text-gray-600 mb-6">{error}</p>
          <button
            onClick={() => navigate("/signin")}
            className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2 px-4 rounded-lg transition duration-200"
          >
            Go to Sign In
          </button>
        </div>
      </div>
    );
  }

  return null;
}
