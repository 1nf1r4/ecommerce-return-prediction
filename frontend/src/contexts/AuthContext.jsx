import { useEffect, useState } from "react";
import { client } from "../../supabase/client";
import { AuthContext } from "./auth-context";

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Get initial session
    const getInitialSession = async () => {
      try {
        const {
          data: { session },
          error,
        } = await client.auth.getSession();
        if (error) {
          console.error("Error getting session:", error);
        } else {
          setSession(session);
          setUser(session?.user || null);
        }
      } catch (error) {
        console.error("Error getting session:", error);
      } finally {
        setLoading(false);
      }
    };

    getInitialSession();

    // Listen for auth changes
    const {
      data: { subscription },
    } = client.auth.onAuthStateChange(async (event, session) => {
      console.log("Auth event:", event, session);
      setSession(session);
      setUser(session?.user || null);
      setLoading(false);

      // Handle redirect after email confirmation
      if (event === "SIGNED_IN" && session?.user?.email_confirmed_at) {
        // User just confirmed their email
        console.log("User confirmed email, redirecting to dashboard");
      }
    });

    return () => {
      subscription?.unsubscribe();
    };
  }, []);

  const signUp = async (email, password, options = {}) => {
    setLoading(true);
    try {
      const { data, error } = await client.auth.signUp({
        email,
        password,
        options: {
          ...options,
          emailRedirectTo: `${window.location.origin}/auth/confirm`,
        },
      });

      if (error) {
        console.error("Sign up error:", error);
      } else if (data?.user && !data?.session) {
        console.log("Sign up successful, please check email for confirmation");
      }

      return { data, error };
    } finally {
      setLoading(false);
    }
  };

  const signIn = async (email, password) => {
    setLoading(true);
    try {
      const { data, error } = await client.auth.signInWithPassword({
        email,
        password,
      });

      if (error) {
        console.error("Sign in error:", error);
        // Provide more specific error messages
        if (error.message.includes("Invalid login credentials")) {
          return {
            data,
            error: {
              ...error,
              message:
                "Invalid email or password. Please check your credentials.",
            },
          };
        } else if (error.message.includes("Email not confirmed")) {
          return {
            data,
            error: {
              ...error,
              message:
                "Please check your email and click the confirmation link before signing in.",
            },
          };
        }
      }

      return { data, error };
    } finally {
      setLoading(false);
    }
  };

  const signInWithMagicLink = async (email) => {
    setLoading(true);
    try {
      const { data, error } = await client.auth.signInWithOtp({
        email,
        options: {
          emailRedirectTo: `${window.location.origin}/auth/callback`,
        },
      });
      return { data, error };
    } finally {
      setLoading(false);
    }
  };

  const signOut = async () => {
    setLoading(true);
    try {
      const { error } = await client.auth.signOut();
      return { error };
    } finally {
      setLoading(false);
    }
  };

  const resetPassword = async (email) => {
    try {
      const { data, error } = await client.auth.resetPasswordForEmail(email, {
        redirectTo: `${window.location.origin}/auth/callback?type=recovery`,
      });
      return { data, error };
    } catch (error) {
      return { error };
    }
  };

  const updatePassword = async (password) => {
    try {
      const { data, error } = await client.auth.updateUser({
        password,
      });
      return { data, error };
    } catch (error) {
      return { error };
    }
  };

  const resendConfirmation = async (email) => {
    try {
      const { data, error } = await client.auth.resend({
        type: "signup",
        email,
        options: {
          emailRedirectTo: `${window.location.origin}/auth/callback`,
        },
      });
      return { data, error };
    } catch (error) {
      return { error };
    }
  };

  const value = {
    user,
    session,
    loading,
    signUp,
    signIn,
    signInWithMagicLink,
    signOut,
    resetPassword,
    updatePassword,
    resendConfirmation,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
