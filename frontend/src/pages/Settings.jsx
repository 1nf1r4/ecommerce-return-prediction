import { useState, useEffect } from "react";
import { useAuth } from "../hooks/useAuth";
import DashboardHeader from "../components/layout/DashboardHeader";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";

const Settings = () => {
  const { user } = useAuth();
  const [loading, setLoading] = useState(false);
  const [saved, setSaved] = useState(false);

  // User preferences state
  const [preferences, setPreferences] = useState({
    // Display preferences
    dashboardLayout: "default",
    defaultDateRange: "last_30_days",
    itemsPerPage: 10,
    theme: "light",

    // Notification preferences
    emailNotifications: true,
    highRiskAlerts: true,
    dailySummary: false,
    weeklyReport: true,

    // Export preferences
    defaultExportFormat: "csv",
    includeConfidenceScore: true,
    includeFactors: false,

    // Dashboard widgets
    showRevenueChart: true,
    showTrendsChart: true,
    showAccuracyChart: true,
    showRiskDistribution: true,
  });

  // Load preferences from localStorage
  useEffect(() => {
    const savedPrefs = localStorage.getItem("userPreferences");
    if (savedPrefs) {
      setPreferences(JSON.parse(savedPrefs));
    }
  }, []);

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setPreferences({
      ...preferences,
      [name]: type === "checkbox" ? checked : value,
    });
  };

  const handleSave = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      // Save to localStorage
      localStorage.setItem("userPreferences", JSON.stringify(preferences));

      // In production, you would also save to database via API
      // await apiService.saveUserPreferences(user.id, preferences);

      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (error) {
      console.error("Failed to save preferences:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    const defaultPreferences = {
      dashboardLayout: "default",
      defaultDateRange: "last_30_days",
      itemsPerPage: 10,
      theme: "light",
      emailNotifications: true,
      highRiskAlerts: true,
      dailySummary: false,
      weeklyReport: true,
      defaultExportFormat: "csv",
      includeConfidenceScore: true,
      includeFactors: false,
      showRevenueChart: true,
      showTrendsChart: true,
      showAccuracyChart: true,
      showRiskDistribution: true,
    };
    setPreferences(defaultPreferences);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <DashboardHeader />

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
          <p className="text-gray-600 mt-1">
            Manage your preferences and customize your experience
          </p>
        </div>

        {saved && (
          <Alert className="mb-6 bg-green-50 border-green-200">
            <AlertDescription className="text-green-800">
              Settings saved successfully!
            </AlertDescription>
          </Alert>
        )}

        <form onSubmit={handleSave} className="space-y-6">
          {/* Display Preferences */}
          <Card>
            <CardHeader>
              <CardTitle>Display Preferences</CardTitle>
              <CardDescription>
                Customize how your dashboard appears
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Dashboard Layout
                  </label>
                  <select
                    name="dashboardLayout"
                    value={preferences.dashboardLayout}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border rounded-md"
                  >
                    <option value="default">Default</option>
                    <option value="compact">Compact</option>
                    <option value="expanded">Expanded</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Default Date Range
                  </label>
                  <select
                    name="defaultDateRange"
                    value={preferences.defaultDateRange}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border rounded-md"
                  >
                    <option value="last_7_days">Last 7 Days</option>
                    <option value="last_30_days">Last 30 Days</option>
                    <option value="last_90_days">Last 90 Days</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Items Per Page
                  </label>
                  <select
                    name="itemsPerPage"
                    value={preferences.itemsPerPage}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border rounded-md"
                  >
                    <option value="10">10</option>
                    <option value="25">25</option>
                    <option value="50">50</option>
                    <option value="100">100</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Theme
                  </label>
                  <select
                    name="theme"
                    value={preferences.theme}
                    onChange={handleInputChange}
                    className="w-full px-3 py-2 border rounded-md"
                  >
                    <option value="light">Light</option>
                    <option value="dark">Dark (Coming Soon)</option>
                    <option value="auto">Auto (Coming Soon)</option>
                  </select>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Notification Preferences */}
          <Card>
            <CardHeader>
              <CardTitle>Notification Preferences</CardTitle>
              <CardDescription>
                Choose what notifications you want to receive
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <label className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    name="emailNotifications"
                    checked={preferences.emailNotifications}
                    onChange={handleInputChange}
                    className="w-4 h-4 rounded border-gray-300"
                  />
                  <div>
                    <div className="font-medium">Email Notifications</div>
                    <div className="text-sm text-gray-500">
                      Receive updates via email
                    </div>
                  </div>
                </label>

                <label className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    name="highRiskAlerts"
                    checked={preferences.highRiskAlerts}
                    onChange={handleInputChange}
                    className="w-4 h-4 rounded border-gray-300"
                  />
                  <div>
                    <div className="font-medium">High Risk Alerts</div>
                    <div className="text-sm text-gray-500">
                      Instant alerts for high-risk predictions
                    </div>
                  </div>
                </label>

                <label className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    name="dailySummary"
                    checked={preferences.dailySummary}
                    onChange={handleInputChange}
                    className="w-4 h-4 rounded border-gray-300"
                  />
                  <div>
                    <div className="font-medium">Daily Summary</div>
                    <div className="text-sm text-gray-500">
                      Daily digest of predictions and insights
                    </div>
                  </div>
                </label>

                <label className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    name="weeklyReport"
                    checked={preferences.weeklyReport}
                    onChange={handleInputChange}
                    className="w-4 h-4 rounded border-gray-300"
                  />
                  <div>
                    <div className="font-medium">Weekly Report</div>
                    <div className="text-sm text-gray-500">
                      Comprehensive weekly performance report
                    </div>
                  </div>
                </label>
              </div>
            </CardContent>
          </Card>

          {/* Export Preferences */}
          <Card>
            <CardHeader>
              <CardTitle>Export Preferences</CardTitle>
              <CardDescription>
                Set default export options
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Default Export Format
                </label>
                <select
                  name="defaultExportFormat"
                  value={preferences.defaultExportFormat}
                  onChange={handleInputChange}
                  className="w-full px-3 py-2 border rounded-md"
                >
                  <option value="csv">CSV</option>
                  <option value="json">JSON</option>
                  <option value="xlsx">Excel (XLSX)</option>
                </select>
              </div>

              <div className="space-y-3">
                <label className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    name="includeConfidenceScore"
                    checked={preferences.includeConfidenceScore}
                    onChange={handleInputChange}
                    className="w-4 h-4 rounded border-gray-300"
                  />
                  <div>
                    <div className="font-medium">Include Confidence Score</div>
                    <div className="text-sm text-gray-500">
                      Add confidence metrics to exports
                    </div>
                  </div>
                </label>

                <label className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    name="includeFactors"
                    checked={preferences.includeFactors}
                    onChange={handleInputChange}
                    className="w-4 h-4 rounded border-gray-300"
                  />
                  <div>
                    <div className="font-medium">Include Contributing Factors</div>
                    <div className="text-sm text-gray-500">
                      Add factor analysis to exports
                    </div>
                  </div>
                </label>
              </div>
            </CardContent>
          </Card>

          {/* Dashboard Widgets */}
          <Card>
            <CardHeader>
              <CardTitle>Dashboard Widgets</CardTitle>
              <CardDescription>
                Choose which charts to display on your dashboard
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <label className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    name="showRevenueChart"
                    checked={preferences.showRevenueChart}
                    onChange={handleInputChange}
                    className="w-4 h-4 rounded border-gray-300"
                  />
                  <span className="font-medium">Revenue Impact Chart</span>
                </label>

                <label className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    name="showTrendsChart"
                    checked={preferences.showTrendsChart}
                    onChange={handleInputChange}
                    className="w-4 h-4 rounded border-gray-300"
                  />
                  <span className="font-medium">Return Trends Chart</span>
                </label>

                <label className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    name="showAccuracyChart"
                    checked={preferences.showAccuracyChart}
                    onChange={handleInputChange}
                    className="w-4 h-4 rounded border-gray-300"
                  />
                  <span className="font-medium">Accuracy Tracking Chart</span>
                </label>

                <label className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    name="showRiskDistribution"
                    checked={preferences.showRiskDistribution}
                    onChange={handleInputChange}
                    className="w-4 h-4 rounded border-gray-300"
                  />
                  <span className="font-medium">Risk Distribution Pie</span>
                </label>
              </div>
            </CardContent>
          </Card>

          {/* Action Buttons */}
          <div className="flex gap-4">
            <Button type="submit" disabled={loading} className="flex-1">
              {loading ? "Saving..." : "Save Settings"}
            </Button>
            <Button
              type="button"
              onClick={handleReset}
              variant="outline"
              className="flex-1"
            >
              Reset to Defaults
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Settings;
