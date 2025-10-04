import { useState, useEffect, useCallback } from "react";
import { useAuth } from "../hooks/useAuth";
import DashboardHeader from "../components/layout/DashboardHeader";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import apiService from "../services/apiService";

// Import new components
import TabNavigation from "../components/common/TabNavigation";
import DateRangePicker from "../components/common/DateRangePicker";
import ExportButton from "../components/common/ExportButton";
import BatchUploadZone from "../components/dashboard/BatchUploadZone";
import PredictionResult from "../components/dashboard/PredictionResult";
import RecentPredictionsTable from "../components/dashboard/RecentPredictionsTable";

// Import chart components
import RevenueImpactChart from "../components/charts/RevenueImpactChart";
import ReturnTrendsChart from "../components/charts/ReturnTrendsChart";
import AccuracyChart from "../components/charts/AccuracyChart";
import RiskDistributionPie from "../components/charts/RiskDistributionPie";

import {
  ChartBarIcon,
  CubeIcon,
  DocumentChartBarIcon,
  HomeIcon,
} from "@heroicons/react/24/outline";

const Dashboard = () => {
  const { _user } = useAuth();
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");
  const [backendStatus, setBackendStatus] = useState("checking");

  // Stats state
  const [stats, setStats] = useState({
    totalOrders: 0,
    predictedReturns: 0,
    returnRate: 0,
    revenueSaved: 0,
    accuracy: 0,
  });

  // Predictions state
  const [recentPredictions, setRecentPredictions] = useState([]);
  const [predictionResult, setPredictionResult] = useState(null);

  // Quick prediction form state
  const [quickPrediction, setQuickPrediction] = useState({
    productCategory: "",
    productPrice: "",
    orderQuantity: "",
    userAge: "",
    userGender: "",
    userLocation: "",
    paymentMethod: "",
  });

  // Batch processing state
  const [batchJobId, setBatchJobId] = useState(null);
  const [batchStatus, setBatchStatus] = useState(null);
  const [isProcessingBatch, setIsProcessingBatch] = useState(false);

  // Analytics state
  const [dateRange, setDateRange] = useState({ range: "last_30_days" });
  const [analyticsData, setAnalyticsData] = useState({
    revenue: null,
    trends: null,
    accuracy: null,
    riskDist: null,
  });

  // Load dashboard analytics
  const loadAnalyticsData = useCallback(async () => {
    try {
      const dashboardData = await apiService.getDashboardData();
      if (dashboardData && dashboardData.success && dashboardData.data) {
        const kpis = dashboardData.data.kpis;
        const riskDist = dashboardData.data.risk_distribution_7d;

        const totalRiskOrders =
          (riskDist?.high_risk?.count || 0) +
          (riskDist?.medium_risk?.count || 0);

        const totalAllOrders =
          (riskDist?.high_risk?.count || 0) +
          (riskDist?.medium_risk?.count || 0) +
          (riskDist?.low_risk?.count || 0);

        const returnRate =
          totalAllOrders > 0
            ? (riskDist?.high_risk?.percentage || 0) +
              (riskDist?.medium_risk?.percentage || 0)
            : 0;

        setStats({
          totalOrders: kpis?.total_predictions_30d || 0,
          predictedReturns: totalRiskOrders || 0,
          returnRate: returnRate || 0,
          revenueSaved: kpis?.total_revenue_saved_lifetime || 0,
          accuracy:
            kpis?.model_accuracy_latest !== "N/A"
              ? parseFloat(kpis?.model_accuracy_latest) || 0
              : 0,
        });

        // Set analytics data for charts
        // Transform riskDist for pie chart
        const riskDistForChart = riskDist
          ? [
              {
                name: "Low Risk",
                value: riskDist.low_risk?.percentage || 0,
                color: "#10b981",
              },
              {
                name: "Medium Risk",
                value: riskDist.medium_risk?.percentage || 0,
                color: "#f59e0b",
              },
              {
                name: "High Risk",
                value: riskDist.high_risk?.percentage || 0,
                color: "#ef4444",
              },
            ]
          : null;

        setAnalyticsData((prev) => ({
          ...prev,
          riskDist: riskDistForChart,
        }));
      }
    } catch (error) {
      console.error("Failed to load analytics data:", error);
    }
  }, []);

  // Load recent predictions from database
  const loadRecentPredictions = useCallback(async () => {
    try {
      const response = await apiService.getRecentPredictions(10);
      if (response.success && response.predictions) {
        setRecentPredictions(response.predictions);
      }
    } catch (error) {
      console.error("Failed to load recent predictions:", error);
    }
  }, []);

  // Check backend health
  const checkBackendHealth = useCallback(async () => {
    try {
      const isAvailable = await apiService.isBackendAvailable();
      setBackendStatus(isAvailable ? "healthy" : "unhealthy");

      if (isAvailable) {
        loadAnalyticsData();
        loadRecentPredictions();
      }
    } catch (error) {
      console.error("Health check failed:", error);
      setBackendStatus("unhealthy");
    }
  }, [loadAnalyticsData, loadRecentPredictions]);

  useEffect(() => {
    checkBackendHealth();
  }, [checkBackendHealth]);

  // Handle quick prediction
  const handleQuickPredictionSubmit = useCallback(
    async (e) => {
      e.preventDefault();
      setLoading(true);
      setPredictionResult(null);

      try {
        const orderData = apiService.transformOrderData(quickPrediction);
        const response = await apiService.processOrder(orderData);

        if (response.success) {
          const prediction = response.prediction;
          const probability = Math.round(prediction.return_probability * 100);
          const riskLevel = prediction.risk_level;

          setPredictionResult({
            probability,
            riskLevel,
            confidence: Math.round(prediction.confidence_score * 100),
            recommendations: response.recommendations || [],
            factors: [
              {
                factor: "Product Category",
                impact: "Medium",
                value: quickPrediction.productCategory,
              },
              {
                factor: "Price Range",
                impact: "High",
                value: `$${quickPrediction.productPrice}`,
              },
              {
                factor: "User Location",
                impact: "Low",
                value: quickPrediction.userLocation,
              },
            ],
          });

          // Add to recent predictions
          const newPrediction = {
            timestamp: new Date().toISOString(),
            category: quickPrediction.productCategory,
            orderValue:
              parseFloat(quickPrediction.productPrice) *
              parseInt(quickPrediction.orderQuantity),
            returnProbability: response.prediction.return_probability,
            riskLevel: riskLevel,
            status: "Completed",
          };

          setRecentPredictions((prev) => [newPrediction, ...prev.slice(0, 9)]);

          // Reload analytics data and recent predictions to reflect the new prediction
          await loadAnalyticsData();
          await loadRecentPredictions();
        } else {
          throw new Error(response.error || "Prediction failed");
        }
      } catch (error) {
        console.error("Prediction failed:", error);
      } finally {
        setLoading(false);
      }
    },
    [quickPrediction, loadAnalyticsData, loadRecentPredictions]
  );

  const handleInputChange = (e) => {
    setQuickPrediction({
      ...quickPrediction,
      [e.target.name]: e.target.value,
    });
  };

  // Handle batch file upload
  const handleBatchFileUpload = async (file) => {
    setIsProcessingBatch(true);
    try {
      const response = await apiService.uploadBatchFile(file);
      if (response.success) {
        setBatchJobId(response.job_id);
        setBatchStatus("processing");
        // Poll for status
        pollBatchStatus(response.job_id);
      }
    } catch (error) {
      console.error("Batch upload failed:", error);
      setBatchStatus("failed");
    }
  };

  // Poll batch job status
  const pollBatchStatus = async (jobId) => {
    const checkStatus = async () => {
      try {
        const response = await apiService.getBatchJobStatus(jobId);
        setBatchStatus(response.status);

        if (response.status === "completed") {
          setIsProcessingBatch(false);
        } else if (response.status === "processing") {
          setTimeout(() => checkStatus(), 2000);
        } else {
          setIsProcessingBatch(false);
        }
      } catch (error) {
        console.error("Status check failed:", error);
        setIsProcessingBatch(false);
      }
    };
    await checkStatus();
  };

  // Handle date range change for analytics
  const handleDateRangeChange = (range) => {
    setDateRange(range);
    // Reload analytics with new range
    loadAnalyticsWithRange(range);
  };

  const loadAnalyticsWithRange = async (range) => {
    try {
      const [revenue, trends, accuracy] = await Promise.all([
        apiService.getRevenueImpact(range.range),
        apiService.getReturnTrends(range.range),
        apiService.getAccuracyAnalysis(),
      ]);

      setAnalyticsData({
        revenue: revenue.data || null,
        trends: trends.data || null,
        accuracy: accuracy.data || null,
        riskDist: analyticsData.riskDist,
      });
    } catch (error) {
      console.error("Failed to load analytics:", error);
    }
  };

  // // Export predictions
  // const handleExport = async (format) => {
  //   try {
  //     await apiService.exportPredictions(
  //       { date_range: dateRange.range },
  //       format
  //     );
  //   } catch (error) {
  //     console.error("Export failed:", error);
  //   }
  // };

  // Tab configuration
  const tabs = [
    {
      id: "overview",
      label: "Overview",
      icon: <HomeIcon className="h-4 w-4" />,
    },
    {
      id: "prediction",
      label: "Single Prediction",
      icon: <CubeIcon className="h-4 w-4" />,
    },
    {
      id: "batch",
      label: "Batch Processing",
      icon: <DocumentChartBarIcon className="h-4 w-4" />,
    },
    {
      id: "analytics",
      label: "Analytics",
      icon: <ChartBarIcon className="h-4 w-4" />,
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50/30 to-purple-50/20">
      <DashboardHeader />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Backend Status Alert */}
        {backendStatus === "unhealthy" && (
          <Alert className="mb-6 glass-card border-amber-200/50 shadow-lg animate-pulse-glow">
            <div className="flex items-start gap-3">
              <div className="h-5 w-5 rounded-full bg-amber-500 flex items-center justify-center">
                <span className="h-2 w-2 rounded-full bg-white animate-ping"></span>
              </div>
              <div>
                <AlertTitle className="text-amber-900 font-semibold">
                  Backend Service Unavailable
                </AlertTitle>
                <AlertDescription className="text-amber-700">
                  The prediction service is currently unavailable. Some features
                  may not work.
                </AlertDescription>
              </div>
            </div>
          </Alert>
        )}

        {/* Enhanced Stats Cards with Gradients and Animations */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {/* Total Orders Card */}
          <Card className="hover-lift glass-card border-0 shadow-lg overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 via-transparent to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
            <CardHeader className="pb-3 relative">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-medium text-gray-600">
                  Total Orders
                </CardTitle>
                <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center shadow-lg">
                  <svg
                    className="h-5 w-5 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z"
                    />
                  </svg>
                </div>
              </div>
            </CardHeader>
            <CardContent className="relative">
              <div className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                {stats.totalOrders}
              </div>
              <p className="text-xs text-gray-500 mt-2 flex items-center gap-1">
                <span className="inline-block h-1 w-1 rounded-full bg-green-500"></span>
                Last 30 days
              </p>
            </CardContent>
          </Card>

          {/* Predicted Returns Card */}
          <Card className="hover-lift glass-card border-0 shadow-lg overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-br from-red-500/5 via-transparent to-orange-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
            <CardHeader className="pb-3 relative">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-medium text-gray-600">
                  Predicted Returns
                </CardTitle>
                <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-red-500 to-orange-600 flex items-center justify-center shadow-lg">
                  <svg
                    className="h-5 w-5 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                    />
                  </svg>
                </div>
              </div>
            </CardHeader>
            <CardContent className="relative">
              <div className="text-4xl font-bold bg-gradient-to-r from-red-600 to-orange-600 bg-clip-text text-transparent">
                {stats.predictedReturns}
              </div>
              <p className="text-xs text-gray-500 mt-2 flex items-center gap-1">
                <span className="inline-block h-1 w-1 rounded-full bg-red-500"></span>
                High/Medium risk
              </p>
            </CardContent>
          </Card>

          {/* Return Rate Card */}
          <Card className="hover-lift glass-card border-0 shadow-lg overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-br from-amber-500/5 via-transparent to-yellow-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
            <CardHeader className="pb-3 relative">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-medium text-gray-600">
                  Return Rate
                </CardTitle>
                <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-amber-500 to-yellow-600 flex items-center justify-center shadow-lg">
                  <svg
                    className="h-5 w-5 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                    />
                  </svg>
                </div>
              </div>
            </CardHeader>
            <CardContent className="relative">
              <div className="text-4xl font-bold bg-gradient-to-r from-amber-600 to-yellow-600 bg-clip-text text-transparent">
                {stats.returnRate.toFixed(1)}%
              </div>
              <p className="text-xs text-gray-500 mt-2 flex items-center gap-1">
                <span className="inline-block h-1 w-1 rounded-full bg-amber-500"></span>
                Current period
              </p>
            </CardContent>
          </Card>

          {/* Revenue Saved Card */}
          <Card className="hover-lift glass-card border-0 shadow-lg overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-br from-green-500/5 via-transparent to-emerald-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
            <CardHeader className="pb-3 relative">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-medium text-gray-600">
                  Revenue Saved
                </CardTitle>
                <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center shadow-lg">
                  <svg
                    className="h-5 w-5 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                </div>
              </div>
            </CardHeader>
            <CardContent className="relative">
              <div className="text-4xl font-bold bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent">
                ${stats.revenueSaved.toLocaleString()}
              </div>
              <p className="text-xs text-gray-500 mt-2 flex items-center gap-1">
                <span className="inline-block h-1 w-1 rounded-full bg-green-500"></span>
                Lifetime total
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Tab Navigation */}
        <TabNavigation
          tabs={tabs}
          activeTab={activeTab}
          onTabChange={setActiveTab}
          className="mb-6"
        />

        {/* Tab Content */}
        <div className="space-y-6">
          {/* Overview Tab */}
          {activeTab === "overview" && (
            <div className="space-y-6">
              <div className="flex justify-between items-center">
                <h2 className="text-2xl font-bold text-gray-900">
                  Dashboard Overview
                </h2>
                <ExportButton data={recentPredictions} filename="predictions" />
              </div>

              {recentPredictions.length > 0 ? (
                <RecentPredictionsTable
                  predictions={recentPredictions}
                  onRowClick={(row) => console.log("Clicked:", row)}
                />
              ) : (
                <Card>
                  <CardContent className="py-12 text-center">
                    <p className="text-gray-500">
                      No predictions yet. Start by making a single prediction or
                      uploading a batch file.
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          )}

          {/* Single Prediction Tab */}
          {activeTab === "prediction" && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Quick Order Prediction</CardTitle>
                </CardHeader>
                <CardContent>
                  <form
                    onSubmit={handleQuickPredictionSubmit}
                    className="space-y-4"
                  >
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium mb-1">
                          Product Category
                        </label>
                        <select
                          name="productCategory"
                          value={quickPrediction.productCategory}
                          onChange={handleInputChange}
                          required
                          className="w-full px-3 py-2 border rounded-md"
                        >
                          <option value="">Select category</option>
                          <option value="Electronics">Electronics</option>
                          <option value="Clothing">Clothing</option>
                          <option value="Books">Books</option>
                          <option value="Home & Garden">Home & Garden</option>
                          <option value="Sports">Sports</option>
                          <option value="Beauty">Beauty</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium mb-1">
                          Product Price ($)
                        </label>
                        <input
                          type="number"
                          name="productPrice"
                          value={quickPrediction.productPrice}
                          onChange={handleInputChange}
                          required
                          min="0"
                          step="0.01"
                          className="w-full px-3 py-2 border rounded-md"
                          placeholder="99.99"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium mb-1">
                          Order Quantity
                        </label>
                        <input
                          type="number"
                          name="orderQuantity"
                          value={quickPrediction.orderQuantity}
                          onChange={handleInputChange}
                          required
                          min="1"
                          className="w-full px-3 py-2 border rounded-md"
                          placeholder="1"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium mb-1">
                          Customer Age
                        </label>
                        <input
                          type="number"
                          name="userAge"
                          value={quickPrediction.userAge}
                          onChange={handleInputChange}
                          required
                          min="18"
                          max="100"
                          className="w-full px-3 py-2 border rounded-md"
                          placeholder="25"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium mb-1">
                          Customer Gender
                        </label>
                        <select
                          name="userGender"
                          value={quickPrediction.userGender}
                          onChange={handleInputChange}
                          required
                          className="w-full px-3 py-2 border rounded-md"
                        >
                          <option value="">Select gender</option>
                          <option value="Male">Male</option>
                          <option value="Female">Female</option>
                          <option value="Other">Other</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium mb-1">
                          Location
                        </label>
                        <select
                          name="userLocation"
                          value={quickPrediction.userLocation}
                          onChange={handleInputChange}
                          required
                          className="w-full px-3 py-2 border rounded-md"
                        >
                          <option value="">Select location</option>
                          <option value="New York">New York</option>
                          <option value="California">California</option>
                          <option value="Texas">Texas</option>
                          <option value="Florida">Florida</option>
                          <option value="Washington">Washington</option>
                          <option value="Other">Other</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium mb-1">
                          Payment Method
                        </label>
                        <select
                          name="paymentMethod"
                          value={quickPrediction.paymentMethod}
                          onChange={handleInputChange}
                          required
                          className="w-full px-3 py-2 border rounded-md"
                        >
                          <option value="">Select method</option>
                          <option value="Credit Card">Credit Card</option>
                          <option value="Debit Card">Debit Card</option>
                          <option value="PayPal">PayPal</option>
                          <option value="Bank Transfer">Bank Transfer</option>
                        </select>
                      </div>
                    </div>

                    <Button type="submit" disabled={loading} className="w-full">
                      {loading ? "Predicting..." : "Get Prediction"}
                    </Button>
                  </form>
                </CardContent>
              </Card>

              <div>
                {predictionResult && (
                  <PredictionResult result={predictionResult} />
                )}
              </div>
            </div>
          )}

          {/* Batch Processing Tab */}
          {activeTab === "batch" && (
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Batch Order Processing</CardTitle>
                </CardHeader>
                <CardContent>
                  <BatchUploadZone
                    onFileUpload={handleBatchFileUpload}
                    isProcessing={isProcessingBatch}
                  />

                  {batchStatus && (
                    <div className="mt-6">
                      <Alert>
                        <AlertTitle>Batch Processing Status</AlertTitle>
                        <AlertDescription>
                          Status: {batchStatus}
                          {batchStatus === "completed" && batchJobId && (
                            <div className="mt-2">
                              <Button
                                onClick={() =>
                                  apiService.downloadBatchResults(batchJobId)
                                }
                              >
                                Download Results
                              </Button>
                            </div>
                          )}
                        </AlertDescription>
                      </Alert>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          )}

          {/* Analytics Tab */}
          {activeTab === "analytics" && (
            <div className="space-y-6">
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <h2 className="text-2xl font-bold">Business Analytics</h2>
                <DateRangePicker
                  onDateChange={handleDateRangeChange}
                  defaultRange={dateRange.range}
                />
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Wrap each chart in error boundary */}
                <div>
                  <RevenueImpactChart data={analyticsData.revenue} />
                </div>
                <div>
                  <ReturnTrendsChart data={analyticsData.trends} />
                </div>
                <div>
                  <AccuracyChart data={analyticsData.accuracy} />
                </div>
                <div>
                  <RiskDistributionPie data={analyticsData.riskDist} />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
