/**
 * API Service for E-commerce Return Prediction
 * Handles all communication with the FastAPI backend and Supabase authentication
 */

import { client } from "../../supabase/client";

const API_BASE_URL =
  import.meta.env.VITE_API_URL ||
  (import.meta.env.PROD ? "/api" : "http://localhost:8000");

class ApiService {
  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  async getAuthHeaders() {
    try {
      const {
        data: { session },
      } = await client.auth.getSession();
      if (session?.access_token) {
        return {
          Authorization: `Bearer ${session.access_token}`,
        };
      }
      return {};
    } catch (error) {
      console.warn("Failed to get auth headers:", error);
      return {};
    }
  }

  async makeRequest(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const authHeaders = await this.getAuthHeaders();

    const config = {
      headers: {
        "Content-Type": "application/json",
        ...authHeaders,
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
  }

  // Health Check
  async checkHealth() {
    return this.makeRequest("/health");
  }

  async checkPredictionHealth() {
    return this.makeRequest("/predict/health");
  }

  // Single Prediction
  async makeSinglePrediction(orderData) {
    return this.makeRequest("/predict/single", {
      method: "POST",
      body: JSON.stringify(orderData),
    });
  }

  // Order Processing (maps to prediction for backward compatibility)
  async processOrder(orderData) {
    return this.makeSinglePrediction(orderData);
  }

  // Batch Processing
  async processBatchOrders(orders) {
    return this.makeRequest("/orders/batch-process", {
      method: "POST",
      body: JSON.stringify({ orders }),
    });
  }

  // File Upload for Batch Processing
  async uploadBatchFile(file) {
    const formData = new FormData();
    formData.append("file", file);
    const authHeaders = await this.getAuthHeaders();

    return this.makeRequest("/predict/batch/upload", {
      method: "POST",
      headers: {
        ...authHeaders,
        // Don't set Content-Type for FormData, let browser set it
      },
      body: formData,
    });
  }

  // Batch Job Status
  async getBatchJobStatus(jobId) {
    return this.makeRequest(`/predict/batch/${jobId}`);
  }

  // Batch Results
  async getBatchResults(jobId) {
    return this.makeRequest(`/predict/batch/${jobId}/results`);
  }

  // Download Batch Results
  async downloadBatchResults(jobId, format = "csv") {
    const authHeaders = await this.getAuthHeaders();
    const url = `${this.baseUrl}/predict/batch/${jobId}/download?format=${format}`;

    try {
      const response = await fetch(url, {
        headers: authHeaders,
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = `batch_predictions_${jobId}.${format}`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(downloadUrl);

      return true;
    } catch (error) {
      console.error("Download failed:", error);
      throw error;
    }
  }

  // Model Information
  async getModelInfo() {
    return this.makeRequest("/predict/model-info");
  }

  // Get Example Requests
  async getExampleRequests() {
    return this.makeRequest("/predict/example");
  }

  // Validation Rules
  async getValidationRules() {
    return this.makeRequest("/orders/validation-rules");
  }

  // Agent Statistics
  async getOrderStats() {
    return this.makeRequest("/orders/stats");
  }

  // Transform form data to API format
  transformOrderData(formData) {
    return {
      price: parseFloat(formData.productPrice),
      quantity: parseInt(formData.orderQuantity),
      product_category: formData.productCategory,
      gender: formData.userGender,
      payment_method: formData.paymentMethod,
      age: parseInt(formData.userAge),
      location: formData.userLocation,
    };
  }

  // Utility method to check if backend is available
  async isBackendAvailable() {
    try {
      await this.checkHealth();
      return true;
    } catch {
      return false;
    }
  }

  // User Analytics endpoints (authenticated)
  async getDashboardSummary(days = 30) {
    return await this.makeRequest(`/analytics/dashboard?days=${days}`);
  }

  async getPredictionHistory(page = 1, pageSize = 25, filters = {}) {
    const params = new URLSearchParams({
      page: page.toString(),
      page_size: pageSize.toString(),
      ...filters,
    }).toString();

    return await this.makeRequest(`/analytics/predictions?${params}`);
  }

  async getUserAnalytics(days = 30) {
    return await this.makeRequest(`/analytics/user-analytics?days=${days}`);
  }

  async getBatchJobs(status = null, limit = 20) {
    const params = new URLSearchParams({
      limit: limit.toString(),
      ...(status && { status }),
    }).toString();

    return await this.makeRequest(`/analytics/batch-jobs?${params}`);
  }

  async updateUserPreferences(preferences) {
    return await this.makeRequest("/analytics/preferences", {
      method: "PUT",
      body: JSON.stringify(preferences),
    });
  }

  async getUserProfile() {
    return await this.makeRequest("/analytics/profile");
  }

  async updateUserProfile(profileData) {
    return await this.makeRequest("/analytics/profile", {
      method: "PUT",
      body: JSON.stringify(profileData),
    });
  }

  // Legacy analytics endpoints (for backward compatibility)
  async getDashboardData() {
    return await this.makeRequest("/api/analytics/dashboard");
  }

  async getRevenueImpact(timePeriod = "last_30_days") {
    return await this.makeRequest(
      `/api/analytics/revenue-impact?time_period=${timePeriod}`
    );
  }

  async getDailyReport(date = null) {
    const endpoint = date
      ? `/api/analytics/reports/${date}`
      : "/api/analytics/reports";
    return await this.makeRequest(endpoint);
  }

  async getBusinessInsights(timePeriod = "last_30_days") {
    return await this.makeRequest(
      `/api/analytics/insights?time_period=${timePeriod}`
    );
  }

  async getAccuracyAnalysis() {
    return await this.makeRequest("/api/analytics/accuracy");
  }

  async getReturnTrends(timePeriod = "last_90_days") {
    return await this.makeRequest(
      `/api/analytics/trends?time_period=${timePeriod}`
    );
  }

  async getModelPerformance() {
    return await this.makeRequest("/api/analytics/performance");
  }

  async getBusinessKPIs() {
    return await this.makeRequest("/api/analytics/kpis");
  }

  async getAnalyticsHealth() {
    return await this.makeRequest("/api/analytics/health");
  }

  async getRecentPredictions(limit = 10) {
    return await this.makeRequest(
      `/api/analytics/recent-predictions?limit=${limit}`
    );
  }

  // Export functionality
  async exportPredictions(filters = {}, format = "csv") {
    const queryParams = new URLSearchParams({
      format,
      ...filters,
    }).toString();

    const authHeaders = await this.getAuthHeaders();
    const url = `${this.baseUrl}/api/export/predictions?${queryParams}`;

    try {
      const response = await fetch(url, {
        headers: authHeaders,
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = `predictions_export.${format}`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(downloadUrl);

      return true;
    } catch (error) {
      console.error("Export failed:", error);
      throw error;
    }
  }

  // Notification methods
  async getNotifications(userId) {
    return await this.makeRequest(`/api/notifications?user_id=${userId}`);
  }

  async markNotificationRead(notificationId) {
    return await this.makeRequest(`/api/notifications/${notificationId}/read`, {
      method: "POST",
    });
  }

  async markAllNotificationsRead(userId) {
    return await this.makeRequest(`/api/notifications/read-all`, {
      method: "POST",
      body: JSON.stringify({ user_id: userId }),
    });
  }
}

// Create and export singleton instance
const apiService = new ApiService();
export default apiService;
