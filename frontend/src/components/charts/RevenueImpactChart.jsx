import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

const RevenueImpactChart = ({ data }) => {
  // Default data if none provided
  const defaultData = [
    { date: "Week 1", saved: 2400, atRisk: 4000 },
    { date: "Week 2", saved: 3200, atRisk: 3800 },
    { date: "Week 3", saved: 2800, atRisk: 3600 },
    { date: "Week 4", saved: 3800, atRisk: 3400 },
  ];

  // Use provided data or default
  const chartData = (data && Array.isArray(data) && data.length > 0) ? data : defaultData;

  const totalSaved = chartData.reduce((sum, item) => sum + (item.saved || 0), 0);

  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200">
      <div className="mb-4">
        <h3 className="text-lg font-semibold">Revenue Impact</h3>
        <p className="text-sm text-gray-500">
          Total Saved: <span className="font-semibold text-green-600">${totalSaved.toLocaleString()}</span>
        </p>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip formatter={(value) => `$${value.toLocaleString()}`} />
          <Legend />
          <Line
            type="monotone"
            dataKey="saved"
            stroke="#10b981"
            strokeWidth={2}
            name="Revenue Saved"
          />
          <Line
            type="monotone"
            dataKey="atRisk"
            stroke="#ef4444"
            strokeWidth={2}
            name="Revenue at Risk"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default RevenueImpactChart;
