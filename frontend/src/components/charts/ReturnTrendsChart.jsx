import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

const ReturnTrendsChart = ({ data }) => {
  const defaultData = [
    { category: "Electronics", returns: 45, total: 300 },
    { category: "Clothing", returns: 78, total: 250 },
    { category: "Books", returns: 12, total: 200 },
    { category: "Home", returns: 34, total: 180 },
    { category: "Sports", returns: 28, total: 150 },
  ];

  const chartData = (data && Array.isArray(data) && data.length > 0) ? data : defaultData;

  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200">
      <div className="mb-4">
        <h3 className="text-lg font-semibold">Return Trends by Category</h3>
        <p className="text-sm text-gray-500">
          Analysis of return patterns across product categories
        </p>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="category" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="returns" fill="#ef4444" name="Returns" />
          <Bar dataKey="total" fill="#3b82f6" name="Total Orders" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ReturnTrendsChart;
