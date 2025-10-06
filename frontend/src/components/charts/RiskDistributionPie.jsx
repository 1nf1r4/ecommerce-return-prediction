import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts";

const RiskDistributionPie = ({ data }) => {
  // Default data or use provided data
  const defaultData = [
    { name: "Low Risk", value: 45, color: "#10b981" },
    { name: "Medium Risk", value: 35, color: "#f59e0b" },
    { name: "High Risk", value: 20, color: "#ef4444" },
  ];

  // Ensure data is valid and has the right structure
  let chartData = defaultData;
  
  if (data && Array.isArray(data) && data.length > 0) {
    // Filter out items with zero or invalid values
    const validData = data.filter(item => item && typeof item.value === 'number' && item.value > 0);
    if (validData.length > 0) {
      chartData = validData;
    }
  }

  const COLORS = chartData.map((item) => item.color || "#999999");

  const renderLabel = (entry) => {
    return `${entry.name}: ${entry.value}%`;
  };

  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200">
      <div className="mb-4">
        <h3 className="text-lg font-semibold">Risk Distribution</h3>
        <p className="text-sm text-gray-500">
          Current distribution of prediction risk levels
        </p>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={renderLabel}
            outerRadius={100}
            fill="#8884d8"
            dataKey="value"
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip formatter={(value) => `${value}%`} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
};

export default RiskDistributionPie;
