import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const AccuracyChart = ({ data }) => {
  const defaultData = [
    { date: "Day 1", accuracy: 85 },
    { date: "Day 2", accuracy: 87 },
    { date: "Day 3", accuracy: 86 },
    { date: "Day 4", accuracy: 88 },
    { date: "Day 5", accuracy: 89 },
    { date: "Day 6", accuracy: 87 },
    { date: "Day 7", accuracy: 90 },
  ];

  const chartData = (data && Array.isArray(data) && data.length > 0) ? data : defaultData;

  const avgAccuracy = (
    chartData.reduce((sum, item) => sum + (item.accuracy || 0), 0) / chartData.length
  ).toFixed(1);

  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200">
      <div className="mb-4">
        <h3 className="text-lg font-semibold">Model Accuracy Tracking</h3>
        <p className="text-sm text-gray-500">
          Average Accuracy: <span className="font-semibold text-blue-600">{avgAccuracy}%</span>
        </p>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis domain={[0, 100]} />
          <Tooltip formatter={(value) => `${value}%`} />
          <Line
            type="monotone"
            dataKey="accuracy"
            stroke="#3b82f6"
            strokeWidth={3}
            dot={{ fill: "#3b82f6", r: 5 }}
            name="Accuracy %"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default AccuracyChart;
