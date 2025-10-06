import { Badge } from "@/components/ui/badge";
import DataTable from "../common/DataTable";

const RecentPredictionsTable = ({ predictions, onRowClick }) => {
  const getRiskBadge = (riskLevel) => {
    const colors = {
      HIGH: "bg-red-100 text-red-800",
      MEDIUM: "bg-yellow-100 text-yellow-800",
      LOW: "bg-green-100 text-green-800",
    };
    return (
      <Badge className={colors[riskLevel] || "bg-gray-100 text-gray-800"}>
        {riskLevel}
      </Badge>
    );
  };

  const columns = [
    {
      key: "timestamp",
      label: "Date & Time",
      sortable: true,
      render: (val) => new Date(val).toLocaleString(),
    },
    {
      key: "category",
      label: "Category",
      sortable: true,
    },
    {
      key: "orderValue",
      label: "Order Value",
      sortable: true,
      render: (val) => `$${val?.toFixed(2) || "0.00"}`,
    },
    {
      key: "returnProbability",
      label: "Return Probability",
      sortable: true,
      render: (val) => `${Math.round(val * 100)}%`,
    },
    {
      key: "riskLevel",
      label: "Risk Level",
      sortable: true,
      render: (val) => getRiskBadge(val),
    },
    {
      key: "status",
      label: "Status",
      render: (val) => (
        <Badge variant="outline">
          {val || "Completed"}
        </Badge>
      ),
    },
  ];

  return (
    <div className="bg-white rounded-lg border border-gray-200">
      <div className="p-6 border-b border-gray-200">
        <h3 className="text-lg font-semibold">Recent Predictions</h3>
        <p className="text-sm text-gray-500 mt-1">
          Latest prediction results from your orders
        </p>
      </div>
      <DataTable
        data={predictions}
        columns={columns}
        onRowClick={onRowClick}
      />
    </div>
  );
};

export default RecentPredictionsTable;
