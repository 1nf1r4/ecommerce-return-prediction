import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  CheckCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  SparklesIcon,
  LightBulbIcon,
  ChartBarIcon,
} from "@heroicons/react/24/outline";

const PredictionResult = ({ result, showFactors = true }) => {
  // Empty State - Enhanced
  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center py-16 px-6 bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 rounded-xl border-2 border-dashed border-blue-300 shadow-inner transition-all hover:shadow-md">
        <div className="bg-white p-5 rounded-full shadow-lg mb-5 ring-4 ring-blue-100 animate-pulse">
          <InformationCircleIcon className="h-14 w-14 text-blue-500" />
        </div>
        <h3 className="text-2xl font-bold text-gray-900 mb-3 text-center flex items-center justify-center gap-2">
          <SparklesIcon className="h-7 w-7 text-blue-600" />
          Ready to Predict Returns
        </h3>
        <p className="text-base text-gray-700 text-center max-w-md mb-6 leading-relaxed">
          Fill in the order details above and click the{" "}
          <span className="font-bold text-blue-600">Predict Return Risk</span>{" "}
          button to get instant AI-powered insights.
        </p>
        <div className="flex flex-wrap items-center justify-center gap-4 text-sm text-gray-600 bg-white/50 backdrop-blur-sm px-6 py-3 rounded-full">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 bg-green-500 rounded-full animate-pulse"></span>
            <span className="font-medium">AI-powered</span>
          </div>
          <div className="w-px h-4 bg-gray-300"></div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-pulse"></span>
            <span className="font-medium">Instant results</span>
          </div>
          <div className="w-px h-4 bg-gray-300"></div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 bg-purple-500 rounded-full animate-pulse"></span>
            <span className="font-medium">Actionable insights</span>
          </div>
        </div>
      </div>
    );
  }

  const { probability, riskLevel, confidence, recommendations, factors } =
    result;

  const getRiskColor = (risk) => {
    switch (risk?.toUpperCase()) {
      case "HIGH":
        return "bg-red-100 text-red-800 border-red-200";
      case "MEDIUM":
        return "bg-yellow-100 text-yellow-800 border-yellow-200";
      case "LOW":
        return "bg-green-100 text-green-800 border-green-200";
      default:
        return "bg-gray-100 text-gray-800 border-gray-200";
    }
  };

  const getRiskIcon = (risk) => {
    switch (risk?.toUpperCase()) {
      case "HIGH":
        return <ExclamationTriangleIcon className="h-6 w-6 text-red-600" />;
      case "MEDIUM":
        return <InformationCircleIcon className="h-6 w-6 text-yellow-600" />;
      case "LOW":
        return <CheckCircleIcon className="h-6 w-6 text-green-600" />;
      default:
        return null;
    }
  };

  const getRecommendationText = (risk) => {
    switch (risk?.toUpperCase()) {
      case "HIGH":
        return "Consider quality inspection before shipment and include prepaid return label";
      case "MEDIUM":
        return "Include clear product description and easy return instructions";
      case "LOW":
        return "Process normally - low return risk detected";
      default:
        return "Process according to standard procedures";
    }
  };

  const getUserFriendlyMessage = (risk) => {
    switch (risk?.toUpperCase()) {
      case "HIGH":
        return {
          title: "Higher Chance of Return",
          message:
            "This order has characteristics that suggest it might be returned. We recommend taking extra care with quality checks and packaging.",
        };
      case "MEDIUM":
        return {
          title: "Moderate Return Possibility",
          message:
            "This order shows some patterns that could lead to a return. Standard precautions should be sufficient.",
        };
      case "LOW":
        return {
          title: "Low Return Likelihood",
          message:
            "Great news! This order looks very promising with minimal return indicators. Process as usual.",
        };
      default:
        return {
          title: "Analysis Complete",
          message:
            "The order has been analyzed. Please review the details below.",
        };
    }
  };

  const friendlyMessage = getUserFriendlyMessage(riskLevel);

  return (
    <div className="space-y-3 animate-in fade-in duration-500">
      {/* Main Result Card - Enhanced & Compact */}
      <div
        className={`p-5 rounded-xl border-2 shadow-lg ${getRiskColor(
          riskLevel
        )} transition-all hover:shadow-xl`}
      >
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0 transform transition-transform hover:scale-110">
            {getRiskIcon(riskLevel)}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-xl font-bold text-gray-900 flex items-center gap-2">
                {friendlyMessage.title}
              </h3>
              <Badge
                variant="outline"
                className={`${getRiskColor(
                  riskLevel
                )} font-bold text-xs px-3 py-1 shadow-sm`}
              >
                {riskLevel} RISK
              </Badge>
            </div>
            <p className="text-sm text-gray-800 mb-4 leading-relaxed font-medium">
              {friendlyMessage.message}
            </p>

            {/* Technical Details - Compact Rectangle */}
            <div className="mt-4 pt-3 border-t border-gray-300/50">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-1 h-4 bg-blue-500 rounded-full"></div>
                <p className="text-xs text-gray-600 font-semibold uppercase tracking-wide">
                  Technical Metrics
                </p>
              </div>
              <div className="grid grid-cols-2 gap-3 bg-white/60 backdrop-blur-sm p-3 rounded-lg border border-gray-300/50 shadow-sm">
                <div className="space-y-1">
                  <p className="text-xs text-gray-600 font-medium">
                    Return Probability
                  </p>
                  <div className="flex items-center gap-2">
                    <Progress value={probability} className="flex-1 h-1.5" />
                    <span className="text-sm font-bold text-gray-800 min-w-[35px] text-right">
                      {probability}%
                    </span>
                  </div>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-gray-600 font-medium">
                    Confidence Score
                  </p>
                  <div className="flex items-center gap-2">
                    <Progress value={confidence} className="flex-1 h-1.5" />
                    <span className="text-sm font-bold text-gray-800 min-w-[35px] text-right">
                      {confidence}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recommendations - Enhanced */}
      <div className="bg-gradient-to-br from-blue-50 to-cyan-50 p-4 rounded-xl border border-blue-200 shadow-md">
        <h4 className="text-lg font-bold mb-2 flex items-center gap-2 text-gray-900">
          <div className="bg-blue-500 p-1.5 rounded-lg">
            <LightBulbIcon className="h-5 w-5 text-white" />
          </div>
          <span>What Should You Do?</span>
        </h4>
        <p className="text-xs text-gray-600 mb-3 ml-9">
          Recommended actions based on this prediction:
        </p>
        <div className="space-y-2">
          {recommendations && recommendations.length > 0 ? (
            recommendations.map((rec, idx) => (
              <div
                key={idx}
                className="flex items-start gap-3 bg-white p-3 rounded-lg shadow-sm border border-blue-100 hover:border-blue-300 transition-all"
              >
                <div className="flex-shrink-0 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center mt-0.5">
                  <CheckCircleIcon className="h-4 w-4 text-white stroke-[3]" />
                </div>
                <p className="text-sm text-gray-800 leading-relaxed font-medium">
                  {rec}
                </p>
              </div>
            ))
          ) : (
            <div className="bg-white p-3 rounded-lg shadow-sm border border-blue-100">
              <p className="text-sm text-gray-800 leading-relaxed font-medium">
                {getRecommendationText(riskLevel)}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Contributing Factors - Enhanced */}
      {showFactors && factors && factors.length > 0 && (
        <div className="bg-gradient-to-br from-purple-50 to-pink-50 p-4 rounded-xl border border-purple-200 shadow-md">
          <h4 className="text-lg font-bold mb-2 text-gray-900 flex items-center gap-2">
            <div className="bg-purple-500 p-1.5 rounded-lg">
              <ChartBarIcon className="h-5 w-5 text-white" />
            </div>
            <span>Why This Prediction?</span>
          </h4>
          <p className="text-xs text-gray-600 mb-3">
            Key factors that influenced this prediction:
          </p>
          <div className="space-y-2">
            {factors.map((factor, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between py-2.5 px-3 bg-white rounded-lg border border-purple-100 shadow-sm hover:shadow-md transition-all"
              >
                <div className="flex-1 min-w-0 pr-3">
                  <p className="text-sm font-bold text-gray-900 mb-0.5">
                    {factor.factor}
                  </p>
                  <p className="text-xs text-gray-600 font-medium">
                    {factor.value}
                  </p>
                </div>
                <Badge
                  variant="outline"
                  className={`text-xs font-semibold flex-shrink-0 px-2.5 py-1 shadow-sm ${
                    factor.impact === "High"
                      ? "bg-red-100 text-red-800 border-red-300"
                      : factor.impact === "Medium"
                      ? "bg-yellow-100 text-yellow-800 border-yellow-300"
                      : "bg-green-100 text-green-800 border-green-300"
                  }`}
                >
                  {factor.impact}
                </Badge>
              </div>
            ))}
          </div>
          <div className="mt-3 pt-3 border-t border-purple-200/50">
            <p className="text-xs text-gray-600 italic flex items-center gap-2">
              <span className="w-1.5 h-1.5 bg-purple-400 rounded-full"></span>
              <span>
                These factors help explain how we arrived at this prediction
              </span>
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionResult;
