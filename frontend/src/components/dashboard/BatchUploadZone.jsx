import { useState, useCallback } from "react";
import { CloudArrowUpIcon, DocumentIcon } from "@heroicons/react/24/outline";

const BatchUploadZone = ({ onFileUpload, isProcessing }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    const validTypes = [
      "text/csv",
      "application/vnd.ms-excel",
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ];

    if (!validTypes.includes(file.type) && !file.name.endsWith('.csv')) {
      alert("Please upload a CSV or Excel file");
      return;
    }

    setSelectedFile(file);
  };

  const handleUpload = () => {
    if (selectedFile && onFileUpload) {
      onFileUpload(selectedFile);
    }
  };

  const downloadTemplate = () => {
    const template = `product_category,price,quantity,age,gender,location,payment_method,shipping_method,discount_applied
Electronics,299.99,1,28,Female,Urban,Credit Card,Standard,10
Clothing,49.99,2,35,Male,Suburban,PayPal,Express,15
Books,19.99,1,42,Female,Rural,Debit Card,Standard,0`;

    const blob = new Blob([template], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "batch_upload_template.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-4">
      {/* Template Download */}
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Upload Batch Orders</h3>
        <button
          onClick={downloadTemplate}
          className="text-sm text-blue-600 hover:text-blue-700 underline"
        >
          Download CSV Template
        </button>
      </div>

      {/* Upload Zone */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`
          relative border-2 border-dashed rounded-lg p-8 text-center transition-colors
          ${dragActive ? "border-blue-500 bg-blue-50" : "border-gray-300 bg-gray-50"}
          ${isProcessing ? "opacity-50 pointer-events-none" : ""}
        `}
      >
        <input
          type="file"
          accept=".csv,.xlsx,.xls"
          onChange={handleChange}
          disabled={isProcessing}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />

        <CloudArrowUpIcon className="mx-auto h-12 w-12 text-gray-400" />
        <p className="mt-2 text-sm text-gray-600">
          {dragActive
            ? "Drop your file here"
            : "Drag and drop your CSV/Excel file, or click to browse"}
        </p>
        <p className="mt-1 text-xs text-gray-500">
          Supported formats: CSV, XLSX (Max 10,000 orders)
        </p>
      </div>

      {/* Selected File */}
      {selectedFile && (
        <div className="flex items-center justify-between p-4 bg-white border border-gray-200 rounded-lg">
          <div className="flex items-center gap-3">
            <DocumentIcon className="h-8 w-8 text-gray-400" />
            <div>
              <p className="text-sm font-medium text-gray-900">
                {selectedFile.name}
              </p>
              <p className="text-xs text-gray-500">
                {(selectedFile.size / 1024).toFixed(2)} KB
              </p>
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setSelectedFile(null)}
              disabled={isProcessing}
              className="px-3 py-1 text-sm text-gray-600 hover:text-gray-800"
            >
              Remove
            </button>
            <button
              onClick={handleUpload}
              disabled={isProcessing}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
            >
              {isProcessing ? "Processing..." : "Upload & Process"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default BatchUploadZone;
