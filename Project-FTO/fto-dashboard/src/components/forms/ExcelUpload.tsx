'use client';

import { useRef, useState } from 'react';

interface UploadedData {
  [key: string]: string | number;
}

interface ExcelUploadProps {
  onDataParsed?: (data: UploadedData[]) => void;
  onError?: (error: string) => void;
  acceptedFormats?: string[];
}

export default function ExcelUpload({
  onDataParsed = () => {},
  onError = () => {},
  acceptedFormats = ['.xlsx', '.xls', '.csv'],
}: ExcelUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [previewData, setPreviewData] = useState<UploadedData[]>([]);
  const [fileName, setFileName] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const validateFile = (file: File): boolean => {
    const fileName = file.name.toLowerCase();
    const isValidFormat = acceptedFormats.some((format) =>
      fileName.endsWith(format)
    );

    if (!isValidFormat) {
      onError(
        `Invalid file format. Accepted formats: ${acceptedFormats.join(', ')}`
      );
      return false;
    }

    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      onError('File size exceeds 10MB limit');
      return false;
    }

    return true;
  };

  const parseFile = async (file: File) => {
    try {
      setIsUploading(true);
      setUploadProgress(0);
      setFileName(file.name);

      const reader = new FileReader();

      reader.onprogress = (event) => {
        if (event.lengthComputable) {
          const progress = (event.loaded / event.total) * 100;
          setUploadProgress(progress);
        }
      };

      reader.onload = async (event) => {
        try {
          const content = event.target?.result as string;

          // Simple CSV parsing for demonstration
          const lines = content.split('\n');
          const headers = lines[0].split(',').map((h) => h.trim());
          const data: UploadedData[] = [];

          for (let i = 1; i < Math.min(lines.length, 11); i++) {
            if (lines[i].trim()) {
              const values = lines[i].split(',');
              const row: UploadedData = {};

              headers.forEach((header, index) => {
                const value = values[index]?.trim() || '';
                // Try to convert to number
                row[header] = isNaN(Number(value))
                  ? value
                  : Number(value);
              });

              data.push(row);
            }
          }

          setPreviewData(data);
          setUploadProgress(100);

          // Simulate API call delay
          setTimeout(() => {
            setIsUploading(false);
          }, 500);
        } catch (err) {
          onError('Error parsing file. Please ensure it is a valid CSV format.');
          setIsUploading(false);
        }
      };

      reader.onerror = () => {
        onError('Error reading file');
        setIsUploading(false);
      };

      reader.readAsText(file);
    } catch (err) {
      onError('Error processing file');
      setIsUploading(false);
    }
  };

  const handleFileSelect = (file: File) => {
    if (validateFile(file)) {
      parseFile(file);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.currentTarget.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleConfirmUpload = async () => {
    if (previewData.length === 0) {
      onError('No data to upload');
      return;
    }

    try {
      setIsUploading(true);

      const response = await fetch('/api/upload', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          filename: fileName,
          data: previewData,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Upload failed');
      }

      const result = await response.json();
      onDataParsed(previewData);
      setPreviewData([]);
      setFileName('');
      setUploadProgress(0);

      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (err) {
      onError(
        err instanceof Error ? err.message : 'Error uploading file'
      );
    } finally {
      setIsUploading(false);
    }
  };

  const handleCancel = () => {
    setPreviewData([]);
    setFileName('');
    setUploadProgress(0);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      {previewData.length === 0 && (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-lg p-8 transition-all ${
            isDragging
              ? 'border-blue-500 bg-blue-900/10'
              : 'border-slate-600 bg-slate-800/30 hover:border-slate-500'
          }`}
        >
          <div className="flex flex-col items-center gap-4">
            <div className="text-4xl">📁</div>
            <div className="text-center">
              <h3 className="text-lg font-semibold text-slate-100 mb-2">
                Arrastra archivos aquí o haz clic
              </h3>
              <p className="text-sm text-slate-400 mb-4">
                Formatos aceptados: {acceptedFormats.join(', ')} (máx. 10MB)
              </p>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept={acceptedFormats.join(',')}
              onChange={handleInputChange}
              className="hidden"
            />

            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
            >
              Seleccionar archivo
            </button>
          </div>

          {/* Progress */}
          {isUploading && uploadProgress > 0 && uploadProgress < 100 && (
            <div className="mt-4">
              <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-blue-500 h-full transition-all"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <p className="text-xs text-slate-400 mt-2 text-center">
                {uploadProgress.toFixed(0)}% cargado
              </p>
            </div>
          )}
        </div>
      )}

      {/* Preview */}
      {previewData.length > 0 && (
        <div className="space-y-4">
          <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="text-sm font-semibold text-slate-300">
                  {fileName}
                </p>
                <p className="text-xs text-slate-400 mt-1">
                  {previewData.length} registros encontrados
                </p>
              </div>
              <span className="text-2xl">✓</span>
            </div>

            {/* Data preview table */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700">
                    {Object.keys(previewData[0]).map((key) => (
                      <th
                        key={key}
                        className="text-left px-3 py-2 font-semibold text-slate-300"
                      >
                        {key}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {previewData.slice(0, 5).map((row, idx) => (
                    <tr
                      key={idx}
                      className="border-b border-slate-700 hover:bg-slate-700/50"
                    >
                      {Object.keys(row).map((key) => (
                        <td
                          key={key}
                          className="px-3 py-2 text-slate-200"
                        >
                          {row[key]}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {previewData.length > 5 && (
              <p className="text-xs text-slate-400 mt-2">
                ... y {previewData.length - 5} registros más
              </p>
            )}
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3">
            <button
              onClick={handleConfirmUpload}
              disabled={isUploading}
              className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-slate-600 text-white font-medium rounded-lg transition-colors"
            >
              {isUploading ? 'Enviando...' : 'Confirmar carga'}
            </button>
            <button
              onClick={handleCancel}
              disabled={isUploading}
              className="px-4 py-2 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-600 text-slate-200 font-medium rounded-lg transition-colors"
            >
              Cancelar
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
