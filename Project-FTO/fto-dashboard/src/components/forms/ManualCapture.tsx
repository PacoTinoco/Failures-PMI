'use client';

import { useState } from 'react';

interface KpiDefinition {
  code: string;
  name: string;
  category: string;
  categoryName: string;
  unit: string;
  sortOrder: number;
  minValue?: number;
  maxValue?: number;
  lowerIsBetter?: boolean;
}

interface ManualCaptureProps {
  operators?: { id: string; name: string }[];
  weeks?: { id: string; label: string }[];
  kpiDefinitions?: KpiDefinition[];
  onSubmit?: (data: { operator: string; week: string; values: Record<string, number> }) => void;
  onError?: (error: string) => void;
  isLoading?: boolean;
}

const SQDCM_CATEGORIES = {
  S: { name: 'Safety (Seguridad)', color: 'border-blue-500' },
  Q: { name: 'Quality (Calidad)', color: 'border-green-500' },
  D: { name: 'Delivery (Entrega)', color: 'border-yellow-500' },
  C: { name: 'Cost (Costo)', color: 'border-purple-500' },
  M: { name: 'Morale (Moral)', color: 'border-red-500' },
};

export default function ManualCapture({
  operators = [
    { id: 'op1', name: 'Operador 1' },
    { id: 'op2', name: 'Operador 2' },
    { id: 'op3', name: 'Operador 3' },
  ],
  weeks = [
    { id: 'w1', label: 'Semana 1' },
    { id: 'w2', label: 'Semana 2' },
    { id: 'w3', label: 'Semana 3' },
    { id: 'w4', label: 'Semana 4' },
  ],
  kpiDefinitions = [
    {
      code: 'S001',
      name: 'Accidentes',
      category: 'S',
      categoryName: 'Seguridad',
      unit: 'count',
      sortOrder: 1,
      minValue: 0,
      lowerIsBetter: true,
    },
    {
      code: 'Q001',
      name: 'Defectos',
      category: 'Q',
      categoryName: 'Calidad',
      unit: 'count',
      sortOrder: 2,
      minValue: 0,
      lowerIsBetter: true,
    },
  ],
  onSubmit = () => {},
  onError = () => {},
  isLoading = false,
}: ManualCaptureProps) {
  const [selectedOperator, setSelectedOperator] = useState<string>(
    operators[0]?.id || ''
  );
  const [selectedWeek, setSelectedWeek] = useState<string>(
    weeks[0]?.id || ''
  );
  const [kpiValues, setKpiValues] = useState<Record<string, string>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [successMessage, setSuccessMessage] = useState('');

  // Group KPIs by category
  const groupedKpis = kpiDefinitions.reduce(
    (acc, kpi) => {
      if (!acc[kpi.category]) {
        acc[kpi.category] = [];
      }
      acc[kpi.category].push(kpi);
      return acc;
    },
    {} as Record<string, KpiDefinition[]>
  );

  const categoryOrder = ['S', 'Q', 'D', 'C', 'M'];

  const handleInputChange = (kpiCode: string, value: string) => {
    setKpiValues((prev) => ({
      ...prev,
      [kpiCode]: value,
    }));

    // Clear error for this field
    if (errors[kpiCode]) {
      setErrors((prev) => ({
        ...prev,
        [kpiCode]: '',
      }));
    }
  };

  const validateInputs = (): boolean => {
    const newErrors: Record<string, string> = {};

    kpiDefinitions.forEach((kpi) => {
      const value = kpiValues[kpi.code];

      if (!value || value.trim() === '') {
        newErrors[kpi.code] = 'Este campo es requerido';
        return;
      }

      const numValue = parseFloat(value);

      if (isNaN(numValue)) {
        newErrors[kpi.code] = 'Debe ser un número válido';
        return;
      }

      if (kpi.minValue !== undefined && numValue < kpi.minValue) {
        newErrors[kpi.code] = `Mínimo: ${kpi.minValue}`;
        return;
      }

      if (kpi.maxValue !== undefined && numValue > kpi.maxValue) {
        newErrors[kpi.code] = `Máximo: ${kpi.maxValue}`;
        return;
      }
    });

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSuccessMessage('');

    if (!validateInputs()) {
      onError('Por favor, corrija los errores en el formulario');
      return;
    }

    const numericValues: Record<string, number> = {};
    Object.entries(kpiValues).forEach(([key, value]) => {
      numericValues[key] = parseFloat(value);
    });

    try {
      const response = await fetch('/api/records', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          operator: selectedOperator,
          week: selectedWeek,
          values: numericValues,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Error al guardar los datos');
      }

      setSuccessMessage('Datos guardados exitosamente');
      setKpiValues({});

      // Call the callback
      onSubmit({
        operator: selectedOperator,
        week: selectedWeek,
        values: numericValues,
      });

      // Clear success message after 3 seconds
      setTimeout(() => {
        setSuccessMessage('');
      }, 3000);
    } catch (err) {
      onError(err instanceof Error ? err.message : 'Error al guardar los datos');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-8">
      {/* Header Section */}
      <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
        <h2 className="text-2xl font-bold text-slate-100 mb-6">
          Captura Manual de Datos
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          {/* Operator Selection */}
          <div>
            <label htmlFor="operator" className="block text-sm font-semibold text-slate-300 mb-2">
              Operador
            </label>
            <select
              id="operator"
              value={selectedOperator}
              onChange={(e) => setSelectedOperator(e.target.value)}
              className="w-full bg-slate-700 border border-slate-600 text-slate-100 rounded-lg px-4 py-2 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            >
              {operators.map((op) => (
                <option key={op.id} value={op.id}>
                  {op.name}
                </option>
              ))}
            </select>
          </div>

          {/* Week Selection */}
          <div>
            <label htmlFor="week" className="block text-sm font-semibold text-slate-300 mb-2">
              Semana
            </label>
            <select
              id="week"
              value={selectedWeek}
              onChange={(e) => setSelectedWeek(e.target.value)}
              className="w-full bg-slate-700 border border-slate-600 text-slate-100 rounded-lg px-4 py-2 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            >
              {weeks.map((week) => (
                <option key={week.id} value={week.id}>
                  {week.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Success Message */}
      {successMessage && (
        <div className="bg-green-900/30 border border-green-700 rounded-lg p-4 flex items-center gap-3">
          <span className="text-2xl">✓</span>
          <div>
            <p className="font-semibold text-green-300">{successMessage}</p>
          </div>
        </div>
      )}

      {/* KPI Input Sections */}
      {categoryOrder.map((category) => {
        const kpisInCategory = groupedKpis[category] || [];
        if (kpisInCategory.length === 0) return null;

        const categoryInfo = SQDCM_CATEGORIES[category as keyof typeof SQDCM_CATEGORIES];

        return (
          <div
            key={category}
            className={`border-l-4 ${categoryInfo?.color} bg-slate-800 rounded-lg border border-slate-700 overflow-hidden`}
          >
            {/* Category Header */}
            <div className="bg-slate-750 px-6 py-4 border-b border-slate-700">
              <h3 className="text-lg font-bold text-slate-100">
                {category} - {categoryInfo?.name}
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                {kpisInCategory.length} métricas a registrar
              </p>
            </div>

            {/* KPI Inputs Grid */}
            <div className="p-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {kpisInCategory
                .sort((a, b) => a.sortOrder - b.sortOrder)
                .map((kpi) => (
                  <div key={kpi.code} className="space-y-2">
                    <label
                      htmlFor={kpi.code}
                      className="block text-sm font-semibold text-slate-300"
                    >
                      {kpi.name}
                    </label>
                    <div className="flex gap-2">
                      <input
                        id={kpi.code}
                        type="number"
                        step="0.01"
                        placeholder="Ingrese valor"
                        value={kpiValues[kpi.code] || ''}
                        onChange={(e) =>
                          handleInputChange(kpi.code, e.target.value)
                        }
                        className={`flex-1 bg-slate-700 border rounded-lg px-3 py-2 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 ${
                          errors[kpi.code]
                            ? 'border-red-500 focus:border-red-500 focus:ring-red-500'
                            : 'border-slate-600 focus:border-blue-500 focus:ring-blue-500'
                        }`}
                      />
                      <span className="px-3 py-2 bg-slate-700 rounded-lg text-slate-400 text-sm font-medium whitespace-nowrap">
                        {kpi.unit}
                      </span>
                    </div>
                    {errors[kpi.code] && (
                      <p className="text-xs text-red-400 font-medium">
                        {errors[kpi.code]}
                      </p>
                    )}
                    {kpi.minValue !== undefined && !errors[kpi.code] && (
                      <p className="text-xs text-slate-400">
                        Mín: {kpi.minValue}
                        {kpi.maxValue !== undefined && ` • Máx: ${kpi.maxValue}`}
                      </p>
                    )}
                  </div>
                ))}
            </div>
          </div>
        );
      })}

      {/* Action Buttons */}
      <div className="flex gap-3 pt-4">
        <button
          type="submit"
          disabled={isLoading}
          className="flex-1 px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-600 text-white font-semibold rounded-lg transition-colors flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <>
              <span className="animate-spin">⟳</span>
              Guardando...
            </>
          ) : (
            <>
              <span>✓</span>
              Guardar datos
            </>
          )}
        </button>
        <button
          type="button"
          onClick={() => {
            setKpiValues({});
            setErrors({});
            setSuccessMessage('');
          }}
          className="px-6 py-3 bg-slate-700 hover:bg-slate-600 text-slate-200 font-semibold rounded-lg transition-colors"
        >
          Limpiar
        </button>
      </div>
    </form>
  );
}
