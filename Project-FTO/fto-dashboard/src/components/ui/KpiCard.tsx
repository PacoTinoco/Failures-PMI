'use client';

interface KpiCardProps {
  name: string;
  value: number | null;
  objective: number | null;
  category: string; // S, Q, D, C, M
  categoryName: string;
  trend?: number[];
  onClick?: () => void;
  unit?: string;
  lowerIsBetter?: boolean;
}

const categoryColors: Record<string, { bg: string; border: string; text: string }> = {
  S: { bg: 'from-blue-900 to-blue-800', border: 'border-l-4 border-l-blue-500', text: 'text-blue-400' },
  Q: { bg: 'from-green-900 to-green-800', border: 'border-l-4 border-l-green-500', text: 'text-green-400' },
  D: { bg: 'from-yellow-900 to-yellow-800', border: 'border-l-4 border-l-yellow-500', text: 'text-yellow-400' },
  C: { bg: 'from-purple-900 to-purple-800', border: 'border-l-4 border-l-purple-500', text: 'text-purple-400' },
  M: { bg: 'from-red-900 to-red-800', border: 'border-l-4 border-l-red-500', text: 'text-red-400' },
};

function getTrafficLightStatus(
  value: number | null,
  objective: number | null,
  lowerIsBetter: boolean = false
): 'green' | 'yellow' | 'red' {
  if (value === null || objective === null) return 'red';

  const percentage = lowerIsBetter
    ? (objective / value) * 100
    : (value / objective) * 100;

  if (percentage >= 100) return 'green';
  if (percentage >= 80) return 'yellow';
  return 'red';
}

function getStatusColor(status: 'green' | 'yellow' | 'red'): string {
  const colors = {
    green: 'bg-green-600 text-green-100',
    yellow: 'bg-yellow-600 text-yellow-100',
    red: 'bg-red-600 text-red-100',
  };
  return colors[status];
}

export default function KpiCard({
  name,
  value,
  objective,
  category,
  categoryName,
  trend = [],
  onClick,
  unit = '',
  lowerIsBetter = false,
}: KpiCardProps) {
  const colors = categoryColors[category] || categoryColors.S;
  const status = getTrafficLightStatus(value, objective, lowerIsBetter);
  const statusColor = getStatusColor(status);

  const percentage = value !== null && objective !== null
    ? lowerIsBetter
      ? ((objective / value) * 100).toFixed(1)
      : ((value / objective) * 100).toFixed(1)
    : 'N/A';

  return (
    <div
      onClick={onClick}
      className={`${colors.border} bg-gradient-to-br ${colors.bg} rounded-lg p-4 cursor-pointer hover:shadow-lg transition-all duration-200 hover:scale-105 border border-slate-700`}
    >
      {/* Header with category */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1 min-w-0">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            {categoryName}
          </p>
          <h3 className="text-sm font-bold text-slate-100 truncate">{name}</h3>
        </div>
        <div className={`px-2 py-1 rounded text-xs font-bold whitespace-nowrap ml-2 ${statusColor}`}>
          {status === 'green' ? '✓' : status === 'yellow' ? '⚠' : '✗'}
        </div>
      </div>

      {/* Main Value */}
      <div className="mb-4">
        <div className="flex items-baseline gap-2">
          <span className="text-3xl font-bold text-slate-100">
            {value !== null ? value.toFixed(2) : '—'}
          </span>
          {unit && <span className="text-sm text-slate-400">{unit}</span>}
        </div>
        {objective !== null && (
          <p className="text-xs text-slate-400 mt-1">
            Objetivo: {objective.toFixed(2)} ({percentage}%)
          </p>
        )}
      </div>

      {/* Sparkline visualization */}
      {trend && trend.length > 0 && (
        <div className="mb-3 h-10 flex items-end gap-1">
          {trend.map((val, idx) => {
            const maxVal = Math.max(...trend, objective || 0);
            const height = (val / maxVal) * 100;
            return (
              <div
                key={idx}
                className="flex-1 bg-slate-600 opacity-70 rounded-t"
                style={{ height: `${Math.max(height, 10)}%` }}
                title={`Week ${idx + 1}: ${val.toFixed(2)}`}
              />
            );
          })}
        </div>
      )}

      {/* Performance indicator bar */}
      <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            status === 'green' ? 'bg-green-500' : status === 'yellow' ? 'bg-yellow-500' : 'bg-red-500'
          }`}
          style={{
            width: `${Math.min(
              lowerIsBetter ? ((objective || 0) / (value || 1)) * 100 : ((value || 0) / (objective || 1)) * 100,
              100
            )}%`,
          }}
        />
      </div>
    </div>
  );
}
