'use client';

import KpiCard from '@/components/ui/KpiCard';

interface KpiDefinition {
  code: string;
  name: string;
  category: string;
  categoryName: string;
  unit: string;
  sortOrder: number;
  lowerIsBetter?: boolean;
}

interface SqdcmScorecardProps {
  operatorName: string;
  kpiValues: Record<string, number | null>;
  objectives: Record<string, number | null>;
  kpiDefinitions: KpiDefinition[];
  onKpiClick?: (kpiCode: string) => void;
  height?: string;
}

const SQDCM_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  S: {
    bg: 'bg-blue-900/20',
    text: 'text-blue-400',
    border: 'border-blue-500/30',
  },
  Q: {
    bg: 'bg-green-900/20',
    text: 'text-green-400',
    border: 'border-green-500/30',
  },
  D: {
    bg: 'bg-yellow-900/20',
    text: 'text-yellow-400',
    border: 'border-yellow-500/30',
  },
  C: {
    bg: 'bg-purple-900/20',
    text: 'text-purple-400',
    border: 'border-purple-500/30',
  },
  M: {
    bg: 'bg-red-900/20',
    text: 'text-red-400',
    border: 'border-red-500/30',
  },
};

const categoryNames: Record<string, string> = {
  S: 'Safety (Seguridad)',
  Q: 'Quality (Calidad)',
  D: 'Delivery (Entrega)',
  C: 'Cost (Costo)',
  M: 'Morale (Moral)',
};

export default function SqdcmScorecard({
  operatorName,
  kpiValues,
  objectives,
  kpiDefinitions,
  onKpiClick = () => {},
}: SqdcmScorecardProps) {
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

  // Sort categories in SQDCM order
  const categoryOrder = ['S', 'Q', 'D', 'C', 'M'];
  const sortedCategories = categoryOrder.filter((cat) => groupedKpis[cat]);

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-slate-100 mb-2">{operatorName}</h2>
        <p className="text-sm text-slate-400">Desempeño de KPIs SQDCM</p>
      </div>

      {sortedCategories.map((category) => {
        const kpisInCategory = groupedKpis[category] || [];
        const colors = SQDCM_COLORS[category] || SQDCM_COLORS.S;

        return (
          <div key={category} className="space-y-4">
            {/* Category Header */}
            <div className={`${colors.bg} border ${colors.border} rounded-lg px-4 py-3 border-l-4`}>
              <h3 className={`text-lg font-bold ${colors.text}`}>
                {category} - {categoryNames[category]}
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                {kpisInCategory.length} métricas
              </p>
            </div>

            {/* KPI Cards Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {kpisInCategory
                .sort((a, b) => a.sortOrder - b.sortOrder)
                .map((kpi) => (
                  <KpiCard
                    key={kpi.code}
                    name={kpi.name}
                    value={kpiValues[kpi.code] ?? null}
                    objective={objectives[kpi.code] ?? null}
                    category={category}
                    categoryName={kpi.categoryName}
                    unit={kpi.unit}
                    lowerIsBetter={kpi.lowerIsBetter ?? false}
                    onClick={() => onKpiClick(kpi.code)}
                  />
                ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
