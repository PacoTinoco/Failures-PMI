'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

interface TrendChartProps {
  title: string;
  data: { week: string; [kpi: string]: number | string }[];
  kpis: string[];
  colors: string[];
  objectives?: Record<string, number>;
  height?: number;
}

const defaultColors = ['#3b82f6', '#10b981', '#f59e0b', '#a855f7', '#ef4444'];

export default function TrendChart({
  title,
  data,
  kpis,
  colors = defaultColors,
  objectives = {},
  height = 400,
}: TrendChartProps) {
  const tooltipStyle = {
    backgroundColor: '#1e293b',
    border: '1px solid #475569',
    borderRadius: '8px',
  };

  return (
    <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
      <h2 className="text-xl font-bold text-slate-100 mb-6">{title}</h2>

      <ResponsiveContainer width="100%" height={height}>
        <LineChart
          data={data}
          margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#334155"
            vertical={false}
          />
          <XAxis
            dataKey="week"
            stroke="#94a3b8"
            style={{ fontSize: '12px' }}
          />
          <YAxis
            stroke="#94a3b8"
            style={{ fontSize: '12px' }}
          />
          <Tooltip
            contentStyle={tooltipStyle}
            labelStyle={{ color: '#e2e8f0' }}
            itemStyle={{ color: '#e2e8f0' }}
            formatter={(value: any) => {
              if (typeof value === 'number') return value.toFixed(2);
              return value;
            }}
          />
          <Legend
            wrapperStyle={{ paddingTop: '20px' }}
            iconType="line"
          />

          {/* Reference lines for objectives */}
          {Object.entries(objectives).map(([kpiName, objective]) => (
            <ReferenceLine
              key={`ref-${kpiName}`}
              dataKey={kpiName}
              y={objective}
              stroke={colors[kpis.indexOf(kpiName)] || '#475569'}
              strokeDasharray="5 5"
              strokeOpacity={0.5}
              label={{
                value: `${kpiName} Obj.`,
                position: 'insideTopRight',
                offset: -10,
                fill: '#94a3b8',
                fontSize: 11,
              }}
            />
          ))}

          {/* Data lines */}
          {kpis.map((kpi, idx) => (
            <Line
              key={kpi}
              type="monotone"
              dataKey={kpi}
              stroke={colors[idx] || '#94a3b8'}
              strokeWidth={2}
              dot={{ fill: colors[idx] || '#94a3b8', r: 4 }}
              activeDot={{ r: 6 }}
              isAnimationActive={true}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
