'use client';

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

interface ComparisonChartProps {
  title: string;
  data: { name: string; value: number }[];
  objective?: number;
  color?: string;
  height?: number;
}

export default function ComparisonChart({
  title,
  data,
  objective,
  color = '#3b82f6',
  height = 350,
}: ComparisonChartProps) {
  const tooltipStyle = {
    backgroundColor: '#1e293b',
    border: '1px solid #475569',
    borderRadius: '8px',
  };

  return (
    <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
      <h2 className="text-xl font-bold text-slate-100 mb-6">{title}</h2>

      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={data}
          margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#334155"
            vertical={false}
          />
          <XAxis
            dataKey="name"
            stroke="#94a3b8"
            style={{ fontSize: '12px' }}
            angle={-45}
            textAnchor="end"
            height={80}
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
          <Legend wrapperStyle={{ paddingTop: '20px' }} />

          {/* Reference line for objective */}
          {objective && (
            <ReferenceLine
              y={objective}
              stroke="#94a3b8"
              strokeDasharray="5 5"
              label={{
                value: `Objetivo: ${objective.toFixed(2)}`,
                position: 'insideTopRight',
                offset: -10,
                fill: '#94a3b8',
                fontSize: 11,
              }}
            />
          )}

          <Bar
            dataKey="value"
            fill={color}
            name="Valor"
            radius={[8, 8, 0, 0]}
            isAnimationActive={true}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
