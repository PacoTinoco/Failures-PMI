export interface KpiDefinition {
  code: string
  name: string
  category: string
  categoryName: string
  unit: string
  defaultObjective: number | null
  sortOrder: number
}

export interface WeeklyKpiData {
  week: string
  values: Record<string, number>
}

export interface OperatorData {
  id: string
  name: string
  coordinatorName: string
  kpis: KpiDefinition[]
  objectives: Record<string, number>
  weeklyData: WeeklyKpiData[]
}

export interface DashboardData {
  machine: { id: string; name: string; code: string }
  coordinators: {
    id: string
    name: string
    operators: OperatorData[]
  }[]
}

export const SQDCM_COLORS: Record<string, string> = {
  S: '#10B981',
  Q: '#3B82F6',
  D: '#F59E0B',
  C: '#EF4444',
  M: '#8B5CF6'
}

export const CATEGORY_MAP: Record<string, { letter: string; color: string }> =
  {
    Sustentabilidad: { letter: 'S', color: '#10B981' },
    Calidad: { letter: 'Q', color: '#3B82F6' },
    Desempeño: { letter: 'D', color: '#F59E0B' },
    Costo: { letter: 'C', color: '#EF4444' },
    Moral: { letter: 'M', color: '#8B5CF6' }
  }
