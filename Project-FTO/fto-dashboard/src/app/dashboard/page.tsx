'use client'

import { useState, useEffect } from 'react'
import { useSession } from 'next-auth/react'
import { useRouter } from 'next/navigation'
import Sidebar from '@/components/layout/Sidebar'
import Header from '@/components/layout/Header'
import SqdcmScorecard from '@/components/charts/SqdcmScorecard'
import TrendChart from '@/components/charts/TrendChart'
import ComparisonChart from '@/components/charts/ComparisonChart'

interface Machine {
  id: string
  name: string
}

interface DashboardData {
  machines: Machine[]
  coordinators: string[]
  operators: string[]
  sqdcmScores: Record<string, number>
  trendData: Record<string, any[]>
  comparisonData: any
  weeklyData: any[]
}

export default function DashboardPage() {
  const { data: session, status } = useSession()
  const router = useRouter()

  // State management
  const [selectedMachine, setSelectedMachine] = useState<string>('')
  const [selectedCoordinator, setSelectedCoordinator] = useState<string>('')
  const [selectedOperator, setSelectedOperator] = useState<string>('')
  const [selectedKpi, setSelectedKpi] = useState<string>('seguridad')
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Redirect if not authenticated
  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/login')
    }
  }, [status, router])

  // Fetch dashboard data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true)
        setError(null)

        const params = new URLSearchParams()
        if (selectedMachine) params.append('machineId', selectedMachine)
        if (selectedCoordinator) params.append('coordinator', selectedCoordinator)
        if (selectedOperator) params.append('operator', selectedOperator)

        const response = await fetch(`/api/dashboard?${params.toString()}`)
        if (!response.ok) throw new Error('Error al cargar datos')

        const data = await response.json()
        setDashboardData(data)

        // Set default selections
        if (data.machines.length > 0 && !selectedMachine) {
          setSelectedMachine(data.machines[0].id)
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Error desconocido')
      } finally {
        setIsLoading(false)
      }
    }

    fetchData()
  }, [selectedMachine, selectedCoordinator, selectedOperator])

  if (status === 'loading' || isLoading) {
    return (
      <div className="flex h-screen bg-slate-900">
        <Sidebar user={session?.user} />
        <main className="flex-1 flex items-center justify-center">
          <div className="text-slate-300">Cargando...</div>
        </main>
      </div>
    )
  }

  if (!session) {
    return null
  }

  const machines = dashboardData?.machines || []
  const coordinators = dashboardData?.coordinators || []
  const operators = dashboardData?.operators || []

  return (
    <div className="flex h-screen bg-slate-900">
      <Sidebar user={session.user} />
      <main className="flex-1 flex flex-col overflow-hidden">
        <Header
          title="Panel de Control - KPI"
          userName={session.user?.name || 'Usuario'}
        />

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-6 space-y-6">
            {/* Error Message */}
            {error && (
              <div className="p-4 bg-red-900/20 border border-red-700 rounded-lg">
                <p className="text-red-400">{error}</p>
              </div>
            )}

            {/* Machine Selector */}
            <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
              <label className="block text-sm font-medium text-slate-300 mb-3">
                Seleccionar Máquina
              </label>
              <select
                value={selectedMachine}
                onChange={(e) => {
                  setSelectedMachine(e.target.value)
                  setSelectedCoordinator('')
                  setSelectedOperator('')
                }}
                className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Seleccione una máquina</option>
                {machines.map((machine) => (
                  <option key={machine.id} value={machine.id}>
                    {machine.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Coordinator Tabs */}
            {coordinators.length > 0 && (
              <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                <label className="block text-sm font-medium text-slate-300 mb-3">
                  Coordinadores
                </label>
                <div className="flex flex-wrap gap-2">
                  {coordinators.map((coordinator) => (
                    <button
                      key={coordinator}
                      onClick={() => {
                        setSelectedCoordinator(
                          selectedCoordinator === coordinator ? '' : coordinator
                        )
                        setSelectedOperator('')
                      }}
                      className={`px-4 py-2 rounded-full font-medium transition ${
                        selectedCoordinator === coordinator
                          ? 'bg-blue-600 text-white'
                          : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                      }`}
                    >
                      {coordinator}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Operator Selector */}
            {operators.length > 0 && (
              <div className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                <label className="block text-sm font-medium text-slate-300 mb-3">
                  Operadores
                </label>
                <div className="flex flex-wrap gap-2">
                  {operators.map((operator) => (
                    <button
                      key={operator}
                      onClick={() => {
                        setSelectedOperator(selectedOperator === operator ? '' : operator)
                      }}
                      className={`px-3 py-1 rounded-full text-sm font-medium transition ${
                        selectedOperator === operator
                          ? 'bg-blue-600 text-white'
                          : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                      }`}
                    >
                      {operator}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* SQDCM Scorecard */}
            {dashboardData && (
              <>
                <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                  <h2 className="text-lg font-semibold text-slate-100 mb-4">
                    Tarjeta de Puntuación SQDCM
                  </h2>
                  <SqdcmScorecard scores={dashboardData.sqdcmScores} />
                </div>

                {/* Charts Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Trend Charts */}
                  {Object.entries(dashboardData.trendData).map(([category, data]) => (
                    <div key={category} className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                      <h3 className="text-lg font-semibold text-slate-100 mb-4 capitalize">
                        Tendencia - {category}
                      </h3>
                      <TrendChart
                        data={data as any}
                        title={category}
                      />
                    </div>
                  ))}
                </div>

                {/* KPI Comparison Chart */}
                <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                  <div className="mb-4 flex items-center justify-between">
                    <h2 className="text-lg font-semibold text-slate-100">
                      Comparación de KPI
                    </h2>
                    <select
                      value={selectedKpi}
                      onChange={(e) => setSelectedKpi(e.target.value)}
                      className="px-3 py-1 bg-slate-700 border border-slate-600 rounded text-slate-100 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="seguridad">Seguridad</option>
                      <option value="calidad">Calidad</option>
                      <option value="entrega">Entrega</option>
                      <option value="costo">Costo</option>
                      <option value="moral">Moral</option>
                    </select>
                  </div>
                  <ComparisonChart
                    data={dashboardData.comparisonData}
                    kpi={selectedKpi}
                  />
                </div>

                {/* Weekly Data Summary */}
                <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
                  <h2 className="text-lg font-semibold text-slate-100 mb-4">
                    Datos Semanales
                  </h2>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm text-slate-300">
                      <thead className="border-b border-slate-700">
                        <tr>
                          <th className="px-4 py-2 text-left">Semana</th>
                          <th className="px-4 py-2 text-left">Máquina</th>
                          <th className="px-4 py-2 text-left">Coordinador</th>
                          <th className="px-4 py-2 text-left">Operador</th>
                          <th className="px-4 py-2 text-right">Seguridad</th>
                          <th className="px-4 py-2 text-right">Calidad</th>
                          <th className="px-4 py-2 text-right">Entrega</th>
                          <th className="px-4 py-2 text-right">Costo</th>
                          <th className="px-4 py-2 text-right">Moral</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(dashboardData.weeklyData || []).length > 0 ? (
                          (dashboardData.weeklyData || []).map((row, idx) => (
                            <tr key={idx} className="border-b border-slate-700 hover:bg-slate-700/50">
                              <td className="px-4 py-2">{row.week}</td>
                              <td className="px-4 py-2">{row.machine}</td>
                              <td className="px-4 py-2">{row.coordinator}</td>
                              <td className="px-4 py-2">{row.operator}</td>
                              <td className="px-4 py-2 text-right">{row.seguridad}</td>
                              <td className="px-4 py-2 text-right">{row.calidad}</td>
                              <td className="px-4 py-2 text-right">{row.entrega}</td>
                              <td className="px-4 py-2 text-right">{row.costo}</td>
                              <td className="px-4 py-2 text-right">{row.moral}</td>
                            </tr>
                          ))
                        ) : (
                          <tr>
                            <td colSpan={9} className="px-4 py-2 text-center text-slate-400">No hay datos disponibles</td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}
