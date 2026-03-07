'use client'

import { useState, useEffect } from 'react'
import { useSession } from 'next-auth/react'
import { useRouter } from 'next/navigation'
import Sidebar from '@/components/layout/Sidebar'
import Header from '@/components/layout/Header'
import DataTable from '@/components/common/DataTable'
import LoadingSpinner from '@/components/common/LoadingSpinner'

interface HistoryRecord {
  id: string
  week: string
  machine: string
  coordinator: string
  operator: string
  seguridad: number
  calidad: number
  entrega: number
  costo: number
  moral: number
  capturedAt: string
}

export default function HistoryPage() {
  const { data: session, status } = useSession()
  const router = useRouter()

  // Filter state
  const [machine, setMachine] = useState('')
  const [coordinator, setCoordinator] = useState('')
  const [operator, setOperator] = useState('')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')

  // Data state
  const [records, setRecords] = useState<HistoryRecord[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [currentPage, setCurrentPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const [machines, setMachines] = useState<string[]>([])
  const [coordinators, setCoordinators] = useState<string[]>([])
  const [operators, setOperators] = useState<string[]>([])

  const RECORDS_PER_PAGE = 20

  // Redirect if not authenticated
  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/login')
    }
  }, [status, router])

  // Fetch filter options
  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const response = await fetch('/api/history/options')
        if (!response.ok) throw new Error('Error al cargar opciones')
        const data = await response.json()
        setMachines(data.machines || [])
        setCoordinators(data.coordinators || [])
        setOperators(data.operators || [])
      } catch (err) {
        console.error(err)
      }
    }

    fetchOptions()
  }, [])

  // Fetch history records
  useEffect(() => {
    const fetchRecords = async () => {
      try {
        setIsLoading(true)
        const params = new URLSearchParams()
        params.append('page', currentPage.toString())
        params.append('limit', RECORDS_PER_PAGE.toString())
        if (machine) params.append('machine', machine)
        if (coordinator) params.append('coordinator', coordinator)
        if (operator) params.append('operator', operator)
        if (startDate) params.append('startDate', startDate)
        if (endDate) params.append('endDate', endDate)

        const response = await fetch(`/api/history?${params.toString()}`)
        if (!response.ok) throw new Error('Error al cargar registros')

        const data = await response.json()
        setRecords(data.records || [])
        setTotalPages(Math.ceil((data.total || 0) / RECORDS_PER_PAGE))
      } catch (err) {
        console.error(err)
      } finally {
        setIsLoading(false)
      }
    }

    fetchRecords()
  }, [machine, coordinator, operator, startDate, endDate, currentPage])

  const handleExport = async () => {
    try {
      const params = new URLSearchParams()
      if (machine) params.append('machine', machine)
      if (coordinator) params.append('coordinator', coordinator)
      if (operator) params.append('operator', operator)
      if (startDate) params.append('startDate', startDate)
      if (endDate) params.append('endDate', endDate)

      const response = await fetch(`/api/history/export?${params.toString()}`)
      if (!response.ok) throw new Error('Error al exportar')

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `historia_kpi_${new Date().toISOString().split('T')[0]}.csv`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (err) {
      console.error(err)
    }
  }

  if (status === 'loading') {
    return (
      <div className="flex h-screen bg-slate-900">
        <Sidebar user={session?.user} />
        <main className="flex-1 flex items-center justify-center">
          <LoadingSpinner />
        </main>
      </div>
    )
  }

  if (!session) {
    return null
  }

  return (
    <div className="flex h-screen bg-slate-900">
      <Sidebar user={session.user} />
      <main className="flex-1 flex flex-col overflow-hidden">
        <Header title="Historial de Datos" user={session.user} />

        <div className="flex-1 overflow-y-auto">
          <div className="p-6 space-y-6">
            {/* Filters */}
            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <h2 className="text-lg font-semibold text-slate-100 mb-4">
                Filtros
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                {/* Machine Filter */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Máquina
                  </label>
                  <select
                    value={machine}
                    onChange={(e) => {
                      setMachine(e.target.value)
                      setCurrentPage(1)
                    }}
                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-slate-100 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Todas</option>
                    {machines.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Coordinator Filter */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Coordinador
                  </label>
                  <select
                    value={coordinator}
                    onChange={(e) => {
                      setCoordinator(e.target.value)
                      setCurrentPage(1)
                    }}
                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-slate-100 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Todos</option>
                    {coordinators.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Operator Filter */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Operador
                  </label>
                  <select
                    value={operator}
                    onChange={(e) => {
                      setOperator(e.target.value)
                      setCurrentPage(1)
                    }}
                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-slate-100 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Todos</option>
                    {operators.map((o) => (
                      <option key={o} value={o}>
                        {o}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Start Date */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Desde
                  </label>
                  <input
                    type="date"
                    value={startDate}
                    onChange={(e) => {
                      setStartDate(e.target.value)
                      setCurrentPage(1)
                    }}
                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-slate-100 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                {/* End Date */}
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Hasta
                  </label>
                  <input
                    type="date"
                    value={endDate}
                    onChange={(e) => {
                      setEndDate(e.target.value)
                      setCurrentPage(1)
                    }}
                    className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-slate-100 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
            </div>

            {/* Data Table */}
            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-slate-100">
                  Registros Históricos
                </h2>
                <button
                  onClick={handleExport}
                  className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white font-medium rounded-lg transition"
                >
                  Exportar a CSV
                </button>
              </div>
              <DataTable
                columns={[
                  { key: 'week', label: 'Semana' },
                  { key: 'machine', label: 'Máquina' },
                  { key: 'coordinator', label: 'Coordinador' },
                  { key: 'operator', label: 'Operador' },
                  { key: 'seguridad', label: 'Seguridad' },
                  { key: 'calidad', label: 'Calidad' },
                  { key: 'entrega', label: 'Entrega' },
                  { key: 'costo', label: 'Costo' },
                  { key: 'moral', label: 'Moral' },
                  { key: 'capturedAt', label: 'Capturado' },
                ]}
                data={records}
                isLoading={isLoading}
              />

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="mt-4 flex items-center justify-center gap-2">
                  <button
                    onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                    disabled={currentPage === 1}
                    className="px-3 py-1 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 text-slate-300 rounded transition"
                  >
                    Anterior
                  </button>
                  <span className="text-slate-300 text-sm">
                    Página {currentPage} de {totalPages}
                  </span>
                  <button
                    onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                    disabled={currentPage === totalPages}
                    className="px-3 py-1 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 text-slate-300 rounded transition"
                  >
                    Siguiente
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
