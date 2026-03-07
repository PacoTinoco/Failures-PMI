'use client'

import { useState, useEffect } from 'react'
import { useSession } from 'next-auth/react'
import { useRouter } from 'next/navigation'
import Sidebar from '@/components/layout/Sidebar'
import Header from '@/components/layout/Header'
import DataTable from '@/components/common/DataTable'
import LoadingSpinner from '@/components/common/LoadingSpinner'

interface Machine {
  id: string
  name: string
  code: string
  coordinatorCount: number
  operatorCount: number
  status: 'active' | 'inactive'
  createdAt: string
}

interface AddMachineFormData {
  name: string
  code: string
}

export default function MachinesPage() {
  const { data: session, status } = useSession()
  const router = useRouter()

  // State
  const [machines, setMachines] = useState<Machine[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [showAddModal, setShowAddModal] = useState(false)
  const [formData, setFormData] = useState<AddMachineFormData>({
    name: '',
    code: '',
  })
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [message, setMessage] = useState<{
    type: 'success' | 'error'
    text: string
  } | null>(null)

  // Redirect if not authenticated or not admin
  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/login')
    } else if (session?.user?.role !== 'admin') {
      router.push('/dashboard')
    }
  }, [status, session, router])

  // Fetch machines
  useEffect(() => {
    const fetchMachines = async () => {
      try {
        setIsLoading(true)
        const response = await fetch('/api/admin/machines')
        if (!response.ok) throw new Error('Error al cargar máquinas')
        const data = await response.json()
        setMachines(data)
      } catch (err) {
        console.error(err)
      } finally {
        setIsLoading(false)
      }
    }

    if (session?.user?.role === 'admin') {
      fetchMachines()
    }
  }, [session])

  const handleAddMachine = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    setIsSubmitting(true)
    setMessage(null)

    try {
      const response = await fetch('/api/admin/machines', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.message || 'Error al crear máquina')
      }

      const newMachine = await response.json()
      setMachines([...machines, newMachine])
      setFormData({ name: '', code: '' })
      setShowAddModal(false)
      setMessage({
        type: 'success',
        text: 'Máquina creada exitosamente',
      })
    } catch (err) {
      setMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Error desconocido',
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleDeleteMachine = async (machineId: string) => {
    if (!window.confirm('¿Está seguro que desea eliminar esta máquina?')) {
      return
    }

    try {
      const response = await fetch(`/api/admin/machines/${machineId}`, {
        method: 'DELETE',
      })

      if (!response.ok) throw new Error('Error al eliminar máquina')

      setMachines(machines.filter((m) => m.id !== machineId))
      setMessage({
        type: 'success',
        text: 'Máquina eliminada exitosamente',
      })
    } catch (err) {
      setMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Error desconocido',
      })
    }
  }

  const handleToggleStatus = async (machineId: string, currentStatus: string) => {
    try {
      const response = await fetch(`/api/admin/machines/${machineId}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          status: currentStatus === 'active' ? 'inactive' : 'active',
        }),
      })

      if (!response.ok) throw new Error('Error al actualizar máquina')

      setMachines(
        machines.map((m) =>
          m.id === machineId
            ? {
                ...m,
                status: currentStatus === 'active' ? 'inactive' : 'active',
              }
            : m
        )
      )
      setMessage({
        type: 'success',
        text: 'Máquina actualizada exitosamente',
      })
    } catch (err) {
      setMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Error desconocido',
      })
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

  if (!session || session.user?.role !== 'admin') {
    return null
  }

  return (
    <div className="flex h-screen bg-slate-900">
      <Sidebar user={session.user} />
      <main className="flex-1 flex flex-col overflow-hidden">
        <Header title="Gestión de Máquinas" user={session.user} />

        <div className="flex-1 overflow-y-auto">
          <div className="p-6 space-y-6">
            {/* Status Message */}
            {message && (
              <div
                className={`p-4 rounded-lg ${
                  message.type === 'success'
                    ? 'bg-green-900/20 border border-green-700'
                    : 'bg-red-900/20 border border-red-700'
                }`}
              >
                <p
                  className={
                    message.type === 'success'
                      ? 'text-green-400'
                      : 'text-red-400'
                  }
                >
                  {message.text}
                </p>
              </div>
            )}

            {/* Machines Table */}
            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-slate-100">
                  Máquinas
                </h2>
                <button
                  onClick={() => setShowAddModal(true)}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition"
                >
                  Agregar Máquina
                </button>
              </div>
              <DataTable
                columns={[
                  { key: 'name', label: 'Nombre' },
                  { key: 'code', label: 'Código' },
                  {
                    key: 'coordinatorCount',
                    label: 'Coordinadores',
                    render: (value: number) => (
                      <span className="text-slate-300">{value}</span>
                    ),
                  },
                  {
                    key: 'operatorCount',
                    label: 'Operadores',
                    render: (value: number) => (
                      <span className="text-slate-300">{value}</span>
                    ),
                  },
                  {
                    key: 'status',
                    label: 'Estado',
                    render: (value: string, row: any) => (
                      <button
                        onClick={() => handleToggleStatus(row.id, value)}
                        className={`px-3 py-1 rounded-full text-sm font-medium ${
                          value === 'active'
                            ? 'bg-green-900/30 text-green-400 hover:bg-green-900/50'
                            : 'bg-red-900/30 text-red-400 hover:bg-red-900/50'
                        } transition`}
                      >
                        {value === 'active' ? 'Activa' : 'Inactiva'}
                      </button>
                    ),
                  },
                  { key: 'createdAt', label: 'Creada' },
                  {
                    key: 'actions',
                    label: 'Acciones',
                    render: (_, row: any) => (
                      <button
                        onClick={() => handleDeleteMachine(row.id)}
                        className="px-3 py-1 bg-red-900/30 text-red-400 hover:bg-red-900/50 rounded text-sm font-medium transition"
                      >
                        Eliminar
                      </button>
                    ),
                  },
                ]}
                data={machines}
                isLoading={isLoading}
              />
            </div>
          </div>
        </div>
      </main>

      {/* Add Machine Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-slate-800 rounded-lg p-6 border border-slate-700 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold text-slate-100 mb-4">
              Agregar Nueva Máquina
            </h3>

            <form onSubmit={handleAddMachine} className="space-y-4">
              {/* Name */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Nombre
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) =>
                    setFormData({ ...formData, name: e.target.value })
                  }
                  placeholder="ej. Máquina de Inyección #1"
                  required
                  className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              {/* Code */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Código
                </label>
                <input
                  type="text"
                  value={formData.code}
                  onChange={(e) =>
                    setFormData({ ...formData, code: e.target.value })
                  }
                  placeholder="ej. INY-001"
                  required
                  className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              {/* Actions */}
              <div className="flex gap-3 pt-4">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="flex-1 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-100 font-medium rounded-lg transition"
                >
                  Cancelar
                </button>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-medium rounded-lg transition"
                >
                  {isSubmitting ? 'Creando...' : 'Crear'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
