'use client'

import { useState, useEffect } from 'react'
import { useSession } from 'next-auth/react'
import { useRouter } from 'next/navigation'
import Sidebar from '@/components/layout/Sidebar'
import Header from '@/components/layout/Header'
import DataTable from '@/components/common/DataTable'
import LoadingSpinner from '@/components/common/LoadingSpinner'

interface User {
  id: string
  name: string
  email: string
  role: 'admin' | 'supervisor' | 'operator'
  status: 'active' | 'inactive'
  createdAt: string
}

interface AddUserFormData {
  name: string
  email: string
  role: 'admin' | 'supervisor' | 'operator'
}

export default function UsersPage() {
  const { data: session, status } = useSession()
  const router = useRouter()

  // State
  const [users, setUsers] = useState<User[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [showAddModal, setShowAddModal] = useState(false)
  const [formData, setFormData] = useState<AddUserFormData>({
    name: '',
    email: '',
    role: 'operator',
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

  // Fetch users
  useEffect(() => {
    const fetchUsers = async () => {
      try {
        setIsLoading(true)
        const response = await fetch('/api/admin/users')
        if (!response.ok) throw new Error('Error al cargar usuarios')
        const data = await response.json()
        setUsers(data)
      } catch (err) {
        console.error(err)
      } finally {
        setIsLoading(false)
      }
    }

    if (session?.user?.role === 'admin') {
      fetchUsers()
    }
  }, [session])

  const handleAddUser = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    setIsSubmitting(true)
    setMessage(null)

    try {
      const response = await fetch('/api/admin/users', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.message || 'Error al crear usuario')
      }

      const newUser = await response.json()
      setUsers([...users, newUser])
      setFormData({ name: '', email: '', role: 'operator' })
      setShowAddModal(false)
      setMessage({
        type: 'success',
        text: 'Usuario creado exitosamente',
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

  const handleDeleteUser = async (userId: string) => {
    if (!window.confirm('¿Está seguro que desea eliminar este usuario?')) {
      return
    }

    try {
      const response = await fetch(`/api/admin/users/${userId}`, {
        method: 'DELETE',
      })

      if (!response.ok) throw new Error('Error al eliminar usuario')

      setUsers(users.filter((u) => u.id !== userId))
      setMessage({
        type: 'success',
        text: 'Usuario eliminado exitosamente',
      })
    } catch (err) {
      setMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Error desconocido',
      })
    }
  }

  const handleToggleStatus = async (userId: string, currentStatus: string) => {
    try {
      const response = await fetch(`/api/admin/users/${userId}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          status: currentStatus === 'active' ? 'inactive' : 'active',
        }),
      })

      if (!response.ok) throw new Error('Error al actualizar usuario')

      setUsers(
        users.map((u) =>
          u.id === userId
            ? {
                ...u,
                status: currentStatus === 'active' ? 'inactive' : 'active',
              }
            : u
        )
      )
      setMessage({
        type: 'success',
        text: 'Usuario actualizado exitosamente',
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
        <Header title="Gestión de Usuarios" user={session.user} />

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

            {/* Users Table */}
            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-slate-100">
                  Usuarios del Sistema
                </h2>
                <button
                  onClick={() => setShowAddModal(true)}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition"
                >
                  Agregar Usuario
                </button>
              </div>
              <DataTable
                columns={[
                  { key: 'name', label: 'Nombre' },
                  { key: 'email', label: 'Correo' },
                  {
                    key: 'role',
                    label: 'Rol',
                    render: (value: string) => (
                      <span
                        className={`px-3 py-1 rounded-full text-sm font-medium ${
                          value === 'admin'
                            ? 'bg-red-900/30 text-red-400'
                            : value === 'supervisor'
                              ? 'bg-yellow-900/30 text-yellow-400'
                              : 'bg-blue-900/30 text-blue-400'
                        }`}
                      >
                        {value === 'admin'
                          ? 'Administrador'
                          : value === 'supervisor'
                            ? 'Supervisor'
                            : 'Operador'}
                      </span>
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
                        {value === 'active' ? 'Activo' : 'Inactivo'}
                      </button>
                    ),
                  },
                  { key: 'createdAt', label: 'Creado' },
                  {
                    key: 'actions',
                    label: 'Acciones',
                    render: (_, row: any) => (
                      <button
                        onClick={() => handleDeleteUser(row.id)}
                        className="px-3 py-1 bg-red-900/30 text-red-400 hover:bg-red-900/50 rounded text-sm font-medium transition"
                      >
                        Eliminar
                      </button>
                    ),
                  },
                ]}
                data={users}
                isLoading={isLoading}
              />
            </div>
          </div>
        </div>
      </main>

      {/* Add User Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-slate-800 rounded-lg p-6 border border-slate-700 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold text-slate-100 mb-4">
              Agregar Nuevo Usuario
            </h3>

            <form onSubmit={handleAddUser} className="space-y-4">
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
                  required
                  className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              {/* Email */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Correo Electrónico
                </label>
                <input
                  type="email"
                  value={formData.email}
                  onChange={(e) =>
                    setFormData({ ...formData, email: e.target.value })
                  }
                  required
                  className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              {/* Role */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Rol
                </label>
                <select
                  value={formData.role}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      role: e.target.value as any,
                    })
                  }
                  className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="operator">Operador</option>
                  <option value="supervisor">Supervisor</option>
                  <option value="admin">Administrador</option>
                </select>
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
