'use client'

import { useState, useEffect } from 'react'
import { useSession } from 'next-auth/react'
import { useRouter } from 'next/navigation'
import Sidebar from '@/components/layout/Sidebar'
import Header from '@/components/layout/Header'
import ExcelUpload from '@/components/upload/ExcelUpload'
import DataTable from '@/components/common/DataTable'
import LoadingSpinner from '@/components/common/LoadingSpinner'

interface Upload {
  id: string
  fileName: string
  uploadedAt: string
  uploadedBy: string
  recordCount: number
  status: 'success' | 'error' | 'pending'
}

export default function UploadPage() {
  const { data: session, status } = useSession()
  const router = useRouter()
  const [uploads, setUploads] = useState<Upload[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [refreshTrigger, setRefreshTrigger] = useState(0)

  // Redirect if not authenticated
  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/login')
    }
  }, [status, router])

  // Fetch recent uploads
  useEffect(() => {
    const fetchUploads = async () => {
      try {
        setIsLoading(true)
        const response = await fetch('/api/uploads')
        if (!response.ok) throw new Error('Error al cargar archivos')
        const data = await response.json()
        setUploads(data)
      } catch (err) {
        console.error(err)
      } finally {
        setIsLoading(false)
      }
    }

    fetchUploads()
  }, [refreshTrigger])

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
        <Header title="Carga de Datos" user={session.user} />

        <div className="flex-1 overflow-y-auto">
          <div className="p-6 space-y-6">
            {/* Instructions */}
            <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-slate-100 mb-2">
                Instrucciones de Carga
              </h3>
              <ul className="text-slate-300 space-y-2 text-sm">
                <li>• Descargue la plantilla de Excel desde el panel</li>
                <li>• Complete los datos de KPI (Seguridad, Calidad, Entrega, Costo, Moral)</li>
                <li>• Asegúrese de que todas las máquinas y coordinadores estén registrados</li>
                <li>• Suba el archivo usando el formulario de carga</li>
                <li>• Verifique que la carga sea exitosa en la tabla de abajo</li>
              </ul>
            </div>

            {/* Excel Upload Component */}
            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <h2 className="text-lg font-semibold text-slate-100 mb-4">
                Cargar Archivo Excel
              </h2>
              <ExcelUpload onSuccess={() => setRefreshTrigger(prev => prev + 1)} />
            </div>

            {/* Recent Uploads Table */}
            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <h2 className="text-lg font-semibold text-slate-100 mb-4">
                Cargas Recientes
              </h2>
              <DataTable
                columns={[
                  { key: 'fileName', label: 'Archivo' },
                  { key: 'uploadedAt', label: 'Fecha de Carga' },
                  { key: 'uploadedBy', label: 'Cargado por' },
                  { key: 'recordCount', label: 'Registros' },
                  {
                    key: 'status',
                    label: 'Estado',
                    render: (value: string) => (
                      <span
                        className={`px-3 py-1 rounded-full text-sm font-medium ${
                          value === 'success'
                            ? 'bg-green-900/30 text-green-400'
                            : value === 'error'
                              ? 'bg-red-900/30 text-red-400'
                              : 'bg-yellow-900/30 text-yellow-400'
                        }`}
                      >
                        {value === 'success'
                          ? 'Exitosa'
                          : value === 'error'
                            ? 'Error'
                            : 'Pendiente'}
                      </span>
                    ),
                  },
                ]}
                data={uploads}
                isLoading={isLoading}
              />
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
