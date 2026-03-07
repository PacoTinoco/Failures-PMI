'use client'

import { useState, useEffect } from 'react'
import { useSession } from 'next-auth/react'
import { useRouter } from 'next/navigation'
import Sidebar from '@/components/layout/Sidebar'
import Header from '@/components/layout/Header'
import ManualCapture from '@/components/capture/ManualCapture'
import LoadingSpinner from '@/components/common/LoadingSpinner'

export default function CapturePage() {
  const { data: session, status } = useSession()
  const router = useRouter()
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitMessage, setSubmitMessage] = useState<{
    type: 'success' | 'error'
    text: string
  } | null>(null)

  // Redirect if not authenticated
  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/login')
    }
  }, [status, router])

  const handleSubmit = async (data: any) => {
    try {
      setIsSubmitting(true)
      setSubmitMessage(null)

      const response = await fetch('/api/capture', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      })

      if (!response.ok) {
        throw new Error('Error al guardar datos')
      }

      setSubmitMessage({
        type: 'success',
        text: 'Datos capturados exitosamente',
      })

      // Reset form after 2 seconds
      setTimeout(() => {
        setSubmitMessage(null)
      }, 2000)
    } catch (err) {
      setSubmitMessage({
        type: 'error',
        text: err instanceof Error ? err.message : 'Error desconocido',
      })
    } finally {
      setIsSubmitting(false)
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
        <Header title="Captura Manual de Datos" user={session.user} />

        <div className="flex-1 overflow-y-auto">
          <div className="p-6 space-y-6">
            {/* Instructions */}
            <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-slate-100 mb-2">
                Captura Manual de KPI
              </h3>
              <p className="text-slate-300 text-sm">
                Complete el siguiente formulario para registrar los valores de KPI manualmente.
                Todos los campos son obligatorios.
              </p>
            </div>

            {/* Status Messages */}
            {submitMessage && (
              <div
                className={`p-4 rounded-lg ${
                  submitMessage.type === 'success'
                    ? 'bg-green-900/20 border border-green-700'
                    : 'bg-red-900/20 border border-red-700'
                }`}
              >
                <p
                  className={
                    submitMessage.type === 'success'
                      ? 'text-green-400'
                      : 'text-red-400'
                  }
                >
                  {submitMessage.text}
                </p>
              </div>
            )}

            {/* Manual Capture Form */}
            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <ManualCapture
                onSubmit={handleSubmit}
                isSubmitting={isSubmitting}
              />
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
