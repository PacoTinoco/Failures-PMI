import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { supabase } from '../lib/supabase'

export default function AuthCallback() {
  const navigate = useNavigate()
  const [error, setError] = useState(null)

  useEffect(() => {
    async function handleCallback() {
      try {
        // Supabase automatically handles the hash fragment from the magic link
        const { data, error } = await supabase.auth.getSession()

        if (error) {
          setError(error.message)
          return
        }

        if (data.session) {
          navigate('/captura', { replace: true })
        } else {
          // If no session yet, wait for auth state change
          const { data: { subscription } } = supabase.auth.onAuthStateChange(
            (event, session) => {
              if (event === 'SIGNED_IN' && session) {
                subscription.unsubscribe()
                navigate('/captura', { replace: true })
              }
            }
          )

          // Timeout after 10 seconds
          setTimeout(() => {
            subscription.unsubscribe()
            setError('Tiempo de espera agotado. Intenta iniciar sesion nuevamente.')
          }, 10000)
        }
      } catch (err) {
        setError(err.message)
      }
    }

    handleCallback()
  }, [navigate])

  if (error) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center px-4">
        <div className="bg-slate-800 rounded-2xl border border-slate-700 p-8 max-w-md text-center">
          <div className="inline-flex items-center justify-center w-14 h-14 bg-red-500/20 rounded-full mb-4">
            <svg className="w-7 h-7 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </div>
          <h2 className="text-xl font-semibold text-white mb-2">Error de autenticacion</h2>
          <p className="text-slate-400 text-sm mb-6">{error}</p>
          <button
            onClick={() => navigate('/login', { replace: true })}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors"
          >
            Volver al login
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-slate-900 flex items-center justify-center">
      <div className="flex flex-col items-center gap-4">
        <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
        <p className="text-slate-400">Verificando acceso...</p>
      </div>
    </div>
  )
}
