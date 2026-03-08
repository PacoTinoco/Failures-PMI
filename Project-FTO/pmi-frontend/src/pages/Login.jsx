import { useState } from 'react'
import { Navigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function Login() {
  const { user, signInWithMagicLink } = useAuth()
  const [email, setEmail] = useState('')
  const [status, setStatus] = useState('idle') // idle | sending | sent | error
  const [errorMsg, setErrorMsg] = useState('')

  if (user) {
    return <Navigate to="/captura" replace />
  }

  function validateEmail(e) {
    const val = e.toLowerCase().trim()
    // Acepta cualquier email válido (en producción se restringe a @pmintl.net)
    return val.includes('@') && val.includes('.')
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setErrorMsg('')

    if (!validateEmail(email)) {
      setErrorMsg('Ingresa un correo electrónico válido')
      return
    }

    setStatus('sending')
    try {
      await signInWithMagicLink(email.toLowerCase().trim())
      setStatus('sent')
    } catch (err) {
      setErrorMsg(err.message || 'Error al enviar el Magic Link')
      setStatus('error')
    }
  }

  return (
    <div className="min-h-screen bg-[#0a1628] flex items-center justify-center px-4">
      {/* Background decorative elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 right-10 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl" />
        <div className="absolute bottom-20 left-10 w-80 h-80 bg-green-500/5 rounded-full blur-3xl" />
      </div>

      <div className="relative w-full max-w-md">
        {/* Logo / Branding */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-500 to-green-500 rounded-2xl mb-4 shadow-lg shadow-blue-500/20">
            <span className="text-2xl font-bold text-white">FTO</span>
          </div>
          <h1 className="text-2xl font-bold text-white">PMI <span className="text-blue-400">Plattform</span></h1>
          <p className="text-slate-400 mt-1">FTO Digital — Filtros Cápsula</p>
        </div>

        {/* Card */}
        <div className="bg-[#0f1d32] rounded-2xl border border-white/5 p-8 shadow-xl">
          {status === 'sent' ? (
            <div className="text-center">
              <div className="inline-flex items-center justify-center w-14 h-14 bg-green-500/20 rounded-full mb-4">
                <svg className="w-7 h-7 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-white mb-2">Revisa tu correo</h2>
              <p className="text-slate-400 text-sm mb-6">
                Enviamos un enlace de acceso a <span className="text-blue-400 font-medium">{email}</span>.
                Haz clic en el enlace para iniciar sesion.
              </p>
              <button
                onClick={() => { setStatus('idle'); setEmail('') }}
                className="text-sm text-blue-400 hover:text-blue-300 transition-colors"
              >
                Usar otro correo
              </button>
            </div>
          ) : (
            <>
              <h2 className="text-xl font-semibold text-white mb-2">Iniciar sesion</h2>
              <p className="text-slate-400 text-sm mb-6">
                Ingresa tu correo para recibir un enlace de acceso.
              </p>

              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-slate-300 mb-1.5">
                    Correo electrónico
                  </label>
                  <input
                    id="email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="tu.correo@ejemplo.com"
                    required
                    className="w-full px-4 py-3 bg-[#0a1628] border border-white/10 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  />
                </div>

                {errorMsg && (
                  <div className="flex items-center gap-2 text-red-400 text-sm bg-red-400/10 px-3 py-2 rounded-lg">
                    <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    {errorMsg}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={status === 'sending'}
                  className="w-full py-3 px-4 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-600/50 text-white font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                  {status === 'sending' ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Enviando...
                    </>
                  ) : (
                    'Enviar Magic Link'
                  )}
                </button>
              </form>
            </>
          )}
        </div>

        <p className="text-center text-slate-500 text-xs mt-6">
          Sin contraseñas. Acceso seguro via enlace al correo.
        </p>
      </div>
    </div>
  )
}
