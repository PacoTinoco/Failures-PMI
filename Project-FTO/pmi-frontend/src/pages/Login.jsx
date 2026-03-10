import { useState } from 'react'
import { Navigate, Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function Login() {
  const { user, signUp, signInWithPassword } = useAuth()
  const [mode, setMode] = useState('login') // 'login' | 'signup'
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [status, setStatus] = useState('idle') // idle | submitting | success | error
  const [errorMsg, setErrorMsg] = useState('')

  if (user) {
    return <Navigate to="/captura" replace />
  }

  function validateForm() {
    if (!email.includes('@') || !email.includes('.')) {
      setErrorMsg('Ingresa un correo electrónico válido')
      return false
    }
    if (password.length < 8) {
      setErrorMsg('La contraseña debe tener al menos 8 caracteres')
      return false
    }
    if (mode === 'signup' && password !== confirmPassword) {
      setErrorMsg('Las contraseñas no coinciden')
      return false
    }
    return true
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setErrorMsg('')

    if (!validateForm()) return

    setStatus('submitting')
    try {
      if (mode === 'signup') {
        const result = await signUp(email, password)
        if (result.needsConfirmation) {
          setStatus('needs_confirmation')
        } else {
          setStatus('success')
        }
      } else {
        await signInWithPassword(email, password)
        // Redirect happens via AuthContext → Navigate above
      }
    } catch (err) {
      const msg = err.message || 'Error de autenticación'
      // Traducciones comunes de errores de Supabase
      if (msg.includes('Invalid login credentials')) {
        setErrorMsg('Correo o contraseña incorrectos')
      } else if (msg.includes('User already registered')) {
        setErrorMsg('Este correo ya tiene una cuenta. Inicia sesión.')
      } else if (msg.includes('Password should be')) {
        setErrorMsg('La contraseña debe tener al menos 8 caracteres')
      } else if (msg.includes('Email not confirmed')) {
        setErrorMsg('Tu correo no ha sido confirmado. Revisa tu bandeja de entrada (y spam) para el enlace de confirmación.')
      } else if (msg.includes('Email rate limit exceeded')) {
        setErrorMsg('Demasiados intentos. Espera unos minutos e intenta de nuevo.')
      } else {
        setErrorMsg(msg)
      }
      setStatus('error')
    }
  }

  function toggleMode() {
    setMode(mode === 'login' ? 'signup' : 'login')
    setErrorMsg('')
    setStatus('idle')
    setPassword('')
    setConfirmPassword('')
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
          <Link to="/">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-500 to-green-500 rounded-2xl mb-4 shadow-lg shadow-blue-500/20">
              <span className="text-2xl font-bold text-white">FTO</span>
            </div>
          </Link>
          <h1 className="text-2xl font-bold text-white">PMI <span className="text-blue-400">Plattform</span></h1>
          <p className="text-slate-400 mt-1">FTO Digital — Filtros Cápsula</p>
        </div>

        {/* Card */}
        <div className="bg-[#0f1d32] rounded-2xl border border-white/5 p-8 shadow-xl">
          {status === 'needs_confirmation' && mode === 'signup' ? (
            <div className="text-center">
              <div className="inline-flex items-center justify-center w-14 h-14 bg-yellow-500/20 rounded-full mb-4">
                <svg className="w-7 h-7 text-yellow-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-white mb-2">Confirma tu correo</h2>
              <p className="text-slate-400 text-sm mb-4">
                Te enviamos un enlace de confirmación a <span className="text-blue-400">{email}</span>.
                Revisa tu bandeja de entrada (y la carpeta de spam).
              </p>
              <p className="text-slate-500 text-xs mb-6">
                Una vez confirmado, podrás iniciar sesión con tu correo y contraseña.
              </p>
              <button
                onClick={() => { setMode('login'); setStatus('idle'); setPassword('') }}
                className="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-medium rounded-lg transition-colors"
              >
                Ir a Iniciar Sesión
              </button>
            </div>
          ) : status === 'success' && mode === 'signup' ? (
            <div className="text-center">
              <div className="inline-flex items-center justify-center w-14 h-14 bg-green-500/20 rounded-full mb-4">
                <svg className="w-7 h-7 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-white mb-2">Cuenta creada</h2>
              <p className="text-slate-400 text-sm mb-6">
                Tu cuenta ha sido creada exitosamente. Ya puedes iniciar sesión.
              </p>
              <button
                onClick={() => { setMode('login'); setStatus('idle'); setPassword('') }}
                className="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-medium rounded-lg transition-colors"
              >
                Iniciar Sesión
              </button>
            </div>
          ) : (
            <>
              <h2 className="text-xl font-semibold text-white mb-2">
                {mode === 'login' ? 'Iniciar Sesión' : 'Crear Cuenta'}
              </h2>
              <p className="text-slate-400 text-sm mb-6">
                {mode === 'login'
                  ? 'Ingresa tus credenciales para acceder a la plataforma.'
                  : 'Registra una cuenta nueva para comenzar.'}
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

                <div>
                  <label htmlFor="password" className="block text-sm font-medium text-slate-300 mb-1.5">
                    Contraseña
                  </label>
                  <input
                    id="password"
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder={mode === 'signup' ? 'Mínimo 8 caracteres' : '••••••••'}
                    required
                    minLength={8}
                    className="w-full px-4 py-3 bg-[#0a1628] border border-white/10 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  />
                </div>

                {mode === 'signup' && (
                  <div>
                    <label htmlFor="confirmPassword" className="block text-sm font-medium text-slate-300 mb-1.5">
                      Confirmar contraseña
                    </label>
                    <input
                      id="confirmPassword"
                      type="password"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      placeholder="Repite tu contraseña"
                      required
                      minLength={8}
                      className="w-full px-4 py-3 bg-[#0a1628] border border-white/10 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                    />
                  </div>
                )}

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
                  disabled={status === 'submitting'}
                  className="w-full py-3 px-4 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 disabled:from-blue-600/50 disabled:to-blue-500/50 text-white font-medium rounded-lg transition-all flex items-center justify-center gap-2 shadow-lg shadow-blue-500/20"
                >
                  {status === 'submitting' ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      {mode === 'login' ? 'Entrando...' : 'Creando cuenta...'}
                    </>
                  ) : (
                    mode === 'login' ? 'Iniciar Sesión' : 'Crear Cuenta'
                  )}
                </button>
              </form>

              <div className="mt-6 text-center">
                <button
                  onClick={toggleMode}
                  className="text-sm text-slate-400 hover:text-blue-400 transition-colors"
                >
                  {mode === 'login'
                    ? '¿No tienes cuenta? Crear una'
                    : '¿Ya tienes cuenta? Inicia sesión'}
                </button>
              </div>
            </>
          )}
        </div>

        <p className="text-center text-slate-500 text-xs mt-6">
          Acceso seguro con email y contraseña.
        </p>
      </div>
    </div>
  )
}
