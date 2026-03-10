import { Link } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

// Feature cards data
const features = [
  {
    icon: (
      <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
      </svg>
    ),
    iconBg: 'bg-blue-100',
    iconColor: 'text-blue-600',
    title: 'Dashboard Interactivo',
    desc: 'Visualiza tus indicadores SQDCM con semaforización automática. Verde, amarillo o rojo según el objetivo definido.'
  },
  {
    icon: (
      <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3.375 19.5h17.25m-17.25 0a1.125 1.125 0 01-1.125-1.125M3.375 19.5h7.5c.621 0 1.125-.504 1.125-1.125m-9.75 0V5.625m0 12.75v-1.5c0-.621.504-1.125 1.125-1.125m18.375 2.625V5.625m0 12.75c0 .621-.504 1.125-1.125 1.125m1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125m0 3.75h-7.5A1.125 1.125 0 0112 18.375m9.75-12.75c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125m19.5 0v1.5c0 .621-.504 1.125-1.125 1.125M2.25 5.625v1.5c0 .621.504 1.125 1.125 1.125m0 0h17.25m-17.25 0h7.5c.621 0 1.125.504 1.125 1.125M3.375 8.25c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125m17.25-3.75h-7.5c-.621 0-1.125.504-1.125 1.125m8.625-1.125c.621 0 1.125.504 1.125 1.125v1.5c0 .621-.504 1.125-1.125 1.125m-17.25 0h7.5m-7.5 0c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125M12 10.875v-1.5m0 1.5c0 .621-.504 1.125-1.125 1.125M12 10.875c0 .621.504 1.125 1.125 1.125m-2.25 0c.621 0 1.125.504 1.125 1.125M13.125 12h7.5m-7.5 0c-.621 0-1.125.504-1.125 1.125M20.625 12c.621 0 1.125.504 1.125 1.125v1.5c0 .621-.504 1.125-1.125 1.125m-17.25 0h7.5M12 14.625v-1.5m0 1.5c0 .621-.504 1.125-1.125 1.125M12 14.625c0 .621.504 1.125 1.125 1.125m-2.25 0c.621 0 1.125.504 1.125 1.125m0 0v1.5c0 .621-.504 1.125-1.125 1.125" />
      </svg>
    ),
    iconBg: 'bg-green-100',
    iconColor: 'text-green-600',
    title: 'Captura Tipo Spreadsheet',
    desc: 'Interfaz intuitiva similar a Excel para capturar los 17 indicadores. Selector de semana, operador y validaciones automáticas.'
  },
  {
    icon: (
      <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.25 18L9 11.25l4.306 4.307a11.95 11.95 0 015.814-5.519l2.74-1.22m0 0l-5.94-2.28m5.94 2.28l-2.28 5.941" />
      </svg>
    ),
    iconBg: 'bg-purple-100',
    iconColor: 'text-purple-600',
    title: 'Análisis de Tendencias',
    desc: 'Gráficas de evolución semanal por indicador. Compara operadores y detecta patrones automáticamente.'
  },
  {
    icon: (
      <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
      </svg>
    ),
    iconBg: 'bg-yellow-100',
    iconColor: 'text-yellow-600',
    title: 'Autenticación Segura',
    desc: 'Email y contraseña seguros. Registro rápido con cualquier correo y políticas RLS para seguridad a nivel de fila.'
  },
  {
    icon: (
      <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M18 18.72a9.094 9.094 0 003.741-.479 3 3 0 00-4.682-2.72m.94 3.198l.001.031c0 .225-.012.447-.037.666A11.944 11.944 0 0112 21c-2.17 0-4.207-.576-5.963-1.584A6.062 6.062 0 016 18.719m12 0a5.971 5.971 0 00-.941-3.197m0 0A5.995 5.995 0 0012 12.75a5.995 5.995 0 00-5.058 2.772m0 0a3 3 0 00-4.681 2.72 8.986 8.986 0 003.74.477m.94-3.197a5.971 5.971 0 00-.94 3.197M15 6.75a3 3 0 11-6 0 3 3 0 016 0zm6 3a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0zm-13.5 0a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0z" />
      </svg>
    ),
    iconBg: 'bg-pink-100',
    iconColor: 'text-pink-600',
    title: 'Gestión de Equipos',
    desc: 'Administra Line Coordinators, operadores, máquinas y turnos. Asignaciones claras por cédula.'
  },
  {
    icon: (
      <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
      </svg>
    ),
    iconBg: 'bg-teal-100',
    iconColor: 'text-teal-600',
    title: 'Exportación a Excel',
    desc: 'Genera reportes históricos. Exporta datos por período, cédula o indicador para presentaciones offline.'
  }
]

const steps = [
  { num: '1', color: 'from-blue-500 to-blue-600', title: 'Crea tu Cuenta', desc: 'Regístrate con tu correo y una contraseña. Inicia sesión cuando quieras, tu sesión se mantiene activa.' },
  { num: '2', color: 'from-green-500 to-green-600', title: 'Captura Datos', desc: 'Registra los 17 indicadores SQDCM por operador cada semana. Interfaz intuitiva con validaciones en tiempo real.' },
  { num: '3', color: 'from-purple-500 to-purple-600', title: 'Visualiza Resultados', desc: 'Accede a dashboards interactivos, tendencias históricas y exporta reportes para tus presentaciones.' },
]

export default function Landing() {
  const { user } = useAuth()
  const ctaLink = user ? '/dashboard' : '/login'

  return (
    <div className="min-h-screen bg-[#0a1628] text-white">
      {/* ============ NAVBAR ============ */}
      <nav className="sticky top-0 z-50 backdrop-blur-md bg-[#0a1628]/80 border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <svg className="w-8 h-8" viewBox="0 0 32 32" fill="none">
                <rect width="32" height="32" rx="8" fill="url(#logoGrad)" />
                <path d="M8 12h6v2H8v-2zm0 4h10v2H8v-2zm0 4h8v2H8v-2z" fill="white" fillOpacity="0.9" />
                <path d="M20 10l4 4-4 4" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                <defs><linearGradient id="logoGrad" x1="0" y1="0" x2="32" y2="32"><stop stopColor="#3b82f6" /><stop offset="1" stopColor="#22c55e" /></linearGradient></defs>
              </svg>
              <span className="text-xl font-bold">PMI <span className="text-blue-400">Plattform</span></span>
            </div>
          </div>

          <div className="hidden md:flex items-center gap-8">
            <a href="#features" className="text-sm text-slate-300 hover:text-white transition-colors">Dashboard</a>
            <a href="#features" className="text-sm text-slate-300 hover:text-white transition-colors">Captura de Datos</a>
            <a href="#steps" className="text-sm text-slate-300 hover:text-white transition-colors">Análisis y Tendencias</a>
            <a href="#preview" className="text-sm text-slate-300 hover:text-white transition-colors">Mi Cédula</a>
          </div>

          <Link
            to="/login"
            className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold rounded-xl transition-colors"
          >
            {user ? 'Ir al Dashboard' : 'Iniciar Sesión'}
          </Link>
        </div>
      </nav>

      {/* ============ HERO ============ */}
      <section className="relative overflow-hidden">
        {/* Background decorative elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-20 right-10 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl" />
          <div className="absolute bottom-20 left-10 w-80 h-80 bg-green-500/5 rounded-full blur-3xl" />
          {/* Grid pattern overlay */}
          <div className="absolute inset-0 opacity-[0.03]" style={{
            backgroundImage: 'linear-gradient(rgba(255,255,255,.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.1) 1px, transparent 1px)',
            backgroundSize: '60px 60px'
          }} />
        </div>

        <div className="relative max-w-7xl mx-auto px-6 pt-20 pb-24">
          <div className="max-w-3xl">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-500/10 border border-blue-500/20 rounded-full mb-8">
              <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <span className="text-sm text-blue-300 font-medium">Plataforma FTO - SQDCM</span>
            </div>

            <h1 className="text-5xl md:text-6xl lg:text-7xl font-extrabold leading-[1.1] mb-6">
              Digitaliza tu Control de{' '}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-green-400">
                Indicadores FTO
              </span>
            </h1>

            <p className="text-lg md:text-xl text-slate-400 max-w-2xl mb-10 leading-relaxed">
              Reemplaza tus archivos Excel por una plataforma centralizada. Captura, analiza y visualiza los 17 indicadores SQDCM de tu cédula en tiempo real.
            </p>

            <div className="flex flex-wrap gap-4 mb-12">
              <Link
                to={ctaLink}
                className="inline-flex items-center gap-2 px-7 py-3.5 bg-blue-600 hover:bg-blue-500 text-white font-semibold rounded-xl transition-all shadow-lg shadow-blue-600/25 hover:shadow-blue-500/30"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Acceder a la Plataforma
              </Link>
              <a
                href="#steps"
                className="inline-flex items-center gap-2 px-7 py-3.5 bg-white/5 hover:bg-white/10 border border-white/10 text-white font-semibold rounded-xl transition-all"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Ver Demo
              </a>
            </div>

            <div className="flex flex-wrap items-center gap-6 text-sm text-slate-500">
              <span className="flex items-center gap-2">
                <svg className="w-4 h-4 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
                Acceso Seguro
              </span>
              <span className="flex items-center gap-2">
                <svg className="w-4 h-4 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                </svg>
                PostgreSQL en Supabase
              </span>
              <span className="flex items-center gap-2">
                <svg className="w-4 h-4 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                Dashboards en Tiempo Real
              </span>
            </div>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="flex justify-center pb-8">
          <a href="#features" className="animate-bounce text-slate-500 hover:text-white transition-colors">
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </a>
        </div>
      </section>

      {/* ============ FEATURES ============ */}
      <section id="features" className="py-24 bg-[#f8fafc]">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <p className="text-sm font-bold text-blue-600 tracking-widest uppercase mb-3">Características</p>
            <h2 className="text-3xl md:text-4xl font-extrabold text-slate-900 mb-4">
              Todo lo que necesitas para gestionar tu cédula
            </h2>
            <p className="text-lg text-slate-500 max-w-2xl mx-auto">
              Una plataforma integral diseñada para reemplazar Excel y dar visibilidad total a tus indicadores FTO.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((f, i) => (
              <div key={i} className="bg-white rounded-2xl border border-slate-200 p-8 hover:shadow-lg hover:shadow-slate-200/50 transition-all hover:-translate-y-1">
                <div className={`inline-flex items-center justify-center w-12 h-12 ${f.iconBg} ${f.iconColor} rounded-xl mb-5`}>
                  {f.icon}
                </div>
                <h3 className="text-lg font-bold text-slate-900 mb-2">{f.title}</h3>
                <p className="text-slate-500 text-sm leading-relaxed">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ============ HOW IT WORKS ============ */}
      <section id="steps" className="py-24 bg-[#f8fafc]">
        <div className="max-w-5xl mx-auto px-6">
          <div className="text-center mb-16">
            <p className="text-sm font-bold text-blue-600 tracking-widest uppercase mb-3">Cómo funciona</p>
            <h2 className="text-3xl md:text-4xl font-extrabold text-slate-900 mb-4">
              En 3 pasos hacia la digitalización
            </h2>
            <p className="text-lg text-slate-500 max-w-2xl mx-auto">
              Comienza a usar la plataforma en minutos. Sin instalación, sin configuraciones complejas.
            </p>
          </div>

          <div className="flex flex-col md:flex-row items-start justify-center gap-8 md:gap-4">
            {steps.map((step, i) => (
              <div key={i} className="flex-1 flex flex-col items-center text-center relative">
                {/* Arrow between steps */}
                {i < steps.length - 1 && (
                  <div className="hidden md:block absolute top-8 -right-4 text-slate-300">
                    <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                  </div>
                )}
                <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${step.color} flex items-center justify-center text-white text-2xl font-bold mb-5 shadow-lg`}>
                  {step.num}
                </div>
                <h3 className="text-lg font-bold text-slate-900 mb-2">{step.title}</h3>
                <p className="text-slate-500 text-sm leading-relaxed max-w-xs">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* CTA Banner */}
        <div className="max-w-4xl mx-auto px-6 mt-20">
          <div className="bg-[#0f1d32] rounded-3xl p-10 md:p-14 text-center">
            <h2 className="text-2xl md:text-3xl font-bold text-white mb-4">
              ¿Listo para transformar tu control de indicadores?
            </h2>
            <p className="text-slate-400 mb-8 max-w-lg mx-auto">
              Únete a los equipos que ya están digitalizando su gestión FTO con PMI Plattform.
            </p>
            <Link
              to={ctaLink}
              className="inline-flex items-center gap-2 px-8 py-4 bg-green-500 hover:bg-green-400 text-white font-semibold rounded-xl transition-all shadow-lg shadow-green-500/25"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Comenzar Ahora
            </Link>
          </div>
        </div>
      </section>

      {/* ============ DASHBOARD PREVIEW ============ */}
      <section id="preview" className="py-24 bg-[#0a1628]">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div>
              <p className="text-sm font-bold text-blue-400 tracking-widest uppercase mb-3">Vista previa</p>
              <h2 className="text-3xl md:text-4xl font-extrabold text-white mb-6">
                Dashboard diseñado para la toma de decisiones
              </h2>
              <p className="text-slate-400 text-lg leading-relaxed mb-8">
                Visualiza el rendimiento de tu cédula de un vistazo. Indicadores semaforizados, filtros por semana, LC y operador.
              </p>

              <div className="grid grid-cols-2 gap-4 mb-8">
                <div className="bg-white/5 border border-white/10 rounded-xl p-5">
                  <p className="text-3xl font-bold text-white">17</p>
                  <p className="text-sm text-slate-400">Indicadores SQDCM</p>
                </div>
                <div className="bg-white/5 border border-white/10 rounded-xl p-5">
                  <p className="text-3xl font-bold text-white">100%</p>
                  <p className="text-sm text-slate-400">Tiempo Real</p>
                </div>
                <div className="bg-white/5 border border-white/10 rounded-xl p-5">
                  <p className="text-3xl font-bold text-white">&infin;</p>
                  <p className="text-sm text-slate-400">Histórico de Datos</p>
                </div>
                <div className="bg-white/5 border border-white/10 rounded-xl p-5">
                  <p className="text-3xl font-bold text-white">6-15</p>
                  <p className="text-sm text-slate-400">Cédulas Soportadas</p>
                </div>
              </div>

              <Link to={ctaLink} className="text-blue-400 hover:text-blue-300 font-medium inline-flex items-center gap-1 transition-colors">
                Explorar el Dashboard
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </Link>
            </div>

            {/* Dashboard mockup */}
            <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-2xl border border-white/10 p-6 shadow-2xl">
              <div className="flex items-center gap-2 mb-4">
                <div className="w-3 h-3 rounded-full bg-red-400" />
                <div className="w-3 h-3 rounded-full bg-yellow-400" />
                <div className="w-3 h-3 rounded-full bg-green-400" />
                <span className="ml-3 text-xs text-slate-500">pmi-platform.onrender.com/dashboard</span>
              </div>
              {/* Mini SQDCM cards preview */}
              <div className="grid grid-cols-5 gap-2 mb-4">
                {['S', 'Q', 'D', 'C', 'M'].map((letter, i) => {
                  const colors = ['from-emerald-500 to-emerald-600', 'from-blue-500 to-blue-600', 'from-purple-500 to-purple-600', 'from-orange-500 to-orange-600', 'from-pink-500 to-pink-600']
                  return (
                    <div key={letter} className={`bg-gradient-to-br ${colors[i]} rounded-lg p-3 text-center`}>
                      <p className="text-white font-bold text-lg">{letter}</p>
                      <p className="text-white/70 text-[10px]">
                        {['Sustent.', 'Calidad', 'Desemp.', 'Costo', 'Moral'][i]}
                      </p>
                    </div>
                  )
                })}
              </div>
              {/* Fake table rows */}
              <div className="space-y-1.5">
                {['Ricardo Vargas', 'César Lara', 'Miguel Vidrio'].map((name, i) => (
                  <div key={name} className="flex items-center gap-3 bg-white/5 rounded-lg px-3 py-2">
                    <span className="text-xs text-white font-medium w-28 truncate">{name}</span>
                    <div className="flex-1 flex gap-1.5">
                      {Array.from({ length: 8 }).map((_, j) => {
                        const colors = ['bg-green-400', 'bg-green-400', 'bg-yellow-400', 'bg-green-400', 'bg-red-400', 'bg-green-400', 'bg-green-400', 'bg-yellow-400']
                        return <span key={j} className={`w-2.5 h-2.5 rounded-full ${colors[(i + j) % colors.length]}`} />
                      })}
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-4 flex justify-center">
                <div className="inline-flex items-center gap-2 px-4 py-2 bg-green-500/20 border border-green-500/30 rounded-full">
                  <svg className="w-4 h-4 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-xs text-green-400 font-medium">Datos en Tiempo Real</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ============ FOOTER ============ */}
      <footer className="bg-[#060e1a] border-t border-white/5 py-16">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
            {/* Brand */}
            <div>
              <div className="flex items-center gap-2 mb-4">
                <svg className="w-8 h-8" viewBox="0 0 32 32" fill="none">
                  <rect width="32" height="32" rx="8" fill="url(#logoGrad2)" />
                  <path d="M8 12h6v2H8v-2zm0 4h10v2H8v-2zm0 4h8v2H8v-2z" fill="white" fillOpacity="0.9" />
                  <path d="M20 10l4 4-4 4" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  <defs><linearGradient id="logoGrad2" x1="0" y1="0" x2="32" y2="32"><stop stopColor="#3b82f6" /><stop offset="1" stopColor="#22c55e" /></linearGradient></defs>
                </svg>
                <span className="text-lg font-bold">PMI <span className="text-blue-400">Plattform</span></span>
              </div>
              <p className="text-slate-500 text-sm leading-relaxed max-w-xs">
                Plataforma digital para la gestión de indicadores FTO (SQDCM). Reemplaza tus archivos Excel por un sistema centralizado y seguro.
              </p>
            </div>

            {/* Links */}
            <div>
              <h4 className="text-sm font-bold text-white mb-4">Plataforma</h4>
              <ul className="space-y-2.5">
                <li><Link to="/dashboard" className="text-sm text-slate-500 hover:text-blue-400 transition-colors">Dashboard</Link></li>
                <li><Link to="/captura" className="text-sm text-slate-500 hover:text-blue-400 transition-colors">Captura de Datos</Link></li>
                <li><a href="#steps" className="text-sm text-slate-500 hover:text-blue-400 transition-colors">Análisis y Tendencias</a></li>
                <li><a href="#features" className="text-sm text-slate-500 hover:text-blue-400 transition-colors">Mi Cédula</a></li>
              </ul>
            </div>

            <div>
              <h4 className="text-sm font-bold text-white mb-4">Soporte</h4>
              <ul className="space-y-2.5">
                <li><a href="#" className="text-sm text-slate-500 hover:text-blue-400 transition-colors">Ayuda y Documentación</a></li>
                <li><Link to="/login" className="text-sm text-slate-500 hover:text-blue-400 transition-colors">Iniciar Sesión</Link></li>
                <li><a href="#" className="text-sm text-slate-500 hover:text-blue-400 transition-colors">Configuración</a></li>
              </ul>
            </div>
          </div>

          <div className="border-t border-white/5 mt-12 pt-8 text-center text-sm text-slate-600">
            &copy; {new Date().getFullYear()} PMI Plattform — FTO Digital. Philip Morris International.
          </div>
        </div>
      </footer>
    </div>
  )
}
