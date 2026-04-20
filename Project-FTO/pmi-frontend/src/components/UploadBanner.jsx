import { useState, useEffect } from 'react'

/**
 * UploadBanner — banner de confirmación que se muestra tras un upload exitoso.
 * Se auto-cierra después de `duration` ms (default 6s).
 *
 * Props:
 *   show      — boolean, controla visibilidad
 *   onClose   — callback para cerrar
 *   cedulaName — nombre de la cédula (ej. "Cápsula")
 *   message   — texto principal (ej. "Calendario cargado exitosamente")
 *   detail    — texto secundario opcional (ej. "42 registros, 13 empleados")
 *   type      — 'success' | 'warning' | 'error' (default 'success')
 *   duration  — ms antes de auto-cerrar (default 6000, 0 = no auto-cierra)
 */
export default function UploadBanner({ show, onClose, cedulaName, message, detail, type = 'success', duration = 6000 }) {
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    if (show) {
      setVisible(true)
      if (duration > 0) {
        const t = setTimeout(() => { setVisible(false); onClose?.() }, duration)
        return () => clearTimeout(t)
      }
    } else {
      setVisible(false)
    }
  }, [show])

  if (!visible) return null

  const colors = {
    success: 'from-green-600/90 to-emerald-700/90 border-green-400/30',
    warning: 'from-yellow-600/90 to-amber-700/90 border-yellow-400/30',
    error: 'from-red-600/90 to-rose-700/90 border-red-400/30',
  }

  const icons = {
    success: (
      <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
      </svg>
    ),
    warning: (
      <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    error: (
      <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
      </svg>
    ),
  }

  return (
    <div className={`fixed top-4 left-1/2 -translate-x-1/2 z-[100] w-full max-w-lg animate-slide-down`}>
      <div className={`bg-gradient-to-r ${colors[type]} border rounded-xl shadow-2xl px-5 py-3 flex items-center gap-3`}>
        <div className="bg-white/20 rounded-full p-1.5 flex-shrink-0">
          {icons[type]}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-white text-sm font-semibold">{message}</span>
            {cedulaName && (
              <span className="bg-white/20 text-white/90 text-[10px] font-medium px-2 py-0.5 rounded-full">
                {cedulaName}
              </span>
            )}
          </div>
          {detail && <p className="text-white/80 text-xs mt-0.5">{detail}</p>}
        </div>
        <button onClick={() => { setVisible(false); onClose?.() }}
          className="text-white/60 hover:text-white text-lg flex-shrink-0 transition-colors">
          ✕
        </button>
      </div>
    </div>
  )
}
