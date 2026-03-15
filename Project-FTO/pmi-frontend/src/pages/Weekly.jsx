export default function Weekly() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold text-white">Weekly</h1>
        <p className="text-sm text-slate-400">Seguimiento semanal de indicadores</p>
      </div>
      <div className="bg-[#0f1d32] rounded-xl border border-white/5 px-6 py-16 text-center">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-500/10 rounded-full mb-4">
          <svg className="w-8 h-8 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6.75 3v2.25M17.25 3v2.25M3 18.75V7.5a2.25 2.25 0 012.25-2.25h13.5A2.25 2.25 0 0121 7.5v11.25m-18 0A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75m-18 0v-7.5A2.25 2.25 0 015.25 9h13.5A2.25 2.25 0 0121 11.25v7.5" />
          </svg>
        </div>
        <p className="text-lg font-semibold text-white mb-2">Próximamente</p>
        <p className="text-sm text-slate-400">Esta sección está en desarrollo.</p>
      </div>
    </div>
  )
}
