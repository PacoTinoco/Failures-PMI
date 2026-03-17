export default function QM() {
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold text-white">QM</h1>
        <p className="text-sm text-slate-400">Quality Matrix — Control de calidad</p>
      </div>
      <div className="bg-[#0f1d32] rounded-xl border border-white/5 px-6 py-16 text-center">
        <div className="inline-flex items-center justify-center w-16 h-16 bg-green-500/10 rounded-full mb-4">
          <svg className="w-8 h-8 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
          </svg>
        </div>
        <p className="text-lg font-semibold text-white mb-2">Próximamente</p>
        <p className="text-sm text-slate-400">Esta sección está en desarrollo.</p>
      </div>
    </div>
  )
}
