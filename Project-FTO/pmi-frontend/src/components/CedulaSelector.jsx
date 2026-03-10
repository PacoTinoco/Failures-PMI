export default function CedulaSelector({ cedulas, selectedCedulaId, onChange, loading = false }) {
  if (loading) {
    return (
      <div className="flex items-center gap-2">
        <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
        <span className="text-xs text-slate-400">Cargando cédulas...</span>
      </div>
    )
  }

  return (
    <div className="flex items-center gap-2">
      <label className="text-sm text-slate-400">Cédula:</label>
      <select
        value={selectedCedulaId || ''}
        onChange={(e) => onChange(e.target.value || null)}
        className="px-3 py-2 bg-[#0f1d32] border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        <option value="">Selecciona una cédula</option>
        {(cedulas || []).map((c) => (
          <option key={c.id} value={c.id}>
            {c.nombre}
          </option>
        ))}
      </select>
    </div>
  )
}
