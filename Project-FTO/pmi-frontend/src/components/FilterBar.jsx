export default function FilterBar({ lineCoordinators, selectedLC, onLCChange }) {
  return (
    <div className="flex items-center gap-3">
      <label className="text-sm text-slate-400">Line Coordinator:</label>
      <select
        value={selectedLC || ''}
        onChange={(e) => onLCChange(e.target.value || null)}
        className="px-3 py-2 bg-[#0f1d32] border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        <option value="">Todos</option>
        {lineCoordinators.map((lc) => (
          <option key={lc.id} value={lc.id}>
            {lc.nombre}
          </option>
        ))}
      </select>
    </div>
  )
}
