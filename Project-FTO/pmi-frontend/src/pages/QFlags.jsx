import { useState, useEffect, useRef } from 'react'
import CedulaSelector from '../components/CedulaSelector'
import UploadBanner from '../components/UploadBanner'
import * as api from '../lib/api'

export default function QFlags() {
  const [cedulas, setCedulas] = useState([])
  const [cedulaId, setCedulaId] = useState(null)
  const [cedulasLoading, setCedulasLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [result, setResult] = useState(null)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState(null)
  const [banner, setBanner] = useState(null)
  const inputRef = useRef(null)

  useEffect(() => {
    api.getCedulas()
      .then(res => {
        setCedulas(res.data || [])
        if (res.data?.length > 0) setCedulaId(res.data[0].id)
      })
      .catch(err => setError(err.message))
      .finally(() => setCedulasLoading(false))
  }, [])

  useEffect(() => { setResult(null); setSaved(false) }, [cedulaId])

  async function handleUpload(e) {
    const file = e.target.files[0]
    if (!file || !cedulaId) return
    e.target.value = ''
    setUploading(true); setError(null); setResult(null); setSaved(false)
    try {
      const res = await api.uploadQFlags(cedulaId, file)
      setResult(res)
    } catch (err) { setError(err.message) }
    finally { setUploading(false) }
  }

  async function handleSave() {
    if (!result?.results?.length) return
    setSaving(true); setError(null)
    try {
      await api.saveQFlags(cedulaId, result.results)
      setSaved(true)
      const cName = cedulas.find(c => c.id === cedulaId)?.nombre || ''
      setBanner({ message: 'Q Flags guardados exitosamente', detail: `${result.results.length} registros · ${cName}`, type: 'success' })
    } catch (err) { setError(err.message) }
    finally { setSaving(false) }
  }

  // Group results by semana for display
  const groupedBySemana = (() => {
    if (!result?.results) return []
    const map = {}
    for (const r of result.results) {
      if (!map[r.semana]) map[r.semana] = []
      map[r.semana].push(r)
    }
    return Object.entries(map)
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([sem, rows]) => ({ semana: sem, rows }))
  })()

  return (
    <div className="p-6 space-y-6 min-h-screen bg-[#0a1628]">
      <UploadBanner show={!!banner} onClose={() => setBanner(null)}
        message={banner?.message || ''} detail={banner?.detail} type={banner?.type} />

      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white">Q Flags — ComitDB</h1>
          <p className="text-sm text-slate-400">
            Sube el Excel exportado de ComitDB. La plataforma cuenta las Q Flags por operador y semana y las carga automáticamente al dashboard.
          </p>
        </div>
        <CedulaSelector cedulas={cedulas} selectedCedulaId={cedulaId} onChange={setCedulaId} loading={cedulasLoading} />
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-red-400 text-sm flex items-center gap-2">
          <span className="flex-1">{error}</span>
          <button onClick={() => setError(null)} className="text-red-300 hover:text-white">✕</button>
        </div>
      )}

      {/* Upload card */}
      <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-base font-semibold text-white mb-1">Subir archivo ComitDB</h2>
            <p className="text-xs text-slate-400">
              Columnas esperadas: <code className="text-cyan-400">Created</code>,{' '}
              <code className="text-cyan-400">[username]</code>,{' '}
              <code className="text-cyan-400">Workcenter</code>
            </p>
            <p className="text-[11px] text-slate-500 mt-1">
              El match se hace con aliases configurados o heurístico contra operadores.
            </p>
          </div>
          <div>
            <input
              ref={inputRef}
              type="file"
              accept=".xlsx,.xls"
              onChange={handleUpload}
              className="hidden"
            />
            <button
              onClick={() => inputRef.current?.click()}
              disabled={!cedulaId || uploading}
              className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 disabled:cursor-not-allowed text-white text-sm rounded-lg transition-colors"
            >
              {uploading ? 'Procesando...' : '📤 Cargar Excel'}
            </button>
          </div>
        </div>
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-4">
          {/* Summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <SummaryCard label="Filas leídas" value={result.total_rows} />
            <SummaryCard label="Q Flags contadas" value={result.matched_count} color="text-cyan-400" />
            <SummaryCard label="Operadores" value={result.unique_operators} />
            <SummaryCard label="Semanas" value={result.unique_weeks?.length || 0} />
          </div>

          {/* Unmatched */}
          {result.unmatched_users?.length > 0 && (
            <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-4">
              <p className="text-sm font-semibold text-amber-400 mb-2">
                ⚠ {result.unmatched_users.length} usuarios sin match ({result.unmatched_users.reduce((s, u) => s + u.count, 0)} filas ignoradas)
              </p>
              <div className="flex flex-wrap gap-2">
                {result.unmatched_users.slice(0, 20).map(u => (
                  <span key={u.username} className="text-xs bg-amber-950/40 text-amber-300 px-2 py-1 rounded">
                    {u.username} <span className="text-amber-500/60">({u.count})</span>
                  </span>
                ))}
                {result.unmatched_users.length > 20 && (
                  <span className="text-xs text-amber-400">+{result.unmatched_users.length - 20} más</span>
                )}
              </div>
              <p className="text-[11px] text-amber-300/70 mt-2">
                Puedes agregar estos usuarios en el Excel de aliases (columna <code>username_comitdb</code>) y volver a subir.
              </p>
            </div>
          )}

          {/* Grouped table */}
          {groupedBySemana.map(({ semana, rows }) => (
            <div key={semana} className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
              <div className="px-4 py-2 bg-cyan-950/30 border-b border-cyan-500/20">
                <span className="text-xs font-semibold text-cyan-300">Semana {semana}</span>
                <span className="text-[11px] text-slate-500 ml-2">({rows.length} operadores)</span>
              </div>
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-white/5 text-slate-500">
                    <th className="px-3 py-1.5 text-left font-medium">Operador</th>
                    <th className="px-3 py-1.5 text-left font-medium">Máquina</th>
                    <th className="px-3 py-1.5 text-center font-medium">Q Flags</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.sort((a, b) => b.qflags - a.qflags).map((r, i) => (
                    <tr key={i} className="border-b border-white/5 hover:bg-white/[0.02]">
                      <td className="px-3 py-1.5 text-white">{r.nombre}</td>
                      <td className="px-3 py-1.5 text-slate-400">{r.maquina || '—'}</td>
                      <td className="px-3 py-1.5 text-center">
                        <span className="text-cyan-300 font-semibold">{r.qflags}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}

          {/* Save button */}
          <div className="flex items-center justify-end gap-3">
            {saved && <span className="text-sm text-green-400">✓ Guardado</span>}
            <button
              onClick={handleSave}
              disabled={saving || saved || !result.matched_count}
              className="px-5 py-2 bg-green-600 hover:bg-green-500 disabled:bg-slate-700 disabled:cursor-not-allowed text-white text-sm rounded-lg transition-colors"
            >
              {saving ? 'Guardando...' : saved ? 'Guardado' : '💾 Guardar en Dashboard'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function SummaryCard({ label, value, color = 'text-white' }) {
  return (
    <div className="bg-[#0f1d32] rounded-lg border border-white/5 px-4 py-3">
      <p className="text-[11px] text-slate-500 uppercase tracking-wide">{label}</p>
      <p className={`text-2xl font-bold ${color}`}>{value ?? 0}</p>
    </div>
  )
}
