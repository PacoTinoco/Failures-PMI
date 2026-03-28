import { useState, useEffect, useRef } from 'react'
import CedulaSelector from '../components/CedulaSelector'
import WeekSelector from '../components/WeekSelector'
import * as api from '../lib/api'

function getClosestMonday() {
  const d = new Date()
  const day = d.getDay()
  const diff = d.getDate() - day + (day === 0 ? -6 : 1)
  const monday = new Date(d.getFullYear(), d.getMonth(), diff)
  const yyyy = monday.getFullYear()
  const mm = String(monday.getMonth() + 1).padStart(2, '0')
  const dd = String(monday.getDate()).padStart(2, '0')
  return `${yyyy}-${mm}-${dd}`
}

export default function BOSQBOS() {
  const [cedulas, setCedulas] = useState([])
  const [cedulaId, setCedulaId] = useState(null)
  const [cedulasLoading, setCedulasLoading] = useState(true)
  const [semana, setSemana] = useState(getClosestMonday())
  const [activeTab, setActiveTab] = useState('bos')
  const [error, setError] = useState(null)

  // BOS state
  const [bosUploading, setBosUploading] = useState(false)
  const [bosResult, setBosResult] = useState(null)
  const [bosSaving, setBosSaving] = useState(false)
  const [bosSaved, setBosSaved] = useState(false)
  const bosInputRef = useRef(null)

  // QBOS state
  const [qbosUploading, setQbosUploading] = useState(false)
  const [qbosResult, setQbosResult] = useState(null)
  const [qbosSaving, setQbosSaving] = useState(false)
  const [qbosSaved, setQbosSaved] = useState(false)
  const qbosInputRef = useRef(null)

  // Aliases (mapeo) state
  const [aliasUploading, setAliasUploading] = useState(false)
  const [aliasResult, setAliasResult] = useState(null)
  const aliasInputRef = useRef(null)

  useEffect(() => {
    api.getCedulas()
      .then(res => {
        setCedulas(res.data || [])
        if (res.data?.length > 0) setCedulaId(res.data[0].id)
      })
      .catch(err => setError(err.message))
      .finally(() => setCedulasLoading(false))
  }, [])

  // Reset results when semana changes
  useEffect(() => {
    setBosResult(null); setBosSaved(false)
    setQbosResult(null); setQbosSaved(false)
  }, [semana])

  // ── BOS handlers ──
  async function handleBosUpload(e) {
    const file = e.target.files[0]
    if (!file || !cedulaId) return
    e.target.value = ''
    setBosUploading(true); setError(null); setBosResult(null); setBosSaved(false)
    try {
      const res = await api.uploadBOS(cedulaId, semana, file)
      setBosResult(res)
    } catch (err) { setError(err.message) }
    finally { setBosUploading(false) }
  }

  async function handleBosSave() {
    if (!bosResult?.results?.length) return
    setBosSaving(true); setError(null)
    try {
      await api.saveBOS(cedulaId, semana, bosResult.results)
      setBosSaved(true)
    } catch (err) { setError(err.message) }
    finally { setBosSaving(false) }
  }

  // ── QBOS handlers ──
  async function handleQbosUpload(e) {
    const file = e.target.files[0]
    if (!file || !cedulaId) return
    e.target.value = ''
    setQbosUploading(true); setError(null); setQbosResult(null); setQbosSaved(false)
    try {
      const res = await api.uploadQBOS(cedulaId, semana, file)
      setQbosResult(res)
    } catch (err) { setError(err.message) }
    finally { setQbosUploading(false) }
  }

  async function handleQbosSave() {
    if (!qbosResult?.results?.length) return
    setQbosSaving(true); setError(null)
    try {
      await api.saveQBOS(cedulaId, semana, qbosResult.results)
      setQbosSaved(true)
    } catch (err) { setError(err.message) }
    finally { setQbosSaving(false) }
  }

  // ── Aliases handler ──
  async function handleAliasUpload(e) {
    const file = e.target.files[0]
    if (!file || !cedulaId) return
    e.target.value = ''
    setAliasUploading(true); setError(null); setAliasResult(null)
    try {
      const res = await api.uploadAliases(cedulaId, file)
      setAliasResult(res)
    } catch (err) { setError(err.message) }
    finally { setAliasUploading(false) }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white">BOS / QBOS</h1>
          <p className="text-sm text-slate-400">Carga de archivos BOS y QBOS por semana</p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <CedulaSelector cedulas={cedulas} selectedCedulaId={cedulaId} onChange={setCedulaId} loading={cedulasLoading} />
          <WeekSelector value={semana} onChange={setSemana} />
        </div>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-red-400 text-sm flex items-center gap-2">
          <span className="flex-1">{error}</span>
          <button onClick={() => setError(null)} className="text-red-300 hover:text-white">✕</button>
        </div>
      )}

      {/* Tabs BOS / QBOS */}
      <div className="flex gap-1 bg-[#0a1628] rounded-lg p-1 w-fit">
        {[
          { key: 'bos', label: 'BOS' },
          { key: 'qbos', label: 'QBOS' },
          { key: 'mapeo', label: 'Mapeo' },
        ].map(tab => (
          <button key={tab.key} onClick={() => setActiveTab(tab.key)}
            className={`px-6 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === tab.key ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'
            }`}>
            {tab.label}
          </button>
        ))}
      </div>

      {/* BOS Tab */}
      {activeTab === 'bos' && (
        <UploadPanel
          title="BOS — Behavioural Observation System"
          subtitle={`Subir archivo CSV de BOS para la semana ${semana}`}
          description="Match por email (USER) → columna bos = BOS_CREATED"
          accept=".csv"
          inputRef={bosInputRef}
          uploading={bosUploading}
          onUpload={handleBosUpload}
          result={bosResult}
          saving={bosSaving}
          saved={bosSaved}
          onSave={handleBosSave}
          type="bos"
        />
      )}

      {/* QBOS Tab */}
      {activeTab === 'qbos' && (
        <UploadPanel
          title="QBOS — Quality BOS"
          subtitle={`Subir archivo Excel de QBOS para la semana ${semana}`}
          description="Match por nombre (Personnel Name) → columna qbos = Frecuency"
          accept=".xlsx,.xls"
          inputRef={qbosInputRef}
          uploading={qbosUploading}
          onUpload={handleQbosUpload}
          result={qbosResult}
          saving={qbosSaving}
          saved={qbosSaved}
          onSave={handleQbosSave}
          type="qbos"
        />
      )}

      {/* Mapeo Tab */}
      {activeTab === 'mapeo' && (
        <div className="space-y-4">
          <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-5">
            <h3 className="text-sm font-semibold text-white mb-1">Mapeo de empleados</h3>
            <p className="text-xs text-slate-400 mb-2">
              Sube un Excel con las variaciones de nombre/email por sistema para mejorar el matching automático.
            </p>
            <p className="text-xs text-slate-500 mb-3">
              Columnas: <span className="text-slate-300">nombre_en_bd</span> (obligatorio), <span className="text-slate-300">email_en_bos_csv</span>, <span className="text-slate-300">nombre_en_qbos_excel</span>, <span className="text-slate-300">email_dh</span>
            </p>
            <input ref={aliasInputRef} type="file" accept=".xlsx,.xls" onChange={handleAliasUpload} className="hidden" />
            <button
              onClick={() => aliasInputRef.current?.click()}
              disabled={aliasUploading}
              className="w-full py-2.5 rounded-lg text-sm font-medium transition-colors bg-purple-600 hover:bg-purple-500 text-white disabled:opacity-50"
            >
              {aliasUploading ? 'Procesando...' : 'Subir Excel de mapeo'}
            </button>
          </div>

          {aliasResult && (
            <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-5 space-y-3">
              <div className="flex items-center gap-3">
                <div className="bg-green-500/10 rounded-lg px-4 py-3 text-center flex-1">
                  <p className="text-2xl font-bold text-green-400">{aliasResult.matched}</p>
                  <p className="text-[10px] text-slate-500 uppercase tracking-wider mt-1">Mapeados</p>
                </div>
                <div className="bg-yellow-500/10 rounded-lg px-4 py-3 text-center flex-1">
                  <p className="text-2xl font-bold text-yellow-400">{aliasResult.skipped?.length || 0}</p>
                  <p className="text-[10px] text-slate-500 uppercase tracking-wider mt-1">No encontrados</p>
                </div>
              </div>

              <p className="text-sm text-green-400">{aliasResult.message}</p>

              {aliasResult.skipped?.length > 0 && (
                <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg px-4 py-3">
                  <p className="text-yellow-400 text-sm font-medium mb-1">Nombres no encontrados en BD:</p>
                  {aliasResult.skipped.map((name, i) => (
                    <p key={i} className="text-yellow-300/70 text-xs">{name}</p>
                  ))}
                  <p className="text-xs text-slate-500 mt-2">
                    Estos empleados no están en la tabla operadores. Agrégalos primero en la sección Equipos.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}


function UploadPanel({ title, subtitle, description, accept, inputRef, uploading, onUpload, result, saving, saved, onSave, type }) {
  const valueKey = type === 'bos' ? 'bos' : 'qbos'
  const nameKey = type === 'bos' ? 'email' : 'file_name'

  return (
    <div className="space-y-4">
      {/* Upload zone */}
      <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-5">
        <h3 className="text-sm font-semibold text-white mb-1">{title}</h3>
        <p className="text-xs text-slate-500 mb-1">{subtitle}</p>
        <p className="text-xs text-slate-600 mb-3">{description}</p>
        <input ref={inputRef} type="file" accept={accept} onChange={onUpload} className="hidden" />
        <button
          onClick={() => inputRef.current?.click()}
          disabled={uploading}
          className="w-full py-2.5 rounded-lg text-sm font-medium transition-colors bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-50"
        >
          {uploading ? 'Procesando...' : `Subir archivo ${type.toUpperCase()}`}
        </button>
      </div>

      {/* Results */}
      {result && (
        <>
          {/* Stat cards */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <StatCard label="Total filas" value={result.total_rows} />
            <StatCard label="Mapeados" value={result.matched} color="text-green-400" />
            <StatCard label="Sin match" value={result.unmatched?.length || 0} color={result.unmatched?.length > 0 ? 'text-yellow-400' : 'text-slate-500'} />
            <StatCard label="Nuevos" value={result.new} color="text-blue-400" />
            <StatCard label="Actualizados" value={result.updated} color="text-purple-400" />
          </div>

          {/* Unmatched warnings */}
          {result.unmatched?.length > 0 && (
            <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg px-4 py-3">
              <p className="text-yellow-400 text-sm font-medium mb-1">
                {result.unmatched.length} registro{result.unmatched.length > 1 ? 's' : ''} sin match
              </p>
              <div className="space-y-1">
                {(type === 'bos' ? result.unmatched : result.unmatched.map(u => u.name || u)).map((item, i) => (
                  <p key={i} className="text-yellow-300/70 text-xs">
                    {typeof item === 'string' ? item : item}
                    {type === 'qbos' && result.unmatched[i]?.suggestions?.length > 0 && (
                      <span className="text-slate-500 ml-1">
                        (¿{result.unmatched[i].suggestions.map(s => s.nombre).join(', ')}?)
                      </span>
                    )}
                  </p>
                ))}
              </div>
            </div>
          )}

          {/* Results table */}
          {result.results?.length > 0 && (
            <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
              <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-white">Resultados ({result.results.length})</h3>
                  <p className="text-xs text-slate-500">Semana: {result.semana}</p>
                </div>
                <button
                  onClick={onSave}
                  disabled={saving || saved}
                  className={`px-5 py-2 rounded-lg text-sm font-medium transition-colors ${
                    saved
                      ? 'bg-green-600/20 text-green-400 cursor-default'
                      : 'bg-green-600 hover:bg-green-500 text-white'
                  }`}
                >
                  {saving ? 'Guardando...' : saved ? '✓ Guardado' : 'Guardar en BD'}
                </button>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-white/10">
                      <th className="px-4 py-2 text-left text-xs text-slate-400">Operador</th>
                      <th className="px-3 py-2 text-left text-xs text-slate-400">
                        {type === 'bos' ? 'Email' : 'Nombre archivo'}
                      </th>
                      <th className="px-3 py-2 text-center text-xs text-slate-400">{type.toUpperCase()}</th>
                      {type === 'qbos' && <th className="px-3 py-2 text-center text-xs text-slate-400">Match</th>}
                      <th className="px-3 py-2 text-center text-xs text-slate-400">Estado</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.results.map((r, i) => (
                      <tr key={i} className="border-b border-white/5 hover:bg-slate-700/20">
                        <td className="px-4 py-2 text-white text-sm font-medium">{r.nombre}</td>
                        <td className="px-3 py-2 text-slate-400 text-xs">{r[nameKey]}</td>
                        <td className="px-3 py-2 text-center text-white font-medium">{r[valueKey]}</td>
                        {type === 'qbos' && (
                          <td className="px-3 py-2 text-center">
                            <span className={`text-xs ${r.match_score >= 100 ? 'text-green-400' : 'text-yellow-400'}`}>
                              {r.match_score}%
                            </span>
                          </td>
                        )}
                        <td className="px-3 py-2 text-center">
                          <ActionBadge action={r.action} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

function StatCard({ label, value, color = 'text-white' }) {
  return (
    <div className="bg-[#0f1d32] rounded-xl border border-white/5 px-4 py-3 text-center">
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
      <p className="text-[10px] text-slate-500 uppercase tracking-wider mt-1">{label}</p>
    </div>
  )
}

function ActionBadge({ action }) {
  const styles = {
    nuevo:       'bg-blue-500/10 text-blue-400',
    actualizado: 'bg-purple-500/10 text-purple-400',
    sin_cambio:  'bg-slate-500/10 text-slate-400',
    info:        'bg-cyan-500/10 text-cyan-400',
  }
  const labels = {
    nuevo: 'Nuevo',
    actualizado: 'Actualizado',
    sin_cambio: 'Sin cambio',
    info: 'LC/LS',
  }
  return (
    <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${styles[action] || 'text-slate-500'}`}>
      {labels[action] || action}
    </span>
  )
}
