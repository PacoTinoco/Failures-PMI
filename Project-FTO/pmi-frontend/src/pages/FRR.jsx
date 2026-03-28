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

export default function FRR() {
  const [cedulas, setCedulas] = useState([])
  const [cedulaId, setCedulaId] = useState(null)
  const [cedulasLoading, setCedulasLoading] = useState(true)
  const [semana, setSemana] = useState(getClosestMonday())
  const [activeTab, setActiveTab] = useState('frr')
  const [error, setError] = useState(null)

  // ROL state
  const [rolUploading, setRolUploading] = useState(false)
  const [rolResult, setRolResult] = useState(null)
  const rolInputRef = useRef(null)

  // FRR state
  const [frrUploading, setFrrUploading] = useState(false)
  const [frrResult, setFrrResult] = useState(null)
  const [frrSaving, setFrrSaving] = useState(false)
  const [frrSaved, setFrrSaved] = useState(false)
  const frrInputRef = useRef(null)

  // Turnos view state
  const [turnosData, setTurnosData] = useState(null)
  const [turnosLoading, setTurnosLoading] = useState(false)

  useEffect(() => {
    api.getCedulas()
      .then(res => {
        setCedulas(res.data || [])
        if (res.data?.length > 0) setCedulaId(res.data[0].id)
      })
      .catch(err => setError(err.message))
      .finally(() => setCedulasLoading(false))
  }, [])

  useEffect(() => {
    setFrrResult(null); setFrrSaved(false)
    if (activeTab === 'turnos' && cedulaId && semana) loadTurnos()
  }, [semana])

  useEffect(() => {
    if (activeTab === 'turnos' && cedulaId && semana) loadTurnos()
  }, [activeTab, cedulaId])

  async function loadTurnos() {
    setTurnosLoading(true)
    try {
      const res = await api.getROLSemana(cedulaId, semana)
      setTurnosData(res.data || [])
    } catch (err) { setError(err.message) }
    finally { setTurnosLoading(false) }
  }

  // ── ROL handler ──
  async function handleRolUpload(e) {
    const file = e.target.files[0]
    if (!file || !cedulaId) return
    e.target.value = ''
    setRolUploading(true); setError(null); setRolResult(null)
    try {
      const res = await api.uploadROL(cedulaId, file)
      setRolResult(res)
    } catch (err) { setError(err.message) }
    finally { setRolUploading(false) }
  }

  // ── FRR handlers ──
  async function handleFrrUpload(e) {
    const file = e.target.files[0]
    if (!file || !cedulaId) return
    e.target.value = ''
    setFrrUploading(true); setError(null); setFrrResult(null); setFrrSaved(false)
    try {
      const res = await api.uploadFRR(cedulaId, semana, file)
      setFrrResult(res)
    } catch (err) { setError(err.message) }
    finally { setFrrUploading(false) }
  }

  async function handleFrrSave() {
    if (!frrResult?.results?.length) return
    setFrrSaving(true); setError(null)
    try {
      await api.saveFRR(cedulaId, semana, frrResult.results)
      setFrrSaved(true)
    } catch (err) { setError(err.message) }
    finally { setFrrSaving(false) }
  }

  // ── Override turno ──
  async function handleOverride(operadorId, fecha, nuevoTurno) {
    try {
      await api.overrideTurno(cedulaId, operadorId, fecha, nuevoTurno)
      loadTurnos()
    } catch (err) { setError(err.message) }
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white">FRR — Filter Reject Rate</h1>
          <p className="text-sm text-slate-400">Calendario de turnos y desperdicio semanal</p>
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

      {/* Tabs */}
      <div className="flex gap-1 bg-[#0a1628] rounded-lg p-1 w-fit">
        {[
          { key: 'frr', label: 'FRR Semanal' },
          { key: 'rol', label: 'Calendario ROL' },
          { key: 'turnos', label: 'Ver Turnos' },
        ].map(tab => (
          <button key={tab.key} onClick={() => setActiveTab(tab.key)}
            className={`px-6 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === tab.key ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'
            }`}>
            {tab.label}
          </button>
        ))}
      </div>

      {/* FRR Upload Tab */}
      {activeTab === 'frr' && (
        <div className="space-y-4">
          <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-5">
            <h3 className="text-sm font-semibold text-white mb-1">Subir FRR semanal</h3>
            <p className="text-xs text-slate-400 mb-1">Excel con columnas: CAL_SHIFT, Reject Rate, LINE_NAME</p>
            <p className="text-xs text-slate-500 mb-3">Semana: {semana} — Se cruza automáticamente con el calendario de turnos ROL</p>
            <input ref={frrInputRef} type="file" accept=".xlsx,.xls" onChange={handleFrrUpload} className="hidden" />
            <button
              onClick={() => frrInputRef.current?.click()}
              disabled={frrUploading}
              className="w-full py-2.5 rounded-lg text-sm font-medium transition-colors bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-50"
            >
              {frrUploading ? 'Procesando...' : 'Subir archivo FRR'}
            </button>
          </div>

          {frrResult && (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <StatCard label="Operadores" value={frrResult.matched} color="text-green-400" />
                <StatCard label="Sin datos FRR" value={frrResult.no_data?.length || 0} color={frrResult.no_data?.length > 0 ? 'text-yellow-400' : 'text-slate-500'} />
                <StatCard label="Nuevos" value={frrResult.new} color="text-blue-400" />
                <StatCard label="Actualizados" value={frrResult.updated} color="text-purple-400" />
              </div>

              {frrResult.no_data?.length > 0 && (
                <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg px-4 py-3">
                  <p className="text-yellow-400 text-sm font-medium mb-1">Operadores sin datos FRR esta semana:</p>
                  {frrResult.no_data.map((item, i) => (
                    <p key={i} className="text-yellow-300/70 text-xs">{item.nombre} ({item.kdf}) — {item.reason}</p>
                  ))}
                </div>
              )}

              {frrResult.results?.length > 0 && (
                <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
                  <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-semibold text-white">Resultados FRR ({frrResult.results.length})</h3>
                      <p className="text-xs text-slate-500">Semana: {frrResult.semana}</p>
                    </div>
                    <button
                      onClick={handleFrrSave}
                      disabled={frrSaving || frrSaved}
                      className={`px-5 py-2 rounded-lg text-sm font-medium transition-colors ${
                        frrSaved ? 'bg-green-600/20 text-green-400 cursor-default' : 'bg-green-600 hover:bg-green-500 text-white'
                      }`}
                    >
                      {frrSaving ? 'Guardando...' : frrSaved ? '✓ Guardado' : 'Guardar en BD'}
                    </button>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-white/10">
                          <th className="px-4 py-2 text-left text-xs text-slate-400">Operador</th>
                          <th className="px-3 py-2 text-center text-xs text-slate-400">KDF</th>
                          <th className="px-3 py-2 text-center text-xs text-slate-400">FRR %</th>
                          <th className="px-3 py-2 text-center text-xs text-slate-400">Registros</th>
                          <th className="px-3 py-2 text-center text-xs text-slate-400">Estado</th>
                        </tr>
                      </thead>
                      <tbody>
                        {frrResult.results.map((r, i) => (
                          <tr key={i} className="border-b border-white/5 hover:bg-slate-700/20">
                            <td className="px-4 py-2 text-white text-sm font-medium">{r.nombre}</td>
                            <td className="px-3 py-2 text-center text-slate-400 text-xs">{r.kdf}</td>
                            <td className="px-3 py-2 text-center text-white font-medium">{r.frr}%</td>
                            <td className="px-3 py-2 text-center text-slate-400 text-xs">{r.num_registros}</td>
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
      )}

      {/* ROL Calendar Upload Tab */}
      {activeTab === 'rol' && (
        <div className="space-y-4">
          <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-5">
            <h3 className="text-sm font-semibold text-white mb-1">Calendario ROL de operadores</h3>
            <p className="text-xs text-slate-400 mb-1">Sube el Excel con columnas: Nombre, LC, KDF + fechas con turnos (D/T/N)</p>
            <p className="text-xs text-slate-500 mb-3">Este calendario se usa para cruzar con el FRR y asignar el desperdicio a cada operador</p>
            <input ref={rolInputRef} type="file" accept=".xlsx,.xls" onChange={handleRolUpload} className="hidden" />
            <button
              onClick={() => rolInputRef.current?.click()}
              disabled={rolUploading}
              className="w-full py-2.5 rounded-lg text-sm font-medium transition-colors bg-purple-600 hover:bg-purple-500 text-white disabled:opacity-50"
            >
              {rolUploading ? 'Procesando...' : 'Subir Excel ROL'}
            </button>
          </div>

          {rolResult && (
            <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-5 space-y-3">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <StatCard label="Operadores mapeados" value={rolResult.matched_operators} color="text-green-400" />
                <StatCard label="Registros creados" value={rolResult.total_records} color="text-blue-400" />
                <StatCard label="Fechas encontradas" value={rolResult.dates_found} color="text-purple-400" />
                <StatCard label="No encontrados" value={rolResult.skipped_names?.length || 0} color={rolResult.skipped_names?.length > 0 ? 'text-yellow-400' : 'text-slate-500'} />
              </div>
              <p className="text-xs text-slate-400">Rango: {rolResult.date_range}</p>
              {rolResult.skipped_names?.length > 0 && (
                <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg px-4 py-3">
                  <p className="text-yellow-400 text-sm font-medium mb-1">Nombres no encontrados en BD:</p>
                  {rolResult.skipped_names.map((name, i) => (
                    <p key={i} className="text-yellow-300/70 text-xs">{name}</p>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* View Turnos Tab */}
      {activeTab === 'turnos' && (
        <TurnosView
          data={turnosData}
          loading={turnosLoading}
          semana={semana}
          onOverride={handleOverride}
        />
      )}
    </div>
  )
}


function TurnosView({ data, loading, semana, onOverride }) {
  if (loading) return <div className="text-center py-12 text-slate-500">Cargando turnos...</div>
  if (!data || data.length === 0) return (
    <div className="bg-[#0f1d32] rounded-xl border border-white/5 px-6 py-12 text-center">
      <p className="text-slate-500">No hay turnos cargados para esta semana. Sube primero el calendario ROL.</p>
    </div>
  )

  // Group by operador
  const byOperador = {}
  data.forEach(r => {
    if (!byOperador[r.operador_id]) {
      byOperador[r.operador_id] = { nombre: r.nombre, kdf: r.kdf, dias: {} }
    }
    byOperador[r.operador_id].dias[r.fecha] = { turno: r.turno, override: r.es_override }
  })

  // Get dates for the week
  const baseDate = new Date(semana + 'T00:00:00')
  const weekDates = []
  for (let i = 0; i < 7; i++) {
    const d = new Date(baseDate)
    d.setDate(d.getDate() + i)
    weekDates.push(d.toISOString().split('T')[0])
  }

  const dayNames = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
  const turnoColors = { D: 'bg-amber-500/20 text-amber-300', T: 'bg-blue-500/20 text-blue-300', N: 'bg-indigo-500/20 text-indigo-300' }
  const turnoLabels = { D: 'Día', T: 'Tarde', N: 'Noche' }

  const operadores = Object.entries(byOperador).sort((a, b) => a[1].nombre.localeCompare(b[1].nombre))

  return (
    <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
      <div className="px-4 py-3 border-b border-white/5">
        <h3 className="text-sm font-semibold text-white">Turnos — Semana {semana}</h3>
        <p className="text-xs text-slate-500">Clic en un turno para cambiarlo manualmente</p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-white/10">
              <th className="px-4 py-2 text-left text-xs text-slate-400 sticky left-0 bg-[#0f1d32]">Operador</th>
              <th className="px-2 py-2 text-center text-xs text-slate-400">KDF</th>
              {weekDates.map((d, i) => (
                <th key={d} className="px-2 py-2 text-center text-xs text-slate-400">
                  <div>{dayNames[i]}</div>
                  <div className="text-[10px] text-slate-600">{d.slice(5)}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {operadores.map(([oid, info]) => (
              <tr key={oid} className="border-b border-white/5 hover:bg-white/[0.02]">
                <td className="px-4 py-2 text-white text-xs font-medium sticky left-0 bg-[#0f1d32] whitespace-nowrap">{info.nombre}</td>
                <td className="px-2 py-2 text-center text-slate-500 text-[10px]">{info.kdf}</td>
                {weekDates.map(d => {
                  const dia = info.dias[d]
                  const turno = dia?.turno
                  const isOverride = dia?.override
                  return (
                    <td key={d} className="px-1 py-1 text-center">
                      <button
                        onClick={() => {
                          const opciones = ['D', 'T', 'N', null]
                          const idx = opciones.indexOf(turno)
                          const next = opciones[(idx + 1) % opciones.length]
                          onOverride(oid, d, next)
                        }}
                        className={`w-8 h-7 rounded text-[10px] font-bold transition-colors ${
                          turno
                            ? `${turnoColors[turno]} ${isOverride ? 'ring-1 ring-orange-400/50' : ''}`
                            : 'bg-slate-800/50 text-slate-600'
                        }`}
                        title={turno ? `${turnoLabels[turno]}${isOverride ? ' (editado)' : ''}` : 'Descanso'}
                      >
                        {turno || '—'}
                      </button>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
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
    nuevo: 'bg-blue-500/10 text-blue-400',
    actualizado: 'bg-purple-500/10 text-purple-400',
    sin_cambio: 'bg-slate-500/10 text-slate-400',
  }
  const labels = { nuevo: 'Nuevo', actualizado: 'Actualizado', sin_cambio: 'Sin cambio' }
  return (
    <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${styles[action] || 'text-slate-500'}`}>
      {labels[action] || action}
    </span>
  )
}
