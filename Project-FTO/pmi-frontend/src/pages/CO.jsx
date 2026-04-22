import React, { useState, useEffect, useRef, useMemo } from 'react'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import UploadBanner from '../components/UploadBanner'
import * as api from '../lib/api'

const TABS = ['Dashboard', 'Registros', 'Completitud', 'Nuevo Registro']
const COLORS = ['#06b6d4', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#10b981', '#ec4899', '#f97316']
const MACHINE_COLORS = { 'KDF 7': '#06b6d4', 'KDF 8': '#3b82f6', 'KDF 9': '#8b5cf6', 'KDF 10': '#f59e0b', 'KDF 11': '#ef4444', 'KDF 17': '#10b981' }

export default function CO() {
  const [tab, setTab] = useState('Dashboard')
  const [records, setRecords] = useState([])
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [banner, setBanner] = useState(null)

  // Filters
  const [filterMachine, setFilterMachine] = useState('all')
  const [filterOperator, setFilterOperator] = useState('all')
  const [filterWeek, setFilterWeek] = useState('all')
  const [filterMarcaTermina, setFilterMarcaTermina] = useState('all')
  const [filterMarcaNueva, setFilterMarcaNueva] = useState('all')

  // Completitud expanded operator
  const [expandedOp, setExpandedOp] = useState(null)

  // Seed upload
  const [seeding, setSeeding] = useState(false)
  const seedRef = useRef(null)

  // New record form
  const emptyForm = {
    maquina: 'KDF 7', fecha: '', semana: '', operador: '',
    marca_termina: '', marca_nueva: '', formatos_completados: 'SI',
    tiempo_objetivo: '', tiempo_real: '', runtime_next_2h: '',
    stops_next_2h: '', mtbf: '', variacion_co: '', desperdicio_hora2: '',
    razon_desviacion: '', recomendacion: '',
  }
  const [form, setForm] = useState({ ...emptyForm })
  const [saving, setSaving] = useState(false)

  // Edit modal
  const [editId, setEditId] = useState(null)
  const [editForm, setEditForm] = useState(null)
  const [editSaving, setEditSaving] = useState(false)

  useEffect(() => { loadData() }, [])

  async function loadData() {
    setLoading(true)
    try {
      const [recRes, statsRes] = await Promise.all([api.getCORecords(), api.getCOStats()])
      setRecords(recRes.data || [])
      setStats(statsRes)
    } catch (err) { setError(err.message) }
    finally { setLoading(false) }
  }

  async function handleSeed(e) {
    const file = e.target.files[0]
    if (!file) return
    e.target.value = ''
    let clearExisting = false
    if (records.length > 0) {
      clearExisting = confirm(`Ya hay ${records.length} registros CO. ¿Deseas reemplazarlos? (Cancelar = agregar sin borrar)`)
    }
    setSeeding(true); setError(null)
    try {
      const res = await api.seedCO(file, clearExisting)
      setBanner({ message: 'Seed completado', detail: `${res.inserted} registros importados${res.split_multi_operator ? ` (${res.split_multi_operator} multi-operador)` : ''}`, type: 'success' })
      await loadData()
    } catch (err) { setError(err.message) }
    finally { setSeeding(false) }
  }

  async function handleCreate(e) {
    e.preventDefault()
    setSaving(true); setError(null)
    try {
      const data = { ...form }
      // Convert numeric fields
      for (const k of ['tiempo_objetivo', 'tiempo_real', 'runtime_next_2h', 'mtbf', 'variacion_co', 'desperdicio_hora2']) {
        data[k] = data[k] !== '' ? parseFloat(data[k]) : null
      }
      data.stops_next_2h = data.stops_next_2h !== '' ? parseInt(data.stops_next_2h) : null
      await api.createCORecord(data)
      setBanner({ message: 'Registro creado', detail: `${data.operador} — ${data.maquina}`, type: 'success' })
      setForm({ ...emptyForm })
      await loadData()
    } catch (err) { setError(err.message) }
    finally { setSaving(false) }
  }

  async function handleDelete(id) {
    if (!confirm('¿Eliminar este registro CO?')) return
    try {
      await api.deleteCORecord(id)
      await loadData()
    } catch (err) { setError(err.message) }
  }

  function openEdit(rec) {
    setEditId(rec.id)
    setEditForm({
      maquina: rec.maquina || '', fecha: rec.fecha || '', semana: rec.semana || '',
      operador: rec.operador || '', marca_termina: rec.marca_termina || '',
      marca_nueva: rec.marca_nueva || '', tiempo_objetivo: rec.tiempo_objetivo ?? '',
      tiempo_real: rec.tiempo_real ?? '', razon_desviacion: rec.razon_desviacion || '',
      recomendacion: rec.recomendacion || '', stops_next_2h: rec.stops_next_2h ?? '',
      mtbf: rec.mtbf ?? '', variacion_co: rec.variacion_co ?? '',
      desperdicio_hora2: rec.desperdicio_hora2 ?? '', runtime_next_2h: rec.runtime_next_2h ?? '',
    })
  }

  async function handleEditSave() {
    setEditSaving(true); setError(null)
    try {
      const data = { ...editForm }
      for (const k of ['tiempo_objetivo', 'tiempo_real', 'runtime_next_2h', 'mtbf', 'variacion_co', 'desperdicio_hora2']) {
        data[k] = data[k] !== '' && data[k] !== null ? parseFloat(data[k]) : null
      }
      data.stops_next_2h = data.stops_next_2h !== '' && data.stops_next_2h !== null ? parseInt(data.stops_next_2h) : null
      await api.updateCORecord(editId, data)
      setEditId(null)
      setBanner({ message: 'Registro actualizado', type: 'success' })
      await loadData()
    } catch (err) { setError(err.message) }
    finally { setEditSaving(false) }
  }

  // Derived data
  const machines = useMemo(() => [...new Set(records.map(r => r.maquina).filter(Boolean))].sort(), [records])
  const operators = useMemo(() => [...new Set(records.map(r => r.operador).filter(Boolean))].sort(), [records])
  const weeks = useMemo(() => [...new Set(records.map(r => r.semana).filter(Boolean))].sort((a, b) => parseInt(a) - parseInt(b)), [records])
  const marcasTermina = useMemo(() => stats?.all_marcas_termina || [...new Set(records.map(r => r.marca_termina).filter(Boolean))].sort(), [records, stats])
  const marcasNueva = useMemo(() => stats?.all_marcas_nueva || [...new Set(records.map(r => r.marca_nueva).filter(Boolean))].sort(), [records, stats])

  const filtered = useMemo(() => {
    return records.filter(r => {
      if (filterMachine !== 'all' && r.maquina !== filterMachine) return false
      if (filterOperator !== 'all' && r.operador !== filterOperator) return false
      if (filterWeek !== 'all' && r.semana !== filterWeek) return false
      if (filterMarcaTermina !== 'all' && r.marca_termina !== filterMarcaTermina) return false
      if (filterMarcaNueva !== 'all' && r.marca_nueva !== filterMarcaNueva) return false
      return true
    })
  }, [records, filterMachine, filterOperator, filterWeek, filterMarcaTermina, filterMarcaNueva])

  // Chart data
  const weeklyChart = useMemo(() => {
    const map = {}
    for (const r of filtered) {
      const w = r.semana || '?'
      if (!map[w]) map[w] = { semana: `W${w}`, count: 0, avg_var: 0, sum_var: 0 }
      map[w].count++
      if (r.variacion_co != null) { map[w].sum_var += r.variacion_co }
    }
    for (const w of Object.values(map)) {
      w.avg_var = w.count > 0 ? Math.round((w.sum_var / w.count) * 100) : 0
    }
    return Object.values(map).sort((a, b) => parseInt(a.semana.slice(1)) - parseInt(b.semana.slice(1)))
  }, [filtered])

  const machineChart = useMemo(() => {
    const map = {}
    for (const r of filtered) {
      const m = r.maquina || '?'
      if (!map[m]) map[m] = { name: m, count: 0, avg_tiempo: 0, sum_t: 0, n_t: 0 }
      map[m].count++
      if (r.tiempo_real != null) { map[m].sum_t += r.tiempo_real; map[m].n_t++ }
    }
    for (const m of Object.values(map)) {
      m.avg_tiempo = m.n_t > 0 ? Math.round(m.sum_t / m.n_t) : 0
    }
    return Object.values(map).sort((a, b) => b.count - a.count)
  }, [filtered])

  const operatorChart = useMemo(() => {
    const map = {}
    for (const r of filtered) {
      const op = r.operador || '?'
      if (!map[op]) map[op] = { name: op, count: 0, avg_var: 0, sum_var: 0, n_var: 0 }
      map[op].count++
      if (r.variacion_co != null) { map[op].sum_var += r.variacion_co; map[op].n_var++ }
    }
    for (const o of Object.values(map)) {
      o.avg_var = o.n_var > 0 ? Math.round((o.sum_var / o.n_var) * 100) : 0
    }
    return Object.values(map).sort((a, b) => b.count - a.count)
  }, [filtered])

  if (loading) return (
    <div className="flex items-center justify-center h-64">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500" />
    </div>
  )

  return (
    <div className="p-6 space-y-6 min-h-screen bg-[#0a1628]">
      <UploadBanner show={!!banner} onClose={() => setBanner(null)}
        message={banner?.message || ''} detail={banner?.detail} type={banner?.type} />

      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white">Análisis CO — Changeover</h1>
          <p className="text-sm text-slate-400">
            Análisis de cambios de marca en Filtros Cápsula. Tiempos, desviaciones, MTBF y completitud por operador.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <input ref={seedRef} type="file" accept=".xlsx,.xls" onChange={handleSeed} className="hidden" />
          <button onClick={() => seedRef.current?.click()} disabled={seeding}
            className="px-3 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 text-white text-sm rounded-lg transition-colors">
            {seeding ? 'Importando...' : '📤 Importar Excel'}
          </button>
          <button onClick={() => api.exportCOExcel(filterMachine !== 'all' ? filterMachine : null, filterOperator !== 'all' ? filterOperator : null)}
            className="px-3 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm rounded-lg transition-colors">
            📥 Exportar Excel
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-red-400 text-sm flex items-center gap-2">
          <span className="flex-1">{error}</span>
          <button onClick={() => setError(null)} className="text-red-300 hover:text-white">✕</button>
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 bg-[#0f1d32] rounded-lg p-1 w-fit">
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-4 py-1.5 text-sm rounded-md transition-colors ${tab === t ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}>
            {t}
          </button>
        ))}
      </div>

      {/* Filters (shown on Dashboard and Registros) */}
      {(tab === 'Dashboard' || tab === 'Registros') && (
        <div className="flex flex-wrap gap-3">
          <select value={filterMachine} onChange={e => setFilterMachine(e.target.value)}
            className="px-3 py-2 bg-[#0f1d32] border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500">
            <option value="all">Todas las máquinas</option>
            {machines.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
          <select value={filterOperator} onChange={e => setFilterOperator(e.target.value)}
            className="px-3 py-2 bg-[#0f1d32] border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500">
            <option value="all">Todos los operadores</option>
            {operators.map(o => <option key={o} value={o}>{o}</option>)}
          </select>
          <select value={filterWeek} onChange={e => setFilterWeek(e.target.value)}
            className="px-3 py-2 bg-[#0f1d32] border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500">
            <option value="all">Todas las semanas</option>
            {weeks.map(w => <option key={w} value={w}>W{w}</option>)}
          </select>
          <select value={filterMarcaTermina} onChange={e => setFilterMarcaTermina(e.target.value)}
            className="px-3 py-2 bg-[#0f1d32] border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500">
            <option value="all">Marca origen (todas)</option>
            {marcasTermina.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
          <select value={filterMarcaNueva} onChange={e => setFilterMarcaNueva(e.target.value)}
            className="px-3 py-2 bg-[#0f1d32] border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500">
            <option value="all">Marca destino (todas)</option>
            {marcasNueva.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
      )}

      {/* ═══════ DASHBOARD TAB ═══════ */}
      {tab === 'Dashboard' && stats && (
        <div className="space-y-6">
          {/* Summary cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
            <StatCard label="Total COs" value={filtered.length} color="text-cyan-400" />
            <StatCard label="Operadores" value={stats.total_operators} />
            <StatCard label="Máquinas" value={stats.total_machines} />
            <StatCard label="Semanas" value={stats.weeks_range?.length || 0} />
            <StatCard label="Variación Prom." value={`${Math.round((stats.avg_variacion || 0) * 100)}%`}
              color={stats.avg_variacion > 0 ? 'text-red-400' : 'text-green-400'} />
            <StatCard label="MTBF Prom." value={stats.avg_mtbf?.toFixed(1) || '—'} color="text-blue-400" />
          </div>

          {/* Charts row 1: Weekly + Machine */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* COs by Week */}
            <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-4">
              <h3 className="text-sm font-semibold text-white mb-3">COs por Semana</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={weeklyChart}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                  <XAxis dataKey="semana" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: '#0f1d32', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#fff' }} />
                  <Bar dataKey="count" fill="#06b6d4" name="COs" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* COs by Machine */}
            <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-4">
              <h3 className="text-sm font-semibold text-white mb-3">COs por Máquina</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={machineChart} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                  <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} width={60} />
                  <Tooltip contentStyle={{ background: '#0f1d32', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#fff' }} />
                  <Bar dataKey="count" name="COs" radius={[0, 4, 4, 0]}>
                    {machineChart.map((entry, i) => (
                      <Cell key={i} fill={MACHINE_COLORS[entry.name] || COLORS[i % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Charts row 2: Variación trend + Operator comparison */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Time variation trend */}
            <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-4">
              <h3 className="text-sm font-semibold text-white mb-3">Variación de CO por Semana (%)</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={weeklyChart}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                  <XAxis dataKey="semana" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} unit="%" />
                  <Tooltip contentStyle={{ background: '#0f1d32', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#fff' }}
                    formatter={(v) => [`${v}%`, 'Variación']} />
                  <Line type="monotone" dataKey="avg_var" stroke="#f59e0b" strokeWidth={2} dot={{ r: 4, fill: '#f59e0b' }} name="Variación %" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Operator COs */}
            <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-4">
              <h3 className="text-sm font-semibold text-white mb-3">COs por Operador</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={operatorChart.slice(0, 10)} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                  <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                  <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} width={80} />
                  <Tooltip contentStyle={{ background: '#0f1d32', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#fff' }} />
                  <Bar dataKey="count" fill="#8b5cf6" name="COs" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Brand pairs */}
          {stats.brand_pairs?.length > 0 && (
            <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-4">
              <h3 className="text-sm font-semibold text-white mb-3">Cambios de Marca más frecuentes</h3>
              <div className="flex flex-wrap gap-2">
                {stats.brand_pairs.map((bp, i) => (
                  <span key={i} className="text-xs bg-cyan-950/40 text-cyan-300 px-3 py-1.5 rounded-lg">
                    {bp.pair} <span className="text-cyan-500/70 ml-1">({bp.count})</span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ═══════ REGISTROS TAB ═══════ */}
      {tab === 'Registros' && (
        <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
          <div className="px-4 py-2 border-b border-white/5 flex items-center justify-between">
            <span className="text-xs text-slate-400">{filtered.length} registros</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-white/5 text-slate-500">
                  {['Fecha', 'Sem', 'Máquina', 'Operador', 'Marca Sale', 'Marca Entra', 'T.Obj', 'T.Real', 'Var%', 'MTBF', 'Stops', 'CF', 'Acciones'].map(h => (
                    <th key={h} className="px-2 py-2 text-left font-medium whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map(r => (
                  <tr key={r.id} className="border-b border-white/5 hover:bg-white/[0.02]">
                    <td className="px-2 py-1.5 text-slate-300">{r.fecha || '—'}</td>
                    <td className="px-2 py-1.5 text-slate-400">W{r.semana || '?'}</td>
                    <td className="px-2 py-1.5">
                      <span className="text-xs px-1.5 py-0.5 rounded bg-cyan-950/40 text-cyan-300">{r.maquina}</span>
                    </td>
                    <td className="px-2 py-1.5 text-white">{r.operador}</td>
                    <td className="px-2 py-1.5 text-slate-400">{r.marca_termina || '—'}</td>
                    <td className="px-2 py-1.5 text-slate-400">{r.marca_nueva || '—'}</td>
                    <td className="px-2 py-1.5 text-slate-300">{r.tiempo_objetivo != null ? r.tiempo_objetivo : '—'}</td>
                    <td className="px-2 py-1.5 text-white font-medium">{r.tiempo_real != null ? r.tiempo_real : '—'}</td>
                    <td className="px-2 py-1.5">
                      {r.variacion_co != null ? (
                        <span className={r.variacion_co > 0 ? 'text-red-400' : 'text-green-400'}>
                          {Math.round(r.variacion_co * 100)}%
                        </span>
                      ) : '—'}
                    </td>
                    <td className="px-2 py-1.5 text-blue-400">{r.mtbf != null ? r.mtbf.toFixed(1) : '—'}</td>
                    <td className="px-2 py-1.5 text-slate-300">{r.stops_next_2h ?? '—'}</td>
                    <td className="px-2 py-1.5">
                      {r.analisis_cf != null ? (
                        <span className={`text-xs px-1.5 py-0.5 rounded ${r.analisis_cf >= 1 ? 'bg-green-950/40 text-green-400' : r.analisis_cf > 0 ? 'bg-amber-950/40 text-amber-400' : 'bg-red-950/40 text-red-400'}`}>
                          {Math.round(r.analisis_cf * 100)}%
                        </span>
                      ) : '—'}
                    </td>
                    <td className="px-2 py-1.5 flex gap-1">
                      <button onClick={() => openEdit(r)} className="text-blue-400 hover:text-blue-300 text-xs">✎</button>
                      <button onClick={() => handleDelete(r.id)} className="text-red-400 hover:text-red-300 text-xs">✕</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ═══════ COMPLETITUD TAB ═══════ */}
      {tab === 'Completitud' && stats?.completeness && (
        <div className="space-y-4">
          <p className="text-sm text-slate-400">
            Análisis CF: 100% = Razón de desviación + Recomendación completas. Desperdicio = promedio de "Desperdicio en la hora 2" (%).
            Haz clic en un operador para ver sus detalles.
          </p>
          <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-white/5 text-slate-500">
                  <th className="px-3 py-2 text-left font-medium w-5"></th>
                  <th className="px-3 py-2 text-left font-medium">Operador</th>
                  <th className="px-3 py-2 text-center font-medium">Total COs</th>
                  <th className="px-3 py-2 text-center font-medium">Razón Desv.</th>
                  <th className="px-3 py-2 text-center font-medium">Recomendación</th>
                  <th className="px-3 py-2 text-center font-medium">% Desperdicio Prom.</th>
                  <th className="px-3 py-2 text-center font-medium">Completitud Global</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(stats.completeness)
                  .sort((a, b) => b[1].total - a[1].total)
                  .map(([op, c]) => {
                    const globalPct = Math.round(((c.pct_razon + c.pct_recom + c.pct_desperdicio) / 3))
                    const isExpanded = expandedOp === op
                    return (
                      <React.Fragment key={op}>
                        <tr className={`border-b border-white/5 hover:bg-white/[0.02] cursor-pointer ${isExpanded ? 'bg-white/[0.03]' : ''}`}
                          onClick={() => setExpandedOp(isExpanded ? null : op)}>
                          <td className="px-3 py-2 text-slate-500">
                            <span className={`inline-block transition-transform text-[10px] ${isExpanded ? 'rotate-90' : ''}`}>▶</span>
                          </td>
                          <td className="px-3 py-2 text-white font-medium">{op}</td>
                          <td className="px-3 py-2 text-center text-cyan-400 font-semibold">{c.total}</td>
                          <td className="px-3 py-2 text-center">
                            <CompletionBadge pct={c.pct_razon} filled={c.filled_razon} total={c.total} />
                          </td>
                          <td className="px-3 py-2 text-center">
                            <CompletionBadge pct={c.pct_recom} filled={c.filled_recom} total={c.total} />
                          </td>
                          <td className="px-3 py-2 text-center">
                            {c.avg_desperdicio != null ? (
                              <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                                c.avg_desperdicio <= 2 ? 'bg-green-950/40 text-green-400' :
                                c.avg_desperdicio <= 5 ? 'bg-amber-950/40 text-amber-400' :
                                'bg-red-950/40 text-red-400'
                              }`}>{c.avg_desperdicio}%</span>
                            ) : (
                              <span className="text-slate-600 text-[10px]">Sin datos</span>
                            )}
                          </td>
                          <td className="px-3 py-2 text-center">
                            <div className="flex items-center justify-center gap-2">
                              <div className="w-20 h-2 bg-slate-800 rounded-full overflow-hidden">
                                <div className={`h-full rounded-full ${globalPct >= 80 ? 'bg-green-500' : globalPct >= 50 ? 'bg-amber-500' : 'bg-red-500'}`}
                                  style={{ width: `${globalPct}%` }} />
                              </div>
                              <span className="text-slate-300 text-[11px]">{globalPct}%</span>
                            </div>
                          </td>
                        </tr>
                        {/* Expanded detail rows */}
                        {isExpanded && c.details && (
                          <tr>
                            <td colSpan={7} className="px-0 py-0">
                              <div className="bg-[#0a1628] border-y border-white/5 px-6 py-3">
                                <div className="flex items-center justify-between mb-3">
                                  <p className="text-xs text-slate-400 font-medium">Detalle de COs de {op}</p>
                                  <span className="text-[10px] text-slate-500">{c.details.length} registros</span>
                                </div>
                                <div className="overflow-x-auto max-h-[300px]">
                                  <table className="w-full text-[11px]">
                                    <thead>
                                      <tr className="text-slate-600 border-b border-white/5">
                                        <th className="px-2 py-1.5 text-left font-medium">Fecha</th>
                                        <th className="px-2 py-1.5 text-left font-medium">Máquina</th>
                                        <th className="px-2 py-1.5 text-left font-medium">Marca Sale → Entra</th>
                                        <th className="px-2 py-1.5 text-left font-medium">Razón de Desviación</th>
                                        <th className="px-2 py-1.5 text-left font-medium">Recomendación</th>
                                        <th className="px-2 py-1.5 text-center font-medium">Desp. H2</th>
                                        <th className="px-2 py-1.5 text-center font-medium">Var%</th>
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {c.details.map((d, di) => (
                                        <tr key={di} className="border-b border-white/[0.03] hover:bg-white/[0.02]">
                                          <td className="px-2 py-1.5 text-slate-400 whitespace-nowrap">{d.fecha || '—'}</td>
                                          <td className="px-2 py-1.5">
                                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-950/40 text-cyan-300">{d.maquina}</span>
                                          </td>
                                          <td className="px-2 py-1.5 text-slate-300 whitespace-nowrap">{d.marca_termina || '?'} → {d.marca_nueva || '?'}</td>
                                          <td className="px-2 py-1.5 text-slate-300 max-w-[200px] truncate" title={d.razon_desviacion || ''}>
                                            {d.razon_desviacion || <span className="text-red-400/50">—</span>}
                                          </td>
                                          <td className="px-2 py-1.5 text-slate-300 max-w-[200px] truncate" title={d.recomendacion || ''}>
                                            {d.recomendacion || <span className="text-red-400/50">—</span>}
                                          </td>
                                          <td className="px-2 py-1.5 text-center">
                                            {d.desperdicio_hora2 != null ? (
                                              <span className={d.desperdicio_hora2 <= 2 ? 'text-green-400' : d.desperdicio_hora2 <= 5 ? 'text-amber-400' : 'text-red-400'}>
                                                {d.desperdicio_hora2}%
                                              </span>
                                            ) : <span className="text-slate-600">—</span>}
                                          </td>
                                          <td className="px-2 py-1.5 text-center">
                                            {d.variacion_co != null ? (
                                              <span className={d.variacion_co > 0 ? 'text-red-400' : 'text-green-400'}>
                                                {Math.round(d.variacion_co * 100)}%
                                              </span>
                                            ) : '—'}
                                          </td>
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    )
                  })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ═══════ NUEVO REGISTRO TAB ═══════ */}
      {tab === 'Nuevo Registro' && (
        <form onSubmit={handleCreate} className="bg-[#0f1d32] rounded-xl border border-white/5 p-6 space-y-4 max-w-3xl">
          <h2 className="text-base font-semibold text-white">Nuevo Registro CO</h2>

          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <FormField label="Máquina" required>
              <select value={form.maquina} onChange={e => setForm({ ...form, maquina: e.target.value })} className="form-input">
                {['KDF 7', 'KDF 8', 'KDF 9', 'KDF 10', 'KDF 11', 'KDF 17'].map(m => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </FormField>
            <FormField label="Fecha" required>
              <input type="date" value={form.fecha} onChange={e => setForm({ ...form, fecha: e.target.value })} className="form-input" required />
            </FormField>
            <FormField label="Semana">
              <input type="number" value={form.semana} onChange={e => setForm({ ...form, semana: e.target.value })} className="form-input" placeholder="16" />
            </FormField>
            <FormField label="Operador" required>
              <input value={form.operador} onChange={e => setForm({ ...form, operador: e.target.value })} className="form-input" required placeholder="Nombre" />
            </FormField>
            <FormField label="Marca que termina">
              <input value={form.marca_termina} onChange={e => setForm({ ...form, marca_termina: e.target.value })} className="form-input" />
            </FormField>
            <FormField label="Marca nueva">
              <input value={form.marca_nueva} onChange={e => setForm({ ...form, marca_nueva: e.target.value })} className="form-input" />
            </FormField>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <FormField label="T. Objetivo (min)">
              <input type="number" step="0.1" value={form.tiempo_objetivo} onChange={e => setForm({ ...form, tiempo_objetivo: e.target.value })} className="form-input" />
            </FormField>
            <FormField label="T. Real (min)">
              <input type="number" step="0.1" value={form.tiempo_real} onChange={e => setForm({ ...form, tiempo_real: e.target.value })} className="form-input" />
            </FormField>
            <FormField label="Runtime 2h">
              <input type="number" step="0.01" value={form.runtime_next_2h} onChange={e => setForm({ ...form, runtime_next_2h: e.target.value })} className="form-input" />
            </FormField>
            <FormField label="Stops 2h">
              <input type="number" value={form.stops_next_2h} onChange={e => setForm({ ...form, stops_next_2h: e.target.value })} className="form-input" />
            </FormField>
            <FormField label="MTBF">
              <input type="number" step="0.01" value={form.mtbf} onChange={e => setForm({ ...form, mtbf: e.target.value })} className="form-input" />
            </FormField>
            <FormField label="Variación CO">
              <input type="number" step="0.01" value={form.variacion_co} onChange={e => setForm({ ...form, variacion_co: e.target.value })} className="form-input" placeholder="0.15 = 15%" />
            </FormField>
            <FormField label="Desperdicio H2">
              <input type="number" step="0.01" value={form.desperdicio_hora2} onChange={e => setForm({ ...form, desperdicio_hora2: e.target.value })} className="form-input" />
            </FormField>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <FormField label="Razón de desviación">
              <textarea value={form.razon_desviacion} onChange={e => setForm({ ...form, razon_desviacion: e.target.value })}
                className="form-input min-h-[60px]" rows={2} />
            </FormField>
            <FormField label="Recomendación de mejoras">
              <textarea value={form.recomendacion} onChange={e => setForm({ ...form, recomendacion: e.target.value })}
                className="form-input min-h-[60px]" rows={2} />
            </FormField>
          </div>

          <div className="flex justify-end">
            <button type="submit" disabled={saving}
              className="px-5 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 text-white text-sm rounded-lg transition-colors">
              {saving ? 'Guardando...' : '💾 Crear Registro'}
            </button>
          </div>
        </form>
      )}

      {/* ═══════ EDIT MODAL ═══════ */}
      {editId && editForm && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4" onClick={() => setEditId(null)}>
          <div className="bg-[#0f1d32] rounded-xl border border-white/10 p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto space-y-4"
            onClick={e => e.stopPropagation()}>
            <h2 className="text-base font-semibold text-white">Editar Registro CO</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              <FormField label="Máquina">
                <select value={editForm.maquina} onChange={e => setEditForm({ ...editForm, maquina: e.target.value })} className="form-input">
                  {['KDF 7', 'KDF 8', 'KDF 9', 'KDF 10', 'KDF 11', 'KDF 17'].map(m => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </FormField>
              <FormField label="Fecha">
                <input type="date" value={editForm.fecha} onChange={e => setEditForm({ ...editForm, fecha: e.target.value })} className="form-input" />
              </FormField>
              <FormField label="Semana">
                <input value={editForm.semana} onChange={e => setEditForm({ ...editForm, semana: e.target.value })} className="form-input" />
              </FormField>
              <FormField label="Operador">
                <input value={editForm.operador} onChange={e => setEditForm({ ...editForm, operador: e.target.value })} className="form-input" />
              </FormField>
              <FormField label="T. Objetivo">
                <input type="number" step="0.1" value={editForm.tiempo_objetivo} onChange={e => setEditForm({ ...editForm, tiempo_objetivo: e.target.value })} className="form-input" />
              </FormField>
              <FormField label="T. Real">
                <input type="number" step="0.1" value={editForm.tiempo_real} onChange={e => setEditForm({ ...editForm, tiempo_real: e.target.value })} className="form-input" />
              </FormField>
              <FormField label="Stops 2h">
                <input type="number" value={editForm.stops_next_2h} onChange={e => setEditForm({ ...editForm, stops_next_2h: e.target.value })} className="form-input" />
              </FormField>
              <FormField label="MTBF">
                <input type="number" step="0.01" value={editForm.mtbf} onChange={e => setEditForm({ ...editForm, mtbf: e.target.value })} className="form-input" />
              </FormField>
              <FormField label="Variación CO">
                <input type="number" step="0.01" value={editForm.variacion_co} onChange={e => setEditForm({ ...editForm, variacion_co: e.target.value })} className="form-input" />
              </FormField>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <FormField label="Razón de desviación">
                <textarea value={editForm.razon_desviacion} onChange={e => setEditForm({ ...editForm, razon_desviacion: e.target.value })}
                  className="form-input min-h-[60px]" rows={2} />
              </FormField>
              <FormField label="Recomendación">
                <textarea value={editForm.recomendacion} onChange={e => setEditForm({ ...editForm, recomendacion: e.target.value })}
                  className="form-input min-h-[60px]" rows={2} />
              </FormField>
            </div>
            <div className="flex justify-end gap-2">
              <button onClick={() => setEditId(null)} className="px-4 py-2 text-slate-400 hover:text-white text-sm">Cancelar</button>
              <button onClick={handleEditSave} disabled={editSaving}
                className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 text-white text-sm rounded-lg">
                {editSaving ? 'Guardando...' : 'Guardar'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}


// ═══════ Helper components ═══════

function StatCard({ label, value, color = 'text-white' }) {
  return (
    <div className="bg-[#0f1d32] rounded-lg border border-white/5 px-4 py-3">
      <p className="text-[11px] text-slate-500 uppercase tracking-wide">{label}</p>
      <p className={`text-2xl font-bold ${color}`}>{value ?? 0}</p>
    </div>
  )
}

function CompletionBadge({ pct, filled, total }) {
  const color = pct >= 80 ? 'text-green-400 bg-green-950/40' : pct >= 50 ? 'text-amber-400 bg-amber-950/40' : 'text-red-400 bg-red-950/40'
  return (
    <span className={`text-xs px-2 py-0.5 rounded ${color}`}>
      {filled}/{total} ({pct}%)
    </span>
  )
}

function FormField({ label, required, children }) {
  return (
    <div>
      <label className="block text-xs text-slate-400 mb-1">
        {label} {required && <span className="text-red-400">*</span>}
      </label>
      {children}
    </div>
  )
}
