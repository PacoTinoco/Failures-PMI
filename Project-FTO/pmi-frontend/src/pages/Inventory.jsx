import { useState, useMemo, useRef } from 'react'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Cell, ScatterChart, Scatter, ZAxis
} from 'recharts'
import UploadBanner from '../components/UploadBanner'
import * as api from '../lib/api'

const COLORS = ['#06b6d4', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#10b981', '#ec4899', '#f97316']

export default function Inventory() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [banner, setBanner] = useState(null)
  const fileRef = useRef(null)

  // Filters
  const [filterContainer, setFilterContainer] = useState('all')
  const [filterDest, setFilterDest] = useState('')
  const [filterUser, setFilterUser] = useState('all')
  const [filterTx, setFilterTx] = useState('all')
  const [showAnomaliesOnly, setShowAnomaliesOnly] = useState(false)
  const [threshold, setThreshold] = useState(50)

  // Detail modal
  const [selectedMov, setSelectedMov] = useState(null)

  // Tab
  const [tab, setTab] = useState('timeline')

  async function handleUpload(e) {
    const file = e.target.files[0]
    if (!file) return
    e.target.value = ''
    setLoading(true); setError(null); setData(null)
    try {
      const result = await api.analyzeInventory(file, threshold)
      setData(result)
      setBanner({ message: 'Archivo analizado', detail: `${result.total_movements} movimientos, ${result.total_containers} contenedores, ${result.total_anomalies} anomalías`, type: 'success' })
      // Auto-select first container
      if (result.filters.containers.length > 0) {
        setFilterContainer(result.filters.containers[0])
      }
    } catch (err) { setError(err.message) }
    finally { setLoading(false) }
  }

  // Filtered movements
  const filtered = useMemo(() => {
    if (!data) return []
    return data.movements.filter(m => {
      if (filterContainer !== 'all' && m.container !== filterContainer) return false
      if (filterDest && m.dest_loc && !m.dest_loc.toLowerCase().includes(filterDest.toLowerCase())) return false
      if (filterUser !== 'all' && m.user !== filterUser) return false
      if (filterTx !== 'all' && m.tx_context !== filterTx) return false
      if (showAnomaliesOnly && !m.is_anomaly) return false
      return true
    })
  }, [data, filterContainer, filterDest, filterUser, filterTx, showAnomaliesOnly])

  // Timeline chart data for selected container
  const timelineData = useMemo(() => {
    if (!filtered.length) return []
    return filtered.map((m, i) => ({
      index: i + 1,
      old_qty: m.old_qty,
      new_qty: m.new_qty,
      qty_diff: m.qty_diff != null ? Math.round(m.qty_diff * 1000) / 1000 : 0,
      user: m.user,
      date: m.date ? new Date(m.date).toLocaleString('es-MX', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '?',
      tx_context: m.tx_context,
      is_anomaly: m.is_anomaly,
      is_anomaly_user: m.is_anomaly_user,
      is_anomaly_qty: m.is_anomaly_qty,
      excel_row: m.excel_row,
      fill: m.is_anomaly_qty ? '#ef4444' : m.is_anomaly_user ? '#f59e0b' : '#06b6d4',
    }))
  }, [filtered])

  // Container summary for selected
  const selectedSummary = useMemo(() => {
    if (!data || filterContainer === 'all') return null
    return data.container_summaries.find(s => s.container === filterContainer)
  }, [data, filterContainer])

  if (!data && !loading) {
    return (
      <div className="p-6 space-y-6 min-h-screen bg-[#0a1628]">
        <div>
          <h1 className="text-xl font-bold text-white">Inventory Movements</h1>
          <p className="text-sm text-slate-400 mt-1">Sube un archivo Excel de movimientos de inventario para analizar contenedores, detectar anomalías y rastrear cambios de cantidad.</p>
        </div>
        <div className="bg-[#0f1d32] rounded-xl border-2 border-dashed border-white/10 p-12 flex flex-col items-center gap-4 hover:border-cyan-500/30 transition-colors">
          <div className="text-4xl">📦</div>
          <p className="text-slate-300 text-sm">Arrastra o selecciona tu archivo Excel de Inventory Movements</p>
          <input ref={fileRef} type="file" accept=".xlsx,.xls" onChange={handleUpload} className="hidden" />
          <button onClick={() => fileRef.current?.click()}
            className="px-5 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-white text-sm rounded-lg transition-colors">
            📤 Seleccionar archivo
          </button>
          {error && <p className="text-red-400 text-sm">{error}</p>}
        </div>
      </div>
    )
  }

  if (loading) return (
    <div className="flex items-center justify-center h-64 bg-[#0a1628]">
      <div className="flex flex-col items-center gap-3">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500" />
        <p className="text-slate-400 text-sm">Analizando movimientos...</p>
      </div>
    </div>
  )

  return (
    <div className="p-6 space-y-5 min-h-screen bg-[#0a1628]">
      <UploadBanner show={!!banner} onClose={() => setBanner(null)}
        message={banner?.message || ''} detail={banner?.detail} type={banner?.type} />

      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-white">Inventory Movements</h1>
          <p className="text-xs text-slate-400">{data.total_movements} movimientos · {data.total_containers} contenedores · {data.total_anomalies} anomalías detectadas</p>
        </div>
        <div className="flex items-center gap-2">
          <input ref={fileRef} type="file" accept=".xlsx,.xls" onChange={handleUpload} className="hidden" />
          <button onClick={() => fileRef.current?.click()}
            className="px-3 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-sm rounded-lg transition-colors">
            📤 Nuevo archivo
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-red-400 text-sm flex items-center gap-2">
          <span className="flex-1">{error}</span>
          <button onClick={() => setError(null)} className="text-red-300 hover:text-white">✕</button>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap gap-3 items-end">
        <div>
          <label className="block text-[10px] text-slate-500 mb-1 uppercase tracking-wide">Contenedor</label>
          <select value={filterContainer} onChange={e => setFilterContainer(e.target.value)}
            className="form-input min-w-[220px] text-xs">
            <option value="all">Todos los contenedores</option>
            {data.filters.containers.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-[10px] text-slate-500 mb-1 uppercase tracking-wide">Máquina / Dest Loc</label>
          <input value={filterDest} onChange={e => setFilterDest(e.target.value)}
            placeholder="Escribe o selecciona..."
            list="dest-loc-list"
            className="form-input min-w-[160px] text-xs" />
          <datalist id="dest-loc-list">
            {data.filters.dest_locs.map(d => <option key={d} value={d} />)}
          </datalist>
        </div>
        <div>
          <label className="block text-[10px] text-slate-500 mb-1 uppercase tracking-wide">Usuario</label>
          <select value={filterUser} onChange={e => setFilterUser(e.target.value)}
            className="form-input min-w-[120px] text-xs">
            <option value="all">Todos</option>
            {data.filters.users.map(u => <option key={u} value={u}>{u}</option>)}
          </select>
        </div>
        <div>
          <label className="block text-[10px] text-slate-500 mb-1 uppercase tracking-wide">Transaction</label>
          <select value={filterTx} onChange={e => setFilterTx(e.target.value)}
            className="form-input min-w-[180px] text-xs">
            <option value="all">Todos</option>
            {data.filters.tx_contexts.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>
        <label className="flex items-center gap-2 cursor-pointer pb-1">
          <input type="checkbox" checked={showAnomaliesOnly} onChange={e => setShowAnomaliesOnly(e.target.checked)}
            className="rounded border-white/20 bg-[#0a1628] text-cyan-500 focus:ring-cyan-500" />
          <span className="text-xs text-slate-300">Solo anomalías</span>
        </label>
      </div>

      {/* Container summary card */}
      {selectedSummary && filterContainer !== 'all' && (
        <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-4">
          <div className="flex flex-wrap gap-6 items-center">
            <div>
              <p className="text-[10px] text-slate-500 uppercase">Contenedor</p>
              <p className="text-sm text-white font-mono font-medium">{selectedSummary.container}</p>
            </div>
            <div>
              <p className="text-[10px] text-slate-500 uppercase">Qty Inicial</p>
              <p className="text-lg text-cyan-400 font-bold">{selectedSummary.initial_qty}</p>
            </div>
            <div className="text-slate-600 text-lg">→</div>
            <div>
              <p className="text-[10px] text-slate-500 uppercase">Qty Final</p>
              <p className="text-lg text-white font-bold">{selectedSummary.final_qty}</p>
            </div>
            <div>
              <p className="text-[10px] text-slate-500 uppercase">Cambio Neto</p>
              <p className={`text-lg font-bold ${selectedSummary.net_change < 0 ? 'text-red-400' : selectedSummary.net_change > 0 ? 'text-green-400' : 'text-slate-400'}`}>
                {selectedSummary.net_change > 0 ? '+' : ''}{selectedSummary.net_change}
              </p>
            </div>
            <div>
              <p className="text-[10px] text-slate-500 uppercase">Movimientos</p>
              <p className="text-lg text-white font-bold">{selectedSummary.total_moves}</p>
            </div>
            <div>
              <p className="text-[10px] text-slate-500 uppercase">Anomalías</p>
              <p className={`text-lg font-bold ${selectedSummary.anomaly_count > 0 ? 'text-red-400' : 'text-green-400'}`}>{selectedSummary.anomaly_count}</p>
            </div>
            <div>
              <p className="text-[10px] text-slate-500 uppercase">SAP Batch</p>
              <p className="text-xs text-slate-300">{selectedSummary.sap_batches.join(', ')}</p>
            </div>
            <div>
              <p className="text-[10px] text-slate-500 uppercase">Usuarios</p>
              <div className="flex gap-1 flex-wrap">
                {selectedSummary.users.map(u => (
                  <span key={u} className={`text-[10px] px-1.5 py-0.5 rounded ${
                    u === 'ADMIN' || u === 'System' ? 'bg-slate-800 text-slate-300' : 'bg-amber-950/50 text-amber-400'
                  }`}>{u}</span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 bg-[#0f1d32] rounded-lg p-1 w-fit">
        {['timeline', 'tabla', 'resumen'].map(t => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-4 py-1.5 text-sm rounded-md capitalize transition-colors ${tab === t ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}>
            {t === 'timeline' ? 'Timeline' : t === 'tabla' ? 'Tabla de Movimientos' : 'Resumen Contenedores'}
          </button>
        ))}
      </div>

      {/* ═══════ TIMELINE TAB ═══════ */}
      {tab === 'timeline' && (
        <div className="space-y-4">
          {/* Quantity over time */}
          <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-4">
            <h3 className="text-sm font-semibold text-white mb-1">Cantidad (New Qty) por movimiento</h3>
            <p className="text-[10px] text-slate-500 mb-3">
              <span className="inline-block w-2 h-2 rounded-full bg-cyan-500 mr-1" /> Normal
              <span className="inline-block w-2 h-2 rounded-full bg-amber-500 mr-1 ml-3" /> Usuario no-sistema
              <span className="inline-block w-2 h-2 rounded-full bg-red-500 mr-1 ml-3" /> Pico de cantidad (&gt;{threshold}kg)
            </p>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                <XAxis dataKey="index" tick={{ fill: '#94a3b8', fontSize: 10 }} label={{ value: 'Mov #', position: 'insideBottom', offset: -5, fill: '#64748b', fontSize: 10 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} label={{ value: 'Qty', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10 }} />
                <Tooltip content={<CustomTooltip />} />
                <Line type="stepAfter" dataKey="new_qty" stroke="#06b6d4" strokeWidth={2} dot={(props) => {
                  const { cx, cy, payload } = props
                  if (!cx || !cy) return null
                  const color = payload.is_anomaly_qty ? '#ef4444' : payload.is_anomaly_user ? '#f59e0b' : '#06b6d4'
                  const r = payload.is_anomaly ? 5 : 3
                  return <circle cx={cx} cy={cy} r={r} fill={color} stroke={color} strokeWidth={1} className="cursor-pointer" />
                }} activeDot={{ r: 6, stroke: '#fff', strokeWidth: 2 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Qty diff bar chart */}
          <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-4">
            <h3 className="text-sm font-semibold text-white mb-3">Diferencia de Cantidad por movimiento (New - Old)</h3>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                <XAxis dataKey="index" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <Tooltip content={<DiffTooltip />} />
                <ReferenceLine y={0} stroke="#475569" />
                <ReferenceLine y={threshold} stroke="#ef4444" strokeDasharray="5 5" label={{ value: `+${threshold}`, fill: '#ef4444', fontSize: 10 }} />
                <ReferenceLine y={-threshold} stroke="#ef4444" strokeDasharray="5 5" label={{ value: `-${threshold}`, fill: '#ef4444', fontSize: 10 }} />
                <Bar dataKey="qty_diff" name="Δ Qty" radius={[2, 2, 0, 0]}>
                  {timelineData.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ═══════ TABLA TAB ═══════ */}
      {tab === 'tabla' && (
        <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
          <div className="px-4 py-2 border-b border-white/5 flex items-center justify-between">
            <span className="text-xs text-slate-400">{filtered.length} movimientos{showAnomaliesOnly ? ' (solo anomalías)' : ''}</span>
          </div>
          <div className="overflow-x-auto max-h-[500px]">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-[#0f1d32] z-10">
                <tr className="border-b border-white/5 text-slate-500">
                  {['Fila', 'Fecha', 'Container', 'Dest Loc', 'Old Qty', 'New Qty', 'Δ Qty', 'User', 'Transaction', 'SAP Batch', ''].map(h => (
                    <th key={h} className="px-2 py-2 text-left font-medium whitespace-nowrap">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((m, i) => (
                  <tr key={i} className={`border-b border-white/5 hover:bg-white/[0.03] cursor-pointer ${
                    m.is_anomaly ? 'bg-red-500/[0.03]' : ''
                  }`} onClick={() => setSelectedMov(m)}>
                    <td className="px-2 py-1.5 text-slate-500 font-mono">{m.excel_row}</td>
                    <td className="px-2 py-1.5 text-slate-300 whitespace-nowrap">{m.date ? new Date(m.date).toLocaleString('es-MX', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—'}</td>
                    <td className="px-2 py-1.5 text-white font-mono text-[10px]">{m.container}</td>
                    <td className="px-2 py-1.5"><span className="text-xs px-1.5 py-0.5 rounded bg-cyan-950/40 text-cyan-300">{m.dest_loc}</span></td>
                    <td className="px-2 py-1.5 text-slate-400">{m.old_qty != null ? m.old_qty.toFixed(3) : '—'}</td>
                    <td className="px-2 py-1.5 text-white font-medium">{m.new_qty != null ? m.new_qty.toFixed(3) : '—'}</td>
                    <td className="px-2 py-1.5">
                      {m.qty_diff != null ? (
                        <span className={`font-medium ${m.is_anomaly_qty ? 'text-red-400 font-bold' : m.qty_diff < 0 ? 'text-orange-400' : m.qty_diff > 0 ? 'text-green-400' : 'text-slate-500'}`}>
                          {m.qty_diff > 0 ? '+' : ''}{m.qty_diff.toFixed(3)}
                        </span>
                      ) : '—'}
                    </td>
                    <td className="px-2 py-1.5">
                      <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                        m.is_anomaly_user ? 'bg-amber-950/50 text-amber-400 font-medium' : 'bg-slate-800 text-slate-400'
                      }`}>{m.user}</span>
                    </td>
                    <td className="px-2 py-1.5 text-slate-400 text-[10px]">{m.tx_context}</td>
                    <td className="px-2 py-1.5 text-slate-500 text-[10px]">{m.sap_batch}</td>
                    <td className="px-2 py-1.5">
                      {m.is_anomaly && (
                        <span className="text-[9px] px-1 py-0.5 rounded bg-red-950/50 text-red-400">
                          {m.is_anomaly_qty ? '⚡ QTY' : '👤 USER'}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ═══════ RESUMEN TAB ═══════ */}
      {tab === 'resumen' && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {data.container_summaries.map((cs, i) => (
              <div key={cs.container}
                className={`bg-[#0f1d32] rounded-xl border border-white/5 p-4 cursor-pointer hover:border-cyan-500/30 transition-colors ${
                  filterContainer === cs.container ? 'ring-1 ring-cyan-500/50' : ''
                }`}
                onClick={() => { setFilterContainer(cs.container); setTab('timeline') }}>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-[10px] text-slate-500 font-mono">{cs.container}</p>
                  {cs.anomaly_count > 0 && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-red-950/50 text-red-400">{cs.anomaly_count} anomalías</span>
                  )}
                </div>
                <div className="flex items-center gap-3 mb-3">
                  <div>
                    <p className="text-[9px] text-slate-500">Inicial</p>
                    <p className="text-base text-cyan-400 font-bold">{cs.initial_qty}</p>
                  </div>
                  <span className="text-slate-600">→</span>
                  <div>
                    <p className="text-[9px] text-slate-500">Final</p>
                    <p className="text-base text-white font-bold">{cs.final_qty}</p>
                  </div>
                  <div className="ml-auto">
                    <p className="text-[9px] text-slate-500">Neto</p>
                    <p className={`text-base font-bold ${cs.net_change < 0 ? 'text-red-400' : 'text-green-400'}`}>
                      {cs.net_change > 0 ? '+' : ''}{cs.net_change}
                    </p>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2 text-[10px]">
                  <span className="text-slate-500">{cs.total_moves} movs</span>
                  <span className="text-slate-500">·</span>
                  <span className="text-slate-400">{cs.machines.join(', ')}</span>
                  <span className="text-slate-500">·</span>
                  <span className="text-slate-400">{cs.sap_batches.join(', ')}</span>
                </div>
                <div className="flex gap-1 mt-2">
                  {cs.users.map(u => (
                    <span key={u} className={`text-[9px] px-1 py-0.5 rounded ${
                      u === 'ADMIN' || u === 'System' ? 'bg-slate-800/50 text-slate-500' : 'bg-amber-950/40 text-amber-400'
                    }`}>{u}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ═══════ DETAIL MODAL ═══════ */}
      {selectedMov && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4" onClick={() => setSelectedMov(null)}>
          <div className="bg-[#0f1d32] rounded-xl border border-white/10 p-6 w-full max-w-2xl max-h-[85vh] overflow-y-auto space-y-4"
            onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold text-white">
                Detalle del Movimiento — Fila {selectedMov.excel_row} del Excel
              </h2>
              {selectedMov.is_anomaly && (
                <span className="text-xs px-2 py-1 rounded bg-red-950/50 text-red-400 font-medium">
                  {selectedMov.is_anomaly_qty ? '⚡ Anomalía de cantidad' : '👤 Usuario no-sistema'}
                </span>
              )}
            </div>
            <div className="grid grid-cols-2 gap-3">
              {Object.entries(selectedMov).filter(([k]) =>
                !k.startsWith('is_anomaly') && !k.startsWith('raw_') && k !== 'fill'
              ).map(([key, val]) => (
                <div key={key} className="bg-[#0a1628] rounded-lg px-3 py-2">
                  <p className="text-[10px] text-slate-500 uppercase">{key.replace(/_/g, ' ')}</p>
                  <p className={`text-sm ${key === 'qty_diff' && val != null ? (val < 0 ? 'text-red-400' : 'text-green-400') : 'text-white'}`}>
                    {val != null ? String(val) : '—'}
                  </p>
                </div>
              ))}
            </div>
            {/* Raw original columns */}
            {Object.entries(selectedMov).filter(([k]) => k.startsWith('raw_')).length > 0 && (
              <div>
                <p className="text-xs text-slate-500 mb-2">Columnas adicionales del Excel</p>
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(selectedMov).filter(([k]) => k.startsWith('raw_')).map(([key, val]) => (
                    <div key={key} className="bg-[#0a1628] rounded px-2 py-1">
                      <p className="text-[9px] text-slate-600">{key.replace('raw_', '').replace(/_/g, ' ')}</p>
                      <p className="text-xs text-slate-300">{val != null ? String(val) : '—'}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
            <div className="flex justify-end">
              <button onClick={() => setSelectedMov(null)}
                className="px-4 py-2 text-slate-400 hover:text-white text-sm">Cerrar</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}


// ═══════ Custom Tooltips ═══════

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-[#0f1d32] border border-white/10 rounded-lg p-3 text-xs shadow-xl">
      <p className="text-slate-400 mb-1">{d.date}</p>
      <p className="text-white">Qty: <span className="font-bold text-cyan-400">{d.new_qty}</span></p>
      <p className="text-slate-300">Δ: <span className={d.qty_diff < 0 ? 'text-red-400' : 'text-green-400'}>{d.qty_diff > 0 ? '+' : ''}{d.qty_diff}</span></p>
      <p className="text-slate-400">User: <span className={d.is_anomaly_user ? 'text-amber-400' : 'text-slate-300'}>{d.user}</span></p>
      <p className="text-slate-400">Tx: {d.tx_context}</p>
      <p className="text-slate-500">Fila Excel: {d.excel_row}</p>
      {d.is_anomaly && <p className="text-red-400 font-medium mt-1">{d.is_anomaly_qty ? '⚡ Pico de cantidad' : '👤 Usuario no-sistema'}</p>}
    </div>
  )
}

function DiffTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-[#0f1d32] border border-white/10 rounded-lg p-3 text-xs shadow-xl">
      <p className="text-slate-400">Mov #{d.index} · {d.date}</p>
      <p className={`font-bold ${d.qty_diff < 0 ? 'text-red-400' : 'text-green-400'}`}>
        Δ Qty: {d.qty_diff > 0 ? '+' : ''}{d.qty_diff}
      </p>
      <p className="text-slate-400">User: {d.user} · {d.tx_context}</p>
    </div>
  )
}
