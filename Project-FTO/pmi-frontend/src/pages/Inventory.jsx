import { useState, useMemo, useRef, useCallback } from 'react'
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Brush, ReferenceArea
} from 'recharts'
import UploadBanner from '../components/UploadBanner'
import * as api from '../lib/api'

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

  // Floating consumption panel
  const [showConsumption, setShowConsumption] = useState(false)
  const [consumptionSearch, setConsumptionSearch] = useState('')

  // Manual anomaly toggles (local only, keyed by excel_row)
  const [manualAnomalies, setManualAnomalies] = useState({})

  // Drag-select on anomaly chart
  const [dragStart, setDragStart] = useState(null)
  const [dragEnd, setDragEnd] = useState(null)
  const [isDragging, setIsDragging] = useState(false)

  // Click-to-highlight anomaly from table
  const [selectedAnomalyRow, setSelectedAnomalyRow] = useState(null)
  const [anomalyBrushRange, setAnomalyBrushRange] = useState(null)

  // Confirmation dialog for unmarking anomalies
  const [confirmUnmark, setConfirmUnmark] = useState(null)

  async function handleUpload(e) {
    const file = e.target.files[0]
    if (!file) return
    e.target.value = ''
    setLoading(true); setError(null); setData(null); setManualAnomalies({})
    try {
      const result = await api.analyzeInventory(file, threshold)
      setData(result)
      setBanner({ message: 'Archivo analizado', detail: `${result.total_movements} movimientos, ${result.total_containers} contenedores, ${result.total_anomalies} anomalías`, type: 'success' })
      if (result.filters.containers.length > 0) {
        setFilterContainer(result.filters.containers[0])
      }
    } catch (err) { setError(err.message) }
    finally { setLoading(false) }
  }

  // Check if a movement is anomaly (original OR manually toggled)
  const isAnomaly = useCallback((m) => {
    const row = m.excel_row
    if (row in manualAnomalies) return manualAnomalies[row]
    return m.is_anomaly
  }, [manualAnomalies])

  // Filtered movements — apply ALL filters
  const filtered = useMemo(() => {
    if (!data) return []
    return data.movements.filter(m => {
      if (filterContainer !== 'all' && m.container !== filterContainer) return false
      if (filterDest && m.dest_loc && !m.dest_loc.toLowerCase().includes(filterDest.toLowerCase())) return false
      if (filterUser !== 'all' && m.user !== filterUser) return false
      if (filterTx !== 'all' && m.tx_context !== filterTx) return false
      if (showAnomaliesOnly && !isAnomaly(m)) return false
      return true
    })
  }, [data, filterContainer, filterDest, filterUser, filterTx, showAnomaliesOnly, isAnomaly])

  // Movements filtered by dest_loc only (for consumption table & resumen)
  const destFiltered = useMemo(() => {
    if (!data) return data
    if (!filterDest) return data
    const fMovs = data.movements.filter(m =>
      m.dest_loc && m.dest_loc.toLowerCase().includes(filterDest.toLowerCase())
    )
    const containerIds = new Set(fMovs.map(m => m.container))
    return {
      ...data,
      movements: fMovs,
      container_summaries: data.container_summaries.filter(cs => containerIds.has(cs.container)),
    }
  }, [data, filterDest])

  // Timeline chart data
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
      is_anomaly: isAnomaly(m),
      is_anomaly_user: m.is_anomaly_user,
      is_anomaly_qty: m.is_anomaly_qty,
      excel_row: m.excel_row,
      container: m.container,
      dest_loc: m.dest_loc,
      fill: m.is_anomaly_qty ? '#ef4444' : m.is_anomaly_user ? '#f59e0b' : '#06b6d4',
    }))
  }, [filtered, isAnomaly])

  // Container summary for selected
  const selectedSummary = useMemo(() => {
    if (!data || filterContainer === 'all') return null
    return data.container_summaries.find(s => s.container === filterContainer)
  }, [data, filterContainer])

  // Machines for selected container
  const selectedMachines = useMemo(() => {
    if (!data || filterContainer === 'all') return []
    const movs = data.movements.filter(m => m.container === filterContainer)
    return [...new Set(movs.map(m => m.dest_loc).filter(Boolean))].sort()
  }, [data, filterContainer])

  // Time info for selected container (first real drop & last movement)
  const containerTimeInfo = useMemo(() => {
    if (!data || filterContainer === 'all') return null
    const movs = data.movements.filter(m => m.container === filterContainer)
    if (!movs.length) return null
    const firstDrop = movs.find(m => m.new_qty != null && m.old_qty != null && m.new_qty < m.old_qty)
    const lastMov = movs[movs.length - 1]
    return {
      firstChangeDate: firstDrop?.date || null,
      lastChangeDate: lastMov?.date || null,
    }
  }, [data, filterContainer])

  // Consumption table (filtered by dest_loc + search)
  const consumptionTable = useMemo(() => {
    if (!destFiltered) return []
    let summaries = destFiltered.container_summaries
    if (consumptionSearch) {
      const q = consumptionSearch.toLowerCase()
      summaries = summaries.filter(cs =>
        cs.container.toLowerCase().includes(q) ||
        cs.machines.some(m => m.toLowerCase().includes(q))
      )
    }
    return summaries.map(cs => {
      const movs = destFiltered.movements.filter(m => m.container === cs.container)
      const firstDrop = movs.find(m =>
        m.new_qty != null && m.old_qty != null && m.new_qty < m.old_qty
      )
      const lastMov = movs.length > 0 ? movs[movs.length - 1] : null
      return {
        container: cs.container,
        firstDropDate: firstDrop?.date || null,
        lastMovDate: lastMov?.date || null,
        totalMoves: cs.total_moves,
        machines: cs.machines,
        initialQty: cs.initial_qty,
        finalQty: cs.final_qty,
      }
    })
  }, [destFiltered, consumptionSearch])

  // ─── Anomaly tab data: filtered by container AND dest_loc ───
  const anomalyFilteredMovs = useMemo(() => {
    if (!data) return []
    return data.movements.filter(m => {
      if (filterContainer !== 'all' && m.container !== filterContainer) return false
      if (filterDest && m.dest_loc && !m.dest_loc.toLowerCase().includes(filterDest.toLowerCase())) return false
      return true
    })
  }, [data, filterContainer, filterDest])

  const anomaliesData = useMemo(() => {
    return anomalyFilteredMovs.filter(m => isAnomaly(m))
  }, [anomalyFilteredMovs, isAnomaly])

  // Anomaly chart data (all filtered movements, anomalies highlighted)
  const anomalyChartData = useMemo(() => {
    return anomalyFilteredMovs.map((m, i) => ({
      index: i + 1,
      new_qty: m.new_qty,
      qty_diff: m.qty_diff != null ? Math.round(m.qty_diff * 1000) / 1000 : 0,
      date: m.date ? new Date(m.date).toLocaleString('es-MX', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '?',
      is_anomaly: isAnomaly(m),
      is_manual: m.excel_row in manualAnomalies,
      excel_row: m.excel_row,
      container: m.container,
      user: m.user,
      dest_loc: m.dest_loc,
      fill: isAnomaly(m) ? (m.excel_row in manualAnomalies ? '#a855f7' : '#ef4444') : '#1e3a5f',
    }))
  }, [anomalyFilteredMovs, isAnomaly, manualAnomalies])

  // Toggle manual anomaly
  function toggleManualAnomaly(excelRow, currentIsAnomaly) {
    setManualAnomalies(prev => {
      const next = { ...prev }
      if (excelRow in next) {
        delete next[excelRow]
      } else {
        next[excelRow] = !currentIsAnomaly
      }
      return next
    })
  }

  // Drag-select handlers for anomaly chart
  function handleAnomalyMouseDown(e) {
    if (e && e.activeLabel) {
      setDragStart(e.activeLabel)
      setDragEnd(e.activeLabel)
      setIsDragging(true)
    }
  }
  function handleAnomalyMouseMove(e) {
    if (isDragging && e && e.activeLabel) {
      setDragEnd(e.activeLabel)
    }
  }
  function handleAnomalyMouseUp() {
    if (isDragging && dragStart != null && dragEnd != null) {
      setIsDragging(false)
      // Don't auto-mark, just keep the selection visible
    }
  }
  function confirmDragSelection() {
    if (dragStart == null || dragEnd == null) return
    const start = Math.min(dragStart, dragEnd)
    const end = Math.max(dragStart, dragEnd)
    const movs = anomalyChartData.filter(m => m.index >= start && m.index <= end)
    setManualAnomalies(prev => {
      const next = { ...prev }
      movs.forEach(m => { next[m.excel_row] = true })
      return next
    })
    setDragStart(null)
    setDragEnd(null)
  }
  function clearDragSelection() {
    setDragStart(null)
    setDragEnd(null)
    setIsDragging(false)
  }

  // Click anomaly table row → highlight on chart + scroll Brush
  function handleAnomalyRowClick(excelRow) {
    setSelectedAnomalyRow(prev => prev === excelRow ? null : excelRow)
    const idx = anomalyChartData.findIndex(d => d.excel_row === excelRow)
    if (idx >= 0) {
      const half = 20
      setAnomalyBrushRange({
        startIndex: Math.max(0, idx - half),
        endIndex: Math.min(anomalyChartData.length - 1, idx + half),
      })
    }
  }

  // Handle unmark with confirmation
  function handleUnmarkClick(excelRow, isOriginal) {
    setConfirmUnmark({ excelRow, isOriginal })
  }
  function confirmUnmarkAction() {
    if (!confirmUnmark) return
    toggleManualAnomaly(confirmUnmark.excelRow, confirmUnmark.isOriginal)
    setConfirmUnmark(null)
  }

  // Export anomalies to CSV
  function exportAnomaliesCSV() {
    if (!anomaliesData.length) return
    const headers = ['Fila Excel', 'Fecha', 'Container', 'Dest Loc', 'Old Qty', 'New Qty', 'Δ Qty', 'User', 'Transaction', 'SAP Batch', 'Tipo']
    const rows = anomaliesData.map(m => [
      m.excel_row, m.date || '', m.container || '', m.dest_loc || '',
      m.old_qty ?? '', m.new_qty ?? '', m.qty_diff ?? '', m.user || '',
      m.tx_context || '', m.sap_batch || '',
      m.excel_row in manualAnomalies ? 'Manual' : m.is_anomaly_qty ? 'Qty' : 'User'
    ])
    const csv = [headers.join(','), ...rows.map(r => r.map(v => `"${v}"`).join(','))].join('\n')
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'anomalias_inventario.csv'; a.click()
    URL.revokeObjectURL(url)
  }

  // ═══════ RENDER ═══════

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

  const selMin = dragStart != null && dragEnd != null ? Math.min(dragStart, dragEnd) : null
  const selMax = dragStart != null && dragEnd != null ? Math.max(dragStart, dragEnd) : null

  return (
    <div className="p-6 space-y-5 min-h-screen bg-[#0a1628] relative">
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
          <button onClick={() => setShowConsumption(p => !p)}
            className={`px-3 py-2 text-sm rounded-lg transition-colors ${showConsumption ? 'bg-amber-600 text-white' : 'bg-[#0f1d32] border border-white/10 text-slate-300 hover:text-white hover:border-white/20'}`}>
            📊 Consumo
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
            {selectedMachines.length > 0 && (
              <div>
                <p className="text-[10px] text-slate-500 uppercase">Máquina(s)</p>
                <div className="flex gap-1 flex-wrap">
                  {selectedMachines.map(m => (
                    <span key={m} className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-950/40 text-cyan-300 font-medium">{m}</span>
                  ))}
                </div>
              </div>
            )}
            {containerTimeInfo && (
              <>
                <div>
                  <p className="text-[10px] text-slate-500 uppercase">Primer cambio real</p>
                  <p className="text-xs text-green-400 font-medium">
                    {containerTimeInfo.firstChangeDate
                      ? new Date(containerTimeInfo.firstChangeDate).toLocaleString('es-MX', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' })
                      : '—'}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-slate-500 uppercase">Último movimiento</p>
                  <p className="text-xs text-amber-400 font-medium">
                    {containerTimeInfo.lastChangeDate
                      ? new Date(containerTimeInfo.lastChangeDate).toLocaleString('es-MX', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' })
                      : '—'}
                  </p>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 bg-[#0f1d32] rounded-lg p-1 w-fit">
        {['timeline', 'anomalias', 'tabla', 'resumen'].map(t => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-4 py-1.5 text-sm rounded-md transition-colors ${tab === t ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}>
            {t === 'timeline' ? 'Timeline' : t === 'anomalias' ? 'Anomalías' : t === 'tabla' ? 'Tabla de Movimientos' : 'Resumen Contenedores'}
          </button>
        ))}
      </div>

      {/* ═══════ TIMELINE TAB ═══════ */}
      {tab === 'timeline' && (
        <div className="space-y-4">
          <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-4">
            <h3 className="text-sm font-semibold text-white mb-1">Cantidad (New Qty) por movimiento</h3>
            <p className="text-[10px] text-slate-500 mb-3">
              <span className="inline-block w-2 h-2 rounded-full bg-cyan-500 mr-1" /> Normal
              <span className="inline-block w-2 h-2 rounded-full bg-amber-500 mr-1 ml-3" /> Usuario no-sistema
              <span className="inline-block w-2 h-2 rounded-full bg-red-500 mr-1 ml-3" /> Pico de cantidad (&gt;{threshold}kg)
              <span className="text-slate-600 ml-3">— Arrastra la barra inferior para hacer zoom</span>
            </p>
            <ResponsiveContainer width="100%" height={340}>
              <LineChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                <XAxis dataKey="index" tick={{ fill: '#94a3b8', fontSize: 10 }} label={{ value: 'Mov #', position: 'insideBottom', offset: -15, fill: '#64748b', fontSize: 10 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} label={{ value: 'Qty', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10 }} />
                <Tooltip content={<CustomTooltip />} />
                <Line type="stepAfter" dataKey="new_qty" stroke="#06b6d4" strokeWidth={2} dot={(props) => {
                  const { cx, cy, payload } = props
                  if (!cx || !cy) return null
                  const color = payload.is_anomaly_qty ? '#ef4444' : payload.is_anomaly_user ? '#f59e0b' : '#06b6d4'
                  const r = payload.is_anomaly ? 5 : 3
                  return <circle cx={cx} cy={cy} r={r} fill={color} stroke={color} strokeWidth={1} className="cursor-pointer" />
                }} activeDot={{ r: 6, stroke: '#fff', strokeWidth: 2 }} />
                <Brush dataKey="index" height={25} stroke="#06b6d4" fill="#0a1628" travellerWidth={8} tickFormatter={() => ''} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-4">
            <h3 className="text-sm font-semibold text-white mb-3">Diferencia de Cantidad por movimiento (New - Old)</h3>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                <XAxis dataKey="index" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <Tooltip content={<DiffTooltip />} />
                <ReferenceLine y={0} stroke="#475569" />
                <ReferenceLine y={threshold} stroke="#ef4444" strokeDasharray="5 5" label={{ value: `+${threshold}`, fill: '#ef4444', fontSize: 10 }} />
                <ReferenceLine y={-threshold} stroke="#ef4444" strokeDasharray="5 5" label={{ value: `-${threshold}`, fill: '#ef4444', fontSize: 10 }} />
                <Bar dataKey="qty_diff" name="Δ Qty"
                  shape={(props) => {
                    const { x, y, width, height, payload } = props
                    const fill = payload?.fill || '#06b6d4'
                    const ay = height < 0 ? y + height : y
                    const ah = Math.abs(height)
                    return <rect x={x} y={ay} width={width} height={ah} fill={fill} rx={2} />
                  }}
                />
                <Brush dataKey="index" height={20} stroke="#06b6d4" fill="#0a1628" travellerWidth={8} tickFormatter={() => ''} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ═══════ ANOMALÍAS TAB ═══════ */}
      {tab === 'anomalias' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-400">
                {anomaliesData.length} anomalías detectadas
                {filterContainer !== 'all' && <span className="text-cyan-400 ml-1">(contenedor: ...{filterContainer.slice(-8)})</span>}
                {filterDest && <span className="text-cyan-400 ml-1">(máquina: {filterDest})</span>}
                {Object.keys(manualAnomalies).length > 0 && (
                  <span className="text-purple-400 ml-2">{Object.values(manualAnomalies).filter(v => v).length} marcadas manualmente</span>
                )}
              </p>
            </div>
            <div className="flex gap-2">
              {Object.keys(manualAnomalies).length > 0 && (
                <button onClick={() => setManualAnomalies({})}
                  className="px-3 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg transition-colors">
                  Resetear manuales
                </button>
              )}
              <button onClick={exportAnomaliesCSV} disabled={!anomaliesData.length}
                className="px-3 py-1.5 text-xs bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 text-white rounded-lg transition-colors">
                📥 Exportar CSV
              </button>
            </div>
          </div>

          {/* Anomaly chart with drag-select */}
          <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-white">Mapa de Anomalías</h3>
              <div className="flex items-center gap-3 text-[10px]">
                <span><span className="inline-block w-2 h-2 rounded-full bg-red-500 mr-1" /> Auto-detectada</span>
                <span><span className="inline-block w-2 h-2 rounded-full bg-purple-500 mr-1" /> Manual</span>
                <span className="text-slate-600">|</span>
                <span className="text-slate-500">Arrastra sobre la gráfica para seleccionar un rango</span>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={anomalyChartData}
                onMouseDown={handleAnomalyMouseDown}
                onMouseMove={handleAnomalyMouseMove}
                onMouseUp={handleAnomalyMouseUp}
                style={{ cursor: 'crosshair' }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                <XAxis dataKey="index" tick={{ fill: '#94a3b8', fontSize: 9 }} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <Tooltip content={<AnomalyTooltip />} />
                <ReferenceLine y={0} stroke="#475569" />
                <Bar dataKey="qty_diff" name="Δ Qty"
                  shape={(props) => {
                    const { x, y, width, height, payload } = props
                    const fill = payload?.fill || '#1e3a5f'
                    const ay = height < 0 ? y + height : y
                    const ah = Math.abs(height)
                    const isSel = payload?.excel_row === selectedAnomalyRow
                    return (
                      <g>
                        <rect x={x} y={ay} width={width} height={ah} fill={fill} rx={2} />
                        {isSel && <rect x={x - 2} y={ay - 2} width={width + 4} height={ah + 4} fill="none" stroke="#facc15" strokeWidth={2.5} rx={3} />}
                      </g>
                    )
                  }}
                />
                {selMin != null && selMax != null && (
                  <ReferenceArea x1={selMin} x2={selMax} fill="#a855f7" fillOpacity={0.15} stroke="#a855f7" strokeOpacity={0.4} />
                )}
                <Brush dataKey="index" height={20} stroke="#06b6d4" fill="#0a1628" travellerWidth={8} tickFormatter={() => ''}
                  startIndex={anomalyBrushRange?.startIndex}
                  endIndex={anomalyBrushRange?.endIndex}
                  onChange={(range) => setAnomalyBrushRange(range)}
                />
              </BarChart>
            </ResponsiveContainer>
            {selMin != null && selMax != null && !isDragging && (
              <div className="flex items-center justify-center gap-3 mt-2 bg-purple-500/5 rounded-lg py-2">
                <span className="text-xs text-slate-400">Selección: Mov {selMin} — {selMax} ({selMax - selMin + 1} movimientos)</span>
                <button onClick={confirmDragSelection}
                  className="px-3 py-1 text-xs bg-purple-600 hover:bg-purple-500 text-white rounded-lg transition-colors">
                  ⚡ Marcar como anomalía
                </button>
                <button onClick={clearDragSelection}
                  className="px-3 py-1 text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg transition-colors">
                  ✕ Cancelar
                </button>
              </div>
            )}
          </div>

          {/* Anomalies table */}
          <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
            <div className="px-4 py-2 border-b border-white/5">
              <span className="text-xs text-slate-400">{anomaliesData.length} anomalías</span>
            </div>
            <div className="overflow-x-auto max-h-[400px]">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-[#0f1d32] z-10">
                  <tr className="border-b border-white/5 text-slate-500">
                    {['Fila', 'Fecha', 'Container', 'Dest Loc', 'Old Qty', 'New Qty', 'Δ Qty', 'User', 'Tipo', ''].map(h => (
                      <th key={h} className="px-2 py-2 text-left font-medium whitespace-nowrap">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {anomaliesData.map((m, i) => {
                    const isManual = m.excel_row in manualAnomalies
                    const isSelected = m.excel_row === selectedAnomalyRow
                    return (
                      <tr key={i}
                        className={`border-b border-white/5 hover:bg-white/[0.03] cursor-pointer ${isSelected ? 'bg-yellow-500/[0.08] ring-1 ring-yellow-500/30' : ''}`}
                        onClick={() => handleAnomalyRowClick(m.excel_row)}>
                        <td className="px-2 py-1.5 text-slate-500 font-mono">{m.excel_row}</td>
                        <td className="px-2 py-1.5 text-slate-300 whitespace-nowrap">
                          {m.date ? new Date(m.date).toLocaleString('es-MX', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—'}
                        </td>
                        <td className="px-2 py-1.5 text-white font-mono text-[10px]">...{(m.container || '').slice(-8)}</td>
                        <td className="px-2 py-1.5"><span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-950/40 text-cyan-300">{m.dest_loc}</span></td>
                        <td className="px-2 py-1.5 text-slate-400">{m.old_qty != null ? m.old_qty.toFixed(2) : '—'}</td>
                        <td className="px-2 py-1.5 text-white font-medium">{m.new_qty != null ? m.new_qty.toFixed(2) : '—'}</td>
                        <td className="px-2 py-1.5">
                          <span className={`font-medium ${m.qty_diff < 0 ? 'text-red-400' : 'text-green-400'}`}>
                            {m.qty_diff > 0 ? '+' : ''}{m.qty_diff?.toFixed(2)}
                          </span>
                        </td>
                        <td className="px-2 py-1.5">
                          <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                            m.is_anomaly_user ? 'bg-amber-950/50 text-amber-400' : 'bg-slate-800 text-slate-400'
                          }`}>{m.user}</span>
                        </td>
                        <td className="px-2 py-1.5">
                          <span className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${
                            isManual ? 'bg-purple-950/50 text-purple-400' :
                            m.is_anomaly_qty ? 'bg-red-950/50 text-red-400' : 'bg-amber-950/50 text-amber-400'
                          }`}>
                            {isManual ? '✎ Manual' : m.is_anomaly_qty ? '⚡ QTY' : '👤 USER'}
                          </span>
                        </td>
                        <td className="px-2 py-1.5">
                          <button onClick={(e) => { e.stopPropagation(); handleUnmarkClick(m.excel_row, m.is_anomaly) }}
                            className="text-[10px] px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
                            title={isManual ? 'Revertir a original' : 'Desmarcar anomalía'}>
                            {isManual ? '↩' : '✕'}
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ═══════ TABLA TAB ═══════ */}
      {tab === 'tabla' && (
        <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
          <div className="px-4 py-2 border-b border-white/5">
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
                    isAnomaly(m) ? 'bg-red-500/[0.03]' : ''
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
                      {isAnomaly(m) && (
                        <span className={`text-[9px] px-1 py-0.5 rounded ${
                          m.excel_row in manualAnomalies ? 'bg-purple-950/50 text-purple-400' : 'bg-red-950/50 text-red-400'
                        }`}>
                          {m.excel_row in manualAnomalies ? '✎ Manual' : m.is_anomaly_qty ? '⚡ QTY' : '👤 USER'}
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
          {filterDest && (
            <p className="text-xs text-slate-500">Filtrado por máquina: <span className="text-cyan-400">{filterDest}</span></p>
          )}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {(destFiltered?.container_summaries || []).map((cs) => (
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
                  <span className="text-slate-400" title={cs.machines.join(', ')}>
                    {cs.machines[0] || '—'}{cs.machines.length > 1 && ` +${cs.machines.length - 1}`}
                  </span>
                  <span className="text-slate-500">·</span>
                  <span className="text-slate-400">{cs.sap_batches[0] || '—'}{cs.sap_batches.length > 1 && ` +${cs.sap_batches.length - 1}`}</span>
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

      {/* ═══════ FLOATING CONSUMPTION PANEL ═══════ */}
      {showConsumption && data && (
        <div className="fixed top-20 right-4 z-40 w-[400px] bg-[#0f1d32] rounded-xl border border-white/10 shadow-2xl shadow-black/40 flex flex-col"
          style={{ maxHeight: 'min(50vh, 380px)' }}>
          <div className="flex items-center justify-between px-3 py-2 border-b border-white/5 shrink-0">
            <h4 className="text-xs font-semibold text-white">📊 Consumo ({consumptionTable.length})</h4>
            <button onClick={() => setShowConsumption(false)} className="text-slate-500 hover:text-white text-xs">✕</button>
          </div>
          {/* Search */}
          <div className="px-3 py-1.5 border-b border-white/5 shrink-0">
            <input value={consumptionSearch} onChange={e => setConsumptionSearch(e.target.value)}
              placeholder="Buscar contenedor o máquina..."
              className="w-full px-2 py-1 bg-[#0a1628] border border-white/10 rounded text-[10px] text-white placeholder-slate-600 focus:outline-none focus:border-cyan-500/50" />
          </div>
          <div className="overflow-y-auto flex-1 p-1">
            <table className="w-full text-[10px]">
              <thead className="sticky top-0 bg-[#0f1d32]">
                <tr className="text-slate-600 border-b border-white/5">
                  <th className="px-1.5 py-1 text-left font-medium">ID</th>
                  <th className="px-1.5 py-1 text-left font-medium">Máq.</th>
                  <th className="px-1.5 py-1 text-center font-medium">Ini</th>
                  <th className="px-1.5 py-1 text-left font-medium">Inicio</th>
                  <th className="px-1.5 py-1 text-left font-medium">Fin</th>
                  <th className="px-1.5 py-1 text-center font-medium">Fin Qty</th>
                </tr>
              </thead>
              <tbody>
                {consumptionTable.map(ct => {
                  const isSelected = filterContainer === ct.container
                  return (
                    <tr key={ct.container}
                      className={`border-b border-white/[0.03] hover:bg-white/[0.03] cursor-pointer ${isSelected ? 'bg-cyan-500/[0.06]' : ''}`}
                      onClick={() => { setFilterContainer(ct.container); setTab('timeline') }}>
                      <td className="px-1.5 py-1 text-white font-mono text-[9px]" title={ct.container}>
                        ...{ct.container.slice(-6)}
                        {isSelected && <span className="ml-0.5 text-cyan-400">●</span>}
                      </td>
                      <td className="px-1.5 py-1 text-cyan-300 text-[9px]" title={ct.machines.join(', ')}>
                        {ct.machines[0] || '—'}
                        {ct.machines.length > 1 && <span className="text-slate-600 ml-0.5">+{ct.machines.length - 1}</span>}
                      </td>
                      <td className="px-1.5 py-1 text-center text-cyan-400 font-semibold">{ct.initialQty}</td>
                      <td className="px-1.5 py-1 text-green-400 whitespace-nowrap">
                        {ct.firstDropDate ? new Date(ct.firstDropDate).toLocaleString('es-MX', { day: '2-digit', hour: '2-digit', minute: '2-digit' }) : '—'}
                      </td>
                      <td className="px-1.5 py-1 text-amber-400 whitespace-nowrap">
                        {ct.lastMovDate ? new Date(ct.lastMovDate).toLocaleString('es-MX', { day: '2-digit', hour: '2-digit', minute: '2-digit' }) : '—'}
                      </td>
                      <td className="px-1.5 py-1 text-center text-white font-semibold">{ct.finalQty}</td>
                    </tr>
                  )
                })}
                {consumptionTable.length === 0 && (
                  <tr><td colSpan={6} className="px-3 py-4 text-center text-slate-600 text-[10px]">Sin resultados</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ═══════ CONFIRM UNMARK MODAL ═══════ */}
      {confirmUnmark && (
        <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4" onClick={() => setConfirmUnmark(null)}>
          <div className="bg-[#0f1d32] rounded-xl border border-white/10 p-6 w-full max-w-sm space-y-4" onClick={e => e.stopPropagation()}>
            <div className="flex items-center gap-3">
              <span className="text-2xl">⚠️</span>
              <div>
                <h3 className="text-sm font-semibold text-white">¿Desmarcar esta anomalía?</h3>
                <p className="text-xs text-slate-400 mt-1">
                  Fila Excel: <span className="text-white font-mono">{confirmUnmark.excelRow}</span>
                </p>
                <p className="text-xs text-slate-500 mt-0.5">
                  {confirmUnmark.excelRow in manualAnomalies
                    ? 'Se revertirá al estado original de esta anomalía.'
                    : 'Se desmarcará este registro como anomalía. Podrás volver a marcarlo después.'}
                </p>
              </div>
            </div>
            <div className="flex justify-end gap-2">
              <button onClick={() => setConfirmUnmark(null)}
                className="px-4 py-2 text-sm text-slate-400 hover:text-white rounded-lg transition-colors">
                Cancelar
              </button>
              <button onClick={confirmUnmarkAction}
                className="px-4 py-2 text-sm bg-red-600 hover:bg-red-500 text-white rounded-lg transition-colors">
                Sí, desmarcar
              </button>
            </div>
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
              {isAnomaly(selectedMov) && (
                <span className={`text-xs px-2 py-1 rounded font-medium ${
                  selectedMov.excel_row in manualAnomalies ? 'bg-purple-950/50 text-purple-400' : 'bg-red-950/50 text-red-400'
                }`}>
                  {selectedMov.excel_row in manualAnomalies ? '✎ Anomalía manual' : selectedMov.is_anomaly_qty ? '⚡ Anomalía de cantidad' : '👤 Usuario no-sistema'}
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
            <div className="flex justify-end gap-2">
              <button onClick={() => toggleManualAnomaly(selectedMov.excel_row, selectedMov.is_anomaly)}
                className="px-4 py-2 text-sm rounded-lg bg-purple-600/20 text-purple-400 hover:bg-purple-600/30 transition-colors">
                {selectedMov.excel_row in manualAnomalies ? '↩ Revertir anomalía' : isAnomaly(selectedMov) ? '✕ Desmarcar anomalía' : '⚡ Marcar como anomalía'}
              </button>
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

function AnomalyTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-[#0f1d32] border border-white/10 rounded-lg p-3 text-xs shadow-xl">
      <p className="text-slate-400">Mov #{d.index} · {d.date}</p>
      <p className="text-white">Container: <span className="font-mono text-[10px]">...{(d.container || '').slice(-8)}</span></p>
      <p className={`font-bold ${d.qty_diff < 0 ? 'text-red-400' : 'text-green-400'}`}>
        Δ Qty: {d.qty_diff > 0 ? '+' : ''}{d.qty_diff}
      </p>
      <p className="text-slate-400">User: {d.user} · Dest: {d.dest_loc}</p>
      {d.is_anomaly && (
        <p className={`font-medium mt-1 ${d.is_manual ? 'text-purple-400' : 'text-red-400'}`}>
          {d.is_manual ? '✎ Anomalía manual' : '⚡ Anomalía detectada'}
        </p>
      )}
    </div>
  )
}
