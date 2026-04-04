import { useState, useEffect } from 'react'
import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts'
import CedulaSelector from '../components/CedulaSelector'
import * as api from '../lib/api'

// ══════════════════════════════════════════════════════
// Helpers
// ══════════════════════════════════════════════════════

function getCurrentQuarter() {
  const m = new Date().getMonth()
  return Math.floor(m / 3) + 1
}
function getCurrentYear() {
  return new Date().getFullYear()
}
function getQuarterWeeks(q) {
  // Q1=S1-S13, Q2=S14-S26, Q3=S27-S39, Q4=S40-S52
  const start = (q - 1) * 13 + 1
  const end = q * 13
  return { start, end }
}

// ══════════════════════════════════════════════════════
// Main component
// ══════════════════════════════════════════════════════

export default function Weekly() {
  const [cedulas, setCedulas]               = useState([])
  const [cedulaId, setCedulaId]             = useState(null)
  const [cedulasLoading, setCedulasLoading] = useState(true)
  const [error, setError]                   = useState(null)

  const [year, setYear]       = useState(getCurrentYear())
  const [quarter, setQuarter] = useState(getCurrentQuarter())
  const [weekStart, setWeekStart] = useState(getQuarterWeeks(getCurrentQuarter()).start)
  const [weekEnd, setWeekEnd]     = useState(getQuarterWeeks(getCurrentQuarter()).end)

  const [categories, setCategories]   = useState([])
  const [activeCatId, setActiveCatId] = useState(null)
  const [chartData, setChartData]     = useState(null)
  const [loading, setLoading]         = useState(false)
  const [seeding, setSeeding]         = useState(false)

  const [editMode, setEditMode]             = useState(null) // null | 'values' | 'targets'
  const [pendingChanges, setPendingChanges] = useState({})
  const [saving, setSaving]                 = useState(false)
  const [customYRanges, setCustomYRanges]   = useState({}) // { [indId]: { min, max } }

  const [viewMode, setViewMode]             = useState('category') // 'category' | 'machine'
  const [machineFilter, setMachineFilter]   = useState('all') // 'all' | specific machine
  const [allIndicators, setAllIndicators]   = useState(null) // full chart data for machine view

  // ── Load cédulas ──
  useEffect(() => {
    api.getCedulas()
      .then(res => {
        setCedulas(res.data || [])
        if (res.data?.length > 0) setCedulaId(res.data[0].id)
      })
      .catch(err => setError(err.message))
      .finally(() => setCedulasLoading(false))
  }, [])

  // ── Load categories ──
  useEffect(() => {
    if (!cedulaId) return
    api.getWeeklyCategories(cedulaId)
      .then(res => {
        setCategories(res.data || [])
        if (res.data?.length > 0) setActiveCatId(prev => prev || res.data[0].id)
      })
      .catch(err => setError(err.message))
  }, [cedulaId])

  // ── Helper: seed customYRanges from indicator.y_min / y_max ──
  function initYRangesFromIndicators(indicators) {
    setCustomYRanges(prev => {
      const next = { ...prev }
      for (const ind of indicators) {
        if (ind.y_min != null || ind.y_max != null) {
          next[ind.id] = {
            min: ind.y_min != null ? ind.y_min : null,
            max: ind.y_max != null ? ind.y_max : null,
          }
        }
      }
      return next
    })
  }

  // ── Load chart data (category view) ──
  useEffect(() => {
    if (!cedulaId || !activeCatId || viewMode !== 'category') return
    setLoading(true)
    setChartData(null)
    // NO limpiar pendingChanges aquí — se mantienen al cambiar de pestaña
    api.getWeeklyChartData(cedulaId, year, quarter, activeCatId)
      .then(res => { setChartData(res); initYRangesFromIndicators(res.indicators || []) })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [cedulaId, activeCatId, year, quarter, viewMode])

  // ── Load ALL indicators (machine view) ──
  useEffect(() => {
    if (!cedulaId || viewMode !== 'machine') return
    setLoading(true)
    setAllIndicators(null)
    api.getWeeklyChartData(cedulaId, year, quarter, null)
      .then(res => { setAllIndicators(res); initYRangesFromIndicators(res.indicators || []) })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [cedulaId, year, quarter, viewMode])

  // ── Seed ──
  async function handleSeed() {
    if (!cedulaId) return
    setSeeding(true); setError(null)
    try {
      await api.seedWeekly(cedulaId)
      const cats = await api.getWeeklyCategories(cedulaId)
      setCategories(cats.data || [])
      if (cats.data?.length > 0) setActiveCatId(cats.data[0].id)
    } catch (err) { setError(err.message) }
    finally { setSeeding(false) }
  }

  // ── Seed extras (add missing categories) ──
  async function handleSeedExtras() {
    if (!cedulaId) return
    setSeeding(true); setError(null)
    try {
      const res = await api.seedWeeklyExtras(cedulaId)
      const cats = await api.getWeeklyCategories(cedulaId)
      setCategories(cats.data || [])
      alert(res.message)
    } catch (err) { setError(err.message) }
    finally { setSeeding(false) }
  }

  // ── Save ──
  async function handleSave() {
    if (!cedulaId || Object.keys(pendingChanges).length === 0) return
    setSaving(true); setError(null)
    try {
      if (editMode === 'values') {
        const vals = []
        for (const [indId, weekMap] of Object.entries(pendingChanges)) {
          for (const [wk, val] of Object.entries(weekMap)) {
            if (val === '' || val == null) continue
            vals.push({ indicator_id: indId, cedula_id: cedulaId, year, quarter, week_number: parseInt(wk), actual_value: parseFloat(val) })
          }
        }
        if (vals.length > 0) await api.upsertWeeklyValues(vals)
      } else if (editMode === 'targets') {
        const tgts = []
        for (const [indId, weekMap] of Object.entries(pendingChanges)) {
          for (const [wk, val] of Object.entries(weekMap)) {
            if (val === '' || val == null) continue
            tgts.push({ indicator_id: indId, cedula_id: cedulaId, year, quarter, week_number: parseInt(wk), target_value: parseFloat(val) })
          }
        }
        if (tgts.length > 0) await api.upsertWeeklyTargets(tgts)
      }
      setPendingChanges({})
      // Reload
      const res = await api.getWeeklyChartData(cedulaId, year, quarter, activeCatId)
      setChartData(res)
    } catch (err) { setError(err.message) }
    finally { setSaving(false) }
  }

  // ── Auto-save Y range to DB (debounced 800ms) ──
  const yRangeSaveTimers = {}
  function handleYRangeChange(indId, range) {
    setCustomYRanges(prev => ({ ...prev, [indId]: range }))
    // Clear any pending timer for this indicator
    if (yRangeSaveTimers[indId]) clearTimeout(yRangeSaveTimers[indId])
    // Debounce: save 800ms after the user stops typing
    yRangeSaveTimers[indId] = setTimeout(async () => {
      try {
        await api.updateWeeklyIndicator(indId, {
          y_min: range?.min != null ? range.min : null,
          y_max: range?.max != null ? range.max : null,
        })
      } catch (err) { /* silently ignore save errors for range */ }
    }, 800)
  }

  // ── Auto-save band_size to DB (debounced 800ms) ──
  const bandSaveTimers = {}
  function handleBandSizeChange(indId, bandSize) {
    // Update indicator in local state immediately
    const updateIndicators = (data) => {
      if (!data) return data
      return {
        ...data,
        indicators: data.indicators.map(i => i.id === indId ? { ...i, band_size: bandSize } : i)
      }
    }
    if (viewMode === 'machine') setAllIndicators(updateIndicators)
    else setChartData(updateIndicators)
    // Debounce save
    if (bandSaveTimers[indId]) clearTimeout(bandSaveTimers[indId])
    bandSaveTimers[indId] = setTimeout(async () => {
      try { await api.updateWeeklyIndicator(indId, { band_size: bandSize }) }
      catch (err) { /* silent */ }
    }, 800)
  }

  // Toggle direction (higher_better → lower_better → middle_better → ...) and persist to DB
  async function handleToggleDirection(indId) {
    // Find the indicator in current data source
    const src = viewMode === 'machine' ? allIndicators : chartData
    const ind = src?.indicators?.find(i => i.id === indId)
    if (!ind) return
    const cycle = { higher_better: 'lower_better', lower_better: 'middle_better', middle_better: 'higher_better' }
    const newDir = cycle[ind.direction] || 'higher_better'
    try {
      await api.updateWeeklyIndicator(indId, { direction: newDir })
      // Update local state
      const updateIndicators = (data) => {
        if (!data) return data
        return {
          ...data,
          indicators: data.indicators.map(i => i.id === indId ? { ...i, direction: newDir } : i)
        }
      }
      if (viewMode === 'machine') setAllIndicators(updateIndicators)
      else setChartData(updateIndicators)
    } catch (err) { setError(err.message) }
  }

  function handleCellChange(indId, weekNum, value) {
    setPendingChanges(prev => ({
      ...prev,
      [indId]: { ...(prev[indId] || {}), [weekNum]: value }
    }))
  }

  const hasChanges = Object.keys(pendingChanges).length > 0

  // ── Extract unique machines from indicators (for machine view) ──
  const machineList = (() => {
    const src = viewMode === 'machine' ? allIndicators : chartData
    if (!src?.indicators) return []
    const set = new Set()
    for (const ind of src.indicators) {
      if (ind.subtitle) {
        // Handle combined subtitles like "KDF 7, 9, 10"
        if (ind.subtitle.includes(',')) {
          ind.subtitle.split(',').forEach(part => {
            const trimmed = part.trim()
            // Reconstruct full machine name if needed (e.g. "9" → "KDF 9")
            if (/^\d+$/.test(trimmed)) set.add(`KDF ${trimmed}`)
            else set.add(trimmed)
          })
        } else {
          set.add(ind.subtitle.trim())
        }
      }
    }
    return [...set].sort((a, b) => {
      const numA = parseInt(a.replace(/\D/g, '')) || 0
      const numB = parseInt(b.replace(/\D/g, '')) || 0
      return numA - numB
    })
  })()

  // ── Filter indicators for machine view ──
  const filteredMachineIndicators = (() => {
    if (viewMode !== 'machine' || !allIndicators?.indicators) return []
    if (machineFilter === 'all') return allIndicators.indicators
    return allIndicators.indicators.filter(ind => {
      if (!ind.subtitle) return false
      const sub = ind.subtitle.toLowerCase()
      const filter = machineFilter.toLowerCase()
      // Direct match
      if (sub === filter) return true
      // Partial match for combined subtitles like "KDF 7, 9, 10"
      if (sub.includes(',')) {
        const parts = sub.split(',').map(p => p.trim())
        const filterNum = filter.replace(/\D/g, '')
        return parts.some(p => {
          if (p === filter) return true
          if (/^\d+$/.test(p) && p === filterNum) return true
          return false
        })
      }
      return false
    })
  })()

  // ══════════════════════════════════════════════════════
  // Empty state — no categories yet
  // ══════════════════════════════════════════════════════
  if (!cedulasLoading && categories.length === 0 && cedulaId) {
    return (
      <div className="space-y-6">
        <PageHeader cedulas={cedulas} cedulaId={cedulaId} setCedulaId={setCedulaId} cedulasLoading={cedulasLoading} />
        {error && <ErrorBanner error={error} onClose={() => setError(null)} />}
        <div className="bg-[#0f1d32] rounded-xl border border-white/5 px-6 py-16 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-500/10 rounded-full mb-4">
            <svg className="w-8 h-8 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
            </svg>
          </div>
          <p className="text-lg font-semibold text-white mb-2">Weekly DOS/DCS</p>
          <p className="text-sm text-slate-400 mb-6">
            No hay indicadores configurados. Haz clic para crear las 10 categorías y ~70 indicadores iniciales.
          </p>
          <button onClick={handleSeed} disabled={seeding}
            className="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium text-sm transition-colors disabled:opacity-50">
            {seeding ? 'Creando indicadores...' : 'Inicializar Weekly'}
          </button>
        </div>
      </div>
    )
  }

  // ══════════════════════════════════════════════════════
  // Main render
  // ══════════════════════════════════════════════════════
  return (
    <div className="space-y-4">
      <PageHeader cedulas={cedulas} cedulaId={cedulaId} setCedulaId={setCedulaId} cedulasLoading={cedulasLoading} />
      {error && <ErrorBanner error={error} onClose={() => setError(null)} />}

      {/* Controls bar */}
      <div className="flex flex-wrap items-center gap-3">
        {/* View mode toggle */}
        <div className="flex items-center gap-1 bg-[#0a1628] rounded-lg p-1">
          <button onClick={() => setViewMode('category')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              viewMode === 'category' ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-white'
            }`}>
            Por categoría
          </button>
          <button onClick={() => setViewMode('machine')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              viewMode === 'machine' ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-white'
            }`}>
            Por máquina
          </button>
        </div>

        <div className="w-px h-6 bg-white/10" />

        {/* Quarter */}
        <div className="flex items-center gap-1 bg-[#0a1628] rounded-lg p-1">
          {[1,2,3,4].map(q => (
            <button key={q} onClick={() => { setQuarter(q); const w = getQuarterWeeks(q); setWeekStart(w.start); setWeekEnd(w.end) }}
              className={`px-4 py-1.5 rounded-md text-xs font-medium transition-colors ${
                quarter === q ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'
              }`}>
              Q{q}
            </button>
          ))}
        </div>
        {/* Year */}
        <div className="flex items-center gap-1 bg-[#0a1628] rounded-lg p-1">
          {[year - 1, year, year + 1].map(y => (
            <button key={y} onClick={() => setYear(y)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                year === y ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'
              }`}>
              {y}
            </button>
          ))}
        </div>
        {/* Week range */}
        <div className="flex items-center gap-1 text-xs text-slate-500">
          Sem:
          <input type="number" min={1} max={52} value={weekStart}
            onChange={e => setWeekStart(Math.max(1, Math.min(52, parseInt(e.target.value) || 1)))}
            className="w-12 bg-[#0a1628] border border-white/10 rounded px-2 py-1 text-white text-center" />
          <span>a</span>
          <input type="number" min={1} max={52} value={weekEnd}
            onChange={e => setWeekEnd(Math.max(1, Math.min(52, parseInt(e.target.value) || 13)))}
            className="w-12 bg-[#0a1628] border border-white/10 rounded px-2 py-1 text-white text-center" />
        </div>

        <div className="flex-1" />

        {/* Edit mode */}
        {editMode ? (
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500">
              Modo: <span className={editMode === 'values' ? 'text-blue-400' : 'text-amber-400'}>
                {editMode === 'values' ? 'Captura de valores' : 'Edición de targets'}
              </span>
            </span>
            <button onClick={() => { setEditMode(null); setPendingChanges({}) }}
              className="px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-700/50 text-slate-300 hover:bg-slate-700 transition-colors">
              Cancelar
            </button>
            {hasChanges && (
              <button onClick={handleSave} disabled={saving}
                className="px-4 py-1.5 rounded-lg text-xs font-medium bg-green-600 hover:bg-green-500 text-white transition-colors disabled:opacity-50">
                {saving ? 'Guardando...' : `Guardar (${Object.values(pendingChanges).reduce((s, m) => s + Object.keys(m).length, 0)})`}
              </button>
            )}
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <button onClick={() => setEditMode('values')}
              className="px-3 py-1.5 rounded-lg text-xs font-medium bg-blue-600/20 text-blue-400 hover:bg-blue-600/30 border border-blue-500/30 transition-colors">
              Capturar valores
            </button>
            <button onClick={() => setEditMode('targets')}
              className="px-3 py-1.5 rounded-lg text-xs font-medium bg-amber-600/20 text-amber-400 hover:bg-amber-600/30 border border-amber-500/30 transition-colors">
              Editar targets
            </button>
            <button onClick={handleSeedExtras} disabled={seeding}
              className="px-3 py-1.5 rounded-lg text-xs font-medium bg-emerald-600/20 text-emerald-400 hover:bg-emerald-600/30 border border-emerald-500/30 transition-colors disabled:opacity-50">
              {seeding ? 'Agregando...' : '+ Agregar faltantes'}
            </button>
          </div>
        )}
      </div>

      {/* Category tabs (only in category view) */}
      {viewMode === 'category' && (
        <div className="overflow-x-auto pb-1">
          <div className="flex gap-1 bg-[#0a1628] rounded-lg p-1 w-fit min-w-full">
            {categories.map(cat => (
              <button key={cat.id} onClick={() => setActiveCatId(cat.id)}
                className={`px-4 py-2 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
                  activeCatId === cat.id ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'
                }`}>
                {cat.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Machine filter (only in machine view) */}
      {viewMode === 'machine' && (
        <div className="overflow-x-auto pb-1">
          <div className="flex gap-1 bg-[#0a1628] rounded-lg p-1 w-fit min-w-full">
            <button onClick={() => setMachineFilter('all')}
              className={`px-4 py-2 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
                machineFilter === 'all' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white'
              }`}>
              Todas
            </button>
            {machineList.map(m => (
              <button key={m} onClick={() => setMachineFilter(m)}
                className={`px-4 py-2 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
                  machineFilter === m ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white'
                }`}>
                {m}
              </button>
            ))}
          </div>
          {machineFilter !== 'all' && (
            <p className="text-xs text-cyan-400/70 mt-1 ml-1">
              {filteredMachineIndicators.length} indicador{filteredMachineIndicators.length !== 1 ? 'es' : ''} para {machineFilter}
            </p>
          )}
        </div>
      )}

      {/* Charts grid */}
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : viewMode === 'category' ? (
        // ── Category view ──
        chartData?.indicators?.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {chartData.indicators.map(ind => (
              <WeeklyChart
                key={ind.id}
                indicator={ind}
                targets={chartData.targets[ind.id] || {}}
                values={chartData.values[ind.id] || {}}
                weekStart={weekStart} weekEnd={weekEnd}
                editMode={editMode}
                pendingChanges={pendingChanges[ind.id] || {}}
                onCellChange={(wk, val) => handleCellChange(ind.id, wk, val)}
                customYRange={customYRanges[ind.id]}
                onYRangeChange={(range) => handleYRangeChange(ind.id, range)}
                onToggleDirection={() => handleToggleDirection(ind.id)}
                onBandSizeChange={(bs) => handleBandSizeChange(ind.id, bs)}
              />
            ))}
          </div>
        ) : (
          <div className="bg-[#0f1d32] rounded-xl border border-white/5 px-6 py-12 text-center">
            <p className="text-slate-500 text-sm">No hay indicadores en esta categoría</p>
          </div>
        )
      ) : (
        // ── Machine view ──
        filteredMachineIndicators.length > 0 ? (
          <div className="space-y-6">
            {/* Group by category for cleaner display */}
            {(() => {
              const groups = {}
              for (const ind of filteredMachineIndicators) {
                const catName = categories.find(c => c.id === ind.category_id)?.name || 'Sin categoría'
                if (!groups[catName]) groups[catName] = []
                groups[catName].push(ind)
              }
              return Object.entries(groups).map(([catName, inds]) => (
                <div key={catName}>
                  <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3 px-1">{catName}</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                    {inds.map(ind => (
                      <WeeklyChart
                        key={ind.id}
                        indicator={ind}
                        targets={allIndicators?.targets[ind.id] || {}}
                        values={allIndicators?.values[ind.id] || {}}
                        weekStart={weekStart} weekEnd={weekEnd}
                        editMode={editMode}
                        pendingChanges={pendingChanges[ind.id] || {}}
                        onCellChange={(wk, val) => handleCellChange(ind.id, wk, val)}
                        customYRange={customYRanges[ind.id]}
                        onYRangeChange={(range) => handleYRangeChange(ind.id, range)}
                        onToggleDirection={() => handleToggleDirection(ind.id)}
                        onBandSizeChange={(bs) => handleBandSizeChange(ind.id, bs)}
                      />
                    ))}
                  </div>
                </div>
              ))
            })()}
          </div>
        ) : (
          <div className="bg-[#0f1d32] rounded-xl border border-white/5 px-6 py-12 text-center">
            <p className="text-slate-500 text-sm">
              {allIndicators ? 'No hay indicadores para esta máquina' : 'Cargando indicadores...'}
            </p>
          </div>
        )
      )}
    </div>
  )
}


// ══════════════════════════════════════════════════════
// WeeklyChart — Single indicator card
// ══════════════════════════════════════════════════════

function WeeklyChart({ indicator, targets, values, weekStart, weekEnd, editMode, pendingChanges, onCellChange, customYRange, onYRangeChange, onToggleDirection, onBandSizeChange }) {
  const [showTable, setShowTable] = useState(false)
  const [showYConfig, setShowYConfig] = useState(false)
  const isHigherBetter = indicator.direction === 'higher_better'
  const isMiddleBetter = indicator.direction === 'middle_better'
  const weekNumbers = Array.from({ length: weekEnd - weekStart + 1 }, (_, i) => weekStart + i)

  // Y-axis range — auto or custom
  let allVals = []
  weekNumbers.forEach(w => {
    if (targets[w] != null) allVals.push(parseFloat(targets[w]))
    if (values[w]?.value != null) allVals.push(parseFloat(values[w].value))
    if (pendingChanges[w] != null && pendingChanges[w] !== '') allVals.push(parseFloat(pendingChanges[w]))
  })
  let autoMax = allVals.length > 0 ? Math.max(...allVals) * 1.3 : 100
  if (indicator.unit === '%') autoMax = Math.max(autoMax, 100)
  if (autoMax <= 0) autoMax = 10
  const autoMin = 0

  const yMin = customYRange?.min != null ? customYRange.min : autoMin
  const yMax = customYRange?.max != null ? customYRange.max : autoMax

  const chartArr = weekNumbers.map(w => {
    const tgt = targets[w] != null ? parseFloat(targets[w]) : null
    const val = values[w]?.value != null ? parseFloat(values[w].value) : null
    const displayTarget = editMode === 'targets' && pendingChanges[w] != null && pendingChanges[w] !== ''
      ? parseFloat(pendingChanges[w]) : tgt
    const displayValue = editMode === 'values' && pendingChanges[w] != null && pendingChanges[w] !== ''
      ? parseFloat(pendingChanges[w]) : val

    const clampedTarget = displayTarget != null
      ? Math.min(Math.max(displayTarget, yMin), yMax) : null

    const halfBand = (indicator.band_size ?? 10) / 2

    if (isMiddleBetter) {
      // 3-layer: red bg → green band overlay → red re-cover below band
      // Center on target if available, otherwise default to 0
      const center = clampedTarget ?? 0
      const bandTop = Math.min(center + halfBand, yMax)
      const bandBot = Math.max(center - halfBand, yMin)
      return {
        week: `${w}`, weekNum: w,
        bg: yMax,            // Layer 1: red fills everything
        greenTop: bandTop,   // Layer 2: green covers yMin → bandTop
        redBottom: bandBot,  // Layer 3: red re-covers yMin → bandBot
        actual: displayValue, target: displayTarget,
      }
    }

    // Standard 2-layer: bg color + overlay
    return {
      week: `${w}`, weekNum: w,
      bg: yMax,
      overlay: clampedTarget ?? yMax,
      actual: displayValue, target: displayTarget,
    }
  })

  // Smart Y-axis tick formatter — uses decimals when range is small
  const yRange = yMax - yMin
  const formatYTick = (v) => {
    if (indicator.unit === '%') {
      return yRange < 2 ? `${v.toFixed(2)}%` : `${Math.round(v)}%`
    }
    if (yRange < 0.1)  return v.toFixed(4)
    if (yRange < 1)    return v.toFixed(3)
    if (yRange < 10)   return v.toFixed(1)
    return Math.round(v)
  }

  const dirLabel = isMiddleBetter ? '↔ Medio=Mejor' : isHigherBetter ? '↑ Mayor=Mejor' : '↓ Menor=Mejor'
  const dirColor = isMiddleBetter ? 'text-cyan-400' : isHigherBetter ? 'text-green-400' : 'text-yellow-400'

  const ChartTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null
    const d = payload[0]?.payload
    return (
      <div className="bg-[#0a1628] border border-white/10 rounded-lg px-3 py-2 text-xs">
        <p className="text-white font-medium mb-1">Semana {d?.weekNum}</p>
        {d?.actual != null && <p className="text-blue-300">Valor: {d.actual}{indicator.unit === '%' ? '%' : ` ${indicator.unit}`}</p>}
        {d?.target != null && <p className="text-slate-400">Target: {d.target}{indicator.unit === '%' ? '%' : ` ${indicator.unit}`}</p>}
      </div>
    )
  }

  return (
    <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-4">
      <div className="flex items-start justify-between mb-2">
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-semibold text-white truncate">{indicator.name}</h4>
          <div className="flex items-center gap-2 mt-0.5">
            {indicator.subtitle && <span className="text-xs text-blue-400 font-medium">{indicator.subtitle}</span>}
            <button onClick={onToggleDirection} title="Clic para invertir zonas rojo/verde"
              className={`text-[10px] ${dirColor} hover:opacity-70 transition-opacity cursor-pointer`}>
              {dirLabel}
            </button>
          </div>
        </div>
        <div className="flex items-center gap-1 shrink-0 ml-2">
          <button onClick={() => setShowYConfig(c => !c)}
            title="Configurar eje Y"
            className={`text-xs transition-colors px-1 ${showYConfig || customYRange ? 'text-amber-400' : 'text-slate-600 hover:text-slate-400'}`}>
            ⚙
          </button>
          <button onClick={() => setShowTable(t => !t)}
            className="text-xs text-slate-500 hover:text-slate-300 transition-colors">
            {showTable ? '📊 Gráfica' : '📋 Tabla'}
          </button>
        </div>
      </div>

      {showYConfig && (
        <div className="flex flex-wrap items-center gap-2 mb-2 px-1">
          <span className="text-[10px] text-slate-500">Eje Y:</span>
          <input type="number" step="any" placeholder="Mín"
            value={customYRange?.min ?? ''}
            onChange={e => onYRangeChange({ ...customYRange, min: e.target.value === '' ? null : parseFloat(e.target.value) })}
            className="w-14 bg-[#0a1628] border border-white/10 rounded px-1.5 py-0.5 text-[10px] text-white text-center focus:outline-none focus:border-amber-500/50" />
          <span className="text-[10px] text-slate-600">—</span>
          <input type="number" step="any" placeholder="Máx"
            value={customYRange?.max ?? ''}
            onChange={e => onYRangeChange({ ...customYRange, max: e.target.value === '' ? null : parseFloat(e.target.value) })}
            className="w-14 bg-[#0a1628] border border-white/10 rounded px-1.5 py-0.5 text-[10px] text-white text-center focus:outline-none focus:border-amber-500/50" />
          {customYRange && (
            <button onClick={() => onYRangeChange(undefined)}
              className="text-[10px] text-slate-600 hover:text-red-400 transition-colors">Auto</button>
          )}
          {isMiddleBetter && (
            <>
              <span className="text-[10px] text-slate-600 ml-1">|</span>
              <span className="text-[10px] text-cyan-500">Banda:</span>
              <input type="number" step="any" placeholder="Ancho"
                value={indicator.band_size ?? ''}
                onChange={e => onBandSizeChange(e.target.value === '' ? null : parseFloat(e.target.value))}
                className="w-14 bg-[#0a1628] border border-cyan-500/30 rounded px-1.5 py-0.5 text-[10px] text-cyan-300 text-center focus:outline-none focus:border-cyan-400" />
            </>
          )}
        </div>
      )}

      {showTable ? (
        <DataTable indicator={indicator} weekNumbers={weekNumbers} targets={targets} values={values}
          editMode={editMode} pendingChanges={pendingChanges} onCellChange={onCellChange} />
      ) : (
        <div className="h-[160px]">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartArr} margin={{ left: -15, right: 5, top: 5, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
              <XAxis dataKey="week" tick={{ fill: '#475569', fontSize: 9 }} axisLine={false} tickLine={false}
                interval={Math.max(0, Math.floor(weekNumbers.length / 8) - 1)} />
              <YAxis domain={[yMin, yMax]} allowDataOverflow={true} tick={{ fill: '#475569', fontSize: 9 }} axisLine={false} tickLine={false}
                tickFormatter={formatYTick} />
              <Tooltip content={<ChartTooltip />} />
              {isMiddleBetter ? (
                <>
                  {/* Middle-better: 3 layers — red bg → green band → red re-cover bottom */}
                  <Area type="stepAfter" dataKey="bg" baseValue={yMin}
                    fill="#c82828" fillOpacity={1} stroke="none" isAnimationActive={false} />
                  <Area type="stepAfter" dataKey="greenTop" baseValue={yMin}
                    fill="#22a34d" fillOpacity={1} stroke="none" isAnimationActive={false} />
                  <Area type="stepAfter" dataKey="redBottom" baseValue={yMin}
                    fill="#c82828" fillOpacity={1} stroke="none" isAnimationActive={false} />
                </>
              ) : (
                <>
                  {/* Standard 2-layer: bg color + overlay */}
                  <Area type="stepAfter" dataKey="bg" baseValue={yMin}
                    fill={isHigherBetter ? '#22a34d' : '#c82828'}
                    fillOpacity={1} stroke="none" isAnimationActive={false} />
                  <Area type="stepAfter" dataKey="overlay" baseValue={yMin}
                    fill={isHigherBetter ? '#c82828' : '#22a34d'}
                    fillOpacity={1} stroke="none" isAnimationActive={false} />
                </>
              )}
              {/* Zero reference line — visible when range includes negatives */}
              {yMin < 0 && yMax > 0 && (
                <ReferenceLine y={0} stroke="rgba(255,255,255,0.4)" strokeDasharray="4 3" strokeWidth={1} />
              )}
              <Line type="monotone" dataKey="actual" stroke="#60a5fa" strokeWidth={2}
                dot={{ r: 2.5, fill: '#60a5fa', strokeWidth: 0 }}
                connectNulls={false} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {editMode && !showTable && (
        <p className="text-[10px] text-slate-600 mt-1 text-center">
          Clic en "Tabla" para editar {editMode === 'values' ? 'valores' : 'targets'}
        </p>
      )}
    </div>
  )
}


// ══════════════════════════════════════════════════════
// DataTable — Editable data for one indicator
// ══════════════════════════════════════════════════════

function DataTable({ indicator, weekNumbers, targets, values, editMode, pendingChanges, onCellChange }) {
  const isHigher = indicator.direction === 'higher_better'

  return (
    <div className="overflow-x-auto -mx-1 max-h-[300px]">
      <table className="w-full text-[10px]">
        <thead className="sticky top-0 bg-[#0f1d32] z-10">
          <tr className="border-b border-white/10">
            <th className="px-1.5 py-1 text-left text-slate-500 font-medium w-10">Sem</th>
            <th className="px-1.5 py-1 text-center text-slate-500 font-medium">Target</th>
            <th className="px-1.5 py-1 text-center text-slate-500 font-medium">Valor</th>
            <th className="px-1.5 py-1 text-center text-slate-500 font-medium w-8"></th>
          </tr>
        </thead>
        <tbody>
          {weekNumbers.map(w => {
            const t = targets[w] != null ? parseFloat(targets[w]) : null
            const v = values[w]?.value != null ? parseFloat(values[w].value) : null
            const pendT = editMode === 'targets' && pendingChanges[w] != null ? pendingChanges[w] : null
            const pendV = editMode === 'values' && pendingChanges[w] != null ? pendingChanges[w] : null
            const dispT = pendT != null ? (pendT === '' ? null : parseFloat(pendT)) : t
            const dispV = pendV != null ? (pendV === '' ? null : parseFloat(pendV)) : v

            let dot = null
            if (dispV != null && dispT != null) {
              const isMiddle = indicator.direction === 'middle_better'
              let ok
              if (isMiddle) {
                const hb = (indicator.band_size ?? 10) / 2
                ok = dispV >= (dispT - hb) && dispV <= (dispT + hb)
              } else {
                ok = isHigher ? dispV >= dispT : dispV <= dispT
              }
              dot = ok ? 'text-green-400' : 'text-red-400'
            }

            return (
              <tr key={w} className="border-b border-white/5 hover:bg-white/[0.02]">
                <td className="px-1.5 py-1 text-slate-400 font-medium">S{w}</td>
                <td className="px-1.5 py-1 text-center">
                  {editMode === 'targets' ? (
                    <input type="number" step="any" value={pendingChanges[w] ?? (t ?? '')}
                      onChange={e => onCellChange(w, e.target.value)}
                      className="w-16 bg-amber-950/30 border border-amber-500/30 rounded px-1 py-0.5 text-amber-300 text-center text-[10px] focus:outline-none focus:border-amber-400" />
                  ) : (
                    <span className="text-slate-400">{t != null ? t : '—'}</span>
                  )}
                </td>
                <td className="px-1.5 py-1 text-center">
                  {editMode === 'values' ? (
                    <input type="number" step="any" value={pendingChanges[w] ?? (v ?? '')}
                      onChange={e => onCellChange(w, e.target.value)}
                      className="w-16 bg-blue-950/30 border border-blue-500/30 rounded px-1 py-0.5 text-blue-300 text-center text-[10px] focus:outline-none focus:border-blue-400" />
                  ) : (
                    <span className="text-white font-medium">{v != null ? v : '—'}</span>
                  )}
                </td>
                <td className="px-1.5 py-1 text-center">
                  {dot ? <span className={dot}>●</span> : <span className="text-slate-700">—</span>}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}


// ══════════════════════════════════════════════════════
// Shared
// ══════════════════════════════════════════════════════

function PageHeader({ cedulas, cedulaId, setCedulaId, cedulasLoading }) {
  return (
    <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
      <div>
        <h1 className="text-xl font-bold text-white">Weekly DOS/DCS</h1>
        <p className="text-sm text-slate-400">Indicadores semanales por máquina — gráficas rojo/verde con targets configurables</p>
      </div>
      <CedulaSelector cedulas={cedulas} selectedCedulaId={cedulaId} onChange={setCedulaId} loading={cedulasLoading} />
    </div>
  )
}

function ErrorBanner({ error, onClose }) {
  return (
    <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-red-400 text-sm flex items-center gap-2">
      <span className="flex-1">{error}</span>
      <button onClick={onClose} className="text-red-300 hover:text-white">✕</button>
    </div>
  )
}
