import { useState, useEffect, useCallback, useRef } from 'react'
import WeekSelector from '../components/WeekSelector'
import FilterBar from '../components/FilterBar'
import { getSemaforoColor } from '../components/SemaforoIndicador'
import * as api from '../lib/api'

// Mapping: indicador config → campo API
const INDICADORES = [
  { categoria: 'Sustentabilidad', nombre: 'BOS [#]', campo: 'bos_num', unidad: '#', target: 3, direccion: '≥' },
  { categoria: 'Sustentabilidad', nombre: 'BOS ENG [%]', campo: 'bos_eng', unidad: '%', target: 95, direccion: '≥' },
  { categoria: 'Calidad', nombre: 'QBOS [#]', campo: 'qbos_num', unidad: '#', target: 1, direccion: '≥' },
  { categoria: 'Calidad', nombre: 'QBOS ENG [%]', campo: 'qbos_eng', unidad: '%', target: 95, direccion: '≥' },
  { categoria: 'Calidad', nombre: 'QFlags [#]', campo: 'qflags_num', unidad: '#', target: 6, direccion: '≥' },
  { categoria: 'Calidad', nombre: 'QI/PNC [#]', campo: 'qi_pnc_num', unidad: '#', target: 0, direccion: '=' },
  { categoria: 'Desempeno', nombre: 'DH Enc [#]', campo: 'dh_encontrados', unidad: '#', target: 14, direccion: '≥' },
  { categoria: 'Desempeno', nombre: 'DH Rep [#]', campo: 'dh_reparados', unidad: '#', target: 14, direccion: '≥' },
  { categoria: 'Desempeno', nombre: 'Curva Aut [%]', campo: 'curva_autonomia', unidad: '%', target: 80, direccion: '≥' },
  { categoria: 'Desempeno', nombre: 'Contram [%]', campo: 'contramedidas_defectos', unidad: '%', target: 100, direccion: '≥' },
  { categoria: 'Desempeno', nombre: 'IPS [#]', campo: 'ips_num', unidad: '#', target: 1, direccion: '≥' },
  { categoria: 'Costo', nombre: 'FRR [%]', campo: 'frr', unidad: '%', target: 0.1, direccion: '≤' },
  { categoria: 'Costo', nombre: 'DIM WASTE [%]', campo: 'dim_waste', unidad: '%', target: null, direccion: '≤' },
  { categoria: 'Costo', nombre: 'Sobrep [#]', campo: 'sobrepeso', unidad: '#', target: -20, direccion: '≤' },
  { categoria: 'Costo', nombre: 'LAIKA [#]', campo: 'eventos_laika', unidad: '#', target: 0, direccion: '=' },
  { categoria: 'Moral', nombre: 'Casos Est [#]', campo: 'casos_estudio', unidad: '#', target: 1, direccion: '≥' },
  { categoria: 'Moral', nombre: 'QM Target [%]', campo: 'qm_on_target', unidad: '%', target: 80, direccion: '≥' },
]

const CATEGORIAS = ['Sustentabilidad', 'Calidad', 'Desempeno', 'Costo', 'Moral']
const CAT_LABELS = { Sustentabilidad: 'S', Calidad: 'Q', Desempeno: 'D', Costo: 'C', Moral: 'M' }
const CAT_COLORS = {
  Sustentabilidad: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  Calidad: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  Desempeno: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  Costo: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  Moral: 'bg-pink-500/20 text-pink-400 border-pink-500/30',
}

const semaforoBg = {
  green: 'bg-green-500/10',
  yellow: 'bg-yellow-500/10',
  red: 'bg-red-500/10',
  neutral: ''
}

// Default cédula ID (Filtros Blancos) — will be dynamic in Fase 3
const CEDULA_ID = null // Will be fetched

export default function Captura() {
  const [cedulaId, setCedulaId] = useState(null)
  const [semana, setSemana] = useState(getClosestMonday())
  const [lcs, setLcs] = useState([])
  const [selectedLC, setSelectedLC] = useState(null)
  const [rows, setRows] = useState([]) // { operador, registro, dirty: {} }
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [saveMsg, setSaveMsg] = useState(null)
  const [error, setError] = useState(null)
  const dirtyCount = rows.reduce((n, r) => n + Object.keys(r.dirty || {}).length, 0)

  // Fetch cédula on mount
  useEffect(() => {
    api.getCedulas().then(res => {
      if (res.data?.length > 0) {
        setCedulaId(res.data[0].id)
      }
    }).catch(err => setError('Error cargando cedulas: ' + err.message))
  }, [])

  // Fetch LCs when cédula changes
  useEffect(() => {
    if (!cedulaId) return
    api.getLineCoordinators(cedulaId).then(res => setLcs(res.data || []))
      .catch(err => console.error('Error cargando LCs:', err))
  }, [cedulaId])

  // Fetch data when semana/lc/cedula changes
  useEffect(() => {
    if (!cedulaId || !semana) return
    setLoading(true)
    setError(null)

    api.getOperadoresSemana(cedulaId, semana, selectedLC)
      .then(res => {
        const data = (res.data || []).map(item => ({
          operador: item.operador,
          registro: item.registro,
          dirty: {}
        }))
        setRows(data)
      })
      .catch(err => setError('Error cargando datos: ' + err.message))
      .finally(() => setLoading(false))
  }, [cedulaId, semana, selectedLC])

  function handleCellChange(rowIdx, campo, value) {
    setRows(prev => {
      const updated = [...prev]
      const row = { ...updated[rowIdx] }
      const reg = { ...(row.registro || {}), [campo]: value === '' ? null : Number(value) }
      row.registro = reg
      row.dirty = { ...(row.dirty || {}), [campo]: true }
      updated[rowIdx] = row
      return updated
    })
    setSaveMsg(null)
  }

  async function handleSave() {
    setSaving(true)
    setSaveMsg(null)
    try {
      const registros = rows
        .filter(r => Object.keys(r.dirty || {}).length > 0)
        .map(r => {
          const data = { operador_id: r.operador.id, semana }
          INDICADORES.forEach(ind => {
            const val = r.registro?.[ind.campo]
            if (val != null) data[ind.campo] = val
          })
          return data
        })

      if (registros.length === 0) {
        setSaveMsg({ type: 'info', text: 'No hay cambios para guardar.' })
        setSaving(false)
        return
      }

      await api.crearRegistrosBatch(registros)
      // Clear dirty flags
      setRows(prev => prev.map(r => ({ ...r, dirty: {} })))
      setSaveMsg({ type: 'success', text: `${registros.length} registro(s) guardados correctamente.` })
    } catch (err) {
      setSaveMsg({ type: 'error', text: 'Error al guardar: ' + err.message })
    }
    setSaving(false)
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white">Captura Semanal</h1>
          <p className="text-sm text-slate-400">Registra los indicadores SQDCM por operador</p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <WeekSelector value={semana} onChange={setSemana} />
          <FilterBar lineCoordinators={lcs} selectedLC={selectedLC} onLCChange={setSelectedLC} />
        </div>
      </div>

      {/* Save bar */}
      <div className="flex items-center gap-3 bg-[#0f1d32] rounded-lg px-4 py-3 border border-white/5">
        <button
          onClick={handleSave}
          disabled={saving || dirtyCount === 0}
          className="px-4 py-2 bg-green-600 hover:bg-green-500 disabled:bg-slate-600 disabled:text-slate-400 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-2"
        >
          {saving ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Guardando...
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              Guardar todo
            </>
          )}
        </button>
        {dirtyCount > 0 && (
          <span className="text-xs text-yellow-400">
            {dirtyCount} celda(s) modificada(s)
          </span>
        )}
        {saveMsg && (
          <span className={`text-xs ${
            saveMsg.type === 'success' ? 'text-green-400' :
            saveMsg.type === 'error' ? 'text-red-400' : 'text-slate-400'
          }`}>
            {saveMsg.text}
          </span>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Table */}
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : (
        <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              {/* Category headers */}
              <thead>
                <tr className="border-b border-white/5">
                  <th className="sticky left-0 z-20 bg-[#0f1d32] px-3 py-2 text-left text-slate-400 font-medium" rowSpan={2}>
                    Operador
                  </th>
                  <th className="sticky left-[140px] z-20 bg-[#0f1d32] px-2 py-2 text-left text-slate-400 font-medium text-xs" rowSpan={2}>
                    LC
                  </th>
                  {CATEGORIAS.map(cat => {
                    const count = INDICADORES.filter(i => i.categoria === cat).length
                    return (
                      <th
                        key={cat}
                        colSpan={count}
                        className={`px-2 py-1.5 text-center text-xs font-bold border-b border-white/10 ${CAT_COLORS[cat]} border-l border-r`}
                      >
                        {cat === 'Desempeno' ? 'Desempeño' : cat}
                      </th>
                    )
                  })}
                </tr>
                <tr className="border-b border-white/10">
                  {INDICADORES.map((ind, i) => (
                    <th
                      key={ind.campo}
                      className="px-1 py-1.5 text-center text-xs text-slate-300 font-medium whitespace-nowrap border-l border-white/5/50"
                      title={`Target: ${ind.target ?? 'N/A'} (${ind.direccion})`}
                    >
                      <div>{ind.nombre}</div>
                      <div className="text-[10px] text-slate-500 font-normal">
                        {ind.target != null ? `${ind.direccion}${ind.target}` : '—'}
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.length === 0 ? (
                  <tr>
                    <td colSpan={2 + INDICADORES.length} className="px-4 py-12 text-center text-slate-500">
                      No hay operadores para esta semana / filtro.
                    </td>
                  </tr>
                ) : (
                  rows.map((row, rowIdx) => {
                    const lcName = row.operador?.line_coordinators?.nombre || '—'
                    return (
                      <tr key={row.operador?.id || rowIdx} className="border-b border-white/5/50 hover:bg-slate-700/20">
                        <td className="sticky left-0 z-10 bg-[#0f1d32] px-3 py-2 text-white font-medium whitespace-nowrap max-w-[140px] truncate">
                          {row.operador?.nombre || 'Sin nombre'}
                        </td>
                        <td className="sticky left-[140px] z-10 bg-[#0f1d32] px-2 py-2 text-slate-400 text-xs whitespace-nowrap max-w-[100px] truncate">
                          {lcName}
                        </td>
                        {INDICADORES.map((ind) => {
                          const val = row.registro?.[ind.campo]
                          const isDirty = row.dirty?.[ind.campo]
                          const color = getSemaforoColor(val, ind.target, ind.direccion)
                          return (
                            <td
                              key={ind.campo}
                              className={`px-0.5 py-1 border-l border-white/5/30 semaforo-cell ${semaforoBg[color]}`}
                            >
                              <input
                                type="number"
                                step={ind.unidad === '%' ? '0.1' : '1'}
                                value={val ?? ''}
                                onChange={(e) => handleCellChange(rowIdx, ind.campo, e.target.value)}
                                className={`w-16 px-1.5 py-1 text-center text-xs rounded bg-transparent border ${
                                  isDirty ? 'border-yellow-500/50' : 'border-transparent'
                                } text-white focus:outline-none focus:border-blue-500 focus:bg-slate-900/50 hover:border-slate-500 transition-colors`}
                                placeholder="—"
                              />
                            </td>
                          )
                        })}
                      </tr>
                    )
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-4 text-xs text-slate-500">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full bg-green-400" /> En objetivo
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full bg-yellow-400" /> Cerca del objetivo
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full bg-red-400" /> Fuera de objetivo
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded border border-yellow-500/50" /> Celda modificada
        </span>
      </div>
    </div>
  )
}

function getClosestMonday() {
  const d = new Date()
  const day = d.getDay()
  const diff = d.getDate() - day + (day === 0 ? -6 : 1)
  const monday = new Date(d.setDate(diff))
  return monday.toISOString().split('T')[0]
}
