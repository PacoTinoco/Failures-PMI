import { useState, useEffect } from 'react'
import WeekSelector from '../components/WeekSelector'
import FilterBar from '../components/FilterBar'
import CedulaSelector from '../components/CedulaSelector'
import { getSemaforoColor } from '../components/SemaforoIndicador'
import * as api from '../lib/api'

// tipo: 'auto' = alimentado automáticamente por la plataforma
//       'manual' = el usuario captura el valor manualmente
//       null = sin clasificar por ahora
const CATEGORIAS = [
  {
    key: 'Sustentabilidad',
    label: 'Sustentabilidad',
    letter: 'S',
    color: 'from-emerald-500 to-emerald-600',
    bgLight: 'bg-emerald-500/10',
    textColor: 'text-emerald-400',
    indicadores: [
      { nombre: 'BOS [#]',     campo: 'bos_num',  target: 3,  direccion: '≥', unidad: '#', tipo: 'auto' },
      { nombre: 'BOS ENG [%]', campo: 'bos_eng',  target: 95, direccion: '≥', unidad: '%', tipo: 'auto' },
    ]
  },
  {
    key: 'Calidad',
    label: 'Calidad',
    letter: 'Q',
    color: 'from-blue-500 to-blue-600',
    bgLight: 'bg-blue-500/10',
    textColor: 'text-blue-400',
    indicadores: [
      { nombre: 'QBOS [#]',      campo: 'qbos_num',   target: 1,  direccion: '≥', unidad: '#', tipo: 'auto'   },
      { nombre: 'QBOS ENG [%]',  campo: 'qbos_eng',   target: 95, direccion: '≥', unidad: '%', tipo: 'auto'   },
      { nombre: 'QFlags [#]',    campo: 'qflags_num', target: 6,  direccion: '≥', unidad: '#', tipo: 'manual' },
      { nombre: 'QI/PNC [#]',    campo: 'qi_pnc_num', target: 0,  direccion: '=', unidad: '#', tipo: null     },
    ]
  },
  {
    key: 'Desempeno',
    label: 'Desempeño',
    letter: 'D',
    color: 'from-purple-500 to-purple-600',
    bgLight: 'bg-purple-500/10',
    textColor: 'text-purple-400',
    indicadores: [
      { nombre: 'DH Encontrados [#]', campo: 'dh_encontrados',       target: 14,  direccion: '≥', unidad: '#', tipo: 'auto'   },
      { nombre: 'DH Reparados [#]',   campo: 'dh_reparados',         target: 14,  direccion: '≥', unidad: '#', tipo: 'auto'   },
      { nombre: 'Curva Autonomía [%]',campo: 'curva_autonomia',       target: 80,  direccion: '≥', unidad: '%', tipo: 'auto'   },
      { nombre: 'Contramedidas [%]',  campo: 'contramedidas_defectos',target: 100, direccion: '≥', unidad: '%', tipo: 'auto'   },
      { nombre: 'IPS [#]',            campo: 'ips_num',               target: 1,   direccion: '≥', unidad: '#', tipo: 'manual' },
    ]
  },
  {
    key: 'Costo',
    label: 'Costo',
    letter: 'C',
    color: 'from-orange-500 to-orange-600',
    bgLight: 'bg-orange-500/10',
    textColor: 'text-orange-400',
    indicadores: [
      { nombre: 'FRR [%]',          campo: 'frr',           target: 0.1,  direccion: '≤', unidad: '%', tipo: 'auto' },
      { nombre: 'DIM WASTE [%]',    campo: 'dim_waste',     target: null, direccion: '≤', unidad: '%', tipo: null   },
      { nombre: 'Sobrepeso [#]',    campo: 'sobrepeso',     target: -20,  direccion: '≤', unidad: '#', tipo: null   },
      { nombre: 'Eventos LAIKA [#]',campo: 'eventos_laika', target: 0,    direccion: '=', unidad: '#', tipo: null   },
    ]
  },
  {
    key: 'Moral',
    label: 'Moral',
    letter: 'M',
    color: 'from-pink-500 to-pink-600',
    bgLight: 'bg-pink-500/10',
    textColor: 'text-pink-400',
    indicadores: [
      { nombre: 'Casos Estudio [#]',  campo: 'casos_estudio',  target: 1,  direccion: '≥', unidad: '#', tipo: 'manual' },
      { nombre: 'QM On Target [%]',   campo: 'qm_on_target',   target: 80, direccion: '≥', unidad: '%', tipo: 'auto'   },
    ]
  }
]

function TipoBadge({ tipo, size = 'sm' }) {
  if (!tipo) return null
  if (tipo === 'auto') return (
    <span className={`inline-flex items-center gap-0.5 rounded font-medium bg-cyan-500/15 text-cyan-400 border border-cyan-500/20 ${size === 'xs' ? 'text-[8px] px-1 py-0' : 'text-[9px] px-1.5 py-0.5'}`}>
      ⚡ Auto
    </span>
  )
  return (
    <span className={`inline-flex items-center gap-0.5 rounded font-medium bg-amber-500/15 text-amber-400 border border-amber-500/20 ${size === 'xs' ? 'text-[8px] px-1 py-0' : 'text-[9px] px-1.5 py-0.5'}`}>
      ✏ Manual
    </span>
  )
}

const ALL_INDICADORES = CATEGORIAS.flatMap(c => c.indicadores)

const semaforoColors = {
  green: 'text-green-400',
  yellow: 'text-yellow-400',
  red: 'text-red-400',
  neutral: 'text-slate-500'
}

const semaforoBg = {
  green: 'bg-green-500/15',
  yellow: 'bg-yellow-500/15',
  red: 'bg-red-500/15',
  neutral: ''
}

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

export default function Dashboard() {
  const [cedulas, setCedulas] = useState([])
  const [cedulaId, setCedulaId] = useState(null)
  const [cedulasLoading, setCedulasLoading] = useState(true)
  const [semana, setSemana] = useState(getClosestMonday())
  const [lcs, setLcs] = useState([])
  const [selectedLC, setSelectedLC] = useState(null)
  const [resumenData, setResumenData] = useState([])
  const [operadoresData, setOperadoresData] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Fetch cedulas
  useEffect(() => {
    api.getCedulas().then(res => {
      setCedulas(res.data || [])
      if (res.data?.length > 0) setCedulaId(res.data[0].id)
    }).catch(err => setError(err.message))
      .finally(() => setCedulasLoading(false))
  }, [])

  // Fetch LCs
  useEffect(() => {
    if (!cedulaId) return
    api.getLineCoordinators(cedulaId).then(res => setLcs(res.data || []))
  }, [cedulaId])

  // Fetch data
  useEffect(() => {
    if (!cedulaId || !semana) return
    setLoading(true)
    setError(null)

    Promise.all([
      api.getResumenLC(cedulaId, semana, selectedLC),
      api.getOperadoresSemana(cedulaId, semana, selectedLC)
    ])
      .then(([resumen, operadores]) => {
        setResumenData(resumen.data || [])
        setOperadoresData(operadores.data || [])
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [cedulaId, semana, selectedLC])

  // Calculate averages from operadores data for SQDCM cards
  function calcPromedios() {
    const registros = operadoresData
      .map(d => d.registro)
      .filter(Boolean)

    if (registros.length === 0) return {}

    const promedios = {}
    ALL_INDICADORES.forEach(ind => {
      const values = registros
        .map(r => r[ind.campo])
        .filter(v => v != null && !isNaN(v))
      promedios[ind.campo] = values.length > 0
        ? values.reduce((a, b) => a + b, 0) / values.length
        : null
    })
    return promedios
  }

  const promedios = calcPromedios()
  const totalRegistros = operadoresData.filter(d => d.registro).length
  const totalOperadores = operadoresData.length

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white">Dashboard FTO</h1>
          <p className="text-sm text-slate-400">
            Resumen de indicadores SQDCM — {totalRegistros}/{totalOperadores} operadores con datos
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <CedulaSelector cedulas={cedulas} selectedCedulaId={cedulaId} onChange={setCedulaId} loading={cedulasLoading} />
          <WeekSelector value={semana} onChange={setSemana} />
          <FilterBar lineCoordinators={lcs} selectedLC={selectedLC} onLCChange={setSelectedLC} />
        </div>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-red-400 text-sm">
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : (
        <>
          {/* SQDCM Category Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-4">
            {CATEGORIAS.map(cat => {
              // Count how many indicators are green/yellow/red
              let greenCount = 0, yellowCount = 0, redCount = 0
              cat.indicadores.forEach(ind => {
                const c = getSemaforoColor(promedios[ind.campo], ind.target, ind.direccion)
                if (c === 'green') greenCount++
                else if (c === 'yellow') yellowCount++
                else if (c === 'red') redCount++
              })

              return (
                <div key={cat.key} className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
                  {/* Card header */}
                  <div className={`px-4 py-3 bg-gradient-to-r ${cat.color} flex items-center justify-between`}>
                    <div className="flex items-center gap-2">
                      <span className="w-8 h-8 rounded-lg bg-white/20 flex items-center justify-center text-white font-bold text-sm">
                        {cat.letter}
                      </span>
                      <span className="text-white font-semibold text-sm">{cat.label}</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      {greenCount > 0 && <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-white/20 text-white">{greenCount}✓</span>}
                      {redCount > 0 && <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-red-900/50 text-red-200">{redCount}✗</span>}
                    </div>
                  </div>

                  {/* Indicators list */}
                  <div className="divide-y divide-slate-700/50">
                    {cat.indicadores.map(ind => {
                      const val = promedios[ind.campo]
                      const color = getSemaforoColor(val, ind.target, ind.direccion)
                      const displayVal = val != null ? (ind.unidad === '%' ? val.toFixed(1) : Math.round(val)) : 'NA'

                      return (
                        <div key={ind.campo} className={`px-4 py-2.5 flex items-center justify-between ${semaforoBg[color]}`}>
                          <div>
                            <div className="flex items-center gap-1.5 flex-wrap">
                              <p className="text-xs text-slate-300">{ind.nombre}</p>
                              <TipoBadge tipo={ind.tipo} />
                            </div>
                            <p className="text-[10px] text-slate-500">
                              Target: {ind.target != null ? `${ind.direccion} ${ind.target}${ind.unidad === '%' ? '%' : ''}` : 'N/A'}
                            </p>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className={`text-lg font-bold ${semaforoColors[color]}`}>
                              {displayVal}
                            </span>
                            <span className={`w-2.5 h-2.5 rounded-full ${
                              color === 'green' ? 'bg-green-400' :
                              color === 'yellow' ? 'bg-yellow-400' :
                              color === 'red' ? 'bg-red-400' : 'bg-slate-500'
                            }`} />
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>

          {/* Resumen por Line Coordinator */}
          <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
            <div className="px-4 py-3 border-b border-white/5">
              <h2 className="text-base font-semibold text-white">Resumen por Line Coordinator</h2>
              <p className="text-xs text-slate-400">Promedios de cada LC para la semana seleccionada</p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/5">
                    <th className="px-4 py-2 text-left text-slate-400 font-medium text-xs">Line Coordinator</th>
                    {ALL_INDICADORES.map(ind => (
                      <th key={ind.campo} className="px-2 py-2 text-center text-slate-400 font-medium text-[10px] whitespace-nowrap">
                        <div className="flex flex-col items-center gap-0.5">
                          <span>{ind.nombre.replace(' [#]', '').replace(' [%]', '')}</span>
                          <TipoBadge tipo={ind.tipo} size="xs" />
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {resumenData.length === 0 ? (
                    <tr>
                      <td colSpan={1 + ALL_INDICADORES.length} className="px-4 py-8 text-center text-slate-500 text-sm">
                        No hay datos de resumen para esta semana.
                      </td>
                    </tr>
                  ) : (
                    resumenData.map((lc, idx) => (
                      <tr key={idx} className="border-b border-white/5 hover:bg-slate-700/20">
                        <td className="px-4 py-2.5 text-white font-medium whitespace-nowrap">
                          {lc.lc_nombre || lc.nombre || `LC ${idx + 1}`}
                        </td>
                        {ALL_INDICADORES.map(ind => {
                          // Map indicador campo to resumen field names
                          const val = lc[`avg_${ind.campo}`] ?? lc[ind.campo]
                          const color = getSemaforoColor(val, ind.target, ind.direccion)
                          const displayVal = val != null
                            ? (ind.unidad === '%' ? Number(val).toFixed(1) : Math.round(Number(val)))
                            : 'NA'

                          return (
                            <td key={ind.campo} className={`px-2 py-2.5 text-center text-xs semaforo-cell ${semaforoBg[color]}`}>
                              <span className={semaforoColors[color]}>{displayVal}</span>
                            </td>
                          )
                        })}
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-slate-500">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-green-400" /> En objetivo
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-yellow-400" /> Cerca (&le;10%)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-full bg-red-400" /> Fuera de objetivo
          </span>
        </div>
        <span className="text-slate-700">|</span>
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1.5">
            <span className="inline-flex items-center gap-0.5 text-[9px] px-1.5 py-0.5 rounded font-medium bg-cyan-500/15 text-cyan-400 border border-cyan-500/20">⚡ Auto</span>
            Alimentado automáticamente por la plataforma
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-flex items-center gap-0.5 text-[9px] px-1.5 py-0.5 rounded font-medium bg-amber-500/15 text-amber-400 border border-amber-500/20">✏ Manual</span>
            Captura manual del usuario
          </span>
        </div>
      </div>
    </div>
  )
}
