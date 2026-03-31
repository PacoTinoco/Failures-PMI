import { useState, useEffect, useRef } from 'react'
import CedulaSelector from '../components/CedulaSelector'
import * as api from '../lib/api'

const ACTION_BADGE = {
  nuevo:      { label: 'Nuevo',       cls: 'bg-green-500/15 text-green-400' },
  actualizado:{ label: 'Actualizado', cls: 'bg-blue-500/15 text-blue-400' },
  sin_cambio: { label: 'Sin cambio',  cls: 'bg-slate-500/15 text-slate-400' },
}

export default function DH() {
  const [cedulas, setCedulas]               = useState([])
  const [cedulaId, setCedulaId]             = useState(null)
  const [cedulasLoading, setCedulasLoading] = useState(true)
  const [operators, setOperators]           = useState([])
  const [opsLoading, setOpsLoading]         = useState(false)
  const [uploading, setUploading]           = useState(false)
  const [result, setResult]                 = useState(null)
  const [error, setError]                   = useState(null)
  const [dragOver, setDragOver]             = useState(false)
  const fileInputRef = useRef(null)

  // Cédulas al montar
  useEffect(() => {
    api.getCedulas()
      .then(res => {
        setCedulas(res.data || [])
        if (res.data?.length > 0) setCedulaId(res.data[0].id)
      })
      .catch(err => setError('Error cargando cédulas: ' + err.message))
      .finally(() => setCedulasLoading(false))
  }, [])

  // Operadores cuando cambia cédula
  useEffect(() => {
    if (!cedulaId) return
    setOpsLoading(true)
    api.getDHOperatorEmails(cedulaId)
      .then(res => setOperators(res.data || []))
      .catch(() => {})
      .finally(() => setOpsLoading(false))
  }, [cedulaId])

  async function handleFile(file) {
    if (!file || !cedulaId) return
    if (!file.name.endsWith('.csv')) {
      setError('Solo se aceptan archivos CSV.')
      return
    }
    setUploading(true)
    setError(null)
    setResult(null)
    try {
      const res = await api.uploadDHCSV(cedulaId, file)
      setResult(res)
    } catch (err) {
      setError('Error procesando archivo: ' + err.message)
    }
    setUploading(false)
  }

  function onDrop(e) {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }
  function onDragOver(e) { e.preventDefault(); setDragOver(true) }
  function onDragLeave()  { setDragOver(false) }
  function onFileSelect(e) { handleFile(e.target.files[0]); e.target.value = '' }

  const opsWithEmail    = operators.filter(op => op.email)
  const opsWithoutEmail = operators.filter(op => !op.email)

  return (
    <div className="space-y-6">

      {/* ── Header ── */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white">DH — Defect Handling</h1>
          <p className="text-sm text-slate-400">
            Sube archivos CSV de defectos para cargar DH Encontrados y DH Reparados automáticamente
          </p>
        </div>
        <CedulaSelector cedulas={cedulas} selectedCedulaId={cedulaId} onChange={setCedulaId} loading={cedulasLoading} />
      </div>

      {/* ── Error ── */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-red-400 text-sm flex items-center gap-2">
          <span className="flex-1">{error}</span>
          <button onClick={() => setError(null)} className="text-red-300 hover:text-white">✕</button>
        </div>
      )}

      {/* ── Instrucciones ── */}
      <div className="bg-blue-950/30 border border-blue-500/20 rounded-lg px-4 py-3">
        <p className="text-xs font-semibold text-blue-400 mb-2 uppercase tracking-wider">¿Cómo obtener el archivo?</p>
        <ol className="space-y-1.5">
          {[
            'Ve a Digiperf y selecciona el DMS de DH (Defect Handling).',
            'Filtra por semana y por máquina. Descarga el Excel de cada una de tus máquinas.',
            'Sube los archivos uno por uno aquí. Puedes repetir la operación para cada máquina.',
          ].map((txt, i) => (
            <li key={i} className="flex items-start gap-2 text-xs text-slate-400">
              <span className="flex-shrink-0 w-4 h-4 rounded-full bg-blue-600/30 text-blue-400 text-[10px] flex items-center justify-center font-bold mt-0.5">
                {i + 1}
              </span>
              {txt}
            </li>
          ))}
        </ol>
      </div>

      {/* ── Zona de upload (siempre visible) ── */}
      <div
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => !uploading && fileInputRef.current?.click()}
        className={`
          bg-[#0f1d32] rounded-xl border-2 border-dashed
          ${dragOver ? 'border-blue-400 bg-blue-500/5' : 'border-white/10 hover:border-white/20'}
          px-6 py-10 text-center transition-colors
          ${uploading ? 'cursor-wait' : 'cursor-pointer'}
        `}
      >
        <input ref={fileInputRef} type="file" accept=".csv" onChange={onFileSelect} className="hidden" />
        {uploading ? (
          <div className="flex flex-col items-center gap-3">
            <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-sm text-slate-300">Procesando CSV...</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <svg className="w-10 h-10 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
            </svg>
            <div>
              <p className="text-sm text-slate-300 font-medium">
                {result ? 'Subir otro archivo CSV' : 'Arrastra un archivo CSV aquí o haz clic para seleccionar'}
              </p>
              <p className="text-xs text-slate-500 mt-1">
                Columnas requeridas: REPORTED BY, REPORTED AT, CLOSED BY, CLOSED AT, STATUS, NUMBER
              </p>
            </div>
          </div>
        )}
      </div>

      {/* ── Resultado del procesamiento ── */}
      {result && (
        <div className="space-y-4">

          {/* Tarjetas resumen */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <StatCard label="Filas en CSV"         value={result.stats?.total_rows ?? 0} />
            <StatCard label="Excluidas (DELETED)"  value={result.stats?.excluded_deleted ?? 0}  color="text-slate-400" />
            <StatCard label="Duplicadas ignoradas" value={result.stats?.excluded_duplicate ?? 0} color="text-yellow-400" />
            <StatCard label="DH Enc procesados"    value={result.stats?.matched_encontrados ?? 0} color="text-purple-400" />
          </div>

          {/* Conteo de acciones */}
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-green-500/5 border border-green-500/20 rounded-xl px-4 py-3 text-center">
              <p className="text-2xl font-bold text-green-400">{result.new_count ?? 0}</p>
              <p className="text-xs text-slate-400 mt-1">Registros nuevos</p>
            </div>
            <div className="bg-blue-500/5 border border-blue-500/20 rounded-xl px-4 py-3 text-center">
              <p className="text-2xl font-bold text-blue-400">{result.updated_count ?? 0}</p>
              <p className="text-xs text-slate-400 mt-1">Actualizados</p>
            </div>
            <div className="bg-slate-500/5 border border-slate-500/20 rounded-xl px-4 py-3 text-center">
              <p className="text-2xl font-bold text-slate-400">{result.unchanged_count ?? 0}</p>
              <p className="text-xs text-slate-400 mt-1">Sin cambio (datos iguales)</p>
            </div>
          </div>

          {/* Emails no reconocidos */}
          {result.unmatched_emails?.length > 0 && (
            <div className="bg-yellow-500/5 border border-yellow-500/20 rounded-xl px-5 py-4">
              <h3 className="text-sm font-semibold text-yellow-400 mb-1">
                Emails no reconocidos ({result.unmatched_emails.length})
              </h3>
              <p className="text-xs text-slate-400 mb-3">
                No coinciden con ningún operador registrado — sus defectos no se contaron.
              </p>
              <div className="flex flex-wrap gap-2">
                {result.unmatched_emails.map(email => (
                  <span key={email} className="px-2 py-1 bg-yellow-500/10 text-yellow-300 text-xs rounded-md font-mono">
                    {email}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Detalle por operador/semana */}
          {result.details?.length > 0 && (
            <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
              <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-white">Detalle por operador / semana</h3>
                  <p className="text-xs text-slate-500">
                    Verde = nuevo · Azul = valor actualizado · Gris = sin cambio (no se tocó la BD)
                  </p>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-white/10">
                      <th className="px-4 py-2 text-left text-xs text-slate-400 font-medium">Operador</th>
                      <th className="px-4 py-2 text-left text-xs text-slate-400 font-medium">Semana</th>
                      <th className="px-4 py-2 text-center text-xs text-purple-400 font-medium">DH Enc [#]</th>
                      <th className="px-4 py-2 text-center text-xs text-blue-400 font-medium">DH Rep [#]</th>
                      <th className="px-4 py-2 text-center text-xs text-emerald-400 font-medium">Curva Aut [%]</th>
                      <th className="px-4 py-2 text-center text-xs text-orange-400 font-medium">Contram [%]</th>
                      <th className="px-4 py-2 text-center text-xs text-slate-400 font-medium">Estado</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.details.map((d, idx) => {
                      const badge = ACTION_BADGE[d.accion] || ACTION_BADGE.sin_cambio
                      return (
                        <tr key={idx} className="border-b border-white/5 hover:bg-slate-700/20">
                          <td className="px-4 py-2 text-white text-sm">{d.operador}</td>
                          <td className="px-4 py-2 text-slate-400 text-sm">{d.semana}</td>
                          <td className="px-4 py-2 text-center">
                            <span className="text-purple-400 font-medium">{d.dh_encontrados}</span>
                          </td>
                          <td className="px-4 py-2 text-center">
                            <span className="text-blue-400 font-medium">{d.dh_reparados}</span>
                          </td>
                          <td className="px-4 py-2 text-center">
                            <span className="text-emerald-400 font-medium">{d.curva_autonomia ?? 0}%</span>
                          </td>
                          <td className="px-4 py-2 text-center">
                            <span className="text-orange-400 font-medium">{d.contramedidas_defectos ?? 0}%</span>
                          </td>
                          <td className="px-4 py-2 text-center">
                            <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${badge.cls}`}>
                              {badge.label}
                            </span>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Operadores con email (visible cuando no hay resultado) ── */}
      {!result && (
        <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
          <div className="px-4 py-3 border-b border-white/5">
            <h3 className="text-sm font-semibold text-white">Operadores con email configurado</h3>
            <p className="text-xs text-slate-500">
              {opsLoading
                ? 'Cargando...'
                : `${opsWithEmail.length} de ${operators.length} operadores listos para match automático`}
            </p>
          </div>
          {!opsLoading && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/10">
                    <th className="px-4 py-2 text-left text-xs text-slate-400 font-medium">Operador</th>
                    <th className="px-4 py-2 text-left text-xs text-slate-400 font-medium">Email</th>
                    <th className="px-4 py-2 text-left text-xs text-slate-400 font-medium">LC</th>
                  </tr>
                </thead>
                <tbody>
                  {opsWithEmail.map(op => (
                    <tr key={op.id} className="border-b border-white/5 hover:bg-slate-700/20">
                      <td className="px-4 py-2 text-white text-sm">{op.nombre}</td>
                      <td className="px-4 py-2 text-green-400 text-xs font-mono">{op.email}</td>
                      <td className="px-4 py-2 text-slate-400 text-xs">{op.line_coordinators?.nombre || '—'}</td>
                    </tr>
                  ))}
                  {opsWithoutEmail.map(op => (
                    <tr key={op.id} className="border-b border-white/5 hover:bg-slate-700/20 opacity-40">
                      <td className="px-4 py-2 text-white text-sm">{op.nombre}</td>
                      <td className="px-4 py-2 text-yellow-500 text-xs italic">Sin email — no se hará match</td>
                      <td className="px-4 py-2 text-slate-400 text-xs">{op.line_coordinators?.nombre || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* ── Instrucciones ── */}
      <div className="text-xs text-slate-500 space-y-1 border-t border-white/5 pt-4">
        <p className="font-medium text-slate-400">Cómo funciona:</p>
        <p>1. El sistema deduplica por el ID del defecto (columna NUMBER) — el mismo defecto nunca se cuenta dos veces aunque el CSV tenga filas repetidas.</p>
        <p>2. Los registros con STATUS = DELETED se excluyen automáticamente.</p>
        <p>3. Si ya existe data para un operador/semana, solo actualiza DH Enc y DH Rep — los demás indicadores no se tocan.</p>
        <p>4. Si los valores ya son iguales a lo guardado, no se hace ninguna escritura en la BD (Sin cambio).</p>
      </div>
    </div>
  )
}

function StatCard({ label, value, color = 'text-white' }) {
  return (
    <div className="bg-[#0a1628] rounded-lg px-3 py-2 border border-white/5">
      <p className="text-[10px] text-slate-500 uppercase tracking-wider">{label}</p>
      <p className={`text-xl font-bold ${color}`}>{value}</p>
    </div>
  )
}
