import { useState, useEffect, useRef, Fragment } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'
import CedulaSelector from '../components/CedulaSelector'
import WeekSelector from '../components/WeekSelector'
import UploadBanner from '../components/UploadBanner'
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

export default function QM() {
  const [cedulas, setCedulas]               = useState([])
  const [cedulaId, setCedulaId]             = useState(null)
  const [cedulasLoading, setCedulasLoading] = useState(true)
  const [semana, setSemana]                 = useState(getClosestMonday())

  const [calLoaded, setCalLoaded]       = useState(false)
  const [calUploading, setCalUploading] = useState(false)
  const [calResult, setCalResult]       = useState(null)

  // ── Upload / Preview / Save state ──
  const [dataUploading, setDataUploading] = useState(false)
  const [preview, setPreview]             = useState(null)  // resultado del upload (sin guardar)
  const [saving, setSaving]               = useState(false)
  const [saveResult, setSaveResult]       = useState(null)

  const [semanas, setSemanas]             = useState([])
  const [analisis, setAnalisis]           = useState(null)
  const [analisisLoading, setAnalisisLoading] = useState(false)
  const [expandedEmp, setExpandedEmp]     = useState(null)
  const [showTab, setShowTab]             = useState('employees')

  // ── Por empleado: búsqueda + ordenamiento ──
  const [empSearch, setEmpSearch]         = useState('')
  const [empSortKey, setEmpSortKey]       = useState('name')   // 'name' | 'compliance_pct' | 'below_schedule' | 'on_target'
  const [empSortDir, setEmpSortDir]       = useState('asc')    // 'asc' | 'desc'

  // ── Calendario inline editing ──
  const [calData, setCalData]           = useState([])
  const [calFilter, setCalFilter]       = useState('')
  const [calEditing, setCalEditing]     = useState(null)   // record id being edited
  const [calEditVals, setCalEditVals]   = useState({})
  const [calSaving, setCalSaving]       = useState(false)
  const [calAddMode, setCalAddMode]     = useState(false)
  const [calNewEntry, setCalNewEntry]   = useState({ employee: '', competency: '', role: '', target: 0, current_base: 0 })

  const [deletingSemana, setDeletingSemana] = useState(null)
  const [error, setError] = useState(null)
  const [uploadHistory, setUploadHistory] = useState([])
  const [showHistory, setShowHistory] = useState(false)
  const [syncing, setSyncing] = useState(false)
  const [syncResult, setSyncResult] = useState(null)
  const [deletingLogId, setDeletingLogId] = useState(null)
  const [banner, setBanner] = useState(null) // { message, detail, type }
  const calInputRef  = useRef(null)
  const dataInputRef = useRef(null)

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
    if (!cedulaId) return
    api.getQMCalendario(cedulaId).then(res => {
      setCalLoaded((res.data || []).length > 0)
    }).catch(() => {})
    refreshSemanas()
  }, [cedulaId])

  function refreshSemanas() {
    if (!cedulaId) return
    api.getQMSemanas(cedulaId).then(res => {
      setSemanas(res.data || [])
    }).catch(() => {})
  }

  // Limpiar preview cuando cambia la semana
  useEffect(() => {
    setPreview(null); setSaveResult(null); setSyncResult(null)
  }, [semana])

  // Auto-cargar análisis e historial cuando la semana existe en la lista
  useEffect(() => {
    if (!cedulaId || !semana || !calLoaded) return
    const semanaData = semanas.find(s => s.semana === semana)
    if (!semanaData) {
      // La semana no existe en la lista en absoluto
      setAnalisis(null)
      setUploadHistory([])
      return
    }
    // Siempre intentar cargar análisis (el backend retorna 404 si no hay data,
    // y loadAnalisis lo maneja gracefully limpiando el estado)
    loadAnalisis()
    loadUploadHistory()
  }, [cedulaId, semana, calLoaded, semanas])

  async function loadCalendario() {
    if (!cedulaId) return
    try {
      const res = await api.getQMCalendario(cedulaId)
      setCalData(res.data || [])
    } catch (e) { setError(e.message) }
  }

  async function handleCalSave(recordId) {
    setCalSaving(true)
    try {
      await api.updateQMCalendarioEntry(recordId, calEditVals)
      setCalEditing(null)
      setCalEditVals({})
      await loadCalendario()
    } catch (e) { setError(e.message) }
    finally { setCalSaving(false) }
  }

  async function handleCalAdd() {
    setCalSaving(true)
    try {
      await api.createQMCalendarioEntry({ ...calNewEntry, cedula_id: cedulaId })
      setCalAddMode(false)
      setCalNewEntry({ employee: '', competency: '', role: '', target: 0, current_base: 0 })
      await loadCalendario()
    } catch (e) { setError(e.message) }
    finally { setCalSaving(false) }
  }

  async function handleCalDelete(recordId) {
    if (!confirm('¿Eliminar esta entrada del calendario?')) return
    try {
      await api.deleteQMCalendarioEntry(recordId)
      await loadCalendario()
    } catch (e) { setError(e.message) }
  }

  function loadAnalisis() {
    setAnalisisLoading(true)
    setError(null)
    api.getQMAnalisis(cedulaId, semana)
      .then(res => setAnalisis(res))
      .catch(err => {
        if (!err.message.includes('No hay data')) setError(err.message)
        setAnalisis(null)
      })
      .finally(() => setAnalisisLoading(false))
  }

  function loadUploadHistory() {
    api.getQMUploadHistory(cedulaId, semana)
      .then(res => setUploadHistory(res.data || []))
      .catch(() => setUploadHistory([]))
  }

  async function handleCalUpload(e) {
    const file = e.target.files[0]
    if (!file || !cedulaId) return
    e.target.value = ''
    setCalUploading(true); setError(null); setCalResult(null)
    try {
      const res = await api.uploadQMCalendario(cedulaId, file)
      setCalResult(res)
      setCalLoaded(true)
      const cName = cedulas.find(c => c.id === cedulaId)?.nombre || ''
      setBanner({ message: 'Calendario cargado exitosamente', detail: `${res.total_records} registros · ${res.employees} empleados · ${cName}`, type: 'success' })
    } catch (err) { setError(err.message) }
    setCalUploading(false)
  }

  // Paso 1: subir archivo → genera preview (NO guarda en DB)
  async function handleDataUpload(e) {
    const file = e.target.files[0]
    if (!file || !cedulaId || !semana) return
    e.target.value = ''
    setDataUploading(true)
    setError(null)
    setPreview(null)
    setSaveResult(null)
    setSyncResult(null)
    try {
      const res = await api.uploadQMData(cedulaId, semana, file)
      setPreview(res)  // guarda el resultado (con records[]) en estado
    } catch (err) { setError(err.message) }
    finally { setDataUploading(false) }
  }

  // Paso 2: guardar explícitamente en DB
  async function handleSaveData() {
    if (!preview || !cedulaId || !semana) return
    setSaving(true)
    setError(null)
    setSaveResult(null)
    try {
      const res = await api.saveQMData(
        cedulaId, semana,
        preview.records,
        preview.changed_records,
        preview.new_records,
        preview.changes_detail
      )
      setSaveResult(res)
      const cName = cedulas.find(c => c.id === cedulaId)?.nombre || ''
      setBanner({ message: 'Data semanal guardada', detail: `Semana ${semana} · ${cName}`, type: 'success' })
      // Refrescar lista de semanas y análisis
      refreshSemanas()
      loadAnalisis()
      loadUploadHistory()
      // Auto-sync al dashboard
      try {
        const syncRes = await api.syncQMDashboard(cedulaId, semana)
        setSyncResult(syncRes)
      } catch (syncErr) {
        setSyncResult({ success: false, message: syncErr.message })
      }
    } catch (err) {
      setError(err.message)
    } finally { setSaving(false) }
  }

  async function handleSyncDashboard() {
    if (!cedulaId || !semana) return
    setSyncing(true); setSyncResult(null)
    try {
      const res = await api.syncQMDashboard(cedulaId, semana)
      setSyncResult(res)
    } catch (err) { setSyncResult({ success: false, message: err.message }) }
    finally { setSyncing(false) }
  }

  async function handleDeleteLog(logId) {
    if (!confirm('¿Eliminar este registro del historial?')) return
    setDeletingLogId(logId)
    try {
      await api.deleteQMUploadLog(cedulaId, logId)
      // Actualizar historial local
      setUploadHistory(prev => prev.filter(u => u.id !== logId))
      // También refrescar semanas por si era el único log de esa semana
      refreshSemanas()
    } catch (err) {
      setError(err.message)
    } finally {
      setDeletingLogId(null)
    }
  }

  function handleExportEmployeeAnalysis(data, sem) {
    if (!data?.employees?.length) return
    // Build CSV with BOM for Excel UTF-8 compatibility
    const rows = []
    // Header
    rows.push(['Empleado', 'Rol', 'En Target', 'Debajo', 'Total', '% Cumpl', 'Atrasados', 'Avanzó',
      'Competencia', 'Current', 'Target', 'Pronóstico', 'Sem. Ant.', 'Estado'].join(','))

    for (const emp of data.employees) {
      // Summary row
      rows.push([
        `"${emp.employee}"`, `"${emp.role || ''}"`, emp.on_target, emp.below_target,
        emp.total_competencies, emp.compliance_pct, emp.below_schedule, emp.advanced_this_week || 0,
        '', '', '', '', '', ''
      ].join(','))

      // Detail rows
      if (emp.details) {
        for (const d of emp.details) {
          const estado = d.status === 'on_target' ? 'En target'
            : d.status === 'above_target' ? 'Arriba'
            : d.schedule_status === 'below_schedule' ? 'Atrasado' : 'Pendiente'
          rows.push([
            '', '', '', '', '', '', '', '',
            `"${d.competency}"`, d.current, d.target, d.forecast ?? '', d.previous ?? '', estado
          ].join(','))
        }
      }
    }

    const csv = '\uFEFF' + rows.join('\n')
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `QM_Analisis_${sem}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  async function handleDeleteSemana(s) {
    if (!confirm(`¿Eliminar todos los datos de la semana ${s}? Esta acción no se puede deshacer.`)) return
    setDeletingSemana(s)
    try {
      await api.deleteQMSemana(cedulaId, s)
      refreshSemanas()
      if (semana === s) { setAnalisis(null); setUploadHistory([]); setPreview(null); setSaveResult(null) }
    } catch (err) { setError(err.message) }
    finally { setDeletingSemana(null) }
  }

  const g = analisis?.global || {}

  return (
    <div className="space-y-6">
      <UploadBanner show={!!banner} onClose={() => setBanner(null)}
        cedulaName={banner?.cedulaName} message={banner?.message || ''} detail={banner?.detail} type={banner?.type} />

      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white">Panel QM — Qualification Management</h1>
          <p className="text-sm text-slate-400">Seguimiento de competencias y cumplimiento del equipo</p>
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

      {/* Upload Zone */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-5">
          <h3 className="text-sm font-semibold text-white mb-1">Calendario QM (referencia anual)</h3>
          <p className="text-xs text-slate-500 mb-3">
            {calLoaded ? 'Calendario cargado.' : 'Sube el archivo de calendario QM (.xlsx)'}
          </p>
          <input ref={calInputRef} type="file" accept=".xlsx,.xls" onChange={handleCalUpload} className="hidden" />
          <button
            onClick={() => calInputRef.current?.click()}
            disabled={calUploading}
            className={`w-full py-2.5 rounded-lg text-sm font-medium transition-colors ${
              calLoaded ? 'bg-slate-700 hover:bg-slate-600 text-slate-300' : 'bg-blue-600 hover:bg-blue-500 text-white'
            }`}
          >
            {calUploading ? 'Procesando...' : calLoaded ? 'Reemplazar calendario' : 'Subir calendario'}
          </button>
          {calResult && <p className="text-xs text-green-400 mt-2">{calResult.message}</p>}
        </div>

        <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-5">
          <h3 className="text-sm font-semibold text-white mb-1">Data semanal (semana {semana})</h3>
          <p className="text-xs text-slate-500 mb-3">Sube el archivo de data actualizado para esta semana</p>
          <input ref={dataInputRef} type="file" accept=".xlsx,.xls" onChange={handleDataUpload} className="hidden" />

          {/* Paso 1: Botón subir archivo (genera preview) */}
          <button
            onClick={() => dataInputRef.current?.click()}
            disabled={dataUploading || !calLoaded}
            className={`w-full py-2.5 rounded-lg text-sm font-medium transition-colors ${
              !calLoaded ? 'bg-slate-800 text-slate-500 cursor-not-allowed' : 'bg-purple-600 hover:bg-purple-500 text-white'
            }`}
          >
            {dataUploading ? 'Analizando archivo...' : !calLoaded ? 'Primero sube el calendario' : preview ? 'Re-subir archivo' : 'Subir data semanal'}
          </button>

          {/* Preview: resultado del análisis del archivo (antes de guardar) */}
          {preview && !saveResult && (
            <div className="mt-3 space-y-2">
              {/* Resumen del archivo */}
              <div className="bg-[#0a1628] rounded-lg px-3 py-2 text-xs space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400">{preview.total_records} registros detectados</span>
                  <span className="text-slate-500">{preview.skipped} ignorados</span>
                </div>
                {preview.is_first_upload ? (
                  <p className="text-blue-400">Primera vez que se sube esta semana</p>
                ) : (
                  <div className="flex gap-3">
                    <span className={preview.changed_records > 0 ? 'text-blue-400' : 'text-slate-500'}>
                      {preview.changed_records} cambiaron
                    </span>
                    <span className={preview.new_records > 0 ? 'text-green-400' : 'text-slate-500'}>
                      {preview.new_records} nuevos
                    </span>
                    <span className="text-slate-500">{preview.unchanged_records} sin cambio</span>
                  </div>
                )}
              </div>

              {/* Detalle de cambios */}
              {preview.changed_records > 0 && (
                <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg px-3 py-2">
                  <p className="text-xs text-blue-400 font-medium mb-1">
                    {preview.changed_records} competencia{preview.changed_records !== 1 ? 's' : ''} con cambio de nivel:
                  </p>
                  {preview.changes_detail.slice(0, 5).map((ch, i) => (
                    <p key={i} className="text-[10px] text-blue-300/70">
                      {ch.employee.split(' ')[0]} · {ch.competency.slice(0, 30)}: {ch.old_level} → {ch.new_level}
                    </p>
                  ))}
                  {preview.changes_detail.length > 5 && (
                    <p className="text-[10px] text-blue-300/50">y {preview.changes_detail.length - 5} más...</p>
                  )}
                </div>
              )}

              {/* Botón Guardar */}
              <button
                onClick={handleSaveData}
                disabled={saving}
                className="w-full py-2.5 rounded-lg text-sm font-semibold bg-green-600 hover:bg-green-500 text-white transition-colors"
              >
                {saving ? 'Guardando en plataforma...' : `Guardar ${preview.total_records} registros`}
              </button>
              <p className="text-[10px] text-slate-500 text-center">
                El análisis estará disponible después de guardar
              </p>
            </div>
          )}

          {/* Resultado de guardado exitoso */}
          {saveResult && (
            <div className="mt-2 space-y-1">
              <div className="bg-green-500/10 border border-green-500/20 rounded-lg px-3 py-2">
                <p className="text-xs text-green-400 font-medium">{saveResult.message}</p>
                {syncResult && (
                  <p className={`text-xs mt-1 ${syncResult.success ? 'text-green-300' : 'text-yellow-400'}`}>
                    {syncResult.success
                      ? `Dashboard actualizado: ${syncResult.written} operadores con QM%`
                      : `Sync dashboard: ${syncResult.message}`}
                  </p>
                )}
              </div>
              <button
                onClick={() => { setPreview(null); setSaveResult(null); setSyncResult(null) }}
                className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
              >
                Subir otra versión
              </button>
            </div>
          )}

          {/* Sync manual al dashboard (si ya tiene análisis pero no acaba de guardar) */}
          {analisis && !preview && !saveResult && (
            <button
              onClick={handleSyncDashboard}
              disabled={syncing}
              className="w-full mt-2 py-2 rounded-lg text-xs font-medium bg-green-600/20 hover:bg-green-600/30 text-green-400 border border-green-500/20 transition-colors"
            >
              {syncing ? 'Actualizando...' : 'Actualizar QM% en Dashboard'}
            </button>
          )}
          {syncResult && analisis && !saveResult && (
            <p className={`text-xs mt-1 ${syncResult.success ? 'text-green-400' : 'text-yellow-400'}`}>
              {syncResult.message}
              {syncResult.not_matched?.length > 0 && ` · Sin match: ${syncResult.not_matched.join(', ')}`}
            </p>
          )}

          {/* Historial de guardados de la semana actual */}
          {uploadHistory.length > 0 && (
            <div className="mt-3">
              <button
                onClick={() => setShowHistory(!showHistory)}
                className="text-xs text-slate-400 hover:text-white transition-colors flex items-center gap-1"
              >
                <span>{showHistory ? '▾' : '▸'}</span>
                Historial de guardados ({uploadHistory.length})
              </button>
              {showHistory && (
                <div className="mt-1 space-y-1 max-h-48 overflow-y-auto">
                  {uploadHistory.map((u, i) => (
                    <div key={u.id ?? i} className="bg-[#0a1628] rounded-lg px-3 py-1.5 text-[10px] group">
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-1.5">
                            <span className="text-slate-300">
                              {new Date(u.uploaded_at).toLocaleString('es-MX', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })}
                            </span>
                            <span className="text-slate-600">·</span>
                            <span className="text-slate-500">{u.total_records} reg.</span>
                          </div>
                          {(u.changed_records > 0 || u.new_records > 0) ? (
                            <p className="text-blue-400/70 mt-0.5">
                              {u.changed_records > 0 && `${u.changed_records} cambiaron`}
                              {u.changed_records > 0 && u.new_records > 0 && ' · '}
                              {u.new_records > 0 && `${u.new_records} nuevos`}
                            </p>
                          ) : (
                            <p className="text-slate-600 mt-0.5">Sin cambios respecto al anterior</p>
                          )}
                        </div>
                        <button
                          onClick={() => handleDeleteLog(u.id)}
                          disabled={deletingLogId === u.id}
                          className="opacity-0 group-hover:opacity-100 transition-opacity text-slate-600 hover:text-red-400 px-1 flex-shrink-0"
                          title="Eliminar este registro del historial"
                        >
                          {deletingLogId === u.id ? '···' : '✕'}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Semanas con data cargada — clicables para navegar */}
          {semanas.length > 0 && (
            <div className="mt-3 space-y-1">
              <p className="text-xs text-slate-500 mb-1">Semanas con data cargada:</p>
              {semanas.map(s => (
                <div
                  key={s.semana}
                  className={`flex items-center justify-between rounded-lg px-3 py-2 transition-colors cursor-pointer ${
                    s.semana === semana
                      ? 'bg-purple-500/10 border border-purple-500/30'
                      : 'bg-[#0a1628] hover:bg-slate-700/30 border border-transparent'
                  }`}
                  onClick={() => setSemana(s.semana)}
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      {s.semana === semana && <span className="text-purple-400 text-xs">▶</span>}
                      <span className={`text-xs font-medium ${s.semana === semana ? 'text-purple-400' : 'text-slate-300'}`}>
                        {s.semana}
                      </span>
                      {s.has_data ? (
                        <span className="text-[10px] text-green-400/70 bg-green-500/10 px-1.5 py-0.5 rounded">
                          {s.registros} reg.
                        </span>
                      ) : (
                        <span className="text-[10px] text-yellow-400/70 bg-yellow-500/10 px-1.5 py-0.5 rounded">
                          solo historial
                        </span>
                      )}
                    </div>
                    {s.upload_count > 0 && (
                      <p className="text-[10px] text-slate-500 mt-0.5 ml-4">
                        {s.upload_count} upload{s.upload_count > 1 ? 's' : ''}
                        {s.last_upload && ` · último: ${new Date(s.last_upload).toLocaleString('es-MX', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })}`}
                      </p>
                    )}
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); handleDeleteSemana(s.semana) }}
                    disabled={deletingSemana === s.semana}
                    className="text-slate-600 hover:text-red-400 transition-colors text-xs px-1 py-0.5 rounded ml-2 flex-shrink-0"
                    title="Eliminar data de esta semana"
                  >
                    {deletingSemana === s.semana ? '...' : '✕'}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Analysis Dashboard */}
      {analisisLoading ? (
        <div className="flex items-center justify-center py-16">
          <div className="w-10 h-10 border-4 border-purple-500 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : analisis ? (
        <div className="space-y-5">

          {/* Alerta: empleados sin data esta semana */}
          {analisis.missing_employees?.length > 0 && (
            <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg px-4 py-3">
              <p className="text-yellow-400 text-sm font-medium mb-1">
                ⚠️ {analisis.missing_employees.length} empleado{analisis.missing_employees.length > 1 ? 's' : ''} del calendario sin datos esta semana
              </p>
              <p className="text-yellow-300/70 text-xs">
                {analisis.missing_employees.join(', ')}
              </p>
              <p className="text-yellow-300/50 text-xs mt-1">No aparecen en el análisis. Si faltaron en el archivo de data, re-sube el archivo corregido.</p>
            </div>
          )}

          {/* Global Stats — dos filas: resumen general + dos métricas de cumplimiento */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <GlobeCard label="Empleados" value={g.total_employees} />
            <GlobeCard label="Entradas totales" value={g.total_entries} />
            <GlobeCard label="Avanzaron esta semana" value={g.advanced_this_week} color="text-blue-400" />
            <GlobeCard label="Ref. mes" value={analisis.month_reference?.toUpperCase()} color="text-slate-400" />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <ComplianceCard
              label="Cumplimiento Año"
              sublabel={`Al ritmo del plan (mes ${analisis.month_reference})`}
              pct={g.schedule_compliance_pct}
              count={g.on_schedule}
              total={g.total_entries}
              color="purple"
            />
            <ComplianceCard
              label="Cumplimiento Target"
              sublabel="Alcanzaron el nivel objetivo final"
              pct={g.compliance_pct}
              count={g.on_target}
              total={g.total_entries}
              color="green"
            />
          </div>

          {/* Tabs */}
          <div className="flex gap-1 bg-[#0a1628] rounded-lg p-1 w-fit flex-wrap">
            {[
              { key: 'employees', label: 'Por empleado' },
              { key: 'competencies', label: 'Por competencia' },
              { key: 'monthly', label: 'Cumplimiento mes' },
              { key: 'charts', label: '📊 Gráficas' },
              { key: 'noproj', label: `Sin proyección (${analisis.not_projected_count ?? 0})` },
              { key: 'upcoming', label: 'Próximos cambios' },
              { key: 'calendario', label: '📅 Calendario' },
            ].map(tab => (
              <button key={tab.key} onClick={() => setShowTab(tab.key)}
                className={`px-4 py-1.5 rounded-md text-xs font-medium transition-colors ${
                  showTab === tab.key ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-white'
                }`}>
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab: Employees */}
          {showTab === 'employees' && (() => {
            // Sort helper
            const toggleSort = (key) => {
              if (empSortKey === key) setEmpSortDir(d => d === 'asc' ? 'desc' : 'asc')
              else { setEmpSortKey(key); setEmpSortDir(key === 'name' ? 'asc' : 'desc') }
            }
            const SortIcon = ({ col }) => {
              if (empSortKey !== col) return <span className="text-slate-700 ml-0.5">↕</span>
              return <span className="text-purple-400 ml-0.5">{empSortDir === 'asc' ? '↑' : '↓'}</span>
            }
            // Filter + sort employees
            const filteredEmps = [...(analisis.employees || [])]
              .filter(e => e.employee.toLowerCase().includes(empSearch.toLowerCase()))
              .sort((a, b) => {
                let av, bv
                if (empSortKey === 'name')             { av = a.employee; bv = b.employee; return empSortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av) }
                if (empSortKey === 'compliance_pct')   { av = a.compliance_pct;   bv = b.compliance_pct }
                if (empSortKey === 'below_schedule')   { av = a.below_schedule;   bv = b.below_schedule }
                if (empSortKey === 'on_target')        { av = a.on_target;        bv = b.on_target }
                if (empSortKey === 'below_target')     { av = a.below_target;     bv = b.below_target }
                return empSortDir === 'asc' ? av - bv : bv - av
              })

            return (
              <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
                {/* Header + controls */}
                <div className="px-4 py-3 border-b border-white/5 flex flex-wrap items-center gap-3">
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm font-semibold text-white">Cumplimiento por empleado</h3>
                    <p className="text-xs text-slate-500">Ref: mes {analisis.month_reference} · Clic para ver detalle · {filteredEmps.length} empleado{filteredEmps.length !== 1 ? 's' : ''}</p>
                  </div>
                  {/* Search box */}
                  <div className="relative">
                    <input
                      type="text"
                      placeholder="Buscar empleado..."
                      value={empSearch}
                      onChange={e => setEmpSearch(e.target.value)}
                      className="bg-[#0a1628] border border-white/10 rounded-lg pl-3 pr-7 py-1.5 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-purple-500/50 w-44"
                    />
                    {empSearch && (
                      <button onClick={() => setEmpSearch('')} className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-500 hover:text-white text-xs">✕</button>
                    )}
                  </div>
                  {/* Sort quick buttons */}
                  <div className="flex items-center gap-1 bg-[#0a1628] rounded-lg p-1">
                    {[
                      { key: 'name',           label: 'A-Z' },
                      { key: 'compliance_pct', label: '% Cumpl' },
                      { key: 'below_schedule', label: 'Atrasados' },
                      { key: 'below_target',   label: 'Debajo' },
                    ].map(opt => (
                      <button key={opt.key} onClick={() => toggleSort(opt.key)}
                        className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors flex items-center gap-0.5 ${
                          empSortKey === opt.key ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-white'
                        }`}>
                        {opt.label}
                        {empSortKey === opt.key && <span className="text-[10px]">{empSortDir === 'asc' ? '↑' : '↓'}</span>}
                      </button>
                    ))}
                  </div>
                  <button
                    onClick={() => handleExportEmployeeAnalysis(analisis, semana)}
                    className="px-3 py-1.5 rounded-lg text-xs font-medium bg-green-600/20 text-green-400 hover:bg-green-600/30 border border-green-500/30 transition-colors whitespace-nowrap"
                  >
                    Exportar Excel
                  </button>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-white/10">
                        <th className="px-4 py-2 text-left text-xs text-slate-400 cursor-pointer hover:text-white select-none" onClick={() => toggleSort('name')}>
                          Empleado <SortIcon col="name" />
                        </th>
                        <th className="px-3 py-2 text-center text-xs text-slate-400">Rol</th>
                        <th className="px-3 py-2 text-center text-xs text-green-400 cursor-pointer hover:text-green-300 select-none" onClick={() => toggleSort('on_target')}>
                          En target <SortIcon col="on_target" />
                        </th>
                        <th className="px-3 py-2 text-center text-xs text-red-400 cursor-pointer hover:text-red-300 select-none" onClick={() => toggleSort('below_target')}>
                          Debajo <SortIcon col="below_target" />
                        </th>
                        <th className="px-3 py-2 text-center text-xs text-slate-400">Total</th>
                        <th className="px-3 py-2 text-center text-xs text-purple-400 cursor-pointer hover:text-purple-300 select-none" onClick={() => toggleSort('compliance_pct')}>
                          % Cumpl <SortIcon col="compliance_pct" />
                        </th>
                        <th className="px-3 py-2 text-center text-xs text-yellow-400 cursor-pointer hover:text-yellow-300 select-none" onClick={() => toggleSort('below_schedule')}>
                          Atrasados <SortIcon col="below_schedule" />
                        </th>
                        <th className="px-3 py-2 text-center text-xs text-blue-400">Avanzó</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredEmps.length === 0 ? (
                        <tr><td colSpan={8} className="px-4 py-8 text-center text-slate-500 text-sm">Sin resultados para "{empSearch}"</td></tr>
                      ) : filteredEmps.map((emp, idx) => (
                        <EmployeeRow key={idx} emp={emp}
                          isExpanded={expandedEmp === emp.employee}
                          onToggle={() => setExpandedEmp(expandedEmp === emp.employee ? null : emp.employee)} />
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )
          })()}

          {/* Tab: Competencies */}
          {showTab === 'competencies' && (
            <CompetenciesTab competencies={analisis.competencies} />
          )}

          {/* Tab: Monthly compliance */}
          {showTab === 'monthly' && (
            <MonthlyComplianceTab employees={analisis.employees} semana={semana} />
          )}

          {/* Tab: Gráficas */}
          {showTab === 'charts' && (
            <ChartsTab analisis={analisis} />
          )}

          {/* Tab: Sin proyección */}
          {showTab === 'noproj' && (
            <NotProjectedTab data={analisis.not_projected || []} count={analisis.not_projected_count} />
          )}

          {/* Tab: Upcoming Changes */}
          {showTab === 'upcoming' && (
            <UpcomingChangesTab upcomingChanges={analisis.upcoming_changes || []} semana={semana} />
          )}

          {/* Tab: Calendario inline */}
          {showTab === 'calendario' && (() => {
            // Load on first render of this tab
            if (calData.length === 0 && cedulaId) { loadCalendario() }
            const MONTHS = ['feb','mar','abr','may','jun','jul','ago','sep']
            const filtered = calData.filter(r =>
              r.employee.toLowerCase().includes(calFilter.toLowerCase()) ||
              r.competency.toLowerCase().includes(calFilter.toLowerCase())
            )
            // Group by employee
            const grouped = {}
            filtered.forEach(r => {
              if (!grouped[r.employee]) grouped[r.employee] = []
              grouped[r.employee].push(r)
            })
            return (
              <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-4">
                <div className="flex items-center justify-between gap-3 mb-4 flex-wrap">
                  <h3 className="text-white text-sm font-semibold">Calendario QM — Edición Inline</h3>
                  <div className="flex items-center gap-2">
                    <input type="text" placeholder="Buscar empleado/competencia..."
                      value={calFilter} onChange={e => setCalFilter(e.target.value)}
                      className="bg-[#0a1628] border border-white/10 rounded-lg px-3 py-1.5 text-white text-xs w-56 focus:outline-none focus:ring-1 focus:ring-purple-500" />
                    <button onClick={() => setCalAddMode(!calAddMode)}
                      className="px-3 py-1.5 bg-green-600 hover:bg-green-500 text-white text-xs font-medium rounded-lg transition-colors">
                      + Agregar
                    </button>
                    <button onClick={loadCalendario}
                      className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white text-xs rounded-lg transition-colors">
                      Recargar
                    </button>
                  </div>
                </div>

                {/* Add new entry form */}
                {calAddMode && (
                  <div className="bg-[#0a1628] rounded-lg border border-green-500/30 p-3 mb-4 flex flex-wrap gap-2 items-end">
                    <div>
                      <label className="text-slate-500 text-[10px] block mb-0.5">Empleado</label>
                      <input type="text" value={calNewEntry.employee}
                        onChange={e => setCalNewEntry(p => ({ ...p, employee: e.target.value }))}
                        className="bg-[#0f1d32] border border-white/10 rounded px-2 py-1 text-white text-xs w-40 focus:outline-none focus:ring-1 focus:ring-green-500" />
                    </div>
                    <div>
                      <label className="text-slate-500 text-[10px] block mb-0.5">Competencia</label>
                      <input type="text" value={calNewEntry.competency}
                        onChange={e => setCalNewEntry(p => ({ ...p, competency: e.target.value }))}
                        className="bg-[#0f1d32] border border-white/10 rounded px-2 py-1 text-white text-xs w-40 focus:outline-none focus:ring-1 focus:ring-green-500" />
                    </div>
                    <div>
                      <label className="text-slate-500 text-[10px] block mb-0.5">Rol</label>
                      <input type="text" value={calNewEntry.role}
                        onChange={e => setCalNewEntry(p => ({ ...p, role: e.target.value }))}
                        className="bg-[#0f1d32] border border-white/10 rounded px-2 py-1 text-white text-xs w-32 focus:outline-none focus:ring-1 focus:ring-green-500" />
                    </div>
                    <div>
                      <label className="text-slate-500 text-[10px] block mb-0.5">Base</label>
                      <input type="number" value={calNewEntry.current_base}
                        onChange={e => setCalNewEntry(p => ({ ...p, current_base: Number(e.target.value) }))}
                        className="bg-[#0f1d32] border border-white/10 rounded px-2 py-1 text-white text-xs w-14 focus:outline-none focus:ring-1 focus:ring-green-500" />
                    </div>
                    <div>
                      <label className="text-slate-500 text-[10px] block mb-0.5">Target</label>
                      <input type="number" value={calNewEntry.target}
                        onChange={e => setCalNewEntry(p => ({ ...p, target: Number(e.target.value) }))}
                        className="bg-[#0f1d32] border border-white/10 rounded px-2 py-1 text-white text-xs w-14 focus:outline-none focus:ring-1 focus:ring-green-500" />
                    </div>
                    <button onClick={handleCalAdd} disabled={calSaving || !calNewEntry.employee || !calNewEntry.competency}
                      className="px-3 py-1.5 bg-green-600 hover:bg-green-500 disabled:opacity-40 text-white text-xs font-medium rounded-lg transition-colors">
                      {calSaving ? 'Guardando...' : 'Guardar'}
                    </button>
                    <button onClick={() => setCalAddMode(false)}
                      className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white text-xs rounded-lg transition-colors">
                      Cancelar
                    </button>
                  </div>
                )}

                <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
                  <table className="w-full text-xs">
                    <thead className="sticky top-0 z-10 bg-[#0f1d32]">
                      <tr className="border-b border-white/10">
                        <th className="px-3 py-2 text-left text-slate-400 min-w-[140px]">Empleado</th>
                        <th className="px-2 py-2 text-left text-slate-400 min-w-[120px]">Competencia</th>
                        <th className="px-2 py-2 text-center text-slate-400 w-14">Base</th>
                        <th className="px-2 py-2 text-center text-slate-400 w-14">Target</th>
                        {MONTHS.map(m => (
                          <th key={m} className="px-1 py-2 text-center text-slate-400 w-12">{m.charAt(0).toUpperCase() + m.slice(1)}</th>
                        ))}
                        <th className="px-2 py-2 text-center text-slate-400 w-20">Acciones</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.keys(grouped).sort().map(emp => (
                        <Fragment key={emp}>
                          {grouped[emp].sort((a, b) => a.competency.localeCompare(b.competency)).map((rec, ri) => {
                            const isEditing = calEditing === rec.id
                            return (
                              <tr key={rec.id} className={`border-b border-white/5 hover:bg-slate-700/20 ${ri === 0 ? 'border-t border-white/10' : ''}`}>
                                <td className="px-3 py-1.5 text-white font-medium">
                                  {ri === 0 ? emp : ''}
                                </td>
                                <td className="px-2 py-1.5 text-slate-300">{rec.competency}</td>
                                <td className="px-2 py-1.5 text-center">
                                  {isEditing ? (
                                    <input type="number" defaultValue={rec.current_base || 0}
                                      onChange={e => setCalEditVals(p => ({ ...p, current_base: Number(e.target.value) }))}
                                      className="bg-[#0a1628] border border-purple-500/50 rounded px-1 py-0.5 text-white text-xs w-12 text-center focus:outline-none" />
                                  ) : <span className="text-slate-400">{rec.current_base ?? '—'}</span>}
                                </td>
                                <td className="px-2 py-1.5 text-center">
                                  {isEditing ? (
                                    <input type="number" defaultValue={rec.target || 0}
                                      onChange={e => setCalEditVals(p => ({ ...p, target: Number(e.target.value) }))}
                                      className="bg-[#0a1628] border border-purple-500/50 rounded px-1 py-0.5 text-white text-xs w-12 text-center focus:outline-none" />
                                  ) : <span className="text-cyan-400 font-medium">{rec.target ?? '—'}</span>}
                                </td>
                                {MONTHS.map(m => (
                                  <td key={m} className="px-1 py-1.5 text-center">
                                    {isEditing ? (
                                      <input type="number" defaultValue={rec[m] ?? ''}
                                        onChange={e => setCalEditVals(p => ({ ...p, [m]: e.target.value === '' ? null : Number(e.target.value) }))}
                                        className="bg-[#0a1628] border border-purple-500/50 rounded px-0.5 py-0.5 text-white text-xs w-10 text-center focus:outline-none" />
                                    ) : (
                                      <span className={rec[m] != null ? (rec[m] >= (rec.target || 0) ? 'text-green-400' : 'text-yellow-400') : 'text-slate-600'}>
                                        {rec[m] ?? '·'}
                                      </span>
                                    )}
                                  </td>
                                ))}
                                <td className="px-2 py-1.5 text-center">
                                  {isEditing ? (
                                    <div className="flex gap-1 justify-center">
                                      <button onClick={() => handleCalSave(rec.id)} disabled={calSaving}
                                        className="px-2 py-0.5 bg-green-600 hover:bg-green-500 text-white text-[10px] rounded transition-colors">
                                        {calSaving ? '...' : '✓'}
                                      </button>
                                      <button onClick={() => { setCalEditing(null); setCalEditVals({}) }}
                                        className="px-2 py-0.5 bg-slate-600 hover:bg-slate-500 text-white text-[10px] rounded transition-colors">
                                        ✕
                                      </button>
                                    </div>
                                  ) : (
                                    <div className="flex gap-1 justify-center">
                                      <button onClick={() => { setCalEditing(rec.id); setCalEditVals({}) }}
                                        className="px-2 py-0.5 bg-purple-600/30 hover:bg-purple-600/50 text-purple-300 text-[10px] rounded transition-colors">
                                        Editar
                                      </button>
                                      <button onClick={() => handleCalDelete(rec.id)}
                                        className="px-2 py-0.5 bg-red-600/20 hover:bg-red-600/40 text-red-400 text-[10px] rounded transition-colors">
                                        ✕
                                      </button>
                                    </div>
                                  )}
                                </td>
                              </tr>
                            )
                          })}
                        </Fragment>
                      ))}
                    </tbody>
                  </table>
                  {filtered.length === 0 && (
                    <p className="text-center text-slate-500 py-8 text-sm">
                      {calData.length === 0 ? 'Cargando calendario...' : 'Sin resultados para la búsqueda.'}
                    </p>
                  )}
                </div>
                <p className="text-slate-500 text-[10px] mt-2">{calData.length} registros totales • {Object.keys(grouped).length} empleados mostrados</p>
              </div>
            )
          })()}

        </div>
      ) : calLoaded ? (
        <div className="bg-[#0f1d32] rounded-xl border border-white/5 px-6 py-16 text-center">
          <svg className="w-12 h-12 text-slate-600 mx-auto mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
          </svg>
          <p className="text-slate-400 text-sm">
            {semanas.some(s => s.semana === semana && !s.has_data && s.upload_count > 0)
              ? 'Esta semana tiene historial de uploads pero los datos no están disponibles. Vuelve a subir el archivo.'
              : 'Sube el archivo de data semanal para ver el análisis.'}
          </p>
          <p className="text-slate-500 text-xs mt-1">
            {semanas.length > 0
              ? `Selecciona una semana de la lista (${semanas.length} disponibles) o sube datos para la semana actual.`
              : 'Selecciona la semana correcta y sube el archivo data_WXX.xlsx'}
          </p>
        </div>
      ) : null}
    </div>
  )
}

function MonthlyComplianceTab({ employees, semana }) {
  const [filter, setFilter] = useState('all') // all | met | partial | not
  const [expandedEmp, setExpandedEmp] = useState(null)

  // Calculate monthly compliance for each employee
  // schedule_compliance_pct already exists from backend (vs forecast del mes)
  const enriched = employees.map(e => {
    const total = e.total_competencies
    const onSched = e.on_schedule
    const belowSched = e.below_schedule
    const pct = e.schedule_compliance_pct
    let status = 'not'
    if (pct >= 100) status = 'met'
    else if (pct >= 70) status = 'partial'
    return { ...e, monthly_pct: pct, monthly_status: status, met_count: onSched, gap_count: belowSched }
  })

  const filtered = enriched.filter(e => {
    if (filter === 'all') return true
    return e.monthly_status === filter
  }).sort((a, b) => b.monthly_pct - a.monthly_pct)

  const counts = {
    met: enriched.filter(e => e.monthly_status === 'met').length,
    partial: enriched.filter(e => e.monthly_status === 'partial').length,
    not: enriched.filter(e => e.monthly_status === 'not').length,
  }
  const totalEmps = enriched.length
  const monthName = semana ? new Date(semana + 'T12:00:00').toLocaleDateString('es-MX', { month: 'long', year: 'numeric' }) : ''

  return (
    <div className="space-y-3">
      {/* Top KPI cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <button onClick={() => setFilter('all')}
          className={`bg-[#0f1d32] rounded-xl border p-4 text-left transition-all ${filter === 'all' ? 'border-purple-500/50 ring-1 ring-purple-500/30' : 'border-white/5 hover:border-purple-500/30'}`}>
          <p className="text-2xl font-bold text-white">{totalEmps}</p>
          <p className="text-[10px] text-slate-500 uppercase tracking-wider mt-0.5">Total empleados</p>
        </button>
        <button onClick={() => setFilter(filter === 'met' ? 'all' : 'met')}
          className={`bg-[#0f1d32] rounded-xl border p-4 text-left transition-all ${filter === 'met' ? 'border-green-500/50 ring-1 ring-green-500/30' : 'border-white/5 hover:border-green-500/30'}`}>
          <p className="text-2xl font-bold text-green-400">{counts.met}</p>
          <p className="text-[10px] text-slate-500 uppercase tracking-wider mt-0.5">Cumplieron 100%</p>
        </button>
        <button onClick={() => setFilter(filter === 'partial' ? 'all' : 'partial')}
          className={`bg-[#0f1d32] rounded-xl border p-4 text-left transition-all ${filter === 'partial' ? 'border-yellow-500/50 ring-1 ring-yellow-500/30' : 'border-white/5 hover:border-yellow-500/30'}`}>
          <p className="text-2xl font-bold text-yellow-400">{counts.partial}</p>
          <p className="text-[10px] text-slate-500 uppercase tracking-wider mt-0.5">Parcial (70-99%)</p>
        </button>
        <button onClick={() => setFilter(filter === 'not' ? 'all' : 'not')}
          className={`bg-[#0f1d32] rounded-xl border p-4 text-left transition-all ${filter === 'not' ? 'border-red-500/50 ring-1 ring-red-500/30' : 'border-white/5 hover:border-red-500/30'}`}>
          <p className="text-2xl font-bold text-red-400">{counts.not}</p>
          <p className="text-[10px] text-slate-500 uppercase tracking-wider mt-0.5">No cumplieron (&lt;70%)</p>
        </button>
      </div>

      {/* Table */}
      <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
        <div className="px-4 py-3 border-b border-white/5">
          <h3 className="text-sm font-semibold text-white">Cumplimiento del mes — {monthName}</h3>
          <p className="text-xs text-slate-500">Comparado contra el forecast mensual del calendario · Clic para ver detalle</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/10">
                <th className="px-4 py-2 text-left text-xs text-slate-400">Empleado</th>
                <th className="px-3 py-2 text-center text-xs text-slate-400">Rol</th>
                <th className="px-3 py-2 text-center text-xs text-slate-400">Total</th>
                <th className="px-3 py-2 text-center text-xs text-green-400">Cumplió</th>
                <th className="px-3 py-2 text-center text-xs text-red-400">Falta</th>
                <th className="px-3 py-2 text-center text-xs text-purple-400">% Mes</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((e, idx) => {
                const isExpanded = expandedEmp === e.employee
                const belowDetails = (e.details || []).filter(d => d.schedule_status === 'below_schedule')
                return (
                  <Fragment key={idx}>
                    <tr onClick={() => setExpandedEmp(isExpanded ? null : e.employee)}
                      className={`border-b border-white/5 hover:bg-slate-700/20 cursor-pointer ${isExpanded ? 'bg-purple-950/20' : ''}`}>
                      <td className="px-4 py-2 text-white text-sm">
                        <span className="text-slate-500 text-xs mr-1.5">{isExpanded ? '▾' : '▸'}</span>
                        {e.employee}
                      </td>
                      <td className="px-3 py-2 text-center text-slate-400 text-xs">{e.role || '—'}</td>
                      <td className="px-3 py-2 text-center text-slate-400">{e.total_competencies}</td>
                      <td className="px-3 py-2 text-center text-green-400">{e.met_count}</td>
                      <td className="px-3 py-2 text-center text-red-400">{e.gap_count}</td>
                      <td className="px-3 py-2 text-center"><ComplianceBadge pct={e.monthly_pct} /></td>
                    </tr>
                    {isExpanded && (
                      <tr>
                        <td colSpan={6} className="px-0 py-0">
                          <div className="bg-[#0a1628] border-y border-purple-500/20 px-6 py-3">
                            {belowDetails.length === 0 ? (
                              <p className="text-xs text-green-400">Cumplió todas sus competencias del mes ✓</p>
                            ) : (
                              <>
                                <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">
                                  Competencias debajo del forecast del mes ({belowDetails.length})
                                </p>
                                <table className="w-full text-xs">
                                  <thead>
                                    <tr className="border-b border-white/5 text-slate-500">
                                      <th className="px-3 py-1.5 text-left font-medium">Competencia</th>
                                      <th className="px-3 py-1.5 text-center font-medium">Actual</th>
                                      <th className="px-3 py-1.5 text-center font-medium">Forecast mes</th>
                                      <th className="px-3 py-1.5 text-center font-medium">Target</th>
                                      <th className="px-3 py-1.5 text-center font-medium">Brecha</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {belowDetails.map((d, i) => (
                                      <tr key={i} className="border-b border-white/5">
                                        <td className="px-3 py-1.5 text-slate-300">{d.competency}</td>
                                        <td className="px-3 py-1.5 text-center text-slate-300">{d.current}</td>
                                        <td className="px-3 py-1.5 text-center text-slate-400">{d.forecast}</td>
                                        <td className="px-3 py-1.5 text-center text-slate-400">{d.target}</td>
                                        <td className="px-3 py-1.5 text-center text-red-400 font-semibold">−{d.forecast - d.current}</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                )
              })}
              {filtered.length === 0 && (
                <tr><td colSpan={6} className="px-4 py-8 text-center text-slate-500 text-sm">Sin empleados en esta categoría</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function CompetenciesTab({ competencies }) {
  const [expanded, setExpanded] = useState(null)
  return (
    <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
      <div className="px-4 py-3 border-b border-white/5">
        <h3 className="text-sm font-semibold text-white">Cumplimiento por competencia</h3>
        <p className="text-xs text-slate-500">Ordenado por menor cumplimiento · Clic para ver quiénes están debajo</p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-white/10">
              <th className="px-4 py-2 text-left text-xs text-slate-400">Competencia</th>
              <th className="px-3 py-2 text-center text-xs text-slate-400">Personas</th>
              <th className="px-3 py-2 text-center text-xs text-green-400">En target</th>
              <th className="px-3 py-2 text-center text-xs text-red-400">Debajo</th>
              <th className="px-3 py-2 text-center text-xs text-purple-400">% Cumpl</th>
              <th className="px-2 py-2 w-8"></th>
            </tr>
          </thead>
          <tbody>
            {competencies.map((c, idx) => {
              const isExpanded = expanded === idx
              const hasBelow = (c.below_employees || []).length > 0
              return (
                <Fragment key={idx}>
                  <tr
                    onClick={() => hasBelow && setExpanded(isExpanded ? null : idx)}
                    className={`border-b border-white/5 hover:bg-slate-700/20 ${hasBelow ? 'cursor-pointer' : ''} ${isExpanded ? 'bg-purple-950/20' : ''}`}>
                    <td className="px-4 py-2 text-white text-sm">
                      <span className="text-slate-500 text-xs mr-1.5">{hasBelow ? (isExpanded ? '▾' : '▸') : ' '}</span>
                      {c.competency}
                    </td>
                    <td className="px-3 py-2 text-center text-slate-400">{c.total}</td>
                    <td className="px-3 py-2 text-center text-green-400">{c.on_target}</td>
                    <td className="px-3 py-2 text-center text-red-400">{c.below_target}</td>
                    <td className="px-3 py-2 text-center"><ComplianceBadge pct={c.compliance_pct} /></td>
                    <td className="px-2 py-2 text-center text-slate-600 text-xs">
                      {hasBelow ? `${c.below_employees.length}` : ''}
                    </td>
                  </tr>
                  {isExpanded && hasBelow && (
                    <tr>
                      <td colSpan={6} className="px-0 py-0">
                        <div className="bg-[#0a1628] border-y border-purple-500/20 px-6 py-3">
                          <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">
                            Empleados debajo del target ({c.below_employees.length})
                          </p>
                          <table className="w-full text-xs">
                            <thead>
                              <tr className="border-b border-white/5 text-slate-500">
                                <th className="px-3 py-1.5 text-left font-medium">Empleado</th>
                                <th className="px-3 py-1.5 text-center font-medium">Actual</th>
                                <th className="px-3 py-1.5 text-center font-medium">Forecast mes</th>
                                <th className="px-3 py-1.5 text-center font-medium">Target final</th>
                                <th className="px-3 py-1.5 text-center font-medium">Brecha</th>
                              </tr>
                            </thead>
                            <tbody>
                              {c.below_employees.map((e, i) => (
                                <tr key={i} className="border-b border-white/5 hover:bg-slate-700/10">
                                  <td className="px-3 py-1.5 text-slate-300">{e.employee}</td>
                                  <td className="px-3 py-1.5 text-center text-slate-300">{e.current}</td>
                                  <td className="px-3 py-1.5 text-center text-slate-400">{e.forecast}</td>
                                  <td className="px-3 py-1.5 text-center text-slate-400">{e.target}</td>
                                  <td className="px-3 py-1.5 text-center">
                                    <span className="text-red-400 font-semibold">−{e.gap}</span>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function GlobeCard({ label, value, color = 'text-white' }) {
  return (
    <div className="bg-[#0f1d32] rounded-xl border border-white/5 px-4 py-3 text-center">
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
      <p className="text-[10px] text-slate-500 uppercase tracking-wider mt-1">{label}</p>
    </div>
  )
}

function ComplianceCard({ label, sublabel, pct, count, total, color }) {
  const colorMap = {
    purple: { bar: '#a855f7', bg: 'bg-purple-500/10', text: 'text-purple-400', border: 'border-purple-500/20' },
    green:  { bar: '#22c55e', bg: 'bg-green-500/10',  text: 'text-green-400',  border: 'border-green-500/20' },
  }
  const c = colorMap[color] || colorMap.purple
  const below = total - count

  return (
    <div className={`bg-[#0f1d32] rounded-xl border ${c.border} p-5`}>
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold text-white">{label}</h3>
          <p className="text-xs text-slate-500 mt-0.5">{sublabel}</p>
        </div>
        <span className={`text-3xl font-bold ${c.text}`}>{pct}%</span>
      </div>
      {/* Barra de progreso */}
      <div className="w-full h-2 bg-slate-700/50 rounded-full overflow-hidden mb-3">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: colorMap[color].bar }}
        />
      </div>
      <div className="flex justify-between text-xs">
        <span className={c.text}>{count} al día</span>
        <span className="text-red-400">{below} por debajo</span>
        <span className="text-slate-500">{total} total</span>
      </div>
    </div>
  )
}

function NotProjectedTab({ data, count }) {
  const [expandedEmp, setExpandedEmp] = useState(null)

  if (count === 0) {
    return (
      <div className="bg-[#0f1d32] rounded-xl border border-white/5 px-6 py-16 text-center">
        <p className="text-green-400 text-2xl mb-2">✓</p>
        <p className="text-slate-300 text-sm font-medium">Todos los empleados tienen proyectado alcanzar su target este año</p>
        <p className="text-slate-500 text-xs mt-1">Según el calendario, el nivel de sep ≥ target final para todos.</p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="bg-orange-500/10 border border-orange-500/20 rounded-xl px-5 py-4">
        <p className="text-orange-400 text-sm font-semibold">
          {count} competencia{count > 1 ? 's' : ''} en {data.length} empleado{data.length > 1 ? 's' : ''} no alcanzarán su target final este año según el calendario
        </p>
        <p className="text-orange-300/60 text-xs mt-1">
          El pronóstico de septiembre (último mes del calendario) es menor al nivel objetivo final.
        </p>
      </div>

      <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
        <div className="px-4 py-3 border-b border-white/5">
          <h3 className="text-sm font-semibold text-white">Empleados con brecha al final del año</h3>
          <p className="text-xs text-slate-500">Haz clic para ver qué competencias tienen brecha</p>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-white/10">
              <th className="px-4 py-2 text-left text-xs text-slate-400">Empleado</th>
              <th className="px-3 py-2 text-center text-xs text-orange-400">Competencias con brecha</th>
            </tr>
          </thead>
          <tbody>
            {data.map((emp, idx) => (
              <>
                <tr key={idx}
                  className="border-b border-white/5 hover:bg-slate-700/20 cursor-pointer"
                  onClick={() => setExpandedEmp(expandedEmp === emp.employee ? null : emp.employee)}>
                  <td className="px-4 py-2 text-white text-sm font-medium">
                    <span className="mr-1 text-slate-500">{expandedEmp === emp.employee ? '▾' : '▸'}</span>
                    {emp.employee}
                  </td>
                  <td className="px-3 py-2 text-center">
                    <span className="bg-orange-500/10 text-orange-400 text-xs font-medium px-2 py-0.5 rounded-full">
                      {emp.count} competencia{emp.count > 1 ? 's' : ''}
                    </span>
                  </td>
                </tr>
                {expandedEmp === emp.employee && (
                  <tr key={`${idx}-detail`}>
                    <td colSpan={2} className="px-0 py-0">
                      <div className="bg-[#0a1628] border-y border-white/5">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="border-b border-white/5">
                              <th className="px-6 py-1.5 text-left text-slate-500">Competencia</th>
                              <th className="px-3 py-1.5 text-center text-slate-500">Pronós. Sep</th>
                              <th className="px-3 py-1.5 text-center text-slate-500">Target final</th>
                              <th className="px-3 py-1.5 text-center text-orange-400/70">Brecha</th>
                            </tr>
                          </thead>
                          <tbody>
                            {emp.competencies.sort((a, b) => b.gap - a.gap).map((c, i) => (
                              <tr key={i} className="border-b border-white/5">
                                <td className="px-6 py-1 text-slate-300">{c.competency}</td>
                                <td className="px-3 py-1 text-center text-yellow-400">{c.year_end_forecast}</td>
                                <td className="px-3 py-1 text-center text-slate-400">{c.target}</td>
                                <td className="px-3 py-1 text-center text-orange-400 font-medium">-{c.gap}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </td>
                  </tr>
                )}
              </>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const CHART_COLORS = {
  on:     '#a855f7',
  below:  '#ef4444',
  sched:  '#3b82f6',
  adv:    '#22c55e',
}

function UpcomingChangesTab({ upcomingChanges, semana }) {
  const [filterMonth, setFilterMonth]   = useState('all')
  const [filterEmp, setFilterEmp]       = useState('')
  const [filterStatus, setFilterStatus] = useState('all') // 'all' | 'overdue' | 'pending'

  // Construir tabla consolidada con mes destino y estado (pendiente/atrasado)
  const allChanges = []
  for (const uc of upcomingChanges) {
    for (const ch of uc.changes) {
      allChanges.push({
        month: uc.month || uc.transition,
        employee: ch.employee,
        competency: ch.competency,
        from_level: ch.from_level,
        to_level: ch.to_level,
        current_level: ch.current_level ?? ch.from_level,
        overdue: ch.overdue || uc.is_overdue || false,
      })
    }
  }

  // Unique months for filter tabs
  const uniqueMonths = [...new Set(allChanges.map(c => c.month))]

  // Apply filters
  const visibleChanges = allChanges.filter(c => {
    if (filterMonth !== 'all' && c.month !== filterMonth) return false
    if (filterEmp && !c.employee.toLowerCase().includes(filterEmp.toLowerCase())) return false
    if (filterStatus === 'overdue' && !c.overdue) return false
    if (filterStatus === 'pending' && c.overdue) return false
    return true
  })

  function handleExportExcel() {
    const headers = ['Estado', 'Empleado', 'Mes', 'Competencia', 'Nivel actual', 'Nivel esperado', 'De', 'A']
    const csvRows = [headers.join(',')]
    for (const ch of visibleChanges) {
      csvRows.push([
        ch.overdue ? 'Atrasado' : 'Pendiente',
        `"${ch.employee}"`,
        `"${ch.month}"`,
        `"${ch.competency}"`,
        ch.current_level,
        ch.to_level,
        ch.from_level,
        ch.to_level,
      ].join(','))
    }
    const csvContent = '\uFEFF' + csvRows.join('\n')
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `cambios_pendientes_${semana}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (upcomingChanges.length === 0) {
    return (
      <div className="bg-[#0f1d32] rounded-xl border border-white/5 px-6 py-10 text-center text-slate-500 text-sm">
        No hay cambios pendientes — todos los niveles están al día.
      </div>
    )
  }

  const totalChanges = allChanges.length
  const overdueCount = allChanges.filter(c => c.overdue).length

  return (
    <div className="space-y-4">

      {/* ── KPI cards ARRIBA ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {upcomingChanges.map((uc, idx) => (
          <button key={idx}
            onClick={() => setFilterMonth(filterMonth === (uc.month || uc.transition) ? 'all' : (uc.month || uc.transition))}
            className={`bg-[#0f1d32] rounded-xl border p-4 text-center transition-all ${
              uc.is_overdue ? 'border-red-500/30' : 'border-white/5'
            } ${filterMonth === (uc.month || uc.transition) ? 'ring-2 ring-purple-500/60' : 'hover:border-purple-500/30'}`}>
            <p className={`text-2xl font-bold ${uc.is_overdue ? 'text-red-400' : 'text-purple-400'}`}>{uc.count}</p>
            <p className="text-sm text-white font-medium mt-1">{uc.transition}</p>
            <p className="text-xs text-slate-500 mt-0.5">
              {uc.is_overdue ? 'cambios atrasados' : 'cambios pendientes'}
            </p>
          </button>
        ))}
      </div>

      {/* ── Tabla consolidada ── */}
      <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
        {/* Header */}
        <div className="px-4 py-3 border-b border-white/5 flex flex-wrap items-center gap-3">
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-semibold text-white">Cambios pendientes por mes</h3>
            <p className="text-xs text-slate-500">
              {visibleChanges.length} de {totalChanges} pendientes
              {overdueCount > 0 && <span className="text-red-400 ml-1">({overdueCount} atrasados)</span>}
            </p>
          </div>
          <button onClick={handleExportExcel}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-green-600/20 hover:bg-green-600/30 border border-green-500/20 rounded-lg text-xs text-green-400 font-medium transition-colors">
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Exportar Excel
          </button>
        </div>

        {/* Filter bar */}
        <div className="px-4 py-2.5 border-b border-white/5 flex flex-wrap items-center gap-2 bg-[#0a1628]/50">
          {/* Estado */}
          <div className="flex items-center gap-1 bg-[#0a1628] rounded-lg p-0.5">
            {[
              { key: 'all',     label: 'Todos' },
              { key: 'overdue', label: 'Atrasados' },
              { key: 'pending', label: 'Pendientes' },
            ].map(opt => (
              <button key={opt.key} onClick={() => setFilterStatus(opt.key)}
                className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors ${
                  filterStatus === opt.key
                    ? opt.key === 'overdue' ? 'bg-red-600 text-white' : 'bg-purple-600 text-white'
                    : 'text-slate-400 hover:text-white'
                }`}>
                {opt.label}
              </button>
            ))}
          </div>

          {/* Month tabs */}
          <div className="flex items-center gap-1 bg-[#0a1628] rounded-lg p-0.5 overflow-x-auto">
            <button onClick={() => setFilterMonth('all')}
              className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors whitespace-nowrap ${
                filterMonth === 'all' ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-white'
              }`}>
              Todos los meses
            </button>
            {uniqueMonths.map(m => (
              <button key={m} onClick={() => setFilterMonth(filterMonth === m ? 'all' : m)}
                className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors whitespace-nowrap ${
                  filterMonth === m ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-white'
                }`}>
                {m}
              </button>
            ))}
          </div>

          {/* Employee search */}
          <div className="relative ml-auto">
            <input
              type="text"
              placeholder="Buscar empleado..."
              value={filterEmp}
              onChange={e => setFilterEmp(e.target.value)}
              className="bg-[#0a1628] border border-white/10 rounded-lg pl-3 pr-7 py-1.5 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-purple-500/50 w-40"
            />
            {filterEmp && (
              <button onClick={() => setFilterEmp('')} className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-500 hover:text-white text-xs">✕</button>
            )}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/10">
                <th className="px-4 py-2 text-left text-xs text-slate-400">Estado</th>
                <th className="px-4 py-2 text-left text-xs text-slate-400">Empleado</th>
                <th className="px-4 py-2 text-left text-xs text-slate-400">Mes</th>
                <th className="px-4 py-2 text-left text-xs text-slate-400">Competencia</th>
                <th className="px-3 py-2 text-center text-xs text-slate-400">Actual</th>
                <th className="px-3 py-2 text-center text-xs text-slate-400">Esperado</th>
              </tr>
            </thead>
            <tbody>
              {visibleChanges.length === 0 ? (
                <tr><td colSpan={6} className="px-4 py-8 text-center text-slate-500 text-sm">Sin resultados con los filtros aplicados</td></tr>
              ) : visibleChanges.map((ch, i) => (
                <tr key={i} className={`border-b border-white/5 hover:bg-slate-700/20 ${ch.overdue ? 'bg-red-950/10' : ''}`}>
                  <td className="px-4 py-1.5">
                    <span className={`text-xs px-2 py-0.5 rounded font-medium ${
                      ch.overdue ? 'bg-red-500/15 text-red-400' : 'bg-yellow-500/10 text-yellow-400'
                    }`}>
                      {ch.overdue ? 'Atrasado' : 'Pendiente'}
                    </span>
                  </td>
                  <td className="px-4 py-1.5 text-white text-sm">{ch.employee}</td>
                  <td className="px-4 py-1.5 text-purple-400 text-sm font-medium">{ch.month}</td>
                  <td className="px-4 py-1.5 text-slate-300 text-sm">{ch.competency}</td>
                  <td className="px-3 py-1.5 text-center text-yellow-400 font-medium">{ch.current_level}</td>
                  <td className="px-3 py-1.5 text-center text-green-400 font-medium">{ch.to_level}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function ChartsTab({ analisis }) {
  const [hiddenEmps, setHiddenEmps] = useState(new Set())
  const [empSearch, setEmpSearch]   = useState('')
  const [filterOpen, setFilterOpen] = useState(false)
  const [chartView, setChartView]   = useState('schedule') // 'schedule' | 'target'

  const allEmps = [...analisis.employees].sort((a, b) => a.employee.localeCompare(b.employee))

  const toggleEmp = (name) => {
    setHiddenEmps(prev => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }
  const showAll  = () => setHiddenEmps(new Set())
  const hideAll  = () => setHiddenEmps(new Set(allEmps.map(e => e.employee)))

  const filteredSearch = allEmps.filter(e =>
    e.employee.toLowerCase().includes(empSearch.toLowerCase())
  )
  const visibleCount = allEmps.length - hiddenEmps.size

  // ── 1. Distribución global (donut) ──
  const pieData = [
    { name: 'En target', value: analisis.global.on_target,                         fill: CHART_COLORS.on },
    { name: 'Debajo target', value: analisis.global.total_entries - analisis.global.on_target, fill: CHART_COLORS.below },
  ]

  // ── 2. Cumpl. por empleado (filtrado) ──
  const empData = [...analisis.employees]
    .filter(e => !hiddenEmps.has(e.employee))
    .sort((a, b) => {
      const aVal = chartView === 'schedule' ? a.schedule_compliance_pct : a.compliance_pct
      const bVal = chartView === 'schedule' ? b.schedule_compliance_pct : b.compliance_pct
      return aVal - bVal
    })
    .map(e => ({
      name: e.employee.split(' ').slice(0, 2).join(' '),
      fullName: e.employee,
      schedule: e.schedule_compliance_pct,
      target: e.compliance_pct,
      below: e.below_schedule,
    }))

  // ── 3. Bottom 10 competencias ──
  const compData = analisis.competencies
    .slice(0, 10)
    .map(c => ({
      name: c.competency.length > 22 ? c.competency.slice(0, 20) + '…' : c.competency,
      fullName: c.competency,
      cumpl: c.compliance_pct,
      personas: c.total,
    }))

  // ── 4. Distribución de avances (si hay semana anterior) ──
  const advTotal  = analisis.employees.reduce((s, e) => s + (e.advanced_this_week  || 0), 0)
  const regTotal  = analisis.employees.reduce((s, e) => s + (e.regressed_this_week || 0), 0)
  const unchTotal = analisis.employees.reduce((s, e) => s + (e.unchanged || 0), 0)
  const moveData = [
    { name: 'Avanzó',     value: advTotal,  fill: CHART_COLORS.adv },
    { name: 'Sin cambio', value: unchTotal, fill: '#64748b' },
    { name: 'Retrocedió', value: regTotal,  fill: CHART_COLORS.below },
  ].filter(d => d.value > 0)

  // ── 5. Atrasados por empleado (bar sparkline) ──
  const atrasadosData = [...analisis.employees]
    .filter(e => e.below_schedule > 0 && !hiddenEmps.has(e.employee))
    .sort((a, b) => b.below_schedule - a.below_schedule)
    .slice(0, 12)
    .map(e => ({
      name: e.employee.split(' ').slice(0, 2).join(' '),
      fullName: e.employee,
      atrasados: e.below_schedule,
      total: e.total_competencies,
    }))

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    return (
      <div className="bg-[#0a1628] border border-white/10 rounded-lg px-3 py-2 text-xs">
        <p className="text-white font-medium mb-1">{payload[0]?.payload?.fullName || label}</p>
        {payload.map((p, i) => (
          <p key={i} style={{ color: p.color }}>{p.name}: {p.value}%</p>
        ))}
      </div>
    )
  }

  const CompTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null
    const d = payload[0]?.payload
    return (
      <div className="bg-[#0a1628] border border-white/10 rounded-lg px-3 py-2 text-xs">
        <p className="text-white font-medium mb-1">{d?.fullName}</p>
        <p className="text-purple-400">Cumplimiento: {d?.cumpl}%</p>
        <p className="text-slate-400">{d?.personas} personas</p>
      </div>
    )
  }

  const AtrasadosTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null
    const d = payload[0]?.payload
    return (
      <div className="bg-[#0a1628] border border-white/10 rounded-lg px-3 py-2 text-xs">
        <p className="text-white font-medium mb-1">{d?.fullName}</p>
        <p className="text-yellow-400">Atrasados: {d?.atrasados} de {d?.total}</p>
      </div>
    )
  }

  return (
    <div className="space-y-5">

      {/* ── Filtro de empleados ── */}
      <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2a1 1 0 01-.293.707L13 13.414V19a1 1 0 01-.553.894l-4 2A1 1 0 017 21v-7.586L3.293 6.707A1 1 0 013 6V4z" />
            </svg>
            <span className="text-sm font-medium text-white">Filtrar empleados</span>
            <span className="text-xs text-slate-500">
              {visibleCount === allEmps.length
                ? `Todos (${allEmps.length})`
                : `${visibleCount} de ${allEmps.length} visibles`}
            </span>
            {hiddenEmps.size > 0 && (
              <span className="px-2 py-0.5 rounded-full text-xs bg-yellow-500/15 text-yellow-400">
                {hiddenEmps.size} ocultos
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button onClick={showAll} className="text-xs text-blue-400 hover:text-blue-300 transition-colors">
              Mostrar todos
            </button>
            <span className="text-slate-700">·</span>
            <button onClick={hideAll} className="text-xs text-slate-400 hover:text-slate-300 transition-colors">
              Ocultar todos
            </button>
            <button
              onClick={() => setFilterOpen(o => !o)}
              className="ml-2 px-3 py-1 rounded-lg text-xs font-medium bg-slate-700/50 hover:bg-slate-700 text-slate-300 transition-colors"
            >
              {filterOpen ? 'Ocultar ▲' : 'Seleccionar ▼'}
            </button>
          </div>
        </div>

        {filterOpen && (
          <div className="mt-3 space-y-3">
            <input
              type="text"
              placeholder="Buscar empleado..."
              value={empSearch}
              onChange={e => setEmpSearch(e.target.value)}
              className="w-full bg-[#0a1628] border border-white/10 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-blue-500/50"
            />
            <div className="flex flex-wrap gap-2 max-h-40 overflow-y-auto pr-1">
              {filteredSearch.map(e => (
                <button
                  key={e.employee}
                  onClick={() => toggleEmp(e.employee)}
                  className={`px-2.5 py-1 rounded-full text-xs font-medium border transition-colors ${
                    hiddenEmps.has(e.employee)
                      ? 'bg-slate-800/50 border-slate-700 text-slate-500 line-through'
                      : 'bg-blue-600/20 border-blue-500/40 text-blue-300 hover:bg-blue-600/30'
                  }`}
                >
                  {e.employee.split(' ')[0]}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Row 1: Donuts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

        {/* Donut — En target vs Debajo */}
        <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-5">
          <h3 className="text-sm font-semibold text-white mb-1">Distribución vs Target final</h3>
          <p className="text-xs text-slate-500 mb-3">¿Cuántos alcanzaron ya el nivel objetivo?</p>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" innerRadius={55} outerRadius={80}
                dataKey="value" paddingAngle={3}>
                {pieData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
              </Pie>
              <Tooltip formatter={(v) => [`${v} entradas`]} contentStyle={{ background: '#0a1628', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }} />
              <Legend iconType="circle" iconSize={8} formatter={(v) => <span style={{ color: '#94a3b8', fontSize: 11 }}>{v}</span>} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Donut — Movimiento semana */}
        <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-5">
          <h3 className="text-sm font-semibold text-white mb-1">Movimiento vs semana anterior</h3>
          <p className="text-xs text-slate-500 mb-3">
            {analisis.has_previous_week ? 'Cambios de nivel detectados esta semana' : 'Sin semana anterior para comparar'}
          </p>
          {analisis.has_previous_week && moveData.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={moveData} cx="50%" cy="50%" innerRadius={55} outerRadius={80}
                  dataKey="value" paddingAngle={3}>
                  {moveData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
                </Pie>
                <Tooltip formatter={(v) => [`${v} entradas`]} contentStyle={{ background: '#0a1628', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }} />
                <Legend iconType="circle" iconSize={8} formatter={(v) => <span style={{ color: '#94a3b8', fontSize: 11 }}>{v}</span>} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-[200px] text-slate-600 text-sm">
              Sube la semana anterior para ver movimientos
            </div>
          )}
        </div>
      </div>

      {/* Row 2: Cumplimiento por empleado con toggle */}
      <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-5">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-sm font-semibold text-white mb-1">Cumplimiento por empleado</h3>
            <p className="text-xs text-slate-500">
              {chartView === 'schedule'
                ? <><span style={{ color: CHART_COLORS.on }}>■</span> % al ritmo del plan anual · ordenado ↑</>
                : <><span style={{ color: CHART_COLORS.adv }}>■</span> % en target final · ordenado ↑</>
              }
            </p>
          </div>
          <div className="flex gap-1 bg-[#0a1628] rounded-lg p-1">
            <button
              onClick={() => setChartView('schedule')}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                chartView === 'schedule' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              Ritmo año
            </button>
            <button
              onClick={() => setChartView('target')}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                chartView === 'target' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              Target final
            </button>
          </div>
        </div>
        {empData.length === 0 ? (
          <p className="text-slate-600 text-sm text-center py-8">Sin empleados visibles — usa el filtro para mostrar algunos</p>
        ) : (
          <ResponsiveContainer width="100%" height={Math.max(200, empData.length * 30)}>
            <BarChart data={empData} layout="vertical" margin={{ left: 4, right: 50, top: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
              <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`}
                tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="name" width={90}
                tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
              {chartView === 'schedule' ? (
                <Bar dataKey="schedule" name="Cumpl. Año" fill={CHART_COLORS.on} radius={[0, 4, 4, 0]} barSize={14}
                  label={{ position: 'right', fill: '#64748b', fontSize: 10, formatter: v => `${v}%` }}
                />
              ) : (
                <Bar dataKey="target" name="Cumpl. Target" fill={CHART_COLORS.adv} radius={[0, 4, 4, 0]} barSize={14}
                  label={{ position: 'right', fill: '#64748b', fontSize: 10, formatter: v => `${v}%` }}
                />
              )}
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Row 3: Atrasados por empleado */}
      {atrasadosData.length > 0 && (
        <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-5">
          <h3 className="text-sm font-semibold text-white mb-1">Competencias atrasadas por empleado</h3>
          <p className="text-xs text-slate-500 mb-4">Top personas con más competencias debajo del ritmo del plan (aplica filtro)</p>
          <ResponsiveContainer width="100%" height={Math.max(160, atrasadosData.length * 28)}>
            <BarChart data={atrasadosData} layout="vertical" margin={{ left: 4, right: 50, top: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
              <XAxis type="number" allowDecimals={false}
                tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="name" width={90}
                tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip content={<AtrasadosTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
              <Bar dataKey="atrasados" name="Atrasados" radius={[0, 4, 4, 0]} barSize={14}
                label={{ position: 'right', fill: '#64748b', fontSize: 10 }}>
                {atrasadosData.map((entry, i) => (
                  <Cell key={i} fill={entry.atrasados >= 5 ? CHART_COLORS.below : '#f59e0b'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Row 4: Bottom 10 competencias */}
      <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-5">
        <h3 className="text-sm font-semibold text-white mb-1">Competencias con menor cumplimiento</h3>
        <p className="text-xs text-slate-500 mb-4">Las 10 áreas donde más personas están debajo del target final</p>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={compData} layout="vertical" margin={{ left: 4, right: 50, top: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
            <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`}
              tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
            <YAxis type="category" dataKey="name" width={140}
              tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} tickLine={false} />
            <Tooltip content={<CompTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
            <Bar dataKey="cumpl" name="% Cumplimiento" radius={[0, 4, 4, 0]} barSize={14}
              label={{ position: 'right', fill: '#64748b', fontSize: 10, formatter: v => `${v}%` }}>
              {compData.map((entry, i) => (
                <Cell key={i} fill={entry.cumpl >= 80 ? CHART_COLORS.adv : entry.cumpl >= 50 ? '#f59e0b' : CHART_COLORS.below} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

    </div>
  )
}

function ComplianceBadge({ pct }) {
  const cls = pct >= 80 ? 'text-green-400 bg-green-500/10' : pct >= 50 ? 'text-yellow-400 bg-yellow-500/10' : 'text-red-400 bg-red-500/10'
  return <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${cls}`}>{pct}%</span>
}

function EmployeeRow({ emp, isExpanded, onToggle }) {
  return (
    <>
      <tr className="border-b border-white/5 hover:bg-slate-700/20 cursor-pointer" onClick={onToggle}>
        <td className="px-4 py-2 text-white text-sm font-medium">
          <span className="mr-1 text-slate-500">{isExpanded ? '▾' : '▸'}</span>
          {emp.employee}
        </td>
        <td className="px-3 py-2 text-center text-slate-400 text-xs max-w-[120px] truncate">{emp.role?.split(' - ')[0] || '—'}</td>
        <td className="px-3 py-2 text-center text-green-400 font-medium">{emp.on_target}</td>
        <td className="px-3 py-2 text-center text-red-400 font-medium">{emp.below_target}</td>
        <td className="px-3 py-2 text-center text-slate-300">{emp.total_competencies}</td>
        <td className="px-3 py-2 text-center"><ComplianceBadge pct={emp.compliance_pct} /></td>
        <td className="px-3 py-2 text-center text-yellow-400">{emp.below_schedule}</td>
        <td className="px-3 py-2 text-center text-blue-400">{emp.advanced_this_week || 0}</td>
      </tr>
      {isExpanded && emp.details && (
        <tr>
          <td colSpan={8} className="px-0 py-0">
            <div className="bg-[#0a1628] border-y border-white/5">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-white/5">
                    <th className="px-4 py-1.5 text-left text-slate-500">Competencia</th>
                    <th className="px-2 py-1.5 text-center text-slate-500">Current</th>
                    <th className="px-2 py-1.5 text-center text-slate-500">Target</th>
                    <th className="px-2 py-1.5 text-center text-slate-500">Pronóstico</th>
                    <th className="px-2 py-1.5 text-center text-slate-500">Sem. ant.</th>
                    <th className="px-2 py-1.5 text-center text-slate-500">Estado</th>
                    <th className="px-2 py-1.5 text-center text-slate-500">Vencido</th>
                  </tr>
                </thead>
                <tbody>
                  {emp.details
                    .sort((a, b) => a.status === 'below_target' ? -1 : b.status === 'below_target' ? 1 : 0)
                    .map((d, i) => (
                      <tr key={i} className="border-b border-white/5">
                        <td className="px-4 py-1 text-slate-300">{d.competency}</td>
                        <td className={`px-2 py-1 text-center font-medium ${
                          d.status === 'on_target' ? 'text-green-400' : d.status === 'above_target' ? 'text-blue-400' : 'text-red-400'
                        }`}>{d.current}</td>
                        <td className="px-2 py-1 text-center text-slate-400">{d.target}</td>
                        <td className="px-2 py-1 text-center text-slate-400">{d.forecast}</td>
                        <td className={`px-2 py-1 text-center ${
                          d.advance === 'avanzó' ? 'text-blue-400' : d.advance === 'retrocedió' ? 'text-red-400' : 'text-slate-500'
                        }`}>
                          {d.prev_level != null ? d.prev_level : '—'}
                          {d.advance === 'avanzó' && ' ↑'}
                        </td>
                        <td className="px-2 py-1 text-center">
                          <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                            d.status === 'on_target' ? 'bg-green-500/10 text-green-400' :
                            d.status === 'above_target' ? 'bg-blue-500/10 text-blue-400' :
                            d.schedule_status === 'below_schedule' ? 'bg-red-500/10 text-red-400' :
                            'bg-yellow-500/10 text-yellow-400'
                          }`}>
                            {d.status === 'on_target' ? 'OK' :
                             d.status === 'above_target' ? 'Sobresaliente' :
                             d.schedule_status === 'below_schedule' ? 'Atrasado' : 'Pendiente'}
                          </span>
                        </td>
                        <td className="px-2 py-1 text-center">
                          {d.due_month ? (
                            <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-orange-500/15 text-orange-400">
                              {d.due_month}
                            </span>
                          ) : '—'}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </td>
        </tr>
      )}
    </>
  )
}
