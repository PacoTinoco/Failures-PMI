import { useState, useEffect, useRef } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'
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

  const [deletingSemana, setDeletingSemana] = useState(null)
  const [error, setError] = useState(null)
  const [uploadHistory, setUploadHistory] = useState([])
  const [showHistory, setShowHistory] = useState(false)
  const [syncing, setSyncing] = useState(false)
  const [syncResult, setSyncResult] = useState(null)
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
    api.getQMSemanas(cedulaId).then(res => {
      setSemanas(res.data || [])
    }).catch(() => {})
  }, [cedulaId])

  // Limpiar preview cuando cambia la semana
  useEffect(() => {
    setPreview(null); setSaveResult(null); setSyncResult(null)
  }, [semana])

  useEffect(() => {
    if (!cedulaId || !semana || !calLoaded) return
    if (!semanas.find(s => s.semana === semana)) return
    loadAnalisis()
    loadUploadHistory()
  }, [cedulaId, semana, calLoaded, semanas])

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
      // Refrescar lista de semanas y análisis
      const semanasRes = await api.getQMSemanas(cedulaId)
      setSemanas(semanasRes.data || [])
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

  async function handleDeleteSemana(s) {
    if (!confirm(`¿Eliminar todos los datos de la semana ${s}? Esta acción no se puede deshacer.`)) return
    setDeletingSemana(s)
    try {
      await api.deleteQMSemana(cedulaId, s)
      const semanasRes = await api.getQMSemanas(cedulaId)
      setSemanas(semanasRes.data || [])
      if (semana === s) { setAnalisis(null); setUploadHistory([]); setPreview(null); setSaveResult(null) }
    } catch (err) { setError(err.message) }
    finally { setDeletingSemana(null) }
  }

  const g = analisis?.global || {}

  return (
    <div className="space-y-6">
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

          {/* Historial de guardados */}
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
                <div className="mt-1 space-y-1 max-h-40 overflow-y-auto">
                  {uploadHistory.map((u, i) => (
                    <div key={i} className="bg-[#0a1628] rounded-lg px-3 py-1.5 text-[10px]">
                      <div className="flex items-center justify-between">
                        <span className="text-slate-300">
                          {new Date(u.uploaded_at).toLocaleString('es-MX', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })}
                        </span>
                        <span className="text-slate-500">{u.total_records} reg.</span>
                      </div>
                      {(u.changed_records > 0 || u.new_records > 0) ? (
                        <p className="text-blue-400/70">
                          {u.changed_records > 0 && `${u.changed_records} cambiaron`}
                          {u.changed_records > 0 && u.new_records > 0 && ' · '}
                          {u.new_records > 0 && `${u.new_records} nuevos`}
                        </p>
                      ) : (
                        <p className="text-slate-500">Sin cambios respecto al anterior</p>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {semanas.length > 0 && (
            <div className="mt-3 space-y-1">
              <p className="text-xs text-slate-500 mb-1">Semanas con data cargada:</p>
              {semanas.map(s => (
                <div key={s.semana} className="flex items-center justify-between bg-[#0a1628] rounded-lg px-3 py-1.5">
                  <span className={`text-xs font-medium ${s.semana === semana ? 'text-purple-400' : 'text-slate-300'}`}>
                    {s.semana === semana && <span className="text-purple-400 mr-1">▶</span>}
                    {s.semana}
                    <span className="text-slate-500 ml-1">({s.registros} reg.)</span>
                  </span>
                  <button
                    onClick={() => handleDeleteSemana(s.semana)}
                    disabled={deletingSemana === s.semana}
                    className="text-slate-500 hover:text-red-400 transition-colors text-xs px-1 py-0.5 rounded"
                    title="Eliminar data de esta semana"
                  >
                    {deletingSemana === s.semana ? '...' : '🗑'}
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
              { key: 'charts', label: '📊 Gráficas' },
              { key: 'noproj', label: `Sin proyección (${analisis.not_projected_count ?? 0})` },
              { key: 'upcoming', label: 'Próximos cambios' },
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
          {showTab === 'employees' && (
            <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
              <div className="px-4 py-3 border-b border-white/5">
                <h3 className="text-sm font-semibold text-white">Cumplimiento por empleado</h3>
                <p className="text-xs text-slate-500">Ref: mes {analisis.month_reference} · Clic para ver detalle</p>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-white/10">
                      <th className="px-4 py-2 text-left text-xs text-slate-400">Empleado</th>
                      <th className="px-3 py-2 text-center text-xs text-slate-400">Rol</th>
                      <th className="px-3 py-2 text-center text-xs text-green-400">En target</th>
                      <th className="px-3 py-2 text-center text-xs text-red-400">Debajo</th>
                      <th className="px-3 py-2 text-center text-xs text-slate-400">Total</th>
                      <th className="px-3 py-2 text-center text-xs text-purple-400">% Cumpl</th>
                      <th className="px-3 py-2 text-center text-xs text-yellow-400">Atrasados</th>
                      <th className="px-3 py-2 text-center text-xs text-blue-400">Avanzó</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analisis.employees.map((emp, idx) => (
                      <EmployeeRow key={idx} emp={emp}
                        isExpanded={expandedEmp === emp.employee}
                        onToggle={() => setExpandedEmp(expandedEmp === emp.employee ? null : emp.employee)} />
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Tab: Competencies */}
          {showTab === 'competencies' && (
            <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
              <div className="px-4 py-3 border-b border-white/5">
                <h3 className="text-sm font-semibold text-white">Cumplimiento por competencia</h3>
                <p className="text-xs text-slate-500">Ordenado por menor cumplimiento</p>
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
                    </tr>
                  </thead>
                  <tbody>
                    {analisis.competencies.map((c, idx) => (
                      <tr key={idx} className="border-b border-white/5 hover:bg-slate-700/20">
                        <td className="px-4 py-2 text-white text-sm">{c.competency}</td>
                        <td className="px-3 py-2 text-center text-slate-400">{c.total}</td>
                        <td className="px-3 py-2 text-center text-green-400">{c.on_target}</td>
                        <td className="px-3 py-2 text-center text-red-400">{c.below_target}</td>
                        <td className="px-3 py-2 text-center"><ComplianceBadge pct={c.compliance_pct} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
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
            <div className="space-y-4">
              {analisis.upcoming_changes.length === 0 ? (
                <div className="bg-[#0f1d32] rounded-xl border border-white/5 px-6 py-10 text-center text-slate-500 text-sm">
                  No hay cambios programados en los próximos meses.
                </div>
              ) : analisis.upcoming_changes.map((uc, idx) => (
                <div key={idx} className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
                  <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-white">{uc.transition}</h3>
                    <span className="text-xs text-purple-400 font-medium">{uc.count} cambios esperados</span>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-white/10">
                          <th className="px-4 py-2 text-left text-xs text-slate-400">Empleado</th>
                          <th className="px-4 py-2 text-left text-xs text-slate-400">Competencia</th>
                          <th className="px-3 py-2 text-center text-xs text-slate-400">De</th>
                          <th className="px-3 py-2 text-center text-xs text-slate-400">A</th>
                        </tr>
                      </thead>
                      <tbody>
                        {uc.changes.map((ch, i) => (
                          <tr key={i} className="border-b border-white/5">
                            <td className="px-4 py-1.5 text-white text-sm">{ch.employee}</td>
                            <td className="px-4 py-1.5 text-slate-300 text-sm">{ch.competency}</td>
                            <td className="px-3 py-1.5 text-center text-yellow-400">{ch.from_level}</td>
                            <td className="px-3 py-1.5 text-center text-green-400">{ch.to_level}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      ) : calLoaded ? (
        <div className="bg-[#0f1d32] rounded-xl border border-white/5 px-6 py-16 text-center">
          <svg className="w-12 h-12 text-slate-600 mx-auto mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
          </svg>
          <p className="text-slate-400 text-sm">Sube el archivo de data semanal para ver el análisis.</p>
          <p className="text-slate-500 text-xs mt-1">Selecciona la semana correcta y sube el archivo data_WXX.xlsx</p>
        </div>
      ) : null}
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

function ChartsTab({ analisis }) {
  // ── 1. Distribución global (donut) ──
  const pieData = [
    { name: 'En target', value: analisis.global.on_target,                         fill: CHART_COLORS.on },
    { name: 'Debajo target', value: analisis.global.total_entries - analisis.global.on_target, fill: CHART_COLORS.below },
  ]

  // ── 2. Cumpl. Año vs Target por empleado (top 15 por nombre) ──
  const empData = [...analisis.employees]
    .sort((a, b) => a.schedule_compliance_pct - b.schedule_compliance_pct)
    .slice(0, 20)
    .map(e => ({
      name: e.employee.split(' ')[0],          // solo primer nombre para el eje
      fullName: e.employee,
      año: e.schedule_compliance_pct,
      target: e.compliance_pct,
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
  const advTotal = analisis.employees.reduce((s, e) => s + (e.advanced_this_week  || 0), 0)
  const regTotal = analisis.employees.reduce((s, e) => s + (e.regressed_this_week || 0), 0)
  const unchTotal = analisis.employees.reduce((s, e) => s + (e.unchanged || 0), 0)
  const moveData = [
    { name: 'Avanzó',    value: advTotal,  fill: CHART_COLORS.adv },
    { name: 'Sin cambio', value: unchTotal, fill: '#64748b' },
    { name: 'Retrocedió', value: regTotal,  fill: CHART_COLORS.below },
  ].filter(d => d.value > 0)

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

  return (
    <div className="space-y-5">

      {/* Row 1: Donut distribución + Avances semana */}
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

      {/* Row 2: Cumplimiento por empleado (barra doble) */}
      <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-5">
        <h3 className="text-sm font-semibold text-white mb-1">Cumplimiento por empleado</h3>
        <p className="text-xs text-slate-500 mb-4">
          <span style={{ color: CHART_COLORS.on }}>■</span> % al ritmo del plan (año) ·&nbsp;
          <span style={{ color: CHART_COLORS.adv }}>■</span> % en target final · Ordenado por cumpl. año ↑
        </p>
        <ResponsiveContainer width="100%" height={Math.max(220, empData.length * 26)}>
          <BarChart data={empData} layout="vertical" margin={{ left: 4, right: 24, top: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
            <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`}
              tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
            <YAxis type="category" dataKey="name" width={70}
              tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
            <Bar dataKey="año" name="Cumpl. Año" fill={CHART_COLORS.on} radius={[0, 3, 3, 0]} barSize={8} />
            <Bar dataKey="target" name="Cumpl. Target" fill={CHART_COLORS.adv} radius={[0, 3, 3, 0]} barSize={8} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Row 3: Bottom 10 competencias */}
      <div className="bg-[#0f1d32] rounded-xl border border-white/5 p-5">
        <h3 className="text-sm font-semibold text-white mb-1">Competencias con menor cumplimiento</h3>
        <p className="text-xs text-slate-500 mb-4">Las 10 áreas donde más personas están debajo del target final</p>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={compData} layout="vertical" margin={{ left: 4, right: 40, top: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
            <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`}
              tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
            <YAxis type="category" dataKey="name" width={140}
              tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} tickLine={false} />
            <Tooltip content={<CompTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
            <Bar dataKey="cumpl" name="% Cumplimiento" radius={[0, 4, 4, 0]} barSize={14}>
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
