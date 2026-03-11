import { useState, useRef, useEffect } from 'react'
import {
  readWorkbook, describeSheets, parseSheet,
  mapColumnsToIndicadores, matchOperadores,
  filterBySemanas, buildRegistros
} from '../lib/excelParser'
import * as api from '../lib/api'

/**
 * ExcelUploadModal — Multi-step modal for uploading Excel data.
 * Steps: upload → select sheets → select semanas → preview → result
 *
 * Now fetches operadores & LCs directly from API (not from parent).
 * Supports multi-sheet selection.
 */
export default function ExcelUploadModal({ isOpen, onClose, onSuccess, cedulaId, semana }) {
  const [step, setStep] = useState('upload')
  const [file, setFile] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [parsing, setParsing] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)

  // DB entities (fetched from API)
  const [dbOperadores, setDbOperadores] = useState([])
  const [dbLCs, setDbLCs] = useState([])
  const [loadingEntities, setLoadingEntities] = useState(false)

  // Workbook data
  const [workbook, setWorkbook] = useState(null)
  const [sheetsInfo, setSheetsInfo] = useState([])
  const [selectedSheets, setSelectedSheets] = useState([])   // multi-select

  // Parsed sheet data (combined from all selected sheets)
  const [allParsedSheets, setAllParsedSheets] = useState([])  // [{sheetName, ...parsedData}]
  const [availableSemanas, setAvailableSemanas] = useState([])
  const [selectedSemanas, setSelectedSemanas] = useState([])

  // Mapping & matching
  const [columnMapping, setColumnMapping] = useState(null)
  const [unmappedCols, setUnmappedCols] = useState([])
  const [matchedRows, setMatchedRows] = useState([])
  const [unmatchedNames, setUnmatchedNames] = useState([])
  const [registros, setRegistros] = useState([])

  // Result
  const [resultMsg, setResultMsg] = useState(null)

  const fileInputRef = useRef(null)

  // Fetch operadores & LCs from API when modal opens
  useEffect(() => {
    if (!isOpen || !cedulaId) return
    setLoadingEntities(true)
    Promise.all([
      api.getOperadores(cedulaId).catch(() => ({ data: [] })),
      api.getLineCoordinators(cedulaId).catch(() => ({ data: [] }))
    ]).then(([opsRes, lcsRes]) => {
      setDbOperadores(opsRes.data || [])
      setDbLCs(lcsRes.data || [])
    }).finally(() => setLoadingEntities(false))
  }, [isOpen, cedulaId])

  function resetState() {
    setStep('upload'); setFile(null); setDragOver(false)
    setParsing(false); setSaving(false); setError(null)
    setWorkbook(null); setSheetsInfo([]); setSelectedSheets([])
    setAllParsedSheets([]); setAvailableSemanas([]); setSelectedSemanas([])
    setColumnMapping(null); setUnmappedCols([]); setMatchedRows([])
    setUnmatchedNames([]); setRegistros([]); setResultMsg(null)
  }

  function handleClose() { resetState(); onClose() }

  // ─── STEP 1: File Upload ───────────────────────────
  function handleFileSelect(f) {
    if (!f) return
    if (!f.name.match(/\.xlsx?$/i)) {
      setError('Solo se aceptan archivos .xlsx o .xls')
      return
    }
    setFile(f)
    setError(null)
    loadWorkbook(f)
  }

  async function loadWorkbook(f) {
    setParsing(true)
    setError(null)
    try {
      const { workbook: wb } = await readWorkbook(f)
      const info = describeSheets(wb)
      setWorkbook(wb)
      setSheetsInfo(info)

      // Pre-select loadable sheets
      const loadable = info.filter(s => s.type === 'operadores' || s.type === 'resumen_lc')
      setSelectedSheets(loadable.map(s => s.name))
      setStep('sheets')
    } catch (err) {
      setError(err.message || 'Error al leer el archivo')
    }
    setParsing(false)
  }

  // ─── STEP 2: Select Sheets (multi) ────────────────
  function toggleSheet(name) {
    setSelectedSheets(prev =>
      prev.includes(name) ? prev.filter(n => n !== name) : [...prev, name]
    )
  }

  function handleSheetsConfirm() {
    if (selectedSheets.length === 0) {
      setError('Selecciona al menos una hoja')
      return
    }
    parseSelectedSheets()
  }

  // ─── Parse all selected sheets ─────────────────────
  function parseSelectedSheets() {
    setParsing(true)
    setError(null)
    try {
      const parsed = []
      const allSemanas = new Set()

      for (const name of selectedSheets) {
        const p = parseSheet(workbook, name)
        parsed.push(p)
        p.semanas.forEach(s => allSemanas.add(s))
      }

      setAllParsedSheets(parsed)
      const sortedSemanas = Array.from(allSemanas).sort()

      if (sortedSemanas.length > 1) {
        setAvailableSemanas(sortedSemanas)
        setSelectedSemanas([...sortedSemanas])
        setStep('semanas')
      } else {
        setSelectedSemanas(sortedSemanas)
        buildPreview(parsed, sortedSemanas)
      }
    } catch (err) {
      setError(err.message || 'Error al procesar las hojas')
    }
    setParsing(false)
  }

  // ─── STEP 3: Select Semanas ────────────────────────
  function handleSemanasConfirm() {
    if (selectedSemanas.length === 0) {
      setError('Selecciona al menos una semana')
      return
    }
    buildPreview(allParsedSheets, selectedSemanas)
  }

  function toggleSemana(s) {
    setSelectedSemanas(prev =>
      prev.includes(s) ? prev.filter(x => x !== s) : [...prev, s]
    )
  }

  function toggleAllSemanas() {
    setSelectedSemanas(prev =>
      prev.length === availableSemanas.length ? [] : [...availableSemanas]
    )
  }

  // ─── Build preview data (combined from all sheets) ─
  function buildPreview(parsedSheets, semanas) {
    setError(null)
    try {
      let allMatched = []
      let allUnmatched = new Set()
      let allMapped = {}
      let allUnmappedCols = new Set()
      let allRegs = []

      for (const parsed of parsedSheets) {
        const sheetInfo = sheetsInfo.find(s => s.name === parsed.sheetName)
        const isLC = sheetInfo?.type === 'resumen_lc'

        // Map columns
        const { mapped, unmapped } = mapColumnsToIndicadores(parsed.headers)
        Object.assign(allMapped, mapped)
        unmapped.forEach(u => allUnmappedCols.add(u))

        if (Object.keys(mapped).length === 0) continue

        // Find entity column
        const entityHeader = isLC
          ? parsed.headers.find(h => h.toLowerCase().trim() === 'line coordinator')
          : parsed.headers.find(h => {
              const n = h.toLowerCase().trim()
              return n === 'operador' || n === 'nombre'
            })

        if (!entityHeader) continue

        // Find semana column
        const semanaHeader = parsed.headers.find(h => h.toLowerCase().trim() === 'semana') || null

        // Filter by selected semanas
        const filteredRows = filterBySemanas(parsed.rows, semanaHeader, semanas)

        if (isLC) {
          // For Resumen LC sheet: match LC names → find their operators → create registros per operator
          // Note: Resumen LC in DB is a VIEW (auto-calculated from operator data).
          // We'll show this info to the user but can't directly insert LC summaries.
          // Instead we show what would be loaded if it were operator data.

          // Match LC names
          const lcLookup = {}
          dbLCs.forEach(lc => { lcLookup[lc.nombre.toLowerCase().trim()] = lc })

          let lcMatchCount = 0
          filteredRows.forEach(row => {
            const name = String(row[entityHeader] || '').trim()
            if (!name) return
            if (lcLookup[name.toLowerCase()]) {
              lcMatchCount++
            } else {
              allUnmatched.add(name + ' (LC)')
            }
          })

          // We can't insert LC-level data directly (it's a VIEW).
          // But we inform the user that LC summary auto-calculates from operator data.
          // Skip adding registros for LC sheet.
          continue
        }

        // Operadores sheet: match and build registros
        const { matched, unmatched } = matchOperadores(filteredRows, dbOperadores, entityHeader)
        allMatched = allMatched.concat(matched)
        unmatched.forEach(n => allUnmatched.add(n))

        const regs = buildRegistros(matched, mapped, semana, semanaHeader)
        allRegs = allRegs.concat(regs)
      }

      if (Object.keys(allMapped).length === 0) {
        setError('No se encontraron columnas de indicadores SQDCM reconocidas en ninguna hoja.')
        return
      }

      setColumnMapping(allMapped)
      setUnmappedCols(Array.from(allUnmappedCols))
      setMatchedRows(allMatched)
      setUnmatchedNames(Array.from(allUnmatched))
      setRegistros(allRegs)
      setStep('preview')
    } catch (err) {
      setError(err.message || 'Error procesando datos')
    }
  }

  // ─── STEP 5: Confirm & Save ────────────────────────
  async function handleConfirm() {
    if (registros.length === 0) return
    setSaving(true)
    setError(null)
    try {
      const result = await api.crearRegistrosBatch(registros)
      setResultMsg({
        type: 'success',
        count: result.count || registros.length,
        text: `${result.count || registros.length} registro(s) guardados correctamente.`
      })
      setStep('result')
      if (onSuccess) onSuccess()
    } catch (err) {
      setError('Error al guardar: ' + (err.message || 'Error desconocido'))
    }
    setSaving(false)
  }

  if (!isOpen) return null

  const mappedCount = columnMapping ? Object.keys(columnMapping).length : 0
  const hasLCSheet = selectedSheets.some(name => {
    const info = sheetsInfo.find(s => s.name === name)
    return info?.type === 'resumen_lc'
  })

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={handleClose} />

      <div className="relative bg-[#0f1d32] rounded-2xl border border-white/10 shadow-2xl w-full max-w-3xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
          <div>
            <h2 className="text-lg font-bold text-white">Cargar datos desde Excel</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              {step === 'upload' && 'Sube tu archivo .xlsx con los datos semanales'}
              {step === 'sheets' && 'Selecciona las hojas que quieres cargar'}
              {step === 'semanas' && 'Selecciona las semanas que quieres importar'}
              {step === 'preview' && 'Revisa los datos antes de guardar'}
              {step === 'result' && 'Resultado de la carga'}
            </p>
          </div>
          {/* Step indicators */}
          <div className="flex items-center gap-2 mr-8">
            {['upload', 'sheets', 'semanas', 'preview'].map((s, i) => (
              <div key={s} className={`w-2 h-2 rounded-full transition-colors ${
                step === s ? 'bg-blue-500' :
                ['upload','sheets','semanas','preview','result'].indexOf(step) > i ? 'bg-green-500' : 'bg-slate-600'
              }`} />
            ))}
          </div>
          <button onClick={handleClose} className="p-2 text-slate-400 hover:text-white transition-colors rounded-lg hover:bg-white/5">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-6 py-5">
          {/* Error banner */}
          {error && (
            <div className="mb-4 bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-red-400 text-sm flex items-start gap-2">
              <svg className="w-5 h-5 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.072 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
              <span>{error}</span>
            </div>
          )}

          {/* Loading entities */}
          {loadingEntities && (
            <div className="mb-4 bg-blue-500/10 border border-blue-500/30 rounded-lg px-4 py-3 text-blue-400 text-sm flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
              Cargando operadores y LCs de la base de datos...
            </div>
          )}

          {/* ──── STEP: Upload ──── */}
          {step === 'upload' && (
            <div>
              <div
                onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFileSelect(e.dataTransfer.files?.[0]) }}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                onDragLeave={(e) => { e.preventDefault(); setDragOver(false) }}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${
                  dragOver ? 'border-blue-500 bg-blue-500/10' : 'border-slate-600 hover:border-slate-500 hover:bg-white/5'
                }`}
              >
                {parsing ? (
                  <div className="flex flex-col items-center gap-3">
                    <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
                    <p className="text-sm text-slate-300">Leyendo archivo...</p>
                  </div>
                ) : (
                  <>
                    <svg className="w-12 h-12 text-slate-500 mx-auto mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    <p className="text-sm text-slate-300 mb-1">
                      Arrastra tu archivo Excel aquí o <span className="text-blue-400 underline">haz clic para seleccionar</span>
                    </p>
                    <p className="text-xs text-slate-500">Formatos: .xlsx, .xls — Formato FTO 2026</p>
                  </>
                )}
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".xlsx,.xls"
                onChange={(e) => handleFileSelect(e.target.files?.[0])}
                className="hidden"
              />

              {/* DB status */}
              {!loadingEntities && dbOperadores.length === 0 && (
                <div className="mt-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg px-4 py-3 text-yellow-400 text-sm">
                  No se encontraron operadores en la base de datos para esta cédula.
                  Primero agrega operadores en la sección de Administración para poder hacer match con el Excel.
                </div>
              )}
              {!loadingEntities && dbOperadores.length > 0 && (
                <div className="mt-4 bg-slate-800/50 rounded-lg px-4 py-3 text-xs text-slate-400">
                  Base de datos: <span className="text-white font-medium">{dbOperadores.length} operadores</span> y <span className="text-white font-medium">{dbLCs.length} LCs</span> encontrados en esta cédula.
                </div>
              )}
            </div>
          )}

          {/* ──── STEP: Select Sheets (MULTI) ──── */}
          {step === 'sheets' && (
            <div className="space-y-3">
              <p className="text-sm text-slate-300 mb-4">
                El archivo tiene {sheetsInfo.length} hojas. Selecciona las que quieres cargar:
              </p>
              {sheetsInfo.map(sheet => {
                const loadable = sheet.type === 'operadores' || sheet.type === 'resumen_lc'
                const selected = selectedSheets.includes(sheet.name)
                return (
                  <button
                    key={sheet.name}
                    onClick={() => loadable && toggleSheet(sheet.name)}
                    disabled={!loadable}
                    className={`w-full text-left px-4 py-4 rounded-xl border transition-all ${
                      !loadable
                        ? 'border-white/5 opacity-40 cursor-not-allowed'
                        : selected
                          ? 'border-blue-500/50 bg-blue-500/10'
                          : 'border-white/10 hover:border-slate-500 hover:bg-white/5 cursor-pointer'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      {/* Checkbox */}
                      {loadable && (
                        <div className={`w-5 h-5 rounded border-2 flex items-center justify-center flex-shrink-0 transition-colors ${
                          selected ? 'bg-blue-500 border-blue-500' : 'border-slate-500'
                        }`}>
                          {selected && (
                            <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                            </svg>
                          )}
                        </div>
                      )}
                      <div className="flex-1">
                        <p className="text-white font-medium">{sheet.name}</p>
                        <p className="text-xs text-slate-400 mt-0.5">{sheet.description}</p>
                      </div>
                      {loadable ? (
                        <span className={`px-2 py-0.5 text-[10px] rounded-full font-medium ${
                          sheet.type === 'operadores'
                            ? 'bg-blue-500/20 text-blue-400'
                            : 'bg-green-500/20 text-green-400'
                        }`}>
                          {sheet.type === 'operadores' ? 'Operadores' : 'Resumen LC'}
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 text-[10px] rounded-full bg-slate-700 text-slate-500">
                          No importable
                        </span>
                      )}
                    </div>
                  </button>
                )
              })}

              {/* Info about Resumen LC */}
              {selectedSheets.some(name => sheetsInfo.find(s => s.name === name)?.type === 'resumen_lc') && (
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg px-4 py-3 text-blue-300 text-xs">
                  <span className="font-semibold">Nota:</span> El Resumen por LC en el Dashboard se calcula automáticamente
                  a partir de los datos de operadores. Al cargar la hoja de Operadores, los promedios por LC se actualizan solos.
                </div>
              )}
            </div>
          )}

          {/* ──── STEP: Select Semanas ──── */}
          {step === 'semanas' && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <p className="text-sm text-slate-300">
                  Se encontraron <span className="text-white font-medium">{availableSemanas.length} semanas</span> en las hojas seleccionadas.
                </p>
                <button
                  onClick={toggleAllSemanas}
                  className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
                >
                  {selectedSemanas.length === availableSemanas.length ? 'Deseleccionar todo' : 'Seleccionar todo'}
                </button>
              </div>

              <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 gap-2 max-h-64 overflow-y-auto pr-1">
                {availableSemanas.map(s => {
                  const selected = selectedSemanas.includes(s)
                  return (
                    <button
                      key={s}
                      onClick={() => toggleSemana(s)}
                      className={`px-3 py-2 rounded-lg text-xs font-medium transition-all border ${
                        selected
                          ? 'bg-blue-500/20 border-blue-500/40 text-blue-300'
                          : 'bg-slate-800/50 border-white/5 text-slate-500 hover:text-slate-300'
                      }`}
                    >
                      {formatSemanaLabel(s)}
                    </button>
                  )
                })}
              </div>

              <p className="text-xs text-slate-500">
                {selectedSemanas.length} de {availableSemanas.length} semana(s) seleccionadas
              </p>
            </div>
          )}

          {/* ──── STEP: Preview ──── */}
          {step === 'preview' && (
            <div className="space-y-4">
              {/* Summary stats */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <StatCard label="Hojas" value={selectedSheets.join(', ')} />
                <StatCard label="Semanas" value={selectedSemanas.length} />
                <StatCard label="Indicadores" value={mappedCount} color="text-green-400" />
                <StatCard label="Operadores OK" value={matchedRows.length} color="text-blue-400" />
              </div>

              {/* LC auto-calc note */}
              {hasLCSheet && (
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg px-4 py-3 text-blue-300 text-xs">
                  La hoja "Resumen LC" no se importa directamente — los promedios por LC se calculan automáticamente
                  en el Dashboard a partir de los datos de operadores.
                </div>
              )}

              {/* Warnings */}
              {unmatchedNames.length > 0 && (
                <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg px-4 py-3">
                  <p className="text-xs font-semibold text-yellow-400 mb-1">
                    {unmatchedNames.length} nombre(s) no encontrados en la base de datos:
                  </p>
                  <p className="text-xs text-yellow-300/70">{unmatchedNames.join(', ')}</p>
                  <p className="text-[10px] text-yellow-500/50 mt-1">
                    Se omitirán. Agrégalos en Administración para incluirlos.
                  </p>
                </div>
              )}

              {registros.length === 0 && matchedRows.length === 0 && (
                <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3">
                  <p className="text-sm font-semibold text-red-400 mb-1">No hay registros para guardar</p>
                  <p className="text-xs text-red-300/70">
                    Ningún operador del Excel coincide con los de la base de datos.
                    Verifica que los nombres sean exactamente iguales o agrega los operadores en Administración.
                  </p>
                </div>
              )}

              {unmappedCols.length > 0 && (
                <div className="bg-slate-700/30 border border-slate-600/30 rounded-lg px-4 py-3">
                  <p className="text-xs text-slate-400">
                    Columnas no reconocidas: {unmappedCols.join(', ')}
                  </p>
                </div>
              )}

              {/* Column mapping */}
              {columnMapping && Object.keys(columnMapping).length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold text-slate-300 mb-2">Mapeo de columnas</h3>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(columnMapping).map(([excel, db]) => (
                      <span key={excel} className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-green-500/10 border border-green-500/20 text-xs">
                        <span className="text-slate-300">{excel}</span>
                        <svg className="w-3 h-3 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                        </svg>
                        <span className="text-green-400 font-medium">{db}</span>
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Data preview table */}
              {registros.length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold text-slate-300 mb-2">
                    Vista previa ({Math.min(registros.length, 8)} de {registros.length} registros)
                  </h3>
                  <div className="overflow-x-auto rounded-lg border border-white/5">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="bg-slate-800/50 border-b border-white/5">
                          <th className="px-3 py-2 text-left text-slate-400 font-medium">Operador</th>
                          <th className="px-3 py-2 text-left text-slate-400 font-medium">Semana</th>
                          {columnMapping && Object.values(columnMapping).slice(0, 8).map(field => (
                            <th key={field} className="px-2 py-2 text-center text-slate-400 font-medium whitespace-nowrap">{field}</th>
                          ))}
                          {columnMapping && Object.keys(columnMapping).length > 8 && (
                            <th className="px-2 py-2 text-center text-slate-500">+{Object.keys(columnMapping).length - 8}</th>
                          )}
                        </tr>
                      </thead>
                      <tbody>
                        {registros.slice(0, 8).map((reg, i) => {
                          const opName = matchedRows[i]?.operador?.nombre || '—'
                          return (
                            <tr key={i} className="border-b border-white/5 hover:bg-slate-700/20">
                              <td className="px-3 py-2 text-white whitespace-nowrap">{opName}</td>
                              <td className="px-3 py-2 text-slate-300">{reg.semana}</td>
                              {columnMapping && Object.values(columnMapping).slice(0, 8).map(field => (
                                <td key={field} className="px-2 py-2 text-center text-slate-300">
                                  {reg[field] != null ? reg[field] : <span className="text-slate-600">—</span>}
                                </td>
                              ))}
                              {columnMapping && Object.keys(columnMapping).length > 8 && <td />}
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

          {/* ──── STEP: Result ──── */}
          {step === 'result' && resultMsg && (
            <div className="flex flex-col items-center justify-center py-8">
              <div className="w-16 h-16 rounded-full bg-green-500/20 flex items-center justify-center mb-4">
                <svg className="w-8 h-8 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <p className="text-lg font-semibold text-white mb-2">Datos cargados</p>
              <p className="text-sm text-slate-400">{resultMsg.text}</p>
              <p className="text-xs text-slate-500 mt-2">
                El Resumen por LC se actualizará automáticamente en el Dashboard.
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-white/5">
          <div>
            {step !== 'upload' && step !== 'result' && (
              <button
                onClick={() => {
                  setError(null)
                  if (step === 'sheets') resetState()
                  else if (step === 'semanas') setStep('sheets')
                  else if (step === 'preview') setStep(availableSemanas.length > 1 ? 'semanas' : 'sheets')
                }}
                className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors flex items-center gap-1"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                Atrás
              </button>
            )}
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={handleClose}
              className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors rounded-lg hover:bg-white/5"
            >
              {step === 'result' ? 'Cerrar' : 'Cancelar'}
            </button>

            {step === 'sheets' && (
              <button
                onClick={handleSheetsConfirm}
                disabled={selectedSheets.length === 0 || parsing}
                className="px-5 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-600 disabled:text-slate-400 text-white text-sm font-medium rounded-lg transition-colors"
              >
                Continuar ({selectedSheets.length} {selectedSheets.length === 1 ? 'hoja' : 'hojas'})
              </button>
            )}

            {step === 'semanas' && (
              <button
                onClick={handleSemanasConfirm}
                disabled={selectedSemanas.length === 0}
                className="px-5 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-600 disabled:text-slate-400 text-white text-sm font-medium rounded-lg transition-colors"
              >
                Continuar ({selectedSemanas.length} semanas)
              </button>
            )}

            {step === 'preview' && (
              <button
                onClick={handleConfirm}
                disabled={saving || registros.length === 0}
                className="px-5 py-2 bg-green-600 hover:bg-green-500 disabled:bg-slate-600 disabled:text-slate-400 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-2"
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
                    Guardar {registros.length} registro(s)
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function StatCard({ label, value, color = 'text-white' }) {
  return (
    <div className="bg-slate-800/50 rounded-lg px-3 py-2">
      <p className="text-[10px] text-slate-500 uppercase tracking-wide">{label}</p>
      <p className={`text-sm font-bold ${color} truncate`}>{String(value)}</p>
    </div>
  )
}

function formatSemanaLabel(s) {
  try {
    const d = new Date(s + 'T00:00:00')
    const months = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
    return `${d.getDate()} ${months[d.getMonth()]} ${d.getFullYear()}`
  } catch {
    return s
  }
}
