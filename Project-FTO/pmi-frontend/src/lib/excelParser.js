import * as XLSX from 'xlsx'

// Column name → DB field mapping (lowercase normalized)
const COLUMN_MAP = {
  'bos [#]': 'bos_num',
  'bos num': 'bos_num',
  'bos_num': 'bos_num',
  'bos eng [%]': 'bos_eng',
  'bos eng': 'bos_eng',
  'bos_eng': 'bos_eng',
  'qbos [#]': 'qbos_num',
  'qbos num': 'qbos_num',
  'qbos_num': 'qbos_num',
  'qbos eng [%]': 'qbos_eng',
  'qbos eng': 'qbos_eng',
  'qbos_eng': 'qbos_eng',
  'qflags [#]': 'qflags_num',
  'qflags num': 'qflags_num',
  'qflags_num': 'qflags_num',
  'qi/pnc [#]': 'qi_pnc_num',
  'qi pnc [#]': 'qi_pnc_num',
  'qi_pnc_num': 'qi_pnc_num',
  'dh encontrados [#]': 'dh_encontrados',
  'dh encontrados': 'dh_encontrados',
  'dh enc [#]': 'dh_encontrados',
  'dh_encontrados': 'dh_encontrados',
  'dh reparados [#]': 'dh_reparados',
  'dh reparados': 'dh_reparados',
  'dh rep [#]': 'dh_reparados',
  'dh_reparados': 'dh_reparados',
  'curva autonomía [%]': 'curva_autonomia',
  'curva autonomia [%]': 'curva_autonomia',
  'curva aut [%]': 'curva_autonomia',
  'curva de autonomía [%]': 'curva_autonomia',
  'curva_autonomia': 'curva_autonomia',
  'contramedidas [%]': 'contramedidas_defectos',
  'contramedidas defectos [%]': 'contramedidas_defectos',
  'contram [%]': 'contramedidas_defectos',
  'contramedidas_defectos': 'contramedidas_defectos',
  'ips [#]': 'ips_num',
  'ips num': 'ips_num',
  'ips_num': 'ips_num',
  'frr [%]': 'frr',
  'frr': 'frr',
  'dim waste [%]': 'dim_waste',
  'dim_waste': 'dim_waste',
  'sobrepeso [#]': 'sobrepeso',
  'sobrepeso': 'sobrepeso',
  'eventos laika [#]': 'eventos_laika',
  'laika [#]': 'eventos_laika',
  'eventos_laika': 'eventos_laika',
  'casos estudio [#]': 'casos_estudio',
  'casos est [#]': 'casos_estudio',
  'casos de estudios [#]': 'casos_estudio',
  'casos_estudio': 'casos_estudio',
  'qm on target [%]': 'qm_on_target',
  'qm target [%]': 'qm_on_target',
  'qm_on_target': 'qm_on_target',
}

// Known non-indicator columns to skip
const SKIP_COLUMNS = new Set([
  'semana', 'operador', 'line coordinator', 'lc', 'turno',
  'maquina', 'máquina', 'estatus', 'grupo', 'id', 'nombre',
  'nombre operador', 'notas', ''
])

/**
 * Read workbook from File and return sheet names + workbook reference.
 * @param {File} file - The uploaded .xlsx file
 * @returns {Promise<{workbook: object, sheetNames: string[]}>}
 */
export function readWorkbook(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const data = new Uint8Array(e.target.result)
        const workbook = XLSX.read(data, { type: 'array', cellDates: true })
        resolve({ workbook, sheetNames: workbook.SheetNames })
      } catch (err) {
        reject(new Error('Error al leer el archivo Excel: ' + err.message))
      }
    }
    reader.onerror = () => reject(new Error('Error leyendo el archivo'))
    reader.readAsArrayBuffer(file)
  })
}

/**
 * Describe sheets in the workbook (name, row count, detected type).
 */
export function describeSheets(workbook) {
  return workbook.SheetNames.map(name => {
    const sheet = workbook.Sheets[name]
    const raw = XLSX.utils.sheet_to_json(sheet, { header: 1, defval: null })

    // Detect sheet type by scanning first rows
    let type = 'unknown'
    let dataRows = 0
    let headerRowIdx = -1

    for (let i = 0; i < Math.min(raw.length, 10); i++) {
      const row = (raw[i] || []).map(c => String(c || '').toLowerCase().trim())
      if (row.includes('operador') && row.includes('semana')) {
        type = 'operadores'
        headerRowIdx = i
        break
      }
      if (row.includes('line coordinator') && row.includes('semana')) {
        type = 'resumen_lc'
        headerRowIdx = i
        break
      }
      if (row.includes('nombre operador') || row.includes('id')) {
        type = 'catalogo'
        headerRowIdx = i
        break
      }
    }

    if (headerRowIdx >= 0) {
      // Count actual data rows (skip OBJETIVO row and empties)
      const headers = raw[headerRowIdx]
      for (let i = headerRowIdx + 1; i < raw.length; i++) {
        const row = raw[i] || []
        const first = String(row[0] || '').toLowerCase().trim()
        if (first.startsWith('objetivo') || first === '') continue
        if (row.some(c => c != null && c !== '')) dataRows++
      }
    }

    return {
      name,
      type,
      headerRowIdx,
      totalRows: raw.length,
      dataRows,
      description:
        type === 'operadores' ? `Datos por operador (${dataRows} filas)` :
        type === 'resumen_lc' ? `Resumen por LC (${dataRows} filas)` :
        type === 'catalogo' ? `Catálogo de configuración` :
        `Hoja desconocida (${raw.length} filas)`
    }
  })
}

/**
 * Parse a specific sheet and extract structured data.
 * Handles the FTO format: Row 1=title, Row 2=instructions, Row 3=categories,
 * Row 4=headers, Row 5=OBJETIVO, Row 6+=data
 *
 * @param {object} workbook - XLSX workbook
 * @param {string} sheetName - Name of sheet to parse
 * @returns {{headers: string[], rows: object[], sheetName: string, objectives: object, semanas: string[]}}
 */
export function parseSheet(workbook, sheetName) {
  const sheet = workbook.Sheets[sheetName]
  const raw = XLSX.utils.sheet_to_json(sheet, { header: 1, defval: null })

  // Find the header row (contains "Semana" and either "Operador" or "Line Coordinator")
  let headerRowIdx = -1
  for (let i = 0; i < Math.min(raw.length, 10); i++) {
    const row = (raw[i] || []).map(c => String(c || '').toLowerCase().trim())
    if (row.includes('semana') && (row.includes('operador') || row.includes('line coordinator'))) {
      headerRowIdx = i
      break
    }
  }

  if (headerRowIdx === -1) {
    throw new Error(`No se encontraron encabezados válidos en la hoja "${sheetName}". Se necesita una fila con "Semana" y "Operador" o "Line Coordinator".`)
  }

  const headers = raw[headerRowIdx].map(h => String(h || '').trim())

  // Parse OBJETIVO row (usually headerRowIdx + 1)
  const objectives = {}
  const nextRow = raw[headerRowIdx + 1] || []
  const firstCell = String(nextRow[0] || '').toLowerCase().trim()
  let dataStartIdx = headerRowIdx + 1

  if (firstCell.includes('objetivo')) {
    headers.forEach((h, i) => {
      if (nextRow[i] != null && !isNaN(Number(nextRow[i]))) {
        objectives[h] = Number(nextRow[i])
      }
    })
    dataStartIdx = headerRowIdx + 2
  }

  // Parse data rows
  const dataRows = []
  const semanasSet = new Set()

  for (let i = dataStartIdx; i < raw.length; i++) {
    const rawRow = raw[i] || []
    if (!rawRow.some(c => c != null && c !== '')) continue // skip empty rows

    const obj = {}
    headers.forEach((h, idx) => {
      if (!h) return
      let val = rawRow[idx]

      // Convert dates to ISO string
      if (val instanceof Date) {
        val = val.toISOString().split('T')[0]
      }

      obj[h] = val ?? null
    })

    // Track semanas
    const semanaCol = headers.find(h => h.toLowerCase() === 'semana')
    if (semanaCol && obj[semanaCol]) {
      semanasSet.add(obj[semanaCol])
    }

    dataRows.push(obj)
  }

  return {
    headers,
    rows: dataRows,
    sheetName,
    objectives,
    semanas: Array.from(semanasSet).sort()
  }
}

/**
 * Map Excel column headers to database indicator fields.
 * @param {string[]} headers - Column headers from Excel
 * @returns {{mapped: object, unmapped: string[]}}
 */
export function mapColumnsToIndicadores(headers) {
  const mapped = {} // { excelHeader: dbField }
  const unmapped = []

  headers.forEach(header => {
    const normalized = header.toLowerCase().trim()
    if (COLUMN_MAP[normalized]) {
      mapped[header] = COLUMN_MAP[normalized]
    } else if (SKIP_COLUMNS.has(normalized)) {
      // Known non-indicator columns, skip silently
    } else if (normalized === '' || normalized === 'undefined') {
      // Empty header, skip
    } else {
      unmapped.push(header)
    }
  })

  return { mapped, unmapped }
}

/**
 * Match Excel operator names to database operators (fuzzy by name).
 * @param {object[]} excelRows - Parsed Excel rows
 * @param {object[]} dbOperadores - Operators from API
 * @param {string} operadorHeader - The Excel column name for operator
 * @returns {{matched: object[], unmatched: string[]}}
 */
export function matchOperadores(excelRows, dbOperadores, operadorHeader) {
  const matched = []
  const unmatchedNames = new Set()

  // Build lookup by normalized name
  const opLookup = {}
  dbOperadores.forEach(op => {
    opLookup[op.nombre.toLowerCase().trim()] = op
  })

  excelRows.forEach(row => {
    const name = String(row[operadorHeader] || '').trim()
    if (!name) return

    const normalized = name.toLowerCase()
    const dbOp = opLookup[normalized]

    if (dbOp) {
      matched.push({ excelRow: row, operador: dbOp })
    } else {
      unmatchedNames.add(name)
    }
  })

  return { matched, unmatched: Array.from(unmatchedNames) }
}

/**
 * Filter rows by selected semanas.
 */
export function filterBySemanas(rows, semanaHeader, selectedSemanas) {
  if (!selectedSemanas || selectedSemanas.length === 0) return rows
  const allowed = new Set(selectedSemanas)
  return rows.filter(row => {
    const val = row[semanaHeader]
    return val && allowed.has(val)
  })
}

/**
 * Convert matched Excel data to registros ready for batch API.
 * @param {object[]} matchedRows - From matchOperadores
 * @param {object} columnMapping - From mapColumnsToIndicadores
 * @param {string} defaultSemana - Fallback week date string (YYYY-MM-DD)
 * @param {string|null} semanaHeader - Excel column for week
 * @returns {object[]} Array of registro objects for crearRegistrosBatch
 */
export function buildRegistros(matchedRows, columnMapping, defaultSemana, semanaHeader = null) {
  return matchedRows.map(({ excelRow, operador }) => {
    const registro = {
      operador_id: operador.id,
      semana: defaultSemana
    }

    // Use Excel's semana if available
    if (semanaHeader && excelRow[semanaHeader]) {
      const excelDate = excelRow[semanaHeader]
      if (typeof excelDate === 'string' && excelDate.match(/^\d{4}-\d{2}-\d{2}/)) {
        registro.semana = excelDate.split('T')[0]
      }
    }

    // Map indicator values
    Object.entries(columnMapping).forEach(([excelHeader, dbField]) => {
      const val = excelRow[excelHeader]
      if (val != null && val !== '' && !isNaN(Number(val))) {
        registro[dbField] = Number(val)
      }
    })

    return registro
  })
}
