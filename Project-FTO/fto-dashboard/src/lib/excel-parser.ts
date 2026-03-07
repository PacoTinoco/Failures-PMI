import * as XLSX from 'xlsx'

export interface ParsedOperatorData {
  coordinatorName: string
  operatorName: string
  weekStart: string
  kpiValues: Record<string, number>
  objectives: Record<string, number>
}

export function parseExcelFile(buffer: Buffer): ParsedOperatorData[] {
  const workbook = XLSX.read(buffer, { type: 'buffer' })
  const results: ParsedOperatorData[] = []

  for (const sheetName of workbook.SheetNames) {
    // Skip default or empty sheets
    if (sheetName === 'Sheet1' || sheetName === 'Sheet') {
      continue
    }

    const worksheet = workbook.Sheets[sheetName]
    if (!worksheet) continue

    // Convert sheet to array of arrays
    const rows: any[][] = XLSX.utils.sheet_to_json(worksheet, {
      header: 1,
      defval: ''
    }) as any[][]

    if (rows.length < 4) continue

    // Process rows to find operator blocks
    let rowIdx = 0
    while (rowIdx < rows.length) {
      const currentRow = rows[rowIdx]
      if (!currentRow || currentRow.length === 0) {
        rowIdx++
        continue
      }

      // Check if this row contains 'EO' marker (operator block header)
      const eoIndex = currentRow.findIndex(
        (cell) => String(cell).toUpperCase() === 'EO'
      )

      if (eoIndex !== -1 && rowIdx + 3 < rows.length) {
        // Found operator block
        const operatorName = String(
          currentRow[eoIndex + 1] || ''
        ).trim()

        if (!operatorName) {
          rowIdx++
          continue
        }

        const coordinatorName = sheetName.trim()
        const kpiHeaderRow = rows[rowIdx + 2] || []
        const objectivesRow = rows[rowIdx + 3] || []

        // Extract KPI headers and objectives
        const kpiMap: Record<string, number> = {}
        const objectives: Record<string, number> = {}

        for (let colIdx = eoIndex + 1; colIdx < kpiHeaderRow.length; colIdx++) {
          const kpiCode = String(kpiHeaderRow[colIdx] || '').trim()
          if (kpiCode) {
            const objective = parseFloat(
              String(objectivesRow[colIdx] || '')
            )
            kpiMap[kpiCode] = objective || 0
            objectives[kpiCode] = objective || 0
          }
        }

        // Parse weekly data rows (starting from rowIdx + 4)
        for (let dataRowIdx = rowIdx + 4; dataRowIdx < rows.length; dataRowIdx++) {
          const dataRow = rows[dataRowIdx]
          if (!dataRow || !dataRow[eoIndex]) {
            break
          }

          const weekStart = String(dataRow[eoIndex] || '').trim()
          if (!weekStart || weekStart.toLowerCase() === 'sem') {
            continue
          }

          const kpiValues: Record<string, number> = {}
          for (let colIdx = eoIndex + 1; colIdx < dataRow.length; colIdx++) {
            const kpiCode = String(kpiHeaderRow[colIdx] || '').trim()
            if (kpiCode) {
              const value = parseFloat(String(dataRow[colIdx] || ''))
              kpiValues[kpiCode] = isNaN(value) ? 0 : value
            }
          }

          results.push({
            coordinatorName,
            operatorName,
            weekStart,
            kpiValues,
            objectives
          })
        }

        rowIdx += 4
      } else {
        rowIdx++
      }
    }
  }

  return results
}
