const API_URL = import.meta.env.VITE_API_URL || ''

async function apiRequest(path, options = {}) {
  const headers = {
    'Content-Type': 'application/json',
    ...options.headers
  }

  const response = await fetch(`${API_URL}${path}`, {
    ...options,
    headers
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Error de red' }))
    throw new Error(error.detail || `Error ${response.status}`)
  }

  return response.json()
}

// ============================================================
// EQUIPOS
// ============================================================

// --- CÉDULAS ---

export async function getCedulas() {
  return apiRequest('/equipos/cedulas')
}

export async function getDetalleCedula(cedulaId) {
  return apiRequest(`/equipos/cedulas/${cedulaId}`)
}

export async function createCedula(cedula) {
  return apiRequest('/equipos/cedulas', {
    method: 'POST',
    body: JSON.stringify(cedula)
  })
}

export async function updateCedula(cedulaId, cedula) {
  return apiRequest(`/equipos/cedulas/${cedulaId}`, {
    method: 'PUT',
    body: JSON.stringify(cedula)
  })
}

export async function deleteCedula(cedulaId) {
  return apiRequest(`/equipos/cedulas/${cedulaId}`, {
    method: 'DELETE'
  })
}

// --- LINE COORDINATORS ---

export async function getLineCoordinators(cedulaId) {
  return apiRequest(`/equipos/lc?cedula_id=${cedulaId}`)
}

export async function createLC(lc) {
  return apiRequest('/equipos/lc', {
    method: 'POST',
    body: JSON.stringify(lc)
  })
}

export async function updateLC(lcId, lc) {
  return apiRequest(`/equipos/lc/${lcId}`, {
    method: 'PUT',
    body: JSON.stringify(lc)
  })
}

export async function deleteLC(lcId) {
  return apiRequest(`/equipos/lc/${lcId}`, {
    method: 'DELETE'
  })
}

// --- LÍNEA DE ESTRUCTURA (LS) ---

export async function getLS(cedulaId) {
  return apiRequest(`/equipos/ls?cedula_id=${cedulaId}`)
}

export async function createLS(ls) {
  return apiRequest('/equipos/ls', {
    method: 'POST',
    body: JSON.stringify(ls)
  })
}

export async function updateLS(lsId, ls) {
  return apiRequest(`/equipos/ls/${lsId}`, {
    method: 'PUT',
    body: JSON.stringify(ls)
  })
}

export async function deleteLS(lsId) {
  return apiRequest(`/equipos/ls/${lsId}`, {
    method: 'DELETE'
  })
}

// --- OPERADORES ---

export async function getOperadores(cedulaId, lcId = null) {
  let path = `/equipos/operadores?cedula_id=${cedulaId}`
  if (lcId) path += `&lc_id=${lcId}`
  return apiRequest(path)
}

export async function createOperador(operador) {
  return apiRequest('/equipos/operadores', {
    method: 'POST',
    body: JSON.stringify(operador)
  })
}

export async function updateOperador(operadorId, operador) {
  return apiRequest(`/equipos/operadores/${operadorId}`, {
    method: 'PUT',
    body: JSON.stringify(operador)
  })
}

export async function deleteOperador(operadorId) {
  return apiRequest(`/equipos/operadores/${operadorId}`, {
    method: 'DELETE'
  })
}

// ============================================================
// DASHBOARD
// ============================================================

export async function getResumenLC(cedulaId, semana = null, lcId = null) {
  let path = `/dashboard/resumen-lc?cedula_id=${cedulaId}`
  if (semana) path += `&semana=${semana}`
  if (lcId) path += `&lc_id=${lcId}`
  return apiRequest(path)
}

export async function getSemanas(cedulaId) {
  return apiRequest(`/dashboard/semanas?cedula_id=${cedulaId}`)
}

export async function getIndicadores() {
  return apiRequest('/dashboard/indicadores')
}

export async function getOperadoresSemana(cedulaId, semana, lcId = null) {
  let path = `/dashboard/operadores-semana?cedula_id=${cedulaId}&semana=${semana}`
  if (lcId) path += `&lc_id=${lcId}`
  return apiRequest(path)
}

export async function getTendencia(cedulaId, campo, options = {}) {
  let path = `/dashboard/tendencia?cedula_id=${cedulaId}&campo=${campo}`
  if (options.operadorId) path += `&operador_id=${options.operadorId}`
  if (options.lcId) path += `&lc_id=${options.lcId}`
  if (options.semanas) path += `&semanas=${options.semanas}`
  return apiRequest(path)
}

// ============================================================
// REGISTROS
// ============================================================

export async function getRegistros(params = {}) {
  const searchParams = new URLSearchParams()
  Object.entries(params).forEach(([key, val]) => {
    if (val != null) searchParams.set(key, val)
  })
  return apiRequest(`/registros/?${searchParams.toString()}`)
}

export async function crearRegistro(registro) {
  return apiRequest('/registros/', {
    method: 'POST',
    body: JSON.stringify(registro)
  })
}

export async function crearRegistrosBatch(registros) {
  return apiRequest('/registros/batch', {
    method: 'POST',
    body: JSON.stringify(registros)
  })
}

export async function actualizarRegistro(registroId, update) {
  return apiRequest(`/registros/${registroId}`, {
    method: 'PUT',
    body: JSON.stringify(update)
  })
}

// ============================================================
// DH — Defect Handling
// ============================================================

export async function uploadDHCSV(cedulaId, file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_URL}/registros/dh/upload?cedula_id=${cedulaId}`, {
    method: 'POST',
    body: formData
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Error de red' }))
    throw new Error(error.detail || `Error ${response.status}`)
  }

  return response.json()
}

export async function getDHOperatorEmails(cedulaId) {
  return apiRequest(`/registros/dh/preview-operators?cedula_id=${cedulaId}`)
}

// ============================================================
// QM — Qualification Management
// ============================================================

export async function uploadQMCalendario(cedulaId, file) {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(`${API_URL}/qm/calendario/upload?cedula_id=${cedulaId}`, {
    method: 'POST', body: formData
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Error de red' }))
    throw new Error(error.detail || `Error ${response.status}`)
  }
  return response.json()
}

export async function getQMCalendario(cedulaId, employee = null) {
  let path = `/qm/calendario?cedula_id=${cedulaId}`
  if (employee) path += `&employee=${encodeURIComponent(employee)}`
  return apiRequest(path)
}

export async function updateQMCalendarioEntry(recordId, data) {
  return apiRequest(`/qm/calendario/${recordId}`, {
    method: 'PUT', body: JSON.stringify(data)
  })
}

export async function uploadQMData(cedulaId, semana, file) {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(`${API_URL}/qm/data/upload?cedula_id=${cedulaId}&semana=${semana}`, {
    method: 'POST', body: formData
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Error de red' }))
    throw new Error(error.detail || `Error ${response.status}`)
  }
  return response.json()
}

export async function saveQMData(cedulaId, semana, records, changedRecords, newRecords, changesDetail) {
  return apiRequest('/qm/data/save', {
    method: 'POST',
    body: JSON.stringify({
      cedula_id: cedulaId,
      semana,
      records,
      changed_records: changedRecords,
      new_records: newRecords,
      changes_detail: changesDetail,
    })
  })
}

export async function verificarQMData(cedulaId, semana) {
  return apiRequest(`/qm/data/verificar?cedula_id=${cedulaId}&semana=${semana}`)
}

export async function getQMSemanas(cedulaId) {
  return apiRequest(`/qm/data/semanas?cedula_id=${cedulaId}`)
}

export async function deleteQMSemana(cedulaId, semana) {
  return apiRequest(`/qm/data/semana?cedula_id=${cedulaId}&semana=${semana}`, {
    method: 'DELETE'
  })
}

export async function getQMAnalisis(cedulaId, semana) {
  return apiRequest(`/qm/analisis?cedula_id=${cedulaId}&semana=${semana}`)
}

export async function getQMUploadHistory(cedulaId, semana = null) {
  let path = `/qm/data/uploads?cedula_id=${cedulaId}`
  if (semana) path += `&semana=${semana}`
  return apiRequest(path)
}

export async function deleteQMUploadLog(cedulaId, logId) {
  return apiRequest(`/qm/data/upload/${logId}?cedula_id=${cedulaId}`, {
    method: 'DELETE'
  })
}

export async function syncQMDashboard(cedulaId, semana) {
  return apiRequest(`/qm/sync-dashboard?cedula_id=${cedulaId}&semana=${semana}`, {
    method: 'POST'
  })
}

// ============================================================
// BOS / QBOS
// ============================================================

export async function uploadBOS(cedulaId, semana, file) {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(`${API_URL}/registros/bos/upload?cedula_id=${cedulaId}&semana=${semana}`, {
    method: 'POST', body: formData
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Error de red' }))
    throw new Error(error.detail || `Error ${response.status}`)
  }
  return response.json()
}

export async function saveBOS(cedulaId, semana, results) {
  return apiRequest(`/registros/bos/save?cedula_id=${cedulaId}&semana=${semana}`, {
    method: 'POST',
    body: JSON.stringify(results)
  })
}

export async function uploadQBOS(cedulaId, semana, file) {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(`${API_URL}/registros/qbos/upload?cedula_id=${cedulaId}&semana=${semana}`, {
    method: 'POST', body: formData
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Error de red' }))
    throw new Error(error.detail || `Error ${response.status}`)
  }
  return response.json()
}

export async function saveQBOS(cedulaId, semana, results) {
  return apiRequest(`/registros/qbos/save?cedula_id=${cedulaId}&semana=${semana}`, {
    method: 'POST',
    body: JSON.stringify(results)
  })
}

// ============================================================
// FRR — Filter Reject Rate
// ============================================================

export async function uploadROL(cedulaId, file, baseYear = 2026) {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(`${API_URL}/frr/rol/upload?cedula_id=${cedulaId}&base_year=${baseYear}`, {
    method: 'POST', body: formData
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Error de red' }))
    throw new Error(error.detail || `Error ${response.status}`)
  }
  return response.json()
}

export async function getROLSemana(cedulaId, semana) {
  return apiRequest(`/frr/rol/semana?cedula_id=${cedulaId}&semana=${semana}`)
}

export async function overrideTurno(cedulaId, operadorId, fecha, turno, kdf) {
  let path = `/frr/rol/override?cedula_id=${cedulaId}&operador_id=${operadorId}&fecha=${fecha}`
  if (turno !== undefined) path += `&turno=${turno || ''}`
  if (kdf) path += `&kdf=${kdf}`
  return apiRequest(path, { method: 'PUT' })
}

export async function uploadFRR(cedulaId, semana, file) {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(`${API_URL}/frr/upload?cedula_id=${cedulaId}&semana=${semana}`, {
    method: 'POST', body: formData
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Error de red' }))
    throw new Error(error.detail || `Error ${response.status}`)
  }
  return response.json()
}

export async function saveFRR(cedulaId, semana, results) {
  return apiRequest(`/frr/save?cedula_id=${cedulaId}&semana=${semana}`, {
    method: 'POST',
    body: JSON.stringify(results)
  })
}

// --- Aliases (mapeo de empleados) ---

export async function uploadAliases(cedulaId, file) {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(`${API_URL}/registros/aliases/upload?cedula_id=${cedulaId}`, {
    method: 'POST', body: formData
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Error de red' }))
    throw new Error(error.detail || `Error ${response.status}`)
  }
  return response.json()
}


// ============================================================
// WEEKLY
// ============================================================

export async function getWeeklyCategories(cedulaId) {
  return apiRequest(`/weekly/categories?cedula_id=${cedulaId}`)
}

export async function createWeeklyCategory(data) {
  return apiRequest('/weekly/categories', { method: 'POST', body: JSON.stringify(data) })
}

export async function updateWeeklyCategory(catId, data) {
  return apiRequest(`/weekly/categories/${catId}`, { method: 'PUT', body: JSON.stringify(data) })
}

export async function deleteWeeklyCategory(catId) {
  return apiRequest(`/weekly/categories/${catId}`, { method: 'DELETE' })
}

export async function reorderWeeklyCategories(items) {
  return apiRequest('/weekly/categories/reorder', { method: 'POST', body: JSON.stringify({ items }) })
}

export async function getWeeklyIndicators(cedulaId, categoryId) {
  let url = `/weekly/indicators?cedula_id=${cedulaId}`
  if (categoryId) url += `&category_id=${categoryId}`
  return apiRequest(url)
}

export async function createWeeklyIndicator(data) {
  return apiRequest('/weekly/indicators', { method: 'POST', body: JSON.stringify(data) })
}

export async function updateWeeklyIndicator(indId, data) {
  return apiRequest(`/weekly/indicators/${indId}`, { method: 'PUT', body: JSON.stringify(data) })
}

export async function deleteWeeklyIndicator(indId) {
  return apiRequest(`/weekly/indicators/${indId}`, { method: 'DELETE' })
}

export async function reorderWeeklyIndicators(items) {
  return apiRequest('/weekly/indicators/reorder', { method: 'POST', body: JSON.stringify({ items }) })
}

export async function getWeeklyChartData(cedulaId, year, quarter, categoryId) {
  let url = `/weekly/chart-data?cedula_id=${cedulaId}&year=${year}&quarter=${quarter}`
  if (categoryId) url += `&category_id=${categoryId}`
  return apiRequest(url)
}

export async function upsertWeeklyTargets(targets) {
  return apiRequest('/weekly/targets', { method: 'POST', body: JSON.stringify({ targets }) })
}

export async function fillWeeklyTarget(indicatorId, cedulaId, year, quarter, targetValue, weeks = 15) {
  return apiRequest(`/weekly/targets/fill?indicator_id=${indicatorId}&cedula_id=${cedulaId}&year=${year}&quarter=${quarter}&target_value=${targetValue}&weeks=${weeks}`, { method: 'POST' })
}

export async function upsertWeeklyValues(values) {
  return apiRequest('/weekly/values', { method: 'POST', body: JSON.stringify({ values }) })
}

export async function deleteWeeklyValues(entries) {
  return apiRequest('/weekly/values/delete', { method: 'POST', body: JSON.stringify({ entries }) })
}

export async function deleteWeeklyTargets(entries) {
  return apiRequest('/weekly/targets/delete', { method: 'POST', body: JSON.stringify({ entries }) })
}

export async function seedWeekly(cedulaId) {
  return apiRequest(`/weekly/seed?cedula_id=${cedulaId}`, { method: 'POST' })
}

export async function seedWeeklyExtras(cedulaId) {
  return apiRequest(`/weekly/seed-extras?cedula_id=${cedulaId}`, { method: 'POST' })
}

// ══════════════════════════════════════════════════════
// IPS
// ══════════════════════════════════════════════════════

export async function getIPSRecords(cedulaId) {
  return apiRequest(`/ips/records?cedula_id=${cedulaId}`)
}

export async function getIPSRecord(ipsId) {
  return apiRequest(`/ips/records/${ipsId}`)
}

export async function createIPSRecord(data) {
  return apiRequest('/ips/records', { method: 'POST', body: JSON.stringify(data) })
}

export async function updateIPSRecord(ipsId, data) {
  return apiRequest(`/ips/records/${ipsId}`, { method: 'PATCH', body: JSON.stringify(data) })
}

export async function deleteIPSRecord(ipsId) {
  return apiRequest(`/ips/records/${ipsId}`, { method: 'DELETE' })
}

export async function getIPSCountermeasures(ipsId) {
  return apiRequest(`/ips/countermeasures?ips_id=${ipsId}`)
}

export async function getAllCountermeasures(cedulaId) {
  return apiRequest(`/ips/countermeasures/all?cedula_id=${cedulaId}`)
}

export async function createCountermeasure(data) {
  return apiRequest('/ips/countermeasures', { method: 'POST', body: JSON.stringify(data) })
}

export async function updateCountermeasure(cmId, data) {
  return apiRequest(`/ips/countermeasures/${cmId}`, { method: 'PATCH', body: JSON.stringify(data) })
}

export async function deleteCountermeasure(cmId) {
  return apiRequest(`/ips/countermeasures/${cmId}`, { method: 'DELETE' })
}

export async function getIPSStats(cedulaId) {
  return apiRequest(`/ips/stats?cedula_id=${cedulaId}`)
}

export async function dedupIPSCountermeasures(cedulaId) {
  return apiRequest(`/ips/dedup?cedula_id=${cedulaId}`, { method: 'POST' })
}

export async function exportIPSExcel(cedulaId, kdf = null) {
  let url = `${API_URL}/ips/export?cedula_id=${cedulaId}`
  if (kdf != null) url += `&kdf=${kdf}`
  const response = await fetch(url)
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Error de red' }))
    throw new Error(error.detail || `Error ${response.status}`)
  }
  const blob = await response.blob()
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = kdf != null ? `IPS_Export_KDF${kdf}.xlsx` : 'IPS_Export.xlsx'
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(a.href)
}

export async function uploadIPSExcel(cedulaId, file) {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(`${API_URL}/ips/upload?cedula_id=${cedulaId}`, {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Error de red' }))
    throw new Error(error.detail || `Error ${response.status}`)
  }
  return response.json()
}
