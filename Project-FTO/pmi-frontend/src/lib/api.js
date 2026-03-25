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
