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
