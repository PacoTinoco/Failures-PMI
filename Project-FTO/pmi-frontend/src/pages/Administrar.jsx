import { useState, useEffect } from 'react'
import * as api from '../lib/api'

const TABS = [
  { key: 'cedulas', label: 'Cédulas' },
  { key: 'lcs', label: 'Line Coordinators' },
  { key: 'ls', label: 'Línea Estructura' },
  { key: 'operadores', label: 'Operadores' },
]

export default function Administrar() {
  const [activeTab, setActiveTab] = useState('cedulas')
  const [cedulas, setCedulas] = useState([])
  const [lcs, setLcs] = useState([])
  const [lsMembers, setLsMembers] = useState([])
  const [operadores, setOperadores] = useState([])
  const [selectedCedulaId, setSelectedCedulaId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [modal, setModal] = useState(null) // { type: 'create'|'edit', entity: 'cedula'|'lc'|'operador', data?: {} }
  const [saving, setSaving] = useState(false)

  // Fetch cédulas on mount
  useEffect(() => {
    fetchCedulas()
  }, [])

  // Fetch LCs & operadores when cédula changes
  useEffect(() => {
    if (selectedCedulaId) {
      fetchLCs()
      fetchLS()
      fetchOperadores()
    } else {
      setLcs([])
      setLsMembers([])
      setOperadores([])
    }
  }, [selectedCedulaId])

  async function fetchCedulas() {
    setLoading(true)
    try {
      const res = await api.getCedulas()
      setCedulas(res.data || [])
      if (!selectedCedulaId && res.data?.length > 0) {
        setSelectedCedulaId(res.data[0].id)
      }
    } catch (err) {
      setError(err.message)
    }
    setLoading(false)
  }

  async function fetchLCs() {
    try {
      const res = await api.getLineCoordinators(selectedCedulaId)
      setLcs(res.data || [])
    } catch (err) {
      console.error('Error cargando LCs:', err)
    }
  }

  async function fetchLS() {
    try {
      const res = await api.getLS(selectedCedulaId)
      setLsMembers(res.data || [])
    } catch (err) {
      console.error('Error cargando LS:', err)
    }
  }

  async function fetchOperadores() {
    try {
      const res = await api.getOperadores(selectedCedulaId)
      setOperadores(res.data || [])
    } catch (err) {
      console.error('Error cargando operadores:', err)
    }
  }

  async function handleDelete(entity, id, name) {
    if (!window.confirm(`¿Estás seguro de eliminar "${name}"? Esta acción desactivará el registro.`)) return
    try {
      if (entity === 'cedula') {
        await api.deleteCedula(id)
        fetchCedulas()
      } else if (entity === 'lc') {
        await api.deleteLC(id)
        fetchLCs()
      } else if (entity === 'ls') {
        await api.deleteLS(id)
        fetchLS()
      } else if (entity === 'operador') {
        await api.deleteOperador(id)
        fetchOperadores()
      }
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleSave(formData) {
    setSaving(true)
    setError(null)
    try {
      if (modal.entity === 'cedula') {
        if (modal.type === 'create') {
          await api.createCedula(formData)
        } else {
          await api.updateCedula(modal.data.id, formData)
        }
        fetchCedulas()
      } else if (modal.entity === 'lc') {
        if (modal.type === 'create') {
          await api.createLC({ ...formData, cedula_id: selectedCedulaId })
        } else {
          await api.updateLC(modal.data.id, formData)
        }
        fetchLCs()
      } else if (modal.entity === 'ls') {
        if (modal.type === 'create') {
          await api.createLS({ ...formData, cedula_id: selectedCedulaId })
        } else {
          await api.updateLS(modal.data.id, formData)
        }
        fetchLS()
      } else if (modal.entity === 'operador') {
        if (modal.type === 'create') {
          await api.createOperador({ ...formData, cedula_id: selectedCedulaId })
        } else {
          await api.updateOperador(modal.data.id, formData)
        }
        fetchOperadores()
      }
      setModal(null)
    } catch (err) {
      setError(err.message)
    }
    setSaving(false)
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl font-bold text-white">Administración</h1>
        <p className="text-sm text-slate-400">Gestiona cédulas, coordinadores y operadores</p>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-1 bg-[#0f1d32] rounded-lg p-1 border border-white/5 w-fit">
        {TABS.map(tab => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              activeTab === tab.key
                ? 'bg-blue-500/15 text-blue-400'
                : 'text-slate-400 hover:text-white hover:bg-white/5'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Cédula selector for LCs and Operadores tabs */}
      {activeTab !== 'cedulas' && (
        <div className="flex items-center gap-3">
          <label className="text-sm text-slate-400">Cédula:</label>
          <select
            value={selectedCedulaId || ''}
            onChange={e => setSelectedCedulaId(e.target.value || null)}
            className="px-3 py-2 bg-[#0f1d32] border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">Selecciona...</option>
            {cedulas.map(c => (
              <option key={c.id} value={c.id}>{c.nombre}</option>
            ))}
          </select>
        </div>
      )}

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-red-400 text-sm">
          {error}
          <button onClick={() => setError(null)} className="ml-3 text-red-300 hover:text-white">✕</button>
        </div>
      )}

      {/* Content */}
      {activeTab === 'cedulas' && (
        <CedulasTab
          cedulas={cedulas}
          loading={loading}
          onAdd={() => setModal({ type: 'create', entity: 'cedula' })}
          onEdit={(c) => setModal({ type: 'edit', entity: 'cedula', data: c })}
          onDelete={(c) => handleDelete('cedula', c.id, c.nombre)}
        />
      )}
      {activeTab === 'lcs' && (
        <LCsTab
          lcs={lcs}
          loading={loading}
          selectedCedulaId={selectedCedulaId}
          onAdd={() => setModal({ type: 'create', entity: 'lc' })}
          onEdit={(lc) => setModal({ type: 'edit', entity: 'lc', data: lc })}
          onDelete={(lc) => handleDelete('lc', lc.id, lc.nombre)}
        />
      )}
      {activeTab === 'ls' && (
        <LSTab
          lsMembers={lsMembers}
          loading={loading}
          selectedCedulaId={selectedCedulaId}
          onAdd={() => setModal({ type: 'create', entity: 'ls' })}
          onEdit={(ls) => setModal({ type: 'edit', entity: 'ls', data: ls })}
          onDelete={(ls) => handleDelete('ls', ls.id, ls.nombre)}
        />
      )}
      {activeTab === 'operadores' && (
        <OperadoresTab
          operadores={operadores}
          lcs={lcs}
          loading={loading}
          selectedCedulaId={selectedCedulaId}
          onAdd={() => setModal({ type: 'create', entity: 'operador' })}
          onEdit={(op) => setModal({ type: 'edit', entity: 'operador', data: op })}
          onDelete={(op) => handleDelete('operador', op.id, op.nombre)}
        />
      )}

      {/* Modal */}
      {modal && (
        <FormModal
          modal={modal}
          lcs={lcs}
          saving={saving}
          onSave={handleSave}
          onClose={() => setModal(null)}
        />
      )}
    </div>
  )
}

// ============================================================
// TAB COMPONENTS
// ============================================================

function CedulasTab({ cedulas, loading, onAdd, onEdit, onDelete }) {
  return (
    <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/5">
        <h2 className="text-sm font-semibold text-white">Cédulas Activas</h2>
        <button onClick={onAdd} className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium rounded-lg transition-colors">
          + Agregar
        </button>
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-white/5">
            <th className="px-4 py-2 text-left text-slate-400 font-medium text-xs">Nombre</th>
            <th className="px-4 py-2 text-left text-slate-400 font-medium text-xs">Turno</th>
            <th className="px-4 py-2 text-left text-slate-400 font-medium text-xs">Grupo</th>
            <th className="px-4 py-2 text-left text-slate-400 font-medium text-xs">Notas</th>
            <th className="px-4 py-2 text-right text-slate-400 font-medium text-xs">Acciones</th>
          </tr>
        </thead>
        <tbody>
          {cedulas.length === 0 ? (
            <tr><td colSpan={5} className="px-4 py-8 text-center text-slate-500">No hay cédulas.</td></tr>
          ) : cedulas.map(c => (
            <tr key={c.id} className="border-b border-white/5 hover:bg-white/[0.02]">
              <td className="px-4 py-2.5 text-white font-medium">{c.nombre}</td>
              <td className="px-4 py-2.5 text-slate-400">{c.turno || '—'}</td>
              <td className="px-4 py-2.5 text-slate-400">{c.grupo || '—'}</td>
              <td className="px-4 py-2.5 text-slate-400 max-w-[200px] truncate">{c.notas || '—'}</td>
              <td className="px-4 py-2.5 text-right">
                <ActionButtons onEdit={() => onEdit(c)} onDelete={() => onDelete(c)} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function LCsTab({ lcs, loading, selectedCedulaId, onAdd, onEdit, onDelete }) {
  if (!selectedCedulaId) return <EmptyMessage text="Selecciona una cédula para ver sus Line Coordinators." />

  return (
    <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/5">
        <h2 className="text-sm font-semibold text-white">Line Coordinators</h2>
        <button onClick={onAdd} className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium rounded-lg transition-colors">
          + Agregar
        </button>
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-white/5">
            <th className="px-4 py-2 text-left text-slate-400 font-medium text-xs">Nombre</th>
            <th className="px-4 py-2 text-left text-slate-400 font-medium text-xs">Turno</th>
            <th className="px-4 py-2 text-left text-slate-400 font-medium text-xs">Grupo</th>
            <th className="px-4 py-2 text-left text-slate-400 font-medium text-xs">Email</th>
            <th className="px-4 py-2 text-right text-slate-400 font-medium text-xs">Acciones</th>
          </tr>
        </thead>
        <tbody>
          {lcs.length === 0 ? (
            <tr><td colSpan={5} className="px-4 py-8 text-center text-slate-500">No hay LCs para esta cédula.</td></tr>
          ) : lcs.map(lc => (
            <tr key={lc.id} className="border-b border-white/5 hover:bg-white/[0.02]">
              <td className="px-4 py-2.5 text-white font-medium">{lc.nombre}</td>
              <td className="px-4 py-2.5 text-slate-400">{lc.turno || '—'}</td>
              <td className="px-4 py-2.5 text-slate-400">{lc.grupo || '—'}</td>
              <td className="px-4 py-2.5 text-slate-400">{lc.email || '—'}</td>
              <td className="px-4 py-2.5 text-right">
                <ActionButtons onEdit={() => onEdit(lc)} onDelete={() => onDelete(lc)} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function LSTab({ lsMembers, loading, selectedCedulaId, onAdd, onEdit, onDelete }) {
  if (!selectedCedulaId) return <EmptyMessage text="Selecciona una cédula para ver su Línea de Estructura." />

  return (
    <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/5">
        <div>
          <h2 className="text-sm font-semibold text-white">Línea de Estructura ({lsMembers.length})</h2>
          <p className="text-xs text-slate-500">Process Lead, Line Lead, Maintenance Lead, etc.</p>
        </div>
        <button onClick={onAdd} className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium rounded-lg transition-colors">
          + Agregar
        </button>
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-white/5">
            <th className="px-4 py-2 text-left text-slate-400 font-medium text-xs">Nombre</th>
            <th className="px-4 py-2 text-left text-slate-400 font-medium text-xs">Rol</th>
            <th className="px-4 py-2 text-left text-slate-400 font-medium text-xs">Email</th>
            <th className="px-4 py-2 text-left text-slate-400 font-medium text-xs">Turno</th>
            <th className="px-4 py-2 text-right text-slate-400 font-medium text-xs">Acciones</th>
          </tr>
        </thead>
        <tbody>
          {lsMembers.length === 0 ? (
            <tr><td colSpan={5} className="px-4 py-8 text-center text-slate-500">No hay miembros de LS para esta cédula.</td></tr>
          ) : lsMembers.map(ls => (
            <tr key={ls.id} className="border-b border-white/5 hover:bg-white/[0.02]">
              <td className="px-4 py-2.5 text-white font-medium">{ls.nombre}</td>
              <td className="px-4 py-2.5 text-slate-400">{ls.rol || '—'}</td>
              <td className="px-4 py-2.5 text-slate-400">{ls.email || '—'}</td>
              <td className="px-4 py-2.5 text-slate-400">{ls.turno || '—'}</td>
              <td className="px-4 py-2.5 text-right">
                <ActionButtons onEdit={() => onEdit(ls)} onDelete={() => onDelete(ls)} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function OperadoresTab({ operadores, lcs, loading, selectedCedulaId, onAdd, onEdit, onDelete }) {
  if (!selectedCedulaId) return <EmptyMessage text="Selecciona una cédula para ver sus operadores." />

  return (
    <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/5">
        <h2 className="text-sm font-semibold text-white">Operadores ({operadores.length})</h2>
        <button onClick={onAdd} className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium rounded-lg transition-colors">
          + Agregar
        </button>
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-white/5">
            <th className="px-4 py-2 text-left text-slate-400 font-medium text-xs">Nombre</th>
            <th className="px-4 py-2 text-left text-slate-400 font-medium text-xs">Line Coordinator</th>
            <th className="px-4 py-2 text-left text-slate-400 font-medium text-xs">Turno</th>
            <th className="px-4 py-2 text-left text-slate-400 font-medium text-xs">Máquina</th>
            <th className="px-4 py-2 text-right text-slate-400 font-medium text-xs">Acciones</th>
          </tr>
        </thead>
        <tbody>
          {operadores.length === 0 ? (
            <tr><td colSpan={5} className="px-4 py-8 text-center text-slate-500">No hay operadores para esta cédula.</td></tr>
          ) : operadores.map(op => (
            <tr key={op.id} className="border-b border-white/5 hover:bg-white/[0.02]">
              <td className="px-4 py-2.5 text-white font-medium">{op.nombre}</td>
              <td className="px-4 py-2.5 text-slate-400">{op.line_coordinators?.nombre || '—'}</td>
              <td className="px-4 py-2.5 text-slate-400">{op.turno || '—'}</td>
              <td className="px-4 py-2.5 text-slate-400">{op.maquina || '—'}</td>
              <td className="px-4 py-2.5 text-right">
                <ActionButtons onEdit={() => onEdit(op)} onDelete={() => onDelete(op)} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ============================================================
// SHARED COMPONENTS
// ============================================================

function ActionButtons({ onEdit, onDelete }) {
  return (
    <div className="flex items-center justify-end gap-1">
      <button onClick={onEdit} className="p-1.5 text-slate-400 hover:text-blue-400 hover:bg-blue-400/10 rounded-lg transition-colors" title="Editar">
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
        </svg>
      </button>
      <button onClick={onDelete} className="p-1.5 text-slate-400 hover:text-red-400 hover:bg-red-400/10 rounded-lg transition-colors" title="Eliminar">
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
        </svg>
      </button>
    </div>
  )
}

function EmptyMessage({ text }) {
  return (
    <div className="bg-[#0f1d32] rounded-xl border border-white/5 px-6 py-12 text-center">
      <p className="text-slate-500">{text}</p>
    </div>
  )
}

// ============================================================
// FORM MODAL
// ============================================================

function FormModal({ modal, lcs, saving, onSave, onClose }) {
  const isEdit = modal.type === 'edit'
  const entity = modal.entity
  const data = modal.data || {}

  const [form, setForm] = useState(() => {
    if (entity === 'cedula') {
      return { nombre: data.nombre || '', turno: data.turno || '', grupo: data.grupo || '', notas: data.notas || '' }
    } else if (entity === 'lc') {
      return { nombre: data.nombre || '', turno: data.turno || '', grupo: data.grupo || '', email: data.email || '' }
    } else if (entity === 'ls') {
      return { nombre: data.nombre || '', rol: data.rol || '', email: data.email || '', turno: data.turno || '' }
    } else {
      return { nombre: data.nombre || '', lc_id: data.lc_id || '', turno: data.turno || '', maquina: data.maquina || '' }
    }
  })

  function handleChange(field, value) {
    setForm(prev => ({ ...prev, [field]: value }))
  }

  function handleSubmit(e) {
    e.preventDefault()
    // Clean empty strings to null
    const cleaned = {}
    Object.entries(form).forEach(([k, v]) => {
      cleaned[k] = v === '' ? null : v
    })
    // nombre is required
    if (!cleaned.nombre) return
    onSave(cleaned)
  }

  const entityLabels = { cedula: 'Cédula', lc: 'Line Coordinator', ls: 'Miembro LS', operador: 'Operador' }
  const title = `${isEdit ? 'Editar' : 'Agregar'} ${entityLabels[entity] || entity}`

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="bg-[#0f1d32] rounded-2xl border border-white/10 p-6 w-full max-w-md shadow-2xl" onClick={e => e.stopPropagation()}>
        <h3 className="text-lg font-semibold text-white mb-4">{title}</h3>

        <form onSubmit={handleSubmit} className="space-y-4">
          <FormField label="Nombre *" value={form.nombre} onChange={v => handleChange('nombre', v)} required />

          {entity === 'operador' && (
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">Line Coordinator *</label>
              <select
                value={form.lc_id}
                onChange={e => handleChange('lc_id', e.target.value)}
                required
                className="w-full px-3 py-2.5 bg-[#0a1628] border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Selecciona un LC</option>
                {lcs.map(lc => (
                  <option key={lc.id} value={lc.id}>{lc.nombre}</option>
                ))}
              </select>
            </div>
          )}

          <FormField label="Turno" value={form.turno} onChange={v => handleChange('turno', v)} />

          {entity === 'cedula' && (
            <>
              <FormField label="Grupo" value={form.grupo} onChange={v => handleChange('grupo', v)} />
              <FormField label="Notas" value={form.notas} onChange={v => handleChange('notas', v)} multiline />
            </>
          )}

          {entity === 'lc' && (
            <>
              <FormField label="Grupo" value={form.grupo} onChange={v => handleChange('grupo', v)} />
              <FormField label="Email" value={form.email} onChange={v => handleChange('email', v)} type="email" />
            </>
          )}

          {entity === 'ls' && (
            <>
              <FormField label="Rol" value={form.rol} onChange={v => handleChange('rol', v)} placeholder="Ej: Process Lead, Line Lead, Maintenance Lead" />
              <FormField label="Email" value={form.email} onChange={v => handleChange('email', v)} type="email" />
            </>
          )}

          {entity === 'operador' && (
            <FormField label="Máquina" value={form.maquina} onChange={v => handleChange('maquina', v)} />
          )}

          <div className="flex items-center justify-end gap-3 pt-2">
            <button type="button" onClick={onClose} className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors">
              Cancelar
            </button>
            <button
              type="submit"
              disabled={saving || !form.nombre}
              className="px-5 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-600/40 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-2"
            >
              {saving && <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />}
              {isEdit ? 'Guardar' : 'Crear'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

function FormField({ label, value, onChange, required = false, type = 'text', multiline = false, placeholder = '' }) {
  const cls = "w-full px-3 py-2.5 bg-[#0a1628] border border-white/10 rounded-lg text-white text-sm placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"

  return (
    <div>
      <label className="block text-sm font-medium text-slate-300 mb-1">{label}</label>
      {multiline ? (
        <textarea value={value || ''} onChange={e => onChange(e.target.value)} className={cls} rows={3} />
      ) : (
        <input type={type} value={value || ''} onChange={e => onChange(e.target.value)} required={required} placeholder={placeholder} className={cls} />
      )}
    </div>
  )
}
