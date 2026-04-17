import { useState, useEffect, useRef } from 'react'
import CedulaSelector from '../components/CedulaSelector'
import * as api from '../lib/api'

const STATUS_COLORS = {
  Open:      'bg-blue-500/15 text-blue-400',
  Closed:    'bg-green-500/15 text-green-400',
  Cancelled: 'bg-slate-500/15 text-slate-400',
  Merged:    'bg-purple-500/15 text-purple-400',
  BCC:       'bg-amber-500/15 text-amber-400',
  '6W2H':   'bg-cyan-500/15 text-cyan-400',
  Ascended:  'bg-indigo-500/15 text-indigo-400',
  Missing:   'bg-red-500/15 text-red-400',
}

const CM_STATUS_COLORS = {
  Pending:   'bg-yellow-500/15 text-yellow-400',
  Done:      'bg-green-500/15 text-green-400',
  Cancelled: 'bg-slate-500/15 text-slate-400',
  'On going':'bg-blue-500/15 text-blue-400',
}

const EMPTY_IPS = {
  kdf: '', titulo: '', fecha: '', ubicacion: '', participants: '',
  section_6w2h: false, section_bbc: false, section_5w: false, section_res: false,
  status: 'Open', notes: '',
}

const EMPTY_CM = { descripcion: '', owner: '', status: 'Pending', priority: '', due_date: '' }

export default function IPS() {
  const [cedulas, setCedulas]               = useState([])
  const [cedulaId, setCedulaId]             = useState(null)
  const [cedulasLoading, setCedulasLoading] = useState(true)
  const [error, setError]                   = useState(null)

  const [records, setRecords]   = useState([])
  const [allCMs, setAllCMs]     = useState([])
  const [stats, setStats]       = useState(null)
  const [loading, setLoading]   = useState(false)

  // Filters
  const [search, setSearch]             = useState('')
  const [filterKdf, setFilterKdf]       = useState('all')
  const [filterStatus, setFilterStatus] = useState('all')
  const [filterUbi, setFilterUbi]       = useState('all')
  const [filterOwner, setFilterOwner]   = useState([])  // [] = all, array of names = filtered
  const [sortKey, setSortKey]           = useState('fecha')
  const [sortDir, setSortDir]           = useState('desc')

  // Expanded / detail
  const [expandedId, setExpandedId]     = useState(null)
  const [expandedCMs, setExpandedCMs]   = useState([])
  const [loadingCMs, setLoadingCMs]     = useState(false)

  // Upload
  const [uploading, setUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState(null)
  const fileRef = useRef(null)

  // Create modal
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [createForm, setCreateForm] = useState({ ...EMPTY_IPS })
  const [createCMs, setCreateCMs] = useState([{ ...EMPTY_CM }])
  const [creating, setCreating] = useState(false)

  // Add CM modal
  const [showAddCM, setShowAddCM] = useState(null) // ips_id
  const [newCM, setNewCM] = useState({ ...EMPTY_CM })

  // Edit IPS
  const [showEditModal, setShowEditModal] = useState(false)
  const [editForm, setEditForm]           = useState({ ...EMPTY_IPS })
  const [editId, setEditId]               = useState(null)
  const [editSaving, setEditSaving]       = useState(false)
  const [ownerDropOpen, setOwnerDropOpen] = useState(false)

  // Export
  const [exporting, setExporting] = useState(false)

  // Dedup
  const [deduping, setDeduping] = useState(false)

  // ── Load cédulas ──
  useEffect(() => {
    api.getCedulas()
      .then(res => { setCedulas(res.data || []); if (res.data?.length > 0) setCedulaId(res.data[0].id) })
      .catch(err => setError(err.message))
      .finally(() => setCedulasLoading(false))
  }, [])

  // ── Load IPS data ──
  useEffect(() => {
    if (!cedulaId) return
    loadData()
  }, [cedulaId])

  async function loadData() {
    setLoading(true); setError(null)
    try {
      const [recs, cms, st] = await Promise.all([
        api.getIPSRecords(cedulaId),
        api.getAllCountermeasures(cedulaId),
        api.getIPSStats(cedulaId),
      ])
      setRecords(recs.data || [])
      setAllCMs(cms.data || [])
      setStats(st)
    } catch (err) { setError(err.message) }
    finally { setLoading(false) }
  }

  // ── Upload ──
  async function handleUpload(e) {
    const file = e.target.files[0]
    if (!file || !cedulaId) return
    e.target.value = ''
    setUploading(true); setError(null); setUploadResult(null)
    try {
      const res = await api.uploadIPSExcel(cedulaId, file)
      setUploadResult(res)
      await loadData()
    } catch (err) { setError(err.message) }
    finally { setUploading(false) }
  }

  // ── Dedup ──
  async function handleDedup() {
    if (!cedulaId) return
    setDeduping(true); setError(null)
    try {
      const res = await api.dedupIPSCountermeasures(cedulaId)
      setUploadResult(res)
      await loadData()
    } catch (err) { setError(err.message) }
    finally { setDeduping(false) }
  }

  // ── Export ──
  async function handleExport() {
    if (!cedulaId) return
    setExporting(true); setError(null)
    try {
      const kdfParam = filterKdf !== 'all' ? parseInt(filterKdf) : null
      await api.exportIPSExcel(cedulaId, kdfParam)
    } catch (err) { setError(err.message) }
    finally { setExporting(false) }
  }

  // ── Create IPS ──
  async function handleCreateIPS() {
    if (!cedulaId || !createForm.kdf || !createForm.titulo) return
    setCreating(true); setError(null)
    try {
      const participants = createForm.participants
        .split(',').map(p => p.trim()).filter(Boolean)
      const body = {
        cedula_id: cedulaId,
        kdf: parseInt(createForm.kdf),
        titulo: createForm.titulo,
        fecha: createForm.fecha || null,
        ubicacion: createForm.ubicacion || null,
        participants,
        section_6w2h: createForm.section_6w2h,
        section_bbc: createForm.section_bbc,
        section_5w: createForm.section_5w,
        section_res: createForm.section_res,
        status: createForm.status,
        notes: createForm.notes || null,
      }
      const result = await api.createIPSRecord(body)
      const ipsId = result.data?.id

      // Create CMs if any
      if (ipsId) {
        for (const cm of createCMs) {
          if (!cm.descripcion.trim()) continue
          await api.createCountermeasure({
            ips_id: ipsId,
            descripcion: cm.descripcion,
            owner: cm.owner || null,
            status: cm.status,
            priority: cm.priority || null,
            due_date: cm.due_date || null,
          })
        }
      }

      setShowCreateModal(false)
      setCreateForm({ ...EMPTY_IPS })
      setCreateCMs([{ ...EMPTY_CM }])
      await loadData()
    } catch (err) { setError(err.message) }
    finally { setCreating(false) }
  }

  // ── Add CM to existing IPS ──
  async function handleAddCM(ipsId) {
    if (!newCM.descripcion.trim()) return
    try {
      await api.createCountermeasure({
        ips_id: ipsId,
        descripcion: newCM.descripcion,
        owner: newCM.owner || null,
        status: newCM.status,
        priority: newCM.priority || null,
        due_date: newCM.due_date || null,
      })
      setShowAddCM(null)
      setNewCM({ ...EMPTY_CM })
      // Reload CMs for expanded row
      if (expandedId === ipsId) {
        const res = await api.getIPSCountermeasures(ipsId)
        setExpandedCMs(res.data || [])
      }
      // Reload all CMs for counts
      const allRes = await api.getAllCountermeasures(cedulaId)
      setAllCMs(allRes.data || [])
    } catch (err) { setError(err.message) }
  }

  // ── Expand IPS detail ──
  async function handleExpand(ipsId) {
    if (expandedId === ipsId) { setExpandedId(null); return }
    setExpandedId(ipsId)
    setLoadingCMs(true)
    try {
      const res = await api.getIPSCountermeasures(ipsId)
      setExpandedCMs(res.data || [])
    } catch (err) { setExpandedCMs([]) }
    finally { setLoadingCMs(false) }
  }

  // ── Update CM status inline ──
  async function handleCMStatusChange(cmId, newStatus) {
    try {
      await api.updateCountermeasure(cmId, { status: newStatus })
      setExpandedCMs(prev => prev.map(c => c.id === cmId ? { ...c, status: newStatus } : c))
      setAllCMs(prev => prev.map(c => c.id === cmId ? { ...c, status: newStatus } : c))
    } catch (err) { setError(err.message) }
  }

  // ── Update IPS status inline ──
  async function handleIPSStatusChange(ipsId, newStatus) {
    try {
      await api.updateIPSRecord(ipsId, { status: newStatus })
      setRecords(prev => prev.map(r => r.id === ipsId ? { ...r, status: newStatus } : r))
    } catch (err) { setError(err.message) }
  }

  // ── Section toggle ──
  async function handleSectionToggle(ipsId, section, currentVal) {
    try {
      await api.updateIPSRecord(ipsId, { [section]: !currentVal })
      setRecords(prev => prev.map(r => r.id === ipsId ? { ...r, [section]: !currentVal } : r))
    } catch (err) { setError(err.message) }
  }

  // ── Delete IPS ──
  async function handleDeleteIPS(ipsId) {
    if (!confirm('¿Eliminar este registro IPS y todas sus contramedidas?')) return
    try {
      await api.deleteIPSRecord(ipsId)
      if (expandedId === ipsId) setExpandedId(null)
      await loadData()
    } catch (err) { setError(err.message) }
  }

  // ── Delete CM ──
  async function handleDeleteCM(cmId) {
    try {
      await api.deleteCountermeasure(cmId)
      setExpandedCMs(prev => prev.filter(c => c.id !== cmId))
      setAllCMs(prev => prev.filter(c => c.id !== cmId))
    } catch (err) { setError(err.message) }
  }

  // ── Edit IPS record ──
  function openEditModal(rec) {
    setEditId(rec.id)
    setEditForm({
      kdf: rec.kdf || '',
      titulo: rec.titulo || '',
      fecha: rec.fecha || '',
      ubicacion: rec.ubicacion || '',
      participants: Array.isArray(rec.participants) ? rec.participants.join(', ') : (rec.participants || ''),
      section_6w2h: rec.section_6w2h || false,
      section_bbc: rec.section_bbc || false,
      section_5w: rec.section_5w || false,
      section_res: rec.section_res || false,
      status: rec.status || 'Open',
      notes: rec.notes || '',
    })
    setShowEditModal(true)
  }

  async function handleEditSave() {
    setEditSaving(true)
    try {
      const data = {
        ...editForm,
        kdf: parseInt(editForm.kdf) || 0,
        participants: typeof editForm.participants === 'string'
          ? editForm.participants.split(',').map(p => p.trim()).filter(Boolean)
          : editForm.participants,
      }
      await api.updateIPSRecord(editId, data)
      setShowEditModal(false)
      setEditId(null)
      await loadData()
    } catch (err) { setError(err.message) }
    finally { setEditSaving(false) }
  }

  // ── Filter & sort ──
  const uniqueKdfs = [...new Set(records.map(r => r.kdf))].sort((a, b) => a - b)
  const uniqueStatuses = [...new Set(records.map(r => r.status))].sort()
  const uniqueUbis = [...new Set(records.map(r => r.ubicacion).filter(Boolean))].sort()
  const uniqueOwners = [...new Set(allCMs.map(c => c.owner).filter(Boolean))].sort()

  // Build CM ownership map: ips_id → [owners]
  const cmOwnerMap = {}
  for (const cm of allCMs) {
    if (!cmOwnerMap[cm.ips_id]) cmOwnerMap[cm.ips_id] = new Set()
    if (cm.owner) cmOwnerMap[cm.ips_id].add(cm.owner)
  }

  const filtered = records
    .filter(r => {
      if (search && !r.titulo.toLowerCase().includes(search.toLowerCase()) &&
          !(r.participants || []).some(p => p.toLowerCase().includes(search.toLowerCase()))) return false
      if (filterKdf !== 'all' && r.kdf !== parseInt(filterKdf)) return false
      if (filterStatus !== 'all' && r.status !== filterStatus) return false
      if (filterUbi !== 'all' && r.ubicacion !== filterUbi) return false
      if (filterOwner.length > 0) {
        const owners = cmOwnerMap[r.id]
        if (!owners || !filterOwner.some(fo => owners.has(fo))) return false
      }
      return true
    })
    .sort((a, b) => {
      let av, bv
      if (sortKey === 'fecha') { av = a.fecha || ''; bv = b.fecha || '' }
      else if (sortKey === 'kdf') { av = a.kdf; bv = b.kdf }
      else if (sortKey === 'titulo') { av = a.titulo; bv = b.titulo }
      else if (sortKey === 'status') { av = a.status; bv = b.status }
      else if (sortKey === 'cms') {
        av = allCMs.filter(c => c.ips_id === a.id).length
        bv = allCMs.filter(c => c.ips_id === b.id).length
      }
      if (typeof av === 'string') return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av)
      return sortDir === 'asc' ? av - bv : bv - av
    })

  const toggleSort = (key) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir(key === 'titulo' ? 'asc' : 'desc') }
  }
  const SortIcon = ({ col }) => {
    if (sortKey !== col) return <span className="text-slate-700 ml-0.5">↕</span>
    return <span className="text-purple-400 ml-0.5">{sortDir === 'asc' ? '↑' : '↓'}</span>
  }

  // CM counts per IPS
  const cmCountMap = {}
  for (const cm of allCMs) { cmCountMap[cm.ips_id] = (cmCountMap[cm.ips_id] || 0) + 1 }
  const cmDoneMap = {}
  for (const cm of allCMs) { if (cm.status === 'Done') cmDoneMap[cm.ips_id] = (cmDoneMap[cm.ips_id] || 0) + 1 }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white">IPS — Issue Problem Solving</h1>
          <p className="text-sm text-slate-400">Registro, seguimiento y contramedidas de problemas por máquina</p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <CedulaSelector cedulas={cedulas} selectedCedulaId={cedulaId} onChange={setCedulaId} loading={cedulasLoading} />
          <button onClick={() => setShowCreateModal(true)} disabled={!cedulaId}
            className="px-4 py-2 rounded-lg text-xs font-medium bg-green-600 hover:bg-green-500 text-white transition-colors disabled:opacity-50 whitespace-nowrap">
            + Nuevo IPS
          </button>
          <button onClick={handleExport} disabled={exporting || !cedulaId || records.length === 0}
            className="px-4 py-2 rounded-lg text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white transition-colors disabled:opacity-50 whitespace-nowrap">
            {exporting ? 'Exportando...' : 'Exportar Excel'}
          </button>
          <input ref={fileRef} type="file" accept=".xlsx,.xls" onChange={handleUpload} className="hidden" />
          <button onClick={() => fileRef.current?.click()} disabled={uploading || !cedulaId}
            className="px-4 py-2 rounded-lg text-xs font-medium bg-purple-600 hover:bg-purple-500 text-white transition-colors disabled:opacity-50 whitespace-nowrap">
            {uploading ? 'Importando...' : 'Importar Excel'}
          </button>
          <button onClick={handleDedup} disabled={deduping || !cedulaId || allCMs.length === 0}
            className="px-3 py-2 rounded-lg text-xs font-medium bg-amber-600 hover:bg-amber-500 text-white transition-colors disabled:opacity-50 whitespace-nowrap"
            title="Eliminar contramedidas duplicadas">
            {deduping ? 'Limpiando...' : 'Limpiar duplicados'}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-4 py-3 text-red-400 text-sm flex items-center gap-2">
          <span className="flex-1">{error}</span>
          <button onClick={() => setError(null)} className="text-red-300 hover:text-white">✕</button>
        </div>
      )}

      {uploadResult && (
        <div className="bg-green-500/10 border border-green-500/30 rounded-lg px-4 py-3 text-green-400 text-sm flex items-center gap-2">
          <span className="flex-1">{uploadResult.message}</span>
          <button onClick={() => setUploadResult(null)} className="text-green-300 hover:text-white">✕</button>
        </div>
      )}

      {/* Stats cards */}
      {stats && stats.total > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-3">
          <StatCard label="Total IPS" value={stats.total} color="text-white" />
          {Object.entries(stats.by_status).sort((a, b) => b[1] - a[1]).map(([s, c]) => (
            <StatCard key={s} label={s} value={c}
              color={s === 'Closed' ? 'text-green-400' : s === 'Cancelled' ? 'text-slate-400' : s === 'Open' ? 'text-blue-400' : 'text-purple-400'}
              onClick={() => setFilterStatus(filterStatus === s ? 'all' : s)}
              active={filterStatus === s} />
          ))}
          <StatCard label="Contramedidas" value={stats.total_countermeasures} color="text-cyan-400" />
        </div>
      )}

      {/* Filters */}
      {records.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-slate-500">KDF:</span>
          <div className="flex items-center gap-1 bg-[#0a1628] rounded-lg p-0.5">
            <button onClick={() => setFilterKdf('all')}
              className={`px-3 py-1 rounded-md text-[11px] font-medium transition-colors ${filterKdf === 'all' ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-white'}`}>
              Todas
            </button>
            {uniqueKdfs.map(k => (
              <button key={k} onClick={() => setFilterKdf(filterKdf === String(k) ? 'all' : String(k))}
                className={`px-3 py-1 rounded-md text-[11px] font-medium transition-colors ${filterKdf === String(k) ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-white'}`}>
                KDF {k}
              </button>
            ))}
          </div>

          <span className="text-xs text-slate-500 ml-2">Ubi:</span>
          <div className="flex items-center gap-1 bg-[#0a1628] rounded-lg p-0.5">
            <button onClick={() => setFilterUbi('all')}
              className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors ${filterUbi === 'all' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white'}`}>
              Todas
            </button>
            {uniqueUbis.map(u => (
              <button key={u} onClick={() => setFilterUbi(filterUbi === u ? 'all' : u)}
                className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors ${filterUbi === u ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white'}`}>
                {u}
              </button>
            ))}
          </div>

          {/* Owner multi-select filter */}
          <span className="text-xs text-slate-500 ml-2">Owner:</span>
          <div className="relative">
            <button onClick={() => setOwnerDropOpen(!ownerDropOpen)}
              className={`bg-[#0a1628] border ${filterOwner.length > 0 ? 'border-purple-500/60' : 'border-white/10'} rounded-lg px-2.5 py-1 text-[11px] text-white focus:outline-none min-w-[120px] text-left flex items-center justify-between gap-1`}>
              <span className="truncate max-w-[140px]">
                {filterOwner.length === 0 ? 'Todos' : filterOwner.length === 1 ? filterOwner[0] : `${filterOwner.length} seleccionados`}
              </span>
              <span className="text-slate-500 text-[9px]">▾</span>
            </button>
            {ownerDropOpen && (
              <div className="absolute top-full left-0 mt-1 bg-[#0f1d32] border border-white/10 rounded-lg shadow-xl z-50 w-56 max-h-60 overflow-y-auto">
                <button onClick={() => { setFilterOwner([]); setOwnerDropOpen(false) }}
                  className="w-full px-3 py-1.5 text-left text-[11px] text-slate-400 hover:bg-slate-700/50 border-b border-white/5">
                  Todos (limpiar filtro)
                </button>
                {uniqueOwners.map(o => (
                  <label key={o} className="flex items-center gap-2 px-3 py-1.5 hover:bg-slate-700/30 cursor-pointer">
                    <input type="checkbox" checked={filterOwner.includes(o)}
                      onChange={() => {
                        setFilterOwner(prev =>
                          prev.includes(o) ? prev.filter(x => x !== o) : [...prev, o]
                        )
                      }}
                      className="rounded border-slate-600 text-purple-500 focus:ring-purple-500 w-3 h-3" />
                    <span className="text-[11px] text-white truncate">{o}</span>
                  </label>
                ))}
              </div>
            )}
          </div>

          <div className="flex-1" />

          {/* Search */}
          <div className="relative">
            <input type="text" placeholder="Buscar título o persona..."
              value={search} onChange={e => setSearch(e.target.value)}
              className="bg-[#0a1628] border border-white/10 rounded-lg pl-3 pr-7 py-1.5 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-purple-500/50 w-52" />
            {search && <button onClick={() => setSearch('')} className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-500 hover:text-white text-xs">✕</button>}
          </div>
          <span className="text-xs text-slate-500">{filtered.length} de {records.length}</span>
        </div>
      )}

      {/* Main table */}
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="w-8 h-8 border-4 border-purple-500 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : records.length === 0 ? (
        <div className="bg-[#0f1d32] rounded-xl border border-white/5 px-6 py-16 text-center">
          <p className="text-lg font-semibold text-white mb-2">IPS Dashboard</p>
          <p className="text-sm text-slate-400 mb-4">No hay registros IPS. Crea uno nuevo o importa desde Excel.</p>
          <div className="flex items-center justify-center gap-3">
            <button onClick={() => setShowCreateModal(true)} disabled={!cedulaId}
              className="px-6 py-2.5 bg-green-600 hover:bg-green-500 text-white rounded-lg font-medium text-sm transition-colors disabled:opacity-50">
              + Nuevo IPS
            </button>
            <button onClick={() => fileRef.current?.click()} disabled={uploading || !cedulaId}
              className="px-6 py-2.5 bg-purple-600 hover:bg-purple-500 text-white rounded-lg font-medium text-sm transition-colors disabled:opacity-50">
              {uploading ? 'Importando...' : 'Importar Excel IPS'}
            </button>
          </div>
        </div>
      ) : (
        <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
          <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-white">Registro IPS</h3>
              <p className="text-xs text-slate-500">{filtered.length} IPS · Clic para ver contramedidas</p>
            </div>
            <div className="flex items-center gap-1 bg-[#0a1628] rounded-lg p-0.5">
              <button onClick={() => setFilterStatus('all')}
                className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors ${filterStatus === 'all' ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-white'}`}>
                Todos
              </button>
              {uniqueStatuses.map(s => (
                <button key={s} onClick={() => setFilterStatus(filterStatus === s ? 'all' : s)}
                  className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors ${filterStatus === s ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-white'}`}>
                  {s}
                </button>
              ))}
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  <th className="px-4 py-2 text-left text-xs text-slate-400 cursor-pointer select-none hover:text-white" onClick={() => toggleSort('kdf')}>
                    KDF <SortIcon col="kdf" />
                  </th>
                  <th className="px-4 py-2 text-left text-xs text-slate-400 cursor-pointer select-none hover:text-white" onClick={() => toggleSort('titulo')}>
                    Título <SortIcon col="titulo" />
                  </th>
                  <th className="px-3 py-2 text-center text-xs text-slate-400">Personas</th>
                  <th className="px-3 py-2 text-center text-xs text-slate-400 cursor-pointer select-none hover:text-white" onClick={() => toggleSort('fecha')}>
                    Fecha <SortIcon col="fecha" />
                  </th>
                  <th className="px-3 py-2 text-center text-xs text-slate-400">Ubi</th>
                  <th className="px-2 py-2 text-center text-xs text-slate-400" title="6W2H">6W</th>
                  <th className="px-2 py-2 text-center text-xs text-slate-400" title="BBC">BB</th>
                  <th className="px-2 py-2 text-center text-xs text-slate-400" title="5W">5W</th>
                  <th className="px-2 py-2 text-center text-xs text-slate-400" title="Res">Rs</th>
                  <th className="px-3 py-2 text-center text-xs text-slate-400 cursor-pointer select-none hover:text-white" onClick={() => toggleSort('status')}>
                    Status <SortIcon col="status" />
                  </th>
                  <th className="px-3 py-2 text-center text-xs text-slate-400 cursor-pointer select-none hover:text-white" onClick={() => toggleSort('cms')}>
                    CMs <SortIcon col="cms" />
                  </th>
                  <th className="px-2 py-2 text-center text-xs text-slate-400 w-8"></th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(r => {
                  const isExpanded = expandedId === r.id
                  const cmTotal = cmCountMap[r.id] || 0
                  const cmDone = cmDoneMap[r.id] || 0
                  return (
                    <IPSRow key={r.id} record={r} isExpanded={isExpanded}
                      cmTotal={cmTotal} cmDone={cmDone}
                      expandedCMs={expandedCMs} loadingCMs={loadingCMs}
                      filterOwner={filterOwner}
                      onToggle={() => handleExpand(r.id)}
                      onStatusChange={handleIPSStatusChange}
                      onSectionToggle={handleSectionToggle}
                      onCMStatusChange={handleCMStatusChange}
                      onDeleteIPS={handleDeleteIPS}
                      onDeleteCM={handleDeleteCM}
                      onShowAddCM={() => { setShowAddCM(r.id); setNewCM({ ...EMPTY_CM }) }}
                      showAddCM={showAddCM === r.id}
                      newCM={newCM}
                      setNewCM={setNewCM}
                      onAddCM={() => handleAddCM(r.id)}
                      onCancelAddCM={() => setShowAddCM(null)}
                      onEdit={openEditModal} />
                  )
                })}
                {filtered.length === 0 && (
                  <tr><td colSpan={12} className="px-4 py-8 text-center text-slate-500 text-sm">Sin resultados con los filtros aplicados</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Create IPS Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={() => setShowCreateModal(false)}>
          <div className="bg-[#0f1d32] rounded-xl border border-white/10 w-full max-w-2xl max-h-[90vh] overflow-y-auto mx-4 shadow-2xl" onClick={e => e.stopPropagation()}>
            <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between">
              <h2 className="text-lg font-bold text-white">Nuevo Registro IPS</h2>
              <button onClick={() => setShowCreateModal(false)} className="text-slate-400 hover:text-white text-lg">✕</button>
            </div>
            <div className="px-6 py-4 space-y-4">
              {/* Row 1: KDF + Status */}
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs text-slate-400 mb-1">KDF *</label>
                  <input type="number" value={createForm.kdf}
                    onChange={e => setCreateForm(f => ({ ...f, kdf: e.target.value }))}
                    className="w-full bg-[#0a1628] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-purple-500/50"
                    placeholder="Ej: 5" />
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Status</label>
                  <select value={createForm.status}
                    onChange={e => setCreateForm(f => ({ ...f, status: e.target.value }))}
                    className="w-full bg-[#0a1628] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-purple-500/50">
                    {['Open','Closed','BCC','6W2H','Cancelled','Merged','Ascended','Missing'].map(s => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Fecha</label>
                  <input type="date" value={createForm.fecha}
                    onChange={e => setCreateForm(f => ({ ...f, fecha: e.target.value }))}
                    className="w-full bg-[#0a1628] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-purple-500/50" />
                </div>
              </div>

              {/* Row 2: Titulo */}
              <div>
                <label className="block text-xs text-slate-400 mb-1">Título *</label>
                <input type="text" value={createForm.titulo}
                  onChange={e => setCreateForm(f => ({ ...f, titulo: e.target.value }))}
                  className="w-full bg-[#0a1628] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-purple-500/50"
                  placeholder="Descripción del problema" />
              </div>

              {/* Row 3: Ubicacion + Participants */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Ubicación</label>
                  <input type="text" value={createForm.ubicacion}
                    onChange={e => setCreateForm(f => ({ ...f, ubicacion: e.target.value }))}
                    className="w-full bg-[#0a1628] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-purple-500/50"
                    placeholder="Ej: Máquina 3" />
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Participantes (separados por coma)</label>
                  <input type="text" value={createForm.participants}
                    onChange={e => setCreateForm(f => ({ ...f, participants: e.target.value }))}
                    className="w-full bg-[#0a1628] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-purple-500/50"
                    placeholder="Juan, María, Pedro" />
                </div>
              </div>

              {/* Sections */}
              <div className="flex items-center gap-4">
                <span className="text-xs text-slate-400">Secciones:</span>
                {[['section_6w2h','6W2H'],['section_bbc','BBC'],['section_5w','5W'],['section_res','Res']].map(([key, label]) => (
                  <label key={key} className="flex items-center gap-1.5 cursor-pointer">
                    <input type="checkbox" checked={createForm[key]}
                      onChange={() => setCreateForm(f => ({ ...f, [key]: !f[key] }))}
                      className="accent-green-500" />
                    <span className="text-xs text-slate-300">{label}</span>
                  </label>
                ))}
              </div>

              {/* Notes */}
              <div>
                <label className="block text-xs text-slate-400 mb-1">Notas</label>
                <textarea value={createForm.notes}
                  onChange={e => setCreateForm(f => ({ ...f, notes: e.target.value }))}
                  rows={2}
                  className="w-full bg-[#0a1628] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-purple-500/50 resize-none"
                  placeholder="Notas adicionales (opcional)" />
              </div>

              {/* Countermeasures */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-slate-400 uppercase tracking-wider">Contramedidas (opcional)</span>
                  <button onClick={() => setCreateCMs(prev => [...prev, { ...EMPTY_CM }])}
                    className="text-[11px] text-purple-400 hover:text-purple-300">+ Agregar</button>
                </div>
                {createCMs.map((cm, idx) => (
                  <div key={idx} className="flex items-start gap-2 mb-2">
                    <input type="text" placeholder="Descripción de contramedida"
                      value={cm.descripcion}
                      onChange={e => {
                        const arr = [...createCMs]; arr[idx] = { ...arr[idx], descripcion: e.target.value }; setCreateCMs(arr)
                      }}
                      className="flex-1 bg-[#0a1628] border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-purple-500/50" />
                    <input type="text" placeholder="Owner"
                      value={cm.owner}
                      onChange={e => {
                        const arr = [...createCMs]; arr[idx] = { ...arr[idx], owner: e.target.value }; setCreateCMs(arr)
                      }}
                      className="w-28 bg-[#0a1628] border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-purple-500/50" />
                    <select value={cm.status}
                      onChange={e => {
                        const arr = [...createCMs]; arr[idx] = { ...arr[idx], status: e.target.value }; setCreateCMs(arr)
                      }}
                      className="w-24 bg-[#0a1628] border border-white/10 rounded-lg px-2 py-1.5 text-xs text-white focus:outline-none focus:border-purple-500/50">
                      {['Pending','Done','On going','Cancelled'].map(s => <option key={s} value={s}>{s}</option>)}
                    </select>
                    {createCMs.length > 1 && (
                      <button onClick={() => setCreateCMs(prev => prev.filter((_, i) => i !== idx))}
                        className="text-red-400 hover:text-red-300 text-xs px-1 py-1">✕</button>
                    )}
                  </div>
                ))}
              </div>
            </div>
            <div className="px-6 py-4 border-t border-white/5 flex items-center justify-end gap-3">
              <button onClick={() => setShowCreateModal(false)}
                className="px-4 py-2 rounded-lg text-xs text-slate-400 hover:text-white transition-colors">
                Cancelar
              </button>
              <button onClick={handleCreateIPS} disabled={creating || !createForm.kdf || !createForm.titulo}
                className="px-6 py-2 rounded-lg text-xs font-medium bg-green-600 hover:bg-green-500 text-white transition-colors disabled:opacity-50">
                {creating ? 'Creando...' : 'Crear IPS'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Edit IPS Modal ── */}
      {showEditModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={() => setShowEditModal(false)}>
          <div className="bg-[#0f1d32] rounded-xl border border-white/10 w-full max-w-2xl max-h-[90vh] overflow-y-auto mx-4 shadow-2xl" onClick={e => e.stopPropagation()}>
            <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between">
              <h2 className="text-lg font-bold text-white">Editar Registro IPS</h2>
              <button onClick={() => setShowEditModal(false)} className="text-slate-400 hover:text-white text-lg">✕</button>
            </div>
            <div className="px-6 py-4 space-y-4">
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs text-slate-400 mb-1">KDF</label>
                  <input type="number" value={editForm.kdf}
                    onChange={e => setEditForm(f => ({ ...f, kdf: e.target.value }))}
                    className="w-full bg-[#0a1628] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-purple-500/50" />
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Status</label>
                  <select value={editForm.status}
                    onChange={e => setEditForm(f => ({ ...f, status: e.target.value }))}
                    className="w-full bg-[#0a1628] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-purple-500/50">
                    {['Open','Closed','BCC','6W2H','Cancelled','Merged','Ascended','Missing'].map(s => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Fecha</label>
                  <input type="date" value={editForm.fecha}
                    onChange={e => setEditForm(f => ({ ...f, fecha: e.target.value }))}
                    className="w-full bg-[#0a1628] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-purple-500/50" />
                </div>
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1">Título</label>
                <input type="text" value={editForm.titulo}
                  onChange={e => setEditForm(f => ({ ...f, titulo: e.target.value }))}
                  className="w-full bg-[#0a1628] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-purple-500/50" />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Ubicación</label>
                  <input type="text" value={editForm.ubicacion}
                    onChange={e => setEditForm(f => ({ ...f, ubicacion: e.target.value }))}
                    className="w-full bg-[#0a1628] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-purple-500/50" />
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Participantes (separados por coma)</label>
                  <input type="text" value={editForm.participants}
                    onChange={e => setEditForm(f => ({ ...f, participants: e.target.value }))}
                    className="w-full bg-[#0a1628] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-purple-500/50" />
                </div>
              </div>
              <div className="flex items-center gap-4">
                <span className="text-xs text-slate-400">Secciones:</span>
                {[['section_6w2h','6W2H'],['section_bbc','BBC'],['section_5w','5W'],['section_res','Res']].map(([key, label]) => (
                  <label key={key} className="flex items-center gap-1.5 cursor-pointer">
                    <input type="checkbox" checked={editForm[key]}
                      onChange={() => setEditForm(f => ({ ...f, [key]: !f[key] }))}
                      className="accent-green-500" />
                    <span className="text-xs text-slate-300">{label}</span>
                  </label>
                ))}
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1">Notas</label>
                <textarea value={editForm.notes}
                  onChange={e => setEditForm(f => ({ ...f, notes: e.target.value }))}
                  rows={2}
                  className="w-full bg-[#0a1628] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-purple-500/50 resize-none" />
              </div>
            </div>
            <div className="px-6 py-4 border-t border-white/5 flex items-center justify-end gap-3">
              <button onClick={() => setShowEditModal(false)}
                className="px-4 py-2 rounded-lg text-xs text-slate-400 hover:text-white transition-colors">
                Cancelar
              </button>
              <button onClick={handleEditSave} disabled={editSaving}
                className="px-6 py-2 rounded-lg text-xs font-medium bg-purple-600 hover:bg-purple-500 text-white transition-colors disabled:opacity-50">
                {editSaving ? 'Guardando...' : 'Guardar cambios'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}


// ── IPS Row + expandable detail ──
function IPSRow({ record: r, isExpanded, cmTotal, cmDone, expandedCMs, loadingCMs, filterOwner,
  onToggle, onStatusChange, onSectionToggle, onCMStatusChange, onDeleteIPS, onDeleteCM,
  onShowAddCM, showAddCM, newCM, setNewCM, onAddCM, onCancelAddCM, onEdit }) {

  // Filter expanded CMs by owner if filter is active
  const visibleCMs = Array.isArray(filterOwner) && filterOwner.length > 0
    ? expandedCMs.filter(cm => filterOwner.includes(cm.owner))
    : expandedCMs

  return (
    <>
      <tr className={`border-b border-white/5 hover:bg-slate-700/20 cursor-pointer transition-colors ${isExpanded ? 'bg-purple-950/20' : ''}`} onClick={onToggle}>
        <td className="px-4 py-2">
          <span className="bg-blue-500/15 text-blue-300 text-xs font-bold px-2 py-0.5 rounded">{r.kdf}</span>
        </td>
        <td className="px-4 py-2">
          <div className="flex items-center gap-1.5">
            <span className="text-slate-500 text-xs">{isExpanded ? '▾' : '▸'}</span>
            <span className="text-white text-sm font-medium">{r.titulo}</span>
          </div>
        </td>
        <td className="px-3 py-2 text-center">
          {(r.participants?.length || 0) > 0 ? (
            <span className="text-xs text-slate-400" title={r.participants.join(', ')}>
              {r.participants.length} <span className="text-slate-600">pers.</span>
            </span>
          ) : <span className="text-slate-700">—</span>}
        </td>
        <td className="px-3 py-2 text-center text-xs text-slate-400">
          {r.fecha ? new Date(r.fecha + 'T12:00:00').toLocaleDateString('es-MX', { day: '2-digit', month: 'short' }) : <span className="text-slate-700">—</span>}
        </td>
        <td className="px-3 py-2 text-center">
          {r.ubicacion ? <span className="text-xs text-cyan-400">{r.ubicacion}</span> : <span className="text-slate-700">—</span>}
        </td>
        <SectionCell value={r.section_6w2h} onClick={e => { e.stopPropagation(); onSectionToggle(r.id, 'section_6w2h', r.section_6w2h) }} />
        <SectionCell value={r.section_bbc} onClick={e => { e.stopPropagation(); onSectionToggle(r.id, 'section_bbc', r.section_bbc) }} />
        <SectionCell value={r.section_5w} onClick={e => { e.stopPropagation(); onSectionToggle(r.id, 'section_5w', r.section_5w) }} />
        <SectionCell value={r.section_res} onClick={e => { e.stopPropagation(); onSectionToggle(r.id, 'section_res', r.section_res) }} />
        <td className="px-3 py-2 text-center" onClick={e => e.stopPropagation()}>
          <select value={r.status}
            onChange={e => onStatusChange(r.id, e.target.value)}
            className={`text-[11px] font-medium px-2 py-0.5 rounded border-0 cursor-pointer focus:outline-none ${STATUS_COLORS[r.status] || 'bg-slate-500/15 text-slate-400'}`}>
            {['Open','Closed','BCC','6W2H','Cancelled','Merged','Ascended','Missing'].map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </td>
        <td className="px-3 py-2 text-center">
          {cmTotal > 0 ? (
            <span className="text-xs">
              <span className="text-green-400 font-medium">{cmDone}</span>
              <span className="text-slate-600">/{cmTotal}</span>
            </span>
          ) : <span className="text-slate-700">—</span>}
        </td>
        <td className="px-2 py-2 text-center" onClick={e => e.stopPropagation()}>
          <div className="flex gap-1 justify-center">
            <button onClick={() => onEdit(r)}
              className="text-slate-600 hover:text-purple-400 text-xs transition-colors" title="Editar IPS">
              ✎
            </button>
            <button onClick={() => onDeleteIPS(r.id)}
              className="text-slate-600 hover:text-red-400 text-xs transition-colors" title="Eliminar IPS">
              ✕
            </button>
          </div>
        </td>
      </tr>

      {/* Expanded detail */}
      {isExpanded && (
        <tr>
          <td colSpan={12} className="px-0 py-0">
            <div className="bg-[#0a1628] border-y border-purple-500/20">
              {/* Participants */}
              {(r.participants?.length || 0) > 0 && (
                <div className="px-6 py-2 border-b border-white/5">
                  <span className="text-[10px] text-slate-500 uppercase tracking-wider mr-2">Participantes:</span>
                  <div className="inline-flex flex-wrap gap-1.5 mt-1">
                    {r.participants.map((p, i) => (
                      <span key={i} className="bg-slate-700/50 text-slate-300 text-xs px-2 py-0.5 rounded-full">{p}</span>
                    ))}
                  </div>
                </div>
              )}

              {/* Countermeasures */}
              <div className="px-4 py-2">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] text-slate-500 uppercase tracking-wider">
                    Contramedidas ({visibleCMs.length}{filterOwner !== 'all' ? ` de ${expandedCMs.length}` : ''})
                  </span>
                  <button onClick={(e) => { e.stopPropagation(); onShowAddCM() }}
                    className="text-[11px] text-green-400 hover:text-green-300 font-medium">
                    + Agregar CM
                  </button>
                </div>

                {/* Add CM inline form */}
                {showAddCM && (
                  <div className="flex items-center gap-2 mb-2 p-2 bg-slate-800/50 rounded-lg border border-white/5">
                    <input type="text" placeholder="Descripción..."
                      value={newCM.descripcion}
                      onChange={e => setNewCM(prev => ({ ...prev, descripcion: e.target.value }))}
                      className="flex-1 bg-[#0a1628] border border-white/10 rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-purple-500/50"
                      onKeyDown={e => { if (e.key === 'Enter') onAddCM() }} />
                    <input type="text" placeholder="Owner"
                      value={newCM.owner}
                      onChange={e => setNewCM(prev => ({ ...prev, owner: e.target.value }))}
                      className="w-24 bg-[#0a1628] border border-white/10 rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-purple-500/50" />
                    <select value={newCM.status}
                      onChange={e => setNewCM(prev => ({ ...prev, status: e.target.value }))}
                      className="w-24 bg-[#0a1628] border border-white/10 rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-purple-500/50">
                      {['Pending','Done','On going','Cancelled'].map(s => <option key={s} value={s}>{s}</option>)}
                    </select>
                    <button onClick={onAddCM} disabled={!newCM.descripcion.trim()}
                      className="px-3 py-1 bg-green-600 hover:bg-green-500 text-white text-xs rounded transition-colors disabled:opacity-50">
                      Agregar
                    </button>
                    <button onClick={onCancelAddCM} className="text-slate-400 hover:text-white text-xs px-1">✕</button>
                  </div>
                )}

                {loadingCMs ? (
                  <div className="flex items-center gap-2 py-3 px-2">
                    <div className="w-4 h-4 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />
                    <span className="text-xs text-slate-500">Cargando...</span>
                  </div>
                ) : visibleCMs.length === 0 ? (
                  <p className="text-xs text-slate-600 py-2 px-2">
                    {filterOwner !== 'all' ? `Sin contramedidas de ${filterOwner}` : 'Sin contramedidas registradas'}
                  </p>
                ) : (
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-white/5">
                        <th className="px-3 py-1.5 text-left text-slate-500 font-medium">Contramedida</th>
                        <th className="px-3 py-1.5 text-center text-slate-500 font-medium w-24">Owner</th>
                        <th className="px-3 py-1.5 text-center text-slate-500 font-medium w-28">Status</th>
                        <th className="px-2 py-1.5 text-center text-slate-500 font-medium w-8"></th>
                      </tr>
                    </thead>
                    <tbody>
                      {visibleCMs.map(cm => (
                        <tr key={cm.id} className="border-b border-white/5 hover:bg-slate-700/10">
                          <td className="px-3 py-1.5 text-slate-300">{cm.descripcion}</td>
                          <td className="px-3 py-1.5 text-center text-slate-400">{cm.owner || '—'}</td>
                          <td className="px-3 py-1.5 text-center">
                            <select value={cm.status}
                              onChange={e => onCMStatusChange(cm.id, e.target.value)}
                              className={`text-[11px] font-medium px-2 py-0.5 rounded border-0 cursor-pointer focus:outline-none ${CM_STATUS_COLORS[cm.status] || 'bg-slate-500/15 text-slate-400'}`}>
                              {['Pending','Done','On going','Cancelled'].map(s => (
                                <option key={s} value={s}>{s}</option>
                              ))}
                            </select>
                          </td>
                          <td className="px-2 py-1.5 text-center">
                            <button onClick={() => onDeleteCM(cm.id)}
                              className="text-slate-600 hover:text-red-400 text-[10px] transition-colors" title="Eliminar CM">
                              ✕
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  )
}


function SectionCell({ value, onClick }) {
  return (
    <td className="px-2 py-2 text-center">
      <button onClick={onClick}
        className={`w-5 h-5 rounded text-[10px] font-bold transition-colors ${
          value ? 'bg-green-500/20 text-green-400' : 'bg-slate-700/30 text-slate-600 hover:text-slate-400'
        }`}>
        {value ? '✓' : '·'}
      </button>
    </td>
  )
}


function StatCard({ label, value, color = 'text-white', onClick, active }) {
  return (
    <button
      onClick={onClick}
      className={`bg-[#0f1d32] rounded-xl border p-3 text-center transition-all ${
        active ? 'border-purple-500/50 ring-1 ring-purple-500/30' : 'border-white/5 hover:border-purple-500/30'
      }`}>
      <p className={`text-xl font-bold ${color}`}>{value}</p>
      <p className="text-[10px] text-slate-500 mt-0.5">{label}</p>
    </button>
  )
}
