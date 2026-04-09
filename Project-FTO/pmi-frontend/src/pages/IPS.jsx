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
  const [search, setSearch]         = useState('')
  const [filterKdf, setFilterKdf]   = useState('all')
  const [filterStatus, setFilterStatus] = useState('all')
  const [filterUbi, setFilterUbi]   = useState('all')
  const [sortKey, setSortKey]       = useState('fecha')
  const [sortDir, setSortDir]       = useState('desc')

  // Expanded / detail
  const [expandedId, setExpandedId]     = useState(null)
  const [expandedCMs, setExpandedCMs]   = useState([]) // CMs for expanded IPS
  const [loadingCMs, setLoadingCMs]     = useState(false)

  // Upload
  const [uploading, setUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState(null)
  const fileRef = useRef(null)

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

  // ── Filter & sort ──
  const uniqueKdfs = [...new Set(records.map(r => r.kdf))].sort((a, b) => a - b)
  const uniqueStatuses = [...new Set(records.map(r => r.status))].sort()
  const uniqueUbis = [...new Set(records.map(r => r.ubicacion).filter(Boolean))].sort()

  const filtered = records
    .filter(r => {
      if (search && !r.titulo.toLowerCase().includes(search.toLowerCase()) &&
          !(r.participants || []).some(p => p.toLowerCase().includes(search.toLowerCase()))) return false
      if (filterKdf !== 'all' && r.kdf !== parseInt(filterKdf)) return false
      if (filterStatus !== 'all' && r.status !== filterStatus) return false
      if (filterUbi !== 'all' && r.ubicacion !== filterUbi) return false
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
  for (const cm of allCMs) {
    cmCountMap[cm.ips_id] = (cmCountMap[cm.ips_id] || 0) + 1
  }
  const cmDoneMap = {}
  for (const cm of allCMs) {
    if (cm.status === 'Done') cmDoneMap[cm.ips_id] = (cmDoneMap[cm.ips_id] || 0) + 1
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white">IPS — Issue Problem Solving</h1>
          <p className="text-sm text-slate-400">Registro, seguimiento y contramedidas de problemas por máquina</p>
        </div>
        <div className="flex items-center gap-3">
          <CedulaSelector cedulas={cedulas} selectedCedulaId={cedulaId} onChange={setCedulaId} loading={cedulasLoading} />
          <input ref={fileRef} type="file" accept=".xlsx,.xls" onChange={handleUpload} className="hidden" />
          <button onClick={() => fileRef.current?.click()} disabled={uploading || !cedulaId}
            className="px-4 py-2 rounded-lg text-xs font-medium bg-purple-600 hover:bg-purple-500 text-white transition-colors disabled:opacity-50 whitespace-nowrap">
            {uploading ? 'Importando...' : 'Importar Excel'}
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

      {/* KDF quick filter */}
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
          <p className="text-sm text-slate-400 mb-4">No hay registros IPS cargados. Importa tu archivo Excel para comenzar.</p>
          <button onClick={() => fileRef.current?.click()} disabled={uploading || !cedulaId}
            className="px-6 py-2.5 bg-purple-600 hover:bg-purple-500 text-white rounded-lg font-medium text-sm transition-colors disabled:opacity-50">
            {uploading ? 'Importando...' : 'Importar Excel IPS'}
          </button>
        </div>
      ) : (
        <div className="bg-[#0f1d32] rounded-xl border border-white/5 overflow-hidden">
          <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-white">Registro IPS</h3>
              <p className="text-xs text-slate-500">{filtered.length} IPS · Clic para ver contramedidas</p>
            </div>
            {/* Status filter pills */}
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
                      onToggle={() => handleExpand(r.id)}
                      onStatusChange={handleIPSStatusChange}
                      onSectionToggle={handleSectionToggle}
                      onCMStatusChange={handleCMStatusChange} />
                  )
                })}
                {filtered.length === 0 && (
                  <tr><td colSpan={11} className="px-4 py-8 text-center text-slate-500 text-sm">Sin resultados con los filtros aplicados</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}


// ── IPS Row + expandable detail ──
function IPSRow({ record: r, isExpanded, cmTotal, cmDone, expandedCMs, loadingCMs, onToggle, onStatusChange, onSectionToggle, onCMStatusChange }) {
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
        {/* Section checkmarks */}
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
      </tr>

      {/* Expanded detail: participants + countermeasures */}
      {isExpanded && (
        <tr>
          <td colSpan={11} className="px-0 py-0">
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
                    Contramedidas ({expandedCMs.length})
                  </span>
                </div>

                {loadingCMs ? (
                  <div className="flex items-center gap-2 py-3 px-2">
                    <div className="w-4 h-4 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />
                    <span className="text-xs text-slate-500">Cargando...</span>
                  </div>
                ) : expandedCMs.length === 0 ? (
                  <p className="text-xs text-slate-600 py-2 px-2">Sin contramedidas registradas</p>
                ) : (
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-white/5">
                        <th className="px-3 py-1.5 text-left text-slate-500 font-medium">Contramedida</th>
                        <th className="px-3 py-1.5 text-center text-slate-500 font-medium w-24">Owner</th>
                        <th className="px-3 py-1.5 text-center text-slate-500 font-medium w-28">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {expandedCMs.map(cm => (
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
