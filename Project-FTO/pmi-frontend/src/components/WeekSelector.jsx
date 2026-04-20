import { useMemo } from 'react'

function getMondays(year) {
  const mondays = []
  const date = new Date(year, 0, 1)
  // Find first Monday
  while (date.getDay() !== 1) {
    date.setDate(date.getDate() + 1)
  }
  // Collect all Mondays of the year
  while (date.getFullYear() === year) {
    mondays.push(new Date(date))
    date.setDate(date.getDate() + 7)
  }
  return mondays
}

function getISOWeekNumber(date) {
  const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()))
  d.setUTCDate(d.getUTCDate() + 4 - (d.getUTCDay() || 7))
  const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1))
  return Math.ceil(((d - yearStart) / 86400000 + 1) / 7)
}

function formatWeekLabel(date) {
  const endOfWeek = new Date(date)
  endOfWeek.setDate(endOfWeek.getDate() + 6)
  const opts = { day: 'numeric', month: 'short' }
  const start = date.toLocaleDateString('es-MX', opts)
  const end = endOfWeek.toLocaleDateString('es-MX', opts)
  const wn = getISOWeekNumber(date)
  return `W${wn} · ${start} — ${end}`
}

function toISODate(date) {
  const yyyy = date.getFullYear()
  const mm = String(date.getMonth() + 1).padStart(2, '0')
  const dd = String(date.getDate()).padStart(2, '0')
  return `${yyyy}-${mm}-${dd}`
}

export default function WeekSelector({ value, onChange, className = '' }) {
  const mondays = useMemo(() => getMondays(new Date().getFullYear()), [])

  // Find closest Monday to today for default
  const today = new Date()
  const closestMonday = mondays.reduce((closest, monday) => {
    const diff = Math.abs(monday - today)
    const closestDiff = Math.abs(closest - today)
    return diff < closestDiff ? monday : closest
  }, mondays[0])

  const selectedValue = value || toISODate(closestMonday)

  return (
    <select
      value={selectedValue}
      onChange={(e) => onChange(e.target.value)}
      className={`px-3 py-2 bg-[#0f1d32] border border-white/10 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 ${className}`}
    >
      {mondays.map((monday) => (
        <option key={toISODate(monday)} value={toISODate(monday)}>
          {formatWeekLabel(monday)}
        </option>
      ))}
    </select>
  )
}
