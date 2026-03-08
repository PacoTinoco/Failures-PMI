/**
 * Calcula el color del semáforo según el valor, target y dirección.
 *
 * Dirección:
 *  "≥" → verde si valor >= target
 *  "≤" → verde si valor <= target
 *  "=" → verde si valor == target
 */
export function getSemaforoColor(value, target, direction) {
  if (value == null || value === '' || target == null) return 'neutral'

  const val = Number(value)
  const tgt = Number(target)

  if (isNaN(val) || isNaN(tgt)) return 'neutral'

  if (direction === '≥' || direction === '>=' || direction === 'gte') {
    if (val >= tgt) return 'green'
    if (val >= tgt * 0.9) return 'yellow'
    return 'red'
  }

  if (direction === '≤' || direction === '<=' || direction === 'lte') {
    if (val <= tgt) return 'green'
    if (val <= tgt * 1.1) return 'yellow'
    return 'red'
  }

  if (direction === '=' || direction === 'eq') {
    return val === tgt ? 'green' : 'red'
  }

  return 'neutral'
}

const colorMap = {
  green: 'bg-green-500/20 text-green-400 border-green-500/30',
  yellow: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  red: 'bg-red-500/20 text-red-400 border-red-500/30',
  neutral: 'bg-slate-700/50 text-slate-400 border-slate-600'
}

const dotColorMap = {
  green: 'bg-green-400',
  yellow: 'bg-yellow-400',
  red: 'bg-red-400',
  neutral: 'bg-slate-500'
}

export default function SemaforoIndicador({ value, target, direction, showValue = true }) {
  const color = getSemaforoColor(value, target, direction)

  return (
    <div className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-md border ${colorMap[color]}`}>
      <span className={`w-2 h-2 rounded-full ${dotColorMap[color]}`} />
      {showValue && (
        <span className="text-xs font-medium">
          {value != null && value !== '' ? value : '—'}
        </span>
      )}
    </div>
  )
}

export function SemaforoDot({ value, target, direction }) {
  const color = getSemaforoColor(value, target, direction)
  return <span className={`inline-block w-2.5 h-2.5 rounded-full ${dotColorMap[color]}`} />
}
