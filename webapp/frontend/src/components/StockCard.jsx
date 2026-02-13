import { Link } from 'react-router-dom'
import { isStale } from '../utils/freshness'
import { useWorkbench } from '../hooks/useWorkbench'

function changeColor(pct, stale) {
  if (stale) return 'text-gray-400 dark:text-gray-500'
  if (pct > 0) return 'text-green-600 dark:text-green-400'
  if (pct < 0) return 'text-red-600 dark:text-red-400'
  return 'text-gray-500'
}

function changeBg(pct, stale) {
  if (stale) return 'rgba(156, 163, 175, 0.08)'
  const intensity = Math.min(Math.abs(pct) / 5, 1)
  if (pct > 0) return `rgba(34, 197, 94, ${0.04 + intensity * 0.1})`
  if (pct < 0) return `rgba(239, 68, 68, ${0.04 + intensity * 0.1})`
  return undefined
}

export default function StockCard({ symbol, price, change_pct, sectorId, last_updated }) {
  const stale = isStale(last_updated)
  const { has, toggle } = useWorkbench()
  const inBench = has(symbol)
  const arrow = change_pct > 0 ? '+' : ''

  return (
    <div className="relative group">
      <Link to={`/sector/${sectorId}/${symbol}`}>
        <div
          className={`rounded-lg border transition-all duration-200 cursor-pointer p-3 text-center price-transition ${
            stale
              ? 'border-gray-300 dark:border-gray-600'
              : 'border-gray-200 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-500'
          }`}
          style={{ backgroundColor: changeBg(change_pct, stale) }}
        >
          <div className={`font-bold text-sm ${stale ? 'text-gray-400 dark:text-gray-500' : ''}`}>{symbol}</div>
          <div className={`font-semibold mt-1 text-sm ${changeColor(change_pct, stale)}`}>
            {arrow}{change_pct.toFixed(2)}%
          </div>
          {price > 0 && (
            <div className={`text-xs mt-0.5 ${stale ? 'text-gray-300 dark:text-gray-600' : 'text-gray-500 dark:text-gray-400'}`}>
              ${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
          )}
        </div>
      </Link>
      <button
        onClick={(e) => { e.preventDefault(); toggle(symbol) }}
        className={`absolute top-1 right-1 w-5 h-5 rounded-full flex items-center justify-center text-[10px] transition-all ${
          inBench
            ? 'bg-blue-500 text-white opacity-100'
            : 'bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400 opacity-0 group-hover:opacity-100'
        }`}
        aria-label={inBench ? `Remove ${symbol} from workbench` : `Add ${symbol} to workbench`}
      >
        {inBench ? '\u2713' : '+'}
      </button>
    </div>
  )
}
