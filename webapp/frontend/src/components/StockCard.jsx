import { Link } from 'react-router-dom'

function changeColor(pct) {
  if (pct > 0) return 'text-green-600 dark:text-green-400'
  if (pct < 0) return 'text-red-600 dark:text-red-400'
  return 'text-gray-500'
}

function changeBg(pct) {
  const intensity = Math.min(Math.abs(pct) / 5, 1)
  if (pct > 0) return `rgba(34, 197, 94, ${0.04 + intensity * 0.1})`
  if (pct < 0) return `rgba(239, 68, 68, ${0.04 + intensity * 0.1})`
  return undefined
}

export default function StockCard({ symbol, price, change_pct, sectorId }) {
  const arrow = change_pct > 0 ? '+' : ''

  return (
    <Link to={`/sector/${sectorId}/${symbol}`}>
      <div
        className="rounded-lg border border-gray-200 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-500 transition-all duration-200 cursor-pointer p-3 text-center price-transition"
        style={{ backgroundColor: changeBg(change_pct) }}
      >
        <div className="font-bold text-sm">{symbol}</div>
        <div className={`font-semibold mt-1 text-sm ${changeColor(change_pct)}`}>
          {arrow}{change_pct.toFixed(2)}%
        </div>
        {price > 0 && (
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
            ${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
        )}
      </div>
    </Link>
  )
}
