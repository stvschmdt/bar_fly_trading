import { Link } from 'react-router-dom'

function changeColor(pct) {
  if (pct > 0) return 'text-green-600 dark:text-green-400'
  if (pct < 0) return 'text-red-600 dark:text-red-400'
  return 'text-gray-500'
}

function changeBg(pct) {
  const intensity = Math.min(Math.abs(pct) / 3, 1) // cap at 3%
  if (pct > 0) return `rgba(34, 197, 94, ${0.06 + intensity * 0.12})`
  if (pct < 0) return `rgba(239, 68, 68, ${0.06 + intensity * 0.12})`
  return undefined
}

export default function SectorCard({ id, name, change_pct, stock_count, price, large }) {
  const arrow = change_pct > 0 ? '+' : ''
  const bg = changeBg(change_pct)

  const card = (
    <div
      className={`
        rounded-xl border border-gray-200 dark:border-gray-700
        hover:border-gray-400 dark:hover:border-gray-500
        transition-all duration-200 cursor-pointer
        flex flex-col items-center justify-center text-center
        price-transition
        ${large ? 'p-5 min-h-[120px]' : 'p-4 min-h-[100px]'}
      `}
      style={{ backgroundColor: bg }}
    >
      <div className={`font-bold tracking-wide ${large ? 'text-lg' : 'text-base'}`}>
        {id}
      </div>
      {name && (
        <div className="text-xs text-gray-500 dark:text-gray-400 mt-0.5 truncate max-w-full">
          {name}
        </div>
      )}
      <div className={`font-semibold mt-1 ${large ? 'text-xl' : 'text-base'} ${changeColor(change_pct)}`}>
        {arrow}{change_pct.toFixed(2)}%
      </div>
      {price > 0 && (
        <div className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
          ${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </div>
      )}
      {stock_count > 0 && (
        <div className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">
          {stock_count} stocks
        </div>
      )}
    </div>
  )

  // SPY/QQQ link to Big Board
  if (id === 'SPY' || id === 'QQQ') return <Link to="/bigboard">{card}</Link>

  return <Link to={`/sector/${id}`}>{card}</Link>
}
