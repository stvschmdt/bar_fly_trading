import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { getBigBoard } from '../api/client'
import SECTOR_COLORS from '../constants/sectorColors'
import { isStale } from '../utils/freshness'

function changeBg(pct, stale) {
  if (stale) return 'rgba(156, 163, 175, 0.18)'
  if (pct > 0) {
    const intensity = Math.min(pct / 4, 1)
    return `rgba(34, 197, 94, ${0.15 + intensity * 0.55})`
  }
  if (pct < 0) {
    const intensity = Math.min(Math.abs(pct) / 4, 1)
    return `rgba(239, 68, 68, ${0.15 + intensity * 0.55})`
  }
  return 'rgba(156, 163, 175, 0.15)'
}

function changeText(pct, stale) {
  if (stale) return 'text-gray-400 dark:text-gray-500'
  if (pct > 0) return 'text-green-700 dark:text-green-300'
  if (pct < 0) return 'text-red-700 dark:text-red-300'
  return 'text-gray-500 dark:text-gray-400'
}

export default function BigBoard() {
  const [stocks, setStocks] = useState([])
  const [sectors, setSectors] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getBigBoard()
      .then((data) => {
        setStocks(data.stocks || [])
        setSectors(data.sectors || [])
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400">
        Loading...
      </div>
    )
  }

  return (
    <div className="max-w-[1600px] mx-auto px-2 py-4">
      <h1 className="text-2xl font-bold mb-4 px-2">The Big Board</h1>

      <div className="grid grid-cols-7 md:grid-cols-10 lg:grid-cols-14 gap-1">
        {stocks.map((s) => {
          const sector = SECTOR_COLORS[s.sector_id] || { color: '#6b7280' }
          const stale = isStale(s.last_updated)
          return (
            <Link
              key={s.symbol}
              to={`/sector/${s.sector_id}/${s.symbol}`}
              className="block rounded text-center px-1 py-2 hover:opacity-80 transition-opacity"
              style={{
                backgroundColor: changeBg(s.change_pct, stale),
                border: `2px solid ${stale ? '#9ca3af' : sector.color}`,
              }}
            >
              <div className={`text-[11px] font-semibold truncate ${stale ? 'text-gray-400 dark:text-gray-500' : 'dark:text-gray-100'}`}>
                {s.symbol}
              </div>
              <div className={`text-[10px] font-mono ${changeText(s.change_pct, stale)}`}>
                {s.change_pct > 0 ? '+' : ''}
                {s.change_pct.toFixed(1)}%
              </div>
            </Link>
          )
        })}
      </div>

      {/* Sector legend */}
      <div className="mt-6 px-2 flex flex-wrap gap-3 text-xs text-gray-600 dark:text-gray-400">
        {sectors.map((sec) => {
          const c = SECTOR_COLORS[sec.id] || { color: '#6b7280' }
          return (
            <div key={sec.id} className="flex items-center gap-1.5">
              <span
                className="inline-block w-3 h-3 rounded-sm"
                style={{ backgroundColor: c.color }}
              />
              <span>{c.name || sec.name}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
