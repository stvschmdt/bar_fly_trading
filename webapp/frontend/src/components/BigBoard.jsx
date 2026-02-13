import { useState, useEffect, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { getBigBoard } from '../api/client'
import SECTOR_COLORS from '../constants/sectorColors'
import { isStale } from '../utils/freshness'
import { useWorkbench } from '../hooks/useWorkbench'

const SORT_OPTIONS = [
  { key: 'change_pct', label: '% Change', desc: true },
  { key: 'symbol', label: 'A-Z', desc: false },
  { key: 'sector_id', label: 'Sector', desc: false },
]

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
  const [sortKey, setSortKey] = useState('change_pct')
  const [sortDesc, setSortDesc] = useState(true)
  const { has, toggle } = useWorkbench()

  useEffect(() => {
    getBigBoard()
      .then((data) => {
        setStocks(data.stocks || [])
        setSectors(data.sectors || [])
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const sorted = useMemo(() => {
    return [...stocks].sort((a, b) => {
      let av = a[sortKey], bv = b[sortKey]
      if (sortKey === 'sector_id') {
        // Group by sector, then alphabetically within
        const cmp = (av || '').localeCompare(bv || '')
        if (cmp !== 0) return sortDesc ? -cmp : cmp
        return a.symbol.localeCompare(b.symbol)
      }
      if (typeof av === 'string') {
        return sortDesc ? bv.localeCompare(av) : av.localeCompare(bv)
      }
      return sortDesc ? (bv || 0) - (av || 0) : (av || 0) - (bv || 0)
    })
  }, [stocks, sortKey, sortDesc])

  function handleSort(key) {
    if (key === sortKey) {
      setSortDesc(!sortDesc)
    } else {
      setSortKey(key)
      setSortDesc(SORT_OPTIONS.find(o => o.key === key)?.desc ?? true)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400">
        Loading...
      </div>
    )
  }

  return (
    <div className="max-w-[1600px] mx-auto px-2 py-4">
      <div className="flex items-center justify-between mb-4 px-2">
        <h1 className="text-2xl font-bold">The Big Board</h1>
        <div className="flex gap-1">
          {SORT_OPTIONS.map((opt) => (
            <button
              key={opt.key}
              onClick={() => handleSort(opt.key)}
              className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                sortKey === opt.key
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
              }`}
            >
              {opt.label}
              {sortKey === opt.key && (
                <span className="ml-0.5">{sortDesc ? '\u2193' : '\u2191'}</span>
              )}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-7 md:grid-cols-10 lg:grid-cols-14 gap-1">
        {sorted.map((s) => {
          const sector = SECTOR_COLORS[s.sector_id] || { color: '#6b7280' }
          const stale = isStale(s.last_updated)
          const inBench = has(s.symbol)
          return (
            <div key={s.symbol} className="relative group">
              <Link
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
              <button
                onClick={() => toggle(s.symbol)}
                className={`absolute -top-1 -right-1 w-4 h-4 rounded-full flex items-center justify-center text-[8px] leading-none transition-all ${
                  inBench
                    ? 'bg-blue-500 text-white opacity-100'
                    : 'bg-gray-300 dark:bg-gray-600 text-gray-600 dark:text-gray-300 opacity-0 group-hover:opacity-100'
                }`}
              >
                {inBench ? '\u2713' : '+'}
              </button>
            </div>
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
