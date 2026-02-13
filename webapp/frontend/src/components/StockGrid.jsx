import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getSector, getWatchlist } from '../api/client'
import StockCard from './StockCard'

const SORT_OPTIONS = [
  { key: 'change_pct', label: 'Change %', desc: true },
  { key: 'symbol', label: 'Symbol', desc: false },
  { key: 'price', label: 'Price', desc: true },
]

export default function StockGrid() {
  const { sectorId } = useParams()
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [sortKey, setSortKey] = useState('change_pct')
  const [sortDesc, setSortDesc] = useState(true)

  useEffect(() => {
    const fetcher = sectorId === 'CUSTOM' ? getWatchlist : () => getSector(sectorId)
    fetcher()
      .then(setData)
      .catch(e => setError(e.message))
  }, [sectorId])

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-12 text-center text-gray-500">
        Sector data not found. Run build_sector_map.py first.
      </div>
    )
  }

  if (!data) {
    return <div className="max-w-7xl mx-auto px-4 py-12 text-center text-gray-400">Loading...</div>
  }

  const sorted = [...data.stocks].sort((a, b) => {
    let av = a[sortKey], bv = b[sortKey]
    if (typeof av === 'string') {
      return sortDesc ? bv.localeCompare(av) : av.localeCompare(bv)
    }
    return sortDesc ? bv - av : av - bv
  })

  function changeColor(pct) {
    if (pct > 0) return 'text-green-600 dark:text-green-400'
    if (pct < 0) return 'text-red-600 dark:text-red-400'
    return 'text-gray-500'
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-6 space-y-4">
      {/* Back nav + sector header */}
      <div className="flex items-center gap-3">
        <Link to="/" className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </Link>
        <div>
          <h1 className="text-xl font-bold">
            {data.name}
            <span className="ml-2 text-sm font-normal text-gray-400">({sectorId})</span>
          </h1>
          <p className={`text-sm font-semibold ${changeColor(data.change_pct)}`}>
            {data.change_pct > 0 ? '+' : ''}{data.change_pct.toFixed(2)}%
            <span className="ml-2 text-gray-400 font-normal">{data.stocks.length} stocks</span>
          </p>
        </div>
      </div>

      {/* Sort controls */}
      <div className="flex gap-2 text-xs">
        <span className="text-gray-400 py-1">Sort:</span>
        {SORT_OPTIONS.map(opt => (
          <button
            key={opt.key}
            onClick={() => {
              if (sortKey === opt.key) {
                setSortDesc(!sortDesc)
              } else {
                setSortKey(opt.key)
                setSortDesc(opt.desc)
              }
            }}
            className={`px-3 py-1 rounded-full border transition-colors ${
              sortKey === opt.key
                ? 'bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 border-transparent'
                : 'border-gray-300 dark:border-gray-600 hover:border-gray-400'
            }`}
          >
            {opt.label} {sortKey === opt.key ? (sortDesc ? '↓' : '↑') : ''}
          </button>
        ))}
      </div>

      {/* Stock grid */}
      <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 xl:grid-cols-8 gap-3">
        {sorted.map(stock => (
          <StockCard
            key={stock.symbol}
            symbol={stock.symbol}
            price={stock.price}
            change_pct={stock.change_pct}
            sectorId={sectorId}
            last_updated={stock.last_updated}
          />
        ))}
      </div>
    </div>
  )
}
