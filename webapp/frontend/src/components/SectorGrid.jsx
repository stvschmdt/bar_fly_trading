import { useEffect, useState } from 'react'
import { getSectors, getSignals, getWatchlist } from '../api/client'
import SectorCard from './SectorCard'
import WatchlistCard from './WatchlistCard'
import SignalsBanner from './SignalsBanner'

export default function SectorGrid() {
  const [data, setData] = useState(null)
  const [signals, setSignals] = useState([])
  const [watchlist, setWatchlist] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    getSectors()
      .then(setData)
      .catch(e => setError(e.message))

    getSignals()
      .then(d => setSignals(d.signals || []))
      .catch(() => {})

    getWatchlist()
      .then(setWatchlist)
      .catch(() => {})
  }, [])

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-12">
        <div className="text-center text-gray-500 dark:text-gray-400">
          <p className="text-lg font-medium mb-2">No data yet</p>
          <p className="text-sm">Run <code className="bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded">python -m webapp.backend.build_sector_map</code> to generate sector data.</p>
        </div>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-12 text-center text-gray-400">
        Loading...
      </div>
    )
  }

  const hasWatchlist = watchlist && watchlist.stocks && watchlist.stocks.length > 0

  return (
    <div className="max-w-7xl mx-auto px-4 py-6 space-y-6">
      {/* Top row: SPY + QQQ + Custom Watchlist */}
      <div className={`grid gap-4 max-w-2xl mx-auto ${hasWatchlist ? 'grid-cols-3' : 'grid-cols-2 max-w-md'}`}>
        {data.indices.map(idx => (
          <SectorCard
            key={idx.id}
            id={idx.id}
            name={idx.name}
            change_pct={idx.change_pct}
            price={idx.price}
            stock_count={0}
            large
          />
        ))}
        <WatchlistCard watchlist={watchlist} />
      </div>

      {/* Sector grid */}
      <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-3">
        {data.sectors.map(sector => (
          <SectorCard
            key={sector.id}
            id={sector.id}
            name={sector.name}
            change_pct={sector.change_pct}
            stock_count={sector.stock_count}
            price={0}
          />
        ))}
      </div>

      {/* Signals banner */}
      <SignalsBanner signals={signals} />

      {/* Footer info */}
      {data.last_updated && (
        <p className="text-center text-xs text-gray-400 dark:text-gray-500">
          Last updated: {new Date(data.last_updated).toLocaleString()}
        </p>
      )}
    </div>
  )
}
