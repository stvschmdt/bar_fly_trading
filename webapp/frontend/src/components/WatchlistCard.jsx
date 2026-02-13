import { useState } from 'react'
import { Link } from 'react-router-dom'
import { setWatchlist as saveWatchlist } from '../api/client'

function changeBg(pct) {
  const intensity = Math.min(Math.abs(pct) / 3, 1)
  if (pct > 0) return `rgba(34, 197, 94, ${0.06 + intensity * 0.12})`
  if (pct < 0) return `rgba(239, 68, 68, ${0.06 + intensity * 0.12})`
  return undefined
}

function changeColor(pct) {
  if (pct > 0) return 'text-green-600 dark:text-green-400'
  if (pct < 0) return 'text-red-600 dark:text-red-400'
  return 'text-gray-500'
}

export default function WatchlistCard({ watchlist }) {
  const [editing, setEditing] = useState(false)
  const [input, setInput] = useState('')
  const [saving, setSaving] = useState(false)

  const hasStocks = watchlist?.stocks?.length > 0

  function startEditing() {
    // Pre-fill with current symbols
    if (watchlist?.stocks?.length) {
      setInput(watchlist.stocks.map(s => s.symbol).join(', '))
    }
    setEditing(true)
  }

  async function handleSave() {
    if (!input.trim()) return
    setSaving(true)
    try {
      await saveWatchlist(input)
      window.location.reload()
    } catch (e) {
      alert('Failed to save watchlist: ' + e.message)
    }
    setSaving(false)
  }

  function handleFile(e) {
    const file = e.target.files[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      const text = ev.target.result
      // Parse CSV: extract symbols from first column (skip header if present)
      const lines = text.trim().split('\n')
      const symbols = []
      for (const line of lines) {
        const val = line.split(',')[0].trim().toUpperCase()
        // Skip common headers
        if (['SYMBOL', 'TICKER', 'NAME', ''].includes(val)) continue
        if (/^[A-Z]{1,5}$/.test(val)) symbols.push(val)
      }
      setInput(symbols.join(', '))
    }
    reader.readAsText(file)
  }

  // Edit modal
  if (editing) {
    return (
      <div className="rounded-xl border-2 border-dashed border-gray-300 dark:border-gray-600 p-4 flex flex-col items-center justify-center min-h-[120px] space-y-3">
        <p className="text-xs font-medium text-gray-500 dark:text-gray-400">Edit Watchlist</p>
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="AAPL, NVDA, TSLA, JPM..."
          className="w-full text-xs p-2 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 resize-none"
          rows={3}
        />
        <div className="flex gap-2 w-full">
          <label className="flex-1 text-center text-xs py-1 px-2 rounded border border-gray-300 dark:border-gray-600 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors">
            Upload CSV
            <input type="file" accept=".csv,.txt" onChange={handleFile} className="hidden" />
          </label>
        </div>
        <div className="flex gap-2 w-full">
          <button
            onClick={() => setEditing(false)}
            className="flex-1 text-xs py-1 rounded border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={saving || !input.trim()}
            className="flex-1 text-xs py-1 rounded bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 disabled:opacity-50 transition-colors"
          >
            {saving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    )
  }

  // Empty state — show add button
  if (!hasStocks) {
    return (
      <button
        onClick={startEditing}
        className="rounded-xl border-2 border-dashed border-gray-300 dark:border-gray-600 p-5 min-h-[120px] flex flex-col items-center justify-center hover:border-gray-400 dark:hover:border-gray-500 transition-colors cursor-pointer"
      >
        <span className="text-2xl text-gray-300 dark:text-gray-600">+</span>
        <span className="text-xs text-gray-400 dark:text-gray-500 mt-1">Custom Watchlist</span>
      </button>
    )
  }

  // Active watchlist card
  const bg = changeBg(watchlist.change_pct)

  return (
    <div className="relative">
      <Link to="/sector/CUSTOM">
        <div
          className="rounded-xl border border-gray-200 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-500 transition-all duration-200 cursor-pointer flex flex-col items-center justify-center text-center p-5 min-h-[120px] price-transition"
          style={{ backgroundColor: bg }}
        >
          <div className="font-bold tracking-wide text-lg">
            {watchlist.name || 'Custom'}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
            Watchlist
          </div>
          <div className={`font-semibold mt-1 text-xl ${changeColor(watchlist.change_pct)}`}>
            {watchlist.change_pct > 0 ? '+' : ''}{watchlist.change_pct.toFixed(2)}%
          </div>
          <div className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">
            {watchlist.stocks.length} stocks
          </div>
        </div>
      </Link>
      <button
        onClick={(e) => { e.preventDefault(); startEditing() }}
        className="absolute top-1.5 right-1.5 w-5 h-5 rounded-full text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 flex items-center justify-center text-xs transition-colors"
        title="Edit watchlist"
      >
        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
        </svg>
      </button>
    </div>
  )
}
