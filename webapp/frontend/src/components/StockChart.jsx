import { useState, useEffect, useMemo } from 'react'
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts'
import { getSymbolHistory } from '../api/client'

const RANGES = [
  { key: '5D', days: 5 },
  { key: '1M', days: 30 },
  { key: '3M', days: 90 },
  { key: '6M', days: 180 },
  { key: '1Y', days: 365 },
  { key: '3Y', days: 1095 },
]

const OVERLAYS = [
  { key: 'sma_20', label: 'SMA 20', color: '#f59e0b' },
  { key: 'sma_50', label: 'SMA 50', color: '#8b5cf6' },
  { key: 'sma_200', label: 'SMA 200', color: '#ef4444' },
  { key: 'bb', label: 'Bollinger', color: '#6b7280' },
]

function formatDate(dateStr, range) {
  const d = new Date(dateStr + 'T00:00:00')
  if (range <= 30) return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  if (range <= 365) return d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
  return d.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
}

function formatPrice(val) {
  if (val == null) return ''
  return '$' + val.toFixed(2)
}

function formatVol(val) {
  if (val == null) return ''
  if (val >= 1e9) return (val / 1e9).toFixed(1) + 'B'
  if (val >= 1e6) return (val / 1e6).toFixed(1) + 'M'
  if (val >= 1e3) return (val / 1e3).toFixed(0) + 'K'
  return val.toString()
}

function ChartTooltip({ active, payload, range }) {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  if (!d) return null
  return (
    <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-2 text-xs">
      <div className="font-medium text-gray-700 dark:text-gray-200 mb-1">
        {new Date(d.date + 'T00:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
      </div>
      <div className="text-gray-600 dark:text-gray-300">
        Close: <span className="font-mono font-medium">{formatPrice(d.close)}</span>
      </div>
      {d.volume != null && (
        <div className="text-gray-500 dark:text-gray-400">Vol: {formatVol(d.volume)}</div>
      )}
    </div>
  )
}

export default function StockChart({ symbol }) {
  const [history, setHistory] = useState(null)
  const [range, setRange] = useState('1Y')
  const [overlays, setOverlays] = useState(new Set())
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    getSymbolHistory(symbol)
      .then(setHistory)
      .catch(() => setHistory(null))
      .finally(() => setLoading(false))
  }, [symbol])

  const { data, domain } = useMemo(() => {
    if (!history?.length) return { data: [], domain: ['auto', 'auto'] }

    const rangeDays = RANGES.find((r) => r.key === range)?.days || 365
    const cutoff = new Date()
    cutoff.setDate(cutoff.getDate() - rangeDays)
    const cutoffStr = cutoff.toISOString().slice(0, 10)

    const filtered = history.filter((d) => d.date >= cutoffStr)
    if (!filtered.length) return { data: [], domain: ['auto', 'auto'] }

    // Compute Y domain with padding
    let min = Infinity
    let max = -Infinity
    for (const d of filtered) {
      if (d.close != null) {
        if (d.close < min) min = d.close
        if (d.close > max) max = d.close
      }
      if (overlays.has('bb')) {
        if (d.bb_lower != null && d.bb_lower < min) min = d.bb_lower
        if (d.bb_upper != null && d.bb_upper > max) max = d.bb_upper
      }
    }
    const pad = (max - min) * 0.05
    return { data: filtered, domain: [min - pad, max + pad] }
  }, [history, range, overlays])

  function toggleOverlay(key) {
    setOverlays((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  if (loading) {
    return (
      <div className="h-64 flex items-center justify-center text-gray-400 text-sm">
        Loading chart...
      </div>
    )
  }

  if (!history?.length) {
    return (
      <div className="h-32 flex items-center justify-center text-gray-400 text-sm">
        No historical data available. Run generate_history.py.
      </div>
    )
  }

  const isUp = data.length >= 2 && data[data.length - 1].close >= data[0].close
  const areaColor = isUp ? '#22c55e' : '#ef4444'

  return (
    <div className="space-y-3">
      {/* Time range toggles */}
      <div className="flex items-center gap-1">
        {RANGES.map((r) => (
          <button
            key={r.key}
            onClick={() => setRange(r.key)}
            className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
              range === r.key
                ? 'bg-gray-900 dark:bg-white text-white dark:text-gray-900 font-semibold'
                : 'text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
            }`}
          >
            {r.key}
          </button>
        ))}

        <span className="mx-2 w-px h-4 bg-gray-300 dark:bg-gray-600" />

        {/* Overlay toggles */}
        {OVERLAYS.map((o) => (
          <button
            key={o.key}
            onClick={() => toggleOverlay(o.key)}
            className={`px-2 py-1 text-xs rounded-md transition-colors ${
              overlays.has(o.key)
                ? 'font-semibold'
                : 'text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300'
            }`}
            style={overlays.has(o.key) ? { color: o.color } : undefined}
          >
            {o.label}
          </button>
        ))}
      </div>

      {/* Price chart */}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id={`grad-${symbol}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={areaColor} stopOpacity={0.3} />
                <stop offset="100%" stopColor={areaColor} stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="currentColor" opacity={0.08} />
            <XAxis
              dataKey="date"
              tickFormatter={(d) => formatDate(d, RANGES.find((r) => r.key === range)?.days || 365)}
              tick={{ fontSize: 10, fill: '#9ca3af' }}
              axisLine={false}
              tickLine={false}
              minTickGap={40}
            />
            <YAxis
              domain={domain}
              tickFormatter={formatPrice}
              tick={{ fontSize: 10, fill: '#9ca3af' }}
              axisLine={false}
              tickLine={false}
              width={60}
            />
            <Tooltip content={<ChartTooltip range={range} />} />

            {/* Bollinger Bands */}
            {overlays.has('bb') && (
              <>
                <Line
                  dataKey="bb_upper"
                  stroke="#6b7280"
                  strokeWidth={1}
                  strokeDasharray="4 3"
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  dataKey="bb_lower"
                  stroke="#6b7280"
                  strokeWidth={1}
                  strokeDasharray="4 3"
                  dot={false}
                  isAnimationActive={false}
                />
              </>
            )}

            {/* SMA overlays */}
            {overlays.has('sma_20') && (
              <Line dataKey="sma_20" stroke="#f59e0b" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            )}
            {overlays.has('sma_50') && (
              <Line dataKey="sma_50" stroke="#8b5cf6" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            )}
            {overlays.has('sma_200') && (
              <Line dataKey="sma_200" stroke="#ef4444" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            )}

            {/* Price area */}
            <Area
              dataKey="close"
              stroke={areaColor}
              strokeWidth={2}
              fill={`url(#grad-${symbol})`}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Volume chart */}
      <div className="h-16">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 0, right: 5, left: 0, bottom: 0 }}>
            <XAxis dataKey="date" hide />
            <YAxis tickFormatter={formatVol} tick={{ fontSize: 9, fill: '#9ca3af' }} axisLine={false} tickLine={false} width={60} />
            <Bar dataKey="volume" fill="#6b7280" opacity={0.3} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
