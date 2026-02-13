import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getOvernight, getBigBoard } from '../api/client'
import { useWorkbench } from '../hooks/useWorkbench'

const CHART_LABELS = {
  daily_price: 'Price & Technicals',
  daily_volume: 'Volume',
  technical_rsi: 'RSI (14)',
  technical_macd: 'MACD',
  technical_cci: 'CCI (14)',
  technical_off_from_highs: '52-Week Range',
  technical_ttm_pe_ratio: 'P/E Ratio',
}

const CHART_ORDER = [
  'daily_price',
  'daily_volume',
  'technical_rsi',
  'technical_macd',
  'technical_cci',
  'technical_off_from_highs',
  'technical_ttm_pe_ratio',
]

function SignalBadge({ value }) {
  if (value === 1) return <span className="text-green-600 dark:text-green-400 font-bold text-xs">BUY</span>
  if (value === -1) return <span className="text-red-600 dark:text-red-400 font-bold text-xs">SELL</span>
  return <span className="text-gray-400 text-xs">—</span>
}

function DeltaBadge({ delta }) {
  if (delta == null) return null
  const abs = Math.abs(delta)
  let bg = 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'
  if (delta >= 3) bg = 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300'
  else if (delta >= 1) bg = 'bg-green-50 dark:bg-green-950 text-green-600 dark:text-green-400'
  else if (delta <= -3) bg = 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300'
  else if (delta <= -1) bg = 'bg-red-50 dark:bg-red-950 text-red-600 dark:text-red-400'
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-bold ${bg}`}>
      {delta > 0 ? '+' : ''}{delta}
    </span>
  )
}

function CollapsibleSection({ title, defaultOpen = false, count, children }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-5 py-3 bg-gray-50 dark:bg-gray-900 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
      >
        <span className="font-semibold text-sm">
          {title}
          {count != null && <span className="ml-2 text-gray-400 font-normal">({count})</span>}
        </span>
        <svg
          className={`w-4 h-4 text-gray-400 transition-transform ${open ? 'rotate-180' : ''}`}
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {open && <div className="p-4">{children}</div>}
    </div>
  )
}

function StockCard({ symbol, charts, imageBase, signals, sectorMap }) {
  const [expanded, setExpanded] = useState(null)
  const { has, toggle } = useWorkbench()
  const inBench = has(symbol)

  const sig = signals?.[symbol]
  const sectorId = sectorMap?.[symbol] || 'XLK'

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden bg-white dark:bg-gray-900">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 dark:border-gray-800">
        <div className="flex items-center gap-3">
          <Link
            to={`/sector/${sectorId}/${symbol}`}
            className="font-bold text-sm hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
          >
            {symbol}
          </Link>
          {sig && <DeltaBadge delta={sig.bull_bear_delta} />}
          {sig && (
            <div className="hidden sm:flex items-center gap-2 text-xs">
              <span className="text-gray-400">RSI</span><SignalBadge value={sig.rsi_signal} />
              <span className="text-gray-400">MACD</span><SignalBadge value={sig.macd_signal} />
              <span className="text-gray-400">BB</span><SignalBadge value={sig.bollinger_bands_signal} />
              <span className="text-gray-400">CCI</span><SignalBadge value={sig.cci_signal} />
            </div>
          )}
        </div>
        <button
          onClick={() => toggle(symbol)}
          className={`px-2 py-1 text-xs rounded border transition-colors ${
            inBench
              ? 'bg-blue-500 border-blue-500 text-white'
              : 'border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-800'
          }`}
        >
          {inBench ? 'Added' : '+'}
        </button>
      </div>

      {charts.length > 0 ? (
        <div className="divide-y divide-gray-100 dark:divide-gray-800">
          {CHART_ORDER.filter(type => charts.includes(type)).map(type => (
            <div key={type}>
              <button
                onClick={() => setExpanded(expanded === type ? null : type)}
                className="w-full flex items-center justify-between px-4 py-2 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors text-xs"
              >
                <span className="text-gray-600 dark:text-gray-300">{CHART_LABELS[type] || type}</span>
                <svg
                  className={`w-3 h-3 text-gray-400 transition-transform ${expanded === type ? 'rotate-180' : ''}`}
                  fill="none" stroke="currentColor" viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {expanded === type && (
                <div className="px-2 pb-3">
                  <img
                    src={`${imageBase}/${symbol}_${type}.jpg`}
                    alt={`${symbol} ${CHART_LABELS[type]}`}
                    className="w-full rounded-lg"
                    loading="lazy"
                  />
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="px-4 py-3">
          {sig && (
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
              <span className="text-gray-400">SMA Cross</span><SignalBadge value={sig.sma_cross_signal} />
              <span className="text-gray-400">Bollinger</span><SignalBadge value={sig.bollinger_bands_signal} />
              <span className="text-gray-400">RSI</span><SignalBadge value={sig.rsi_signal} />
              <span className="text-gray-400">MACD</span><SignalBadge value={sig.macd_signal} />
              <span className="text-gray-400">MACD Zero</span><SignalBadge value={sig.macd_zero_signal} />
              <span className="text-gray-400">ADX</span><SignalBadge value={sig.adx_signal} />
              <span className="text-gray-400">CCI</span><SignalBadge value={sig.cci_signal} />
              <span className="text-gray-400">ATR</span><SignalBadge value={sig.atr_signal} />
              <span className="text-gray-400">P/E</span><SignalBadge value={sig.pe_ratio_signal} />
              <span className="text-gray-400">PCR</span><SignalBadge value={sig.pcr_signal} />
            </div>
          )}
          <Link
            to={`/sector/${sectorId}/${symbol}`}
            className="mt-2 inline-block text-xs text-blue-500 hover:text-blue-600 transition-colors"
          >
            View full detail &rarr;
          </Link>
        </div>
      )}
    </div>
  )
}

export default function OvernightPage() {
  const [data, setData] = useState(null)
  const [boardData, setBoardData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [filter, setFilter] = useState('all') // all, bullish, bearish

  useEffect(() => {
    Promise.all([
      getOvernight().catch(() => null),
      getBigBoard().catch(() => null),
    ]).then(([overnight, board]) => {
      setData(overnight)
      setBoardData(board)
      setLoading(false)
    }).catch(e => {
      setError(e.message)
      setLoading(false)
    })
  }, [])

  // Build sector lookup from bigboard
  const sectorMap = {}
  if (boardData?.stocks) {
    for (const s of boardData.stocks) {
      sectorMap[s.symbol] = s.sector_id
    }
  }

  // Filter stocks by signal direction
  const signals = data?.signals || {}
  const stocks = data?.stocks || []
  const filteredStocks = stocks.filter(sym => {
    if (filter === 'all') return true
    const sig = signals[sym]
    if (!sig) return false
    if (filter === 'bullish') return sig.bull_bear_delta > 0
    if (filter === 'bearish') return sig.bull_bear_delta < 0
    return true
  })

  // Sort by |delta| descending
  filteredStocks.sort((a, b) => {
    const da = Math.abs(signals[a]?.bull_bear_delta || 0)
    const db = Math.abs(signals[b]?.bull_bear_delta || 0)
    return db - da
  })

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-16 text-center">
        <p className="text-gray-400">Loading overnight data...</p>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-16 text-center">
        <h1 className="text-2xl font-bold mb-4">BFT Overnight</h1>
        <p className="text-gray-500 dark:text-gray-400">
          No overnight screener data available. The screener runs nightly after market close.
        </p>
      </div>
    )
  }

  const imageBase = '/api/overnight/image'
  const bullCount = Object.values(signals).filter(s => s.bull_bear_delta > 0).length
  const bearCount = Object.values(signals).filter(s => s.bull_bear_delta < 0).length

  return (
    <div className="max-w-6xl mx-auto px-4 py-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">BFT Overnight</h1>
          {data.date && (
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Screener date: {data.date}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2 text-sm">
          <span className="text-green-600 dark:text-green-400 font-medium">{bullCount} bullish</span>
          <span className="text-gray-300 dark:text-gray-600">|</span>
          <span className="text-red-600 dark:text-red-400 font-medium">{bearCount} bearish</span>
        </div>
      </div>

      {/* Heatmap Summary Table */}
      <CollapsibleSection title="Signal Heatmap" defaultOpen={true} count={stocks.length}>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="text-left py-2 px-2 font-medium text-gray-500">Symbol</th>
                <th className="text-center py-2 px-1 font-medium text-gray-500">Delta</th>
                <th className="text-center py-2 px-1 font-medium text-gray-500">SMA</th>
                <th className="text-center py-2 px-1 font-medium text-gray-500">BB</th>
                <th className="text-center py-2 px-1 font-medium text-gray-500">RSI</th>
                <th className="text-center py-2 px-1 font-medium text-gray-500">MACD</th>
                <th className="text-center py-2 px-1 font-medium text-gray-500">ADX</th>
                <th className="text-center py-2 px-1 font-medium text-gray-500">CCI</th>
                <th className="text-center py-2 px-1 font-medium text-gray-500">ATR</th>
                <th className="text-center py-2 px-1 font-medium text-gray-500">P/E</th>
                <th className="text-center py-2 px-1 font-medium text-gray-500">PCR</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
              {[...stocks]
                .sort((a, b) => (signals[b]?.bull_bear_delta || 0) - (signals[a]?.bull_bear_delta || 0))
                .map(sym => {
                  const s = signals[sym]
                  if (!s) return null
                  return (
                    <tr key={sym} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                      <td className="py-1.5 px-2 font-medium">
                        <Link
                          to={`/sector/${sectorMap[sym] || 'XLK'}/${sym}`}
                          className="hover:text-blue-500 transition-colors"
                        >
                          {sym}
                        </Link>
                      </td>
                      <td className="text-center py-1.5 px-1"><DeltaBadge delta={s.bull_bear_delta} /></td>
                      <td className="text-center py-1.5 px-1"><SignalBadge value={s.sma_cross_signal} /></td>
                      <td className="text-center py-1.5 px-1"><SignalBadge value={s.bollinger_bands_signal} /></td>
                      <td className="text-center py-1.5 px-1"><SignalBadge value={s.rsi_signal} /></td>
                      <td className="text-center py-1.5 px-1"><SignalBadge value={s.macd_signal} /></td>
                      <td className="text-center py-1.5 px-1"><SignalBadge value={s.adx_signal} /></td>
                      <td className="text-center py-1.5 px-1"><SignalBadge value={s.cci_signal} /></td>
                      <td className="text-center py-1.5 px-1"><SignalBadge value={s.atr_signal} /></td>
                      <td className="text-center py-1.5 px-1"><SignalBadge value={s.pe_ratio_signal} /></td>
                      <td className="text-center py-1.5 px-1"><SignalBadge value={s.pcr_signal} /></td>
                    </tr>
                  )
                })}
            </tbody>
          </table>
        </div>
      </CollapsibleSection>

      {/* Filter buttons */}
      <div className="flex items-center gap-2">
        {['all', 'bullish', 'bearish'].map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1.5 text-xs rounded-lg border transition-colors capitalize ${
              filter === f
                ? 'bg-gray-900 dark:bg-white text-white dark:text-gray-900 border-transparent'
                : 'border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-800'
            }`}
          >
            {f} {f === 'bullish' ? `(${bullCount})` : f === 'bearish' ? `(${bearCount})` : ''}
          </button>
        ))}
      </div>

      {/* Stock Cards */}
      <CollapsibleSection title="Stock Analysis" defaultOpen={true} count={filteredStocks.length}>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredStocks.map(sym => (
            <StockCard
              key={sym}
              symbol={sym}
              charts={data.stock_charts?.[sym] || []}
              imageBase={imageBase}
              signals={signals}
              sectorMap={sectorMap}
            />
          ))}
        </div>
      </CollapsibleSection>

      {/* Sector Analysis */}
      {data.sectors?.length > 0 && (
        <CollapsibleSection title="Sector Analysis" count={data.sectors.length}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {data.sectors.map(s => (
              <div key={s.id} className="border border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden">
                <div className="px-4 py-2 bg-gray-50 dark:bg-gray-900 border-b border-gray-100 dark:border-gray-800">
                  <span className="font-medium text-sm">{s.name}</span>
                  <span className="ml-2 text-xs text-gray-400">{s.id}</span>
                </div>
                <img
                  src={`${imageBase}/${s.filename}`}
                  alt={`${s.name} sector analysis`}
                  className="w-full"
                  loading="lazy"
                />
              </div>
            ))}
          </div>
        </CollapsibleSection>
      )}

      {/* Market Returns */}
      {data.has_market_returns && (
        <CollapsibleSection title="Market Returns Comparison" defaultOpen={true}>
          <img
            src={`${imageBase}/market_returns.jpg`}
            alt="Market sector returns comparison"
            className="w-full rounded-lg"
            loading="lazy"
          />
        </CollapsibleSection>
      )}

      {/* Disclaimer Footer */}
      <footer className="border-t border-gray-200 dark:border-gray-800 pt-4 mt-8">
        <p className="text-xs text-gray-400 dark:text-gray-500 text-center leading-relaxed">
          For educational and research purposes only. Not financial advice.
          Data reflects daily closing values and is not appropriate for intraday trading decisions.
          Past performance does not guarantee future results.
        </p>
      </footer>
    </div>
  )
}
