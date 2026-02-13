import { Link } from 'react-router-dom'

function signalBg(type, intensity = 0.5) {
  if (type === 'BUY') return `rgba(34, 197, 94, ${0.08 + intensity * 0.12})`
  if (type === 'SELL') return `rgba(239, 68, 68, ${0.08 + intensity * 0.12})`
  return undefined
}

export default function SignalsBanner({ signals }) {
  if (!signals || signals.length === 0) return null

  const buys = signals.filter(s => s.type === 'BUY')
  const sells = signals.filter(s => s.type === 'SELL')

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <h2 className="text-sm font-bold tracking-wide text-gray-900 dark:text-gray-100">
          BFT Proprietary Signals
        </h2>
        <div className="flex gap-2 text-xs font-semibold">
          {buys.length > 0 && (
            <span className="text-green-600 dark:text-green-400">{buys.length} BUY</span>
          )}
          {sells.length > 0 && (
            <span className="text-red-600 dark:text-red-400">{sells.length} SELL</span>
          )}
          {buys.length === 0 && sells.length === 0 && (
            <span className="text-gray-400">No signals today</span>
          )}
        </div>
      </div>

      {(buys.length > 0 || sells.length > 0) && (
        <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-10 gap-2">
          {buys.map((s, i) => (
            <Link
              key={`buy-${s.symbol}`}
              to={`/sector/${s.sector || 'CUSTOM'}/${s.symbol}`}
              className="rounded-lg px-2 py-1.5 text-center transition-all hover:scale-105"
              style={{ backgroundColor: signalBg('BUY', Math.max(0.3, 1 - i * 0.1)) }}
            >
              <div className="text-xs font-bold text-green-700 dark:text-green-300">{s.symbol}</div>
              <div className="text-[10px] text-green-600 dark:text-green-400 font-medium">BUY</div>
              {s.strategy && (
                <div className="text-[9px] text-green-500/70 dark:text-green-400/50 truncate">{s.strategy}</div>
              )}
            </Link>
          ))}
          {sells.map((s, i) => (
            <Link
              key={`sell-${s.symbol}`}
              to={`/sector/${s.sector || 'CUSTOM'}/${s.symbol}`}
              className="rounded-lg px-2 py-1.5 text-center transition-all hover:scale-105"
              style={{ backgroundColor: signalBg('SELL', Math.max(0.3, 1 - i * 0.1)) }}
            >
              <div className="text-xs font-bold text-red-700 dark:text-red-300">{s.symbol}</div>
              <div className="text-[10px] text-red-600 dark:text-red-400 font-medium">SELL</div>
              {s.strategy && (
                <div className="text-[9px] text-red-500/70 dark:text-red-400/50 truncate">{s.strategy}</div>
              )}
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
