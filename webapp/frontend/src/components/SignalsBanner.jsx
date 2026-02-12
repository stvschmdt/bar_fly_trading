export default function SignalsBanner({ signals }) {
  if (!signals || signals.length === 0) return null

  const buys = signals.filter(s => s.type === 'BUY').length
  const sells = signals.filter(s => s.type === 'SELL').length

  return (
    <div className="bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg px-4 py-3 text-sm">
      <span className="font-medium">Signals Today: </span>
      {buys > 0 && <span className="text-green-600 dark:text-green-400 font-semibold">{buys} BUY</span>}
      {buys > 0 && sells > 0 && <span className="mx-1">,</span>}
      {sells > 0 && <span className="text-red-600 dark:text-red-400 font-semibold">{sells} SELL</span>}
      {buys === 0 && sells === 0 && <span className="text-gray-500">None</span>}
    </div>
  )
}
