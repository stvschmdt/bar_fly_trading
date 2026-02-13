export default function HomePage() {
  return (
    <div className="max-w-3xl mx-auto px-4 py-16 text-center space-y-8">
      <div>
        <h1 className="text-4xl font-bold mb-4">Bar Fly Trading & Investing</h1>
        <p className="text-lg text-gray-500 dark:text-gray-400 mb-2">Welcome, beta testers.</p>
      </div>

      <div className="text-left bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-xl p-6 space-y-4">
        <p className="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
          Bar Fly Trading & Investing combines proprietary overnight screening with real-time
          market analysis across 600+ equities and 11 sectors. Our platform leverages technical
          indicators, machine learning predictions, and sector momentum to surface actionable
          trading opportunities — updated nightly and monitored in real time during market hours.
        </p>
        <p className="text-sm leading-relaxed text-gray-700 dark:text-gray-300">
          Each stock is evaluated across 10 independent signal dimensions including RSI, MACD,
          Bollinger Bands, CCI, ADX, and options flow. AI-generated summaries synthesize
          technical, fundamental, and sentiment data into concise daily reports.
        </p>
      </div>

      <footer className="border-t border-gray-200 dark:border-gray-800 pt-6">
        <p className="text-xs text-gray-400 dark:text-gray-500 leading-relaxed">
          For educational and research purposes only. Not financial advice.
          Data reflects daily closing values and is not appropriate for intraday trading decisions.
          Past performance does not guarantee future results.
        </p>
      </footer>
    </div>
  )
}
