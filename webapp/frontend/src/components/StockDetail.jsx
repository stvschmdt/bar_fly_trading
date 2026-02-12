import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getSymbol, getSector } from '../api/client'

function changeColor(pct) {
  if (pct > 0) return 'text-green-600 dark:text-green-400'
  if (pct < 0) return 'text-red-600 dark:text-red-400'
  return 'text-gray-500'
}

function Indicator({ label, value, unit, bullish, bearish }) {
  let color = 'text-gray-600 dark:text-gray-300'
  if (bullish) color = 'text-green-600 dark:text-green-400'
  if (bearish) color = 'text-red-600 dark:text-red-400'

  return (
    <div className="flex justify-between py-1.5 border-b border-gray-100 dark:border-gray-800 last:border-0">
      <span className="text-gray-500 dark:text-gray-400 text-sm">{label}</span>
      <span className={`font-mono text-sm font-medium ${color}`}>
        {value != null ? (typeof value === 'number' ? value.toFixed(2) : value) : '—'}
        {unit && <span className="text-gray-400 ml-0.5">{unit}</span>}
      </span>
    </div>
  )
}

export default function StockDetail() {
  const { sectorId, symbol } = useParams()
  const [report, setReport] = useState(null)
  const [stockInfo, setStockInfo] = useState(null)
  const [flipped, setFlipped] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    getSymbol(symbol)
      .then(setReport)
      .catch(e => setError(e.message))

    // Get stock price info from sector data
    getSector(sectorId)
      .then(d => {
        const s = d.stocks.find(st => st.symbol === symbol.toUpperCase())
        if (s) setStockInfo(s)
      })
      .catch(() => {})
  }, [symbol, sectorId])

  const tech = report?.technical
  const news = report?.news

  return (
    <div className="max-w-2xl mx-auto px-4 py-6 space-y-4">
      {/* Back nav */}
      <div className="flex items-center gap-3">
        <Link
          to={`/sector/${sectorId}`}
          className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </Link>
        <div className="flex-1">
          <h1 className="text-xl font-bold">
            {symbol.toUpperCase()}
            {stockInfo && (
              <span className={`ml-3 text-lg ${changeColor(stockInfo.change_pct)}`}>
                ${stockInfo.price}
                <span className="ml-2 text-sm">
                  {stockInfo.change_pct > 0 ? '+' : ''}{stockInfo.change_pct.toFixed(2)}%
                </span>
              </span>
            )}
          </h1>
        </div>
        <button
          onClick={() => setFlipped(!flipped)}
          className="px-3 py-1.5 text-xs rounded-lg border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        >
          {flipped ? 'Technical' : 'News'}
        </button>
      </div>

      {/* Card with flip */}
      <div className="perspective">
        <div className={`relative preserve-3d transition-transform duration-500 ${flipped ? 'rotate-y-180' : ''}`}
          style={{ minHeight: '400px' }}>

          {/* Front: Technical */}
          <div className="absolute inset-0 backface-hidden">
            <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-xl p-6 space-y-4">
              <h2 className="font-semibold text-base border-b border-gray-200 dark:border-gray-700 pb-2">
                BFT Technical Outlook
                {report?.report_date && (
                  <span className="ml-2 text-xs text-gray-400 font-normal">as of {report.report_date}</span>
                )}
              </h2>

              {error && <p className="text-gray-400 text-sm">No report available. Run generate_reports.py.</p>}

              {tech && (
                <div className="space-y-0">
                  <Indicator label="RSI (14)" value={tech.rsi} bullish={tech.rsi < 30} bearish={tech.rsi > 70} />
                  <Indicator label="MACD" value={tech.macd} bullish={tech.macd > tech.macd_signal} bearish={tech.macd < tech.macd_signal} />
                  <Indicator label="MACD Signal" value={tech.macd_signal} />
                  <Indicator label="ADX" value={tech.adx} bullish={tech.adx > 25} bearish={tech.adx < 20} />
                  <Indicator label="CCI (14)" value={tech.cci} bullish={tech.cci < -100} bearish={tech.cci > 100} />
                  <Indicator label="ATR (14)" value={tech.atr} />
                  <Indicator label="SMA 20" value={tech.sma_20} />
                  <Indicator label="SMA 50" value={tech.sma_50} bullish={tech.sma_20 > tech.sma_50} bearish={tech.sma_20 < tech.sma_50} />
                  <Indicator label="BB Upper" value={tech.bbands_upper} />
                  <Indicator label="BB Lower" value={tech.bbands_lower} />
                  <Indicator label="P/E Ratio" value={tech.pe_ratio} bullish={tech.pe_ratio > 0 && tech.pe_ratio < 15} bearish={tech.pe_ratio > 35} />
                </div>
              )}

              {tech?.summary && (
                <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg text-sm leading-relaxed">
                  {tech.summary}
                </div>
              )}

              {!tech && !error && (
                <p className="text-gray-400 text-sm">Loading technical data...</p>
              )}
            </div>
          </div>

          {/* Back: News */}
          <div className="absolute inset-0 backface-hidden rotate-y-180">
            <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-xl p-6 space-y-4">
              <h2 className="font-semibold text-base border-b border-gray-200 dark:border-gray-700 pb-2">
                BFT AI Current Outlook
              </h2>

              {news ? (
                <div className="space-y-3">
                  {news.summary && (
                    <p className="text-sm leading-relaxed">{news.summary}</p>
                  )}
                  {news.sentiment != null && (
                    <div className="flex items-center gap-2 text-sm">
                      <span className="text-gray-500">Sentiment:</span>
                      <span className={`font-semibold ${
                        news.sentiment > 0.6 ? 'text-green-600 dark:text-green-400' :
                        news.sentiment < 0.4 ? 'text-red-600 dark:text-red-400' :
                        'text-yellow-600 dark:text-yellow-400'
                      }`}>
                        {news.sentiment > 0.6 ? 'Bullish' : news.sentiment < 0.4 ? 'Bearish' : 'Neutral'}
                        ({(news.sentiment * 100).toFixed(0)}%)
                      </span>
                    </div>
                  )}
                  {news.bullets && (
                    <ul className="text-sm space-y-1.5 ml-4">
                      {news.bullets.map((b, i) => (
                        <li key={i} className="list-disc text-gray-600 dark:text-gray-300">{b}</li>
                      ))}
                    </ul>
                  )}
                </div>
              ) : (
                <p className="text-gray-400 text-sm">No news data available.</p>
              )}

              {/* Signal section */}
              {report?.signal && (
                <div className="border-t border-gray-200 dark:border-gray-700 pt-3 mt-3">
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                      report.signal.type === 'BUY'
                        ? 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300'
                        : 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300'
                    }`}>
                      {report.signal.type}
                    </span>
                    <span className="text-sm text-gray-500">{report.signal.strategy}</span>
                  </div>
                  {report.signal.reason && (
                    <p className="text-xs text-gray-500 mt-1">{report.signal.reason}</p>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
