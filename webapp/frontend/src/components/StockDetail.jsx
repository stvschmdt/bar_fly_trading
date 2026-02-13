import { useEffect, useState } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { getSymbol, getSector } from '../api/client'
import StockChart from './StockChart'
import { isStale } from '../utils/freshness'
import { useWorkbench } from '../hooks/useWorkbench'

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
  const [activePanel, setActivePanel] = useState('technical')
  const [error, setError] = useState(null)
  const navigate = useNavigate()
  const { has, toggle } = useWorkbench()
  const inBench = has(symbol.toUpperCase())

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
    <div className="max-w-4xl mx-auto px-4 py-6 space-y-4">
      {/* Back nav */}
      <div className="flex items-center gap-3">
        <button
          onClick={() => navigate(-1)}
          className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </button>
        <div className="flex-1">
          <h1 className="text-xl font-bold">
            {symbol.toUpperCase()}
            {stockInfo && (
              <span className={`ml-3 text-lg ${isStale(stockInfo.last_updated) ? 'text-gray-400 dark:text-gray-500' : changeColor(stockInfo.change_pct)}`}>
                ${stockInfo.price}
                <span className="ml-2 text-sm">
                  {stockInfo.change_pct > 0 ? '+' : ''}{stockInfo.change_pct.toFixed(2)}%
                </span>
              </span>
            )}
          </h1>
          {stockInfo && isStale(stockInfo.last_updated) && (
            <p className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">Data may be outdated</p>
          )}
        </div>
        <button
          onClick={() => toggle(symbol.toUpperCase())}
          className={`px-3 py-1.5 text-xs rounded-lg border transition-colors ${
            inBench
              ? 'bg-blue-500 border-blue-500 text-white'
              : 'border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-800'
          }`}
        >
          {inBench ? 'In Workbench' : '+ Workbench'}
        </button>
      </div>

      {/* Price chart */}
      <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-xl p-4">
        <StockChart symbol={symbol} />
      </div>

      {/* Three panels side by side */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

        {/* Technical panel */}
        <div
          onClick={() => setActivePanel('technical')}
          className={`bg-white dark:bg-gray-900 border-2 rounded-xl p-5 space-y-3 cursor-pointer transition-colors ${
            activePanel === 'technical'
              ? 'border-blue-500 dark:border-blue-400'
              : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
          }`}
        >
          <h2 className={`font-semibold text-sm pb-2 border-b ${
            activePanel === 'technical'
              ? 'text-blue-600 dark:text-blue-400 border-blue-200 dark:border-blue-800'
              : 'border-gray-200 dark:border-gray-700'
          }`}>
            Technical
            {report?.report_date && (
              <span className="ml-1 text-xs text-gray-400 font-normal">{report.report_date}</span>
            )}
          </h2>

          {error && <p className="text-gray-400 text-xs">No report available.</p>}

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

          {!tech && !error && (
            <p className="text-gray-400 text-xs">Loading...</p>
          )}
        </div>

        {/* News / AI Outlook panel */}
        <div
          onClick={() => setActivePanel('news')}
          className={`bg-white dark:bg-gray-900 border-2 rounded-xl p-5 space-y-3 cursor-pointer transition-colors ${
            activePanel === 'news'
              ? 'border-blue-500 dark:border-blue-400'
              : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
          }`}
        >
          <h2 className={`font-semibold text-sm pb-2 border-b ${
            activePanel === 'news'
              ? 'text-blue-600 dark:text-blue-400 border-blue-200 dark:border-blue-800'
              : 'border-gray-200 dark:border-gray-700'
          }`}>
            News & Sentiment
          </h2>

          {news ? (
            <div className="space-y-3">
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
                <ul className="text-xs space-y-1.5 ml-4">
                  {news.bullets.map((b, i) => (
                    <li key={i} className="list-disc text-gray-600 dark:text-gray-300">{b}</li>
                  ))}
                </ul>
              )}
              {news.summary && (
                <p className="text-xs leading-relaxed text-gray-600 dark:text-gray-300">{news.summary}</p>
              )}
            </div>
          ) : (
            <p className="text-gray-400 text-xs">No news data available.</p>
          )}

          {/* Signal */}
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
                <span className="text-xs text-gray-500">{report.signal.strategy}</span>
              </div>
              {report.signal.reason && (
                <p className="text-xs text-gray-500 mt-1">{report.signal.reason}</p>
              )}
            </div>
          )}
        </div>

        {/* AI Summary panel */}
        <div
          onClick={() => setActivePanel('summary')}
          className={`bg-white dark:bg-gray-900 border-2 rounded-xl p-5 space-y-3 cursor-pointer transition-colors ${
            activePanel === 'summary'
              ? 'border-blue-500 dark:border-blue-400'
              : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
          }`}
        >
          <h2 className={`font-semibold text-sm pb-2 border-b ${
            activePanel === 'summary'
              ? 'text-blue-600 dark:text-blue-400 border-blue-200 dark:border-blue-800'
              : 'border-gray-200 dark:border-gray-700'
          }`}>
            BFT AI Summary
          </h2>

          {tech?.summary ? (
            <div className="text-xs leading-relaxed text-gray-700 dark:text-gray-300">
              {typeof tech.summary === 'string' ? (
                <p>{tech.summary}</p>
              ) : tech.summary.bullets ? (
                <ul className="space-y-2 ml-4">
                  {tech.summary.bullets.map((b, i) => (
                    <li key={i} className="list-disc">{b}</li>
                  ))}
                </ul>
              ) : (
                <p>{JSON.stringify(tech.summary)}</p>
              )}
            </div>
          ) : (
            <p className="text-gray-400 text-xs">
              {error ? 'No AI summary available.' : 'No AI summary yet. Requires ollama.'}
            </p>
          )}

          {/* Earnings data if available */}
          {report?.earnings && (
            <div className="border-t border-gray-200 dark:border-gray-700 pt-3 mt-3 space-y-1">
              <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400">Earnings</h3>
              {report.earnings.eps_estimate && (
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">EPS Est.</span>
                  <span className="font-mono">{report.earnings.eps_estimate}</span>
                </div>
              )}
              {report.earnings.eps_actual && (
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">EPS Actual</span>
                  <span className="font-mono">{report.earnings.eps_actual}</span>
                </div>
              )}
              {report.earnings.next_date && (
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Next</span>
                  <span>{report.earnings.next_date}</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
