// In dev with Vite proxy, '/api' works. For static serving, point at FastAPI directly.
const BASE = window.location.port === '3000' && !window.__VITE__
  ? 'http://localhost:8000/api'
  : '/api'

async function fetchJson(path) {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    throw new Error(`API ${res.status}: ${res.statusText}`)
  }
  return res.json()
}

export function getSectors() {
  return fetchJson('/sectors')
}

export function getSector(sectorId) {
  return fetchJson(`/sector/${sectorId}`)
}

export function getSymbol(symbol) {
  return fetchJson(`/symbol/${symbol}`)
}

export function getSignals() {
  return fetchJson('/signals/today')
}

export function getWatchlist() {
  return fetchJson('/watchlist')
}

export async function setWatchlist(symbols, name = 'Custom') {
  const res = await fetch(`${BASE}/watchlist`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbols, name }),
  })
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`)
  return res.json()
}
