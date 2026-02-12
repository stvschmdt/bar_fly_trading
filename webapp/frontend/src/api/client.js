const BASE = '/api'

function authHeaders() {
  const headers = {}
  const token = localStorage.getItem('bft-token')
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  return headers
}

function handle401(res) {
  if (res.status === 401) {
    localStorage.removeItem('bft-token')
    window.location.href = '/login'
    throw new Error('Session expired')
  }
}

async function fetchJson(path) {
  const res = await fetch(`${BASE}${path}`, { headers: authHeaders() })
  handle401(res)
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
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ symbols, name }),
  })
  handle401(res)
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`)
  return res.json()
}
