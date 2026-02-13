import { createContext, useContext, useState, useCallback } from 'react'

const WorkbenchContext = createContext()

const STORAGE_KEY = 'bft-workbench'

/** Compute next Friday 7pm EST from now. */
function nextFridayExpiry() {
  const now = new Date()
  // Work in UTC: EST = UTC-5, so 7pm EST = 00:00 UTC Saturday
  const day = now.getUTCDay() // 0=Sun
  // Days until next Saturday 00:00 UTC (= Friday 7pm EST)
  let daysUntil = (6 - day + 7) % 7
  if (daysUntil === 0) {
    // It's Saturday UTC — check if we're past the threshold
    const satMidnightUTC = new Date(now)
    satMidnightUTC.setUTCHours(0, 0, 0, 0)
    if (now >= satMidnightUTC) daysUntil = 7
  }
  const expiry = new Date(now)
  expiry.setUTCDate(expiry.getUTCDate() + daysUntil)
  expiry.setUTCHours(0, 0, 0, 0) // Saturday 00:00 UTC = Friday 7pm EST
  return expiry.toISOString()
}

function loadFromStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const data = JSON.parse(raw)
    if (data.expires && new Date(data.expires) < new Date()) {
      localStorage.removeItem(STORAGE_KEY)
      return []
    }
    return data.symbols || []
  } catch {
    return []
  }
}

function saveToStorage(symbols) {
  if (symbols.length === 0) {
    localStorage.removeItem(STORAGE_KEY)
    return
  }
  // Read existing expiry or compute new one
  let expires
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) expires = JSON.parse(raw).expires
  } catch {}
  if (!expires) expires = nextFridayExpiry()

  localStorage.setItem(STORAGE_KEY, JSON.stringify({ symbols, expires }))
}

export function WorkbenchProvider({ children }) {
  const [symbols, setSymbols] = useState(loadFromStorage)

  const add = useCallback((sym) => {
    setSymbols((prev) => {
      if (prev.includes(sym)) return prev
      const next = [...prev, sym]
      saveToStorage(next)
      return next
    })
  }, [])

  const remove = useCallback((sym) => {
    setSymbols((prev) => {
      const next = prev.filter((s) => s !== sym)
      saveToStorage(next)
      return next
    })
  }, [])

  const clear = useCallback(() => {
    setSymbols([])
    localStorage.removeItem(STORAGE_KEY)
  }, [])

  const has = useCallback((sym) => symbols.includes(sym), [symbols])

  const toggle = useCallback((sym) => {
    setSymbols((prev) => {
      const next = prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym]
      saveToStorage(next)
      return next
    })
  }, [])

  return (
    <WorkbenchContext.Provider value={{ symbols, add, remove, clear, has, toggle }}>
      {children}
    </WorkbenchContext.Provider>
  )
}

export function useWorkbench() {
  return useContext(WorkbenchContext)
}
