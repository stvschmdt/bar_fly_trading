import { createContext, useContext, useEffect, useState } from 'react'

const AuthContext = createContext()

const BASE = '/api'

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [token, setToken] = useState(() => localStorage.getItem('bft-token') || null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!token) {
      setLoading(false)
      return
    }
    fetch(`${BASE}/auth/me`, {
      headers: { 'Authorization': `Bearer ${token}` },
    })
      .then(res => {
        if (!res.ok) throw new Error('Invalid token')
        return res.json()
      })
      .then(data => setUser({ email: data.email }))
      .catch(() => {
        localStorage.removeItem('bft-token')
        setToken(null)
        setUser(null)
      })
      .finally(() => setLoading(false))
  }, [token])

  function login(newToken, email) {
    localStorage.setItem('bft-token', newToken)
    setToken(newToken)
    setUser({ email })
  }

  function logout() {
    localStorage.removeItem('bft-token')
    setToken(null)
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, token, loading, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}
