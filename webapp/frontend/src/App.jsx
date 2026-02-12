import { Routes, Route, Outlet } from 'react-router-dom'
import Header from './components/Header'
import SectorGrid from './components/SectorGrid'
import StockGrid from './components/StockGrid'
import StockDetail from './components/StockDetail'
import LoginPage from './components/LoginPage'
import ProtectedRoute from './components/ProtectedRoute'

function ProtectedLayout() {
  return (
    <ProtectedRoute>
      <Header />
      <main>
        <Outlet />
      </main>
    </ProtectedRoute>
  )
}

export default function App() {
  return (
    <div className="min-h-screen">
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route element={<ProtectedLayout />}>
          <Route path="/" element={<SectorGrid />} />
          <Route path="/sector/:sectorId" element={<StockGrid />} />
          <Route path="/sector/:sectorId/:symbol" element={<StockDetail />} />
        </Route>
      </Routes>
    </div>
  )
}
