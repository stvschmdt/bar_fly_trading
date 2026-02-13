import { Routes, Route, Outlet } from 'react-router-dom'
import Header from './components/Header'
import SectorGrid from './components/SectorGrid'
import StockGrid from './components/StockGrid'
import StockDetail from './components/StockDetail'
import BigBoard from './components/BigBoard'
import HomePage from './components/HomePage'
import AboutPage from './components/AboutPage'
import ContactPage from './components/ContactPage'
import LoginPage from './components/LoginPage'
import ProtectedRoute from './components/ProtectedRoute'
import WorkbenchBar from './components/WorkbenchBar'

function ProtectedLayout() {
  return (
    <ProtectedRoute>
      <Header />
      <main className="pb-12">
        <Outlet />
      </main>
      <WorkbenchBar />
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
          <Route path="/home" element={<HomePage />} />
          <Route path="/bigboard" element={<BigBoard />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/contact" element={<ContactPage />} />
          <Route path="/sector/:sectorId" element={<StockGrid />} />
          <Route path="/sector/:sectorId/:symbol" element={<StockDetail />} />
        </Route>
      </Routes>
    </div>
  )
}
