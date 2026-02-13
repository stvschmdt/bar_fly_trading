import { Routes, Route, Outlet } from 'react-router-dom'
import Header from './components/Header'
import SectorGrid from './components/SectorGrid'
import StockGrid from './components/StockGrid'
import StockDetail from './components/StockDetail'
import BigBoard from './components/BigBoard'
import OvernightPage from './components/OvernightPage'
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
      <footer className="py-3 px-4 text-center">
        <p className="text-[10px] text-gray-400 dark:text-gray-600 leading-relaxed">
          Disclaimer: This is not financial advice. For research and educational purposes only. Data reflects daily closing values and is not suitable for intraday trading decisions. Past performance does not guarantee future results.
        </p>
      </footer>
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
          <Route path="/overnight" element={<OvernightPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/contact" element={<ContactPage />} />
          <Route path="/sector/:sectorId" element={<StockGrid />} />
          <Route path="/sector/:sectorId/:symbol" element={<StockDetail />} />
        </Route>
      </Routes>
    </div>
  )
}
