import { Routes, Route } from 'react-router-dom'
import Header from './components/Header'
import SectorGrid from './components/SectorGrid'
import StockGrid from './components/StockGrid'
import StockDetail from './components/StockDetail'

export default function App() {
  return (
    <div className="min-h-screen">
      <Header />
      <main>
        <Routes>
          <Route path="/" element={<SectorGrid />} />
          <Route path="/sector/:sectorId" element={<StockGrid />} />
          <Route path="/sector/:sectorId/:symbol" element={<StockDetail />} />
        </Routes>
      </main>
    </div>
  )
}
