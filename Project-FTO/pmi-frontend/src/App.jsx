import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Captura from './pages/Captura'
import Dashboard from './pages/Dashboard'
import Administrar from './pages/Administrar'
import DH from './pages/DH'
import Weekly from './pages/Weekly'
import QM from './pages/QM'

export default function App() {
  return (
    <Routes>
      {/* Default → FTO Captura */}
      <Route path="/" element={<Navigate to="/fto/captura" replace />} />

      {/* FTO Section */}
      <Route path="/fto/captura" element={<Layout><Captura /></Layout>} />
      <Route path="/fto/dashboard" element={<Layout><Dashboard /></Layout>} />
      <Route path="/fto/admin" element={<Layout><Administrar /></Layout>} />
      <Route path="/fto/dh" element={<Layout><DH /></Layout>} />

      {/* Weekly Section (placeholder) */}
      <Route path="/weekly" element={<Layout><Weekly /></Layout>} />

      {/* QM Section (placeholder) */}
      <Route path="/qm" element={<Layout><QM /></Layout>} />

      {/* Fallback */}
      <Route path="*" element={<Navigate to="/fto/captura" replace />} />
    </Routes>
  )
}
