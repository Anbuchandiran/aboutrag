import React from "react";
import { Routes, Route, NavLink } from "react-router-dom";
import Dashboard from "./pages/Dashboard.jsx";
import Register from "./pages/Register.jsx";
import Prescription from "./pages/Prescription.jsx";
import History from "./pages/History.jsx";

// Simple icon components (you can replace with your preferred icon library)
const Icons = {
  Dashboard: () => <span>📊</span>,
  Register: () => <span>📝</span>,
  Prescription: () => <span>💊</span>,
  History: () => <span>📋</span>
};

function SidebarLink({ to, label, icon: Icon }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `navLink ${isActive ? "navLinkActive" : ""}`
      }
      end
    >
      <Icon />
      {label}
    </NavLink>
  );
}

export default function App() {
  return (
    <div className="appShell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brandLogo">⚕️</div>
          <div>
            <div className="brandTitle">MEDX CONTROL</div>
            <div className="brandSub">Clinical Intelligence Suite</div>
          </div>
        </div>

        <div className="sidebarLabel">Navigation</div>
        <nav className="nav">
          <SidebarLink to="/" label="Dashboard" icon={Icons.Dashboard} />
          <SidebarLink to="/register" label="Register" icon={Icons.Register} />
          <SidebarLink to="/prescription" label="Prescription" icon={Icons.Prescription} />
          <SidebarLink to="/history" label="History" icon={Icons.History} />
        </nav>

        <div className="sidebarFooter">
          <div className="tinyMuted">⚡ Platform: FastAPI + Mongo + Chroma</div>
          <div className="tinyMuted">🎨 Interface: React + Vite</div>
          <div className="tinyMuted" style={{ marginTop: '8px' }}>🔒 HIPAA Compliant</div>
        </div>
      </aside>

      <main className="main">
        <div className="mainInner">
          <header className="topbar">
            <div>
              <div className="topbarTitle">✨ MedX Clinical Platform</div>
              <div className="topbarMeta">Enterprise Medication Safety and Interaction Validation Workspace</div>
            </div>
            <span className="badge">
              <span style={{ marginRight: '6px' }}>🛡️</span>
              Secure Session
            </span>
          </header>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/register" element={<Register />} />
            <Route path="/prescription" element={<Prescription />} />
            <Route path="/history" element={<History />} />
          </Routes>
        </div>
      </main>
    </div>
  );
}