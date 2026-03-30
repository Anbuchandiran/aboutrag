import React from "react";
import { NavLink, Route, Routes, useLocation } from "react-router-dom";
import Dashboard from "./pages/Dashboard.jsx";
import Register from "./pages/Register.jsx";
import Prescription from "./pages/Prescription.jsx";
import History from "./pages/History.jsx";

function DashboardIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 13.5h6.5V20H4v-6.5Zm9.5-9.5H20v16h-6.5V4ZM4 4h6.5v6.5H4V4Zm9.5 9.5H20v4H13.5v-4Z" />
    </svg>
  );
}

function RegisterIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M6 4h9l5 5v11a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Zm8 1.5V10h4.5L14 5.5ZM8 13h8v1.5H8V13Zm0 3.5h8V18H8v-1.5Zm0-7h3v1.5H8V9.5Z" />
    </svg>
  );
}

function PrescriptionIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M15.8 3.8a4.5 4.5 0 0 1 0 6.4l-1.6 1.6 5 5a2.8 2.8 0 1 1-4 4l-5-5-1.6 1.6a4.5 4.5 0 1 1-6.4-6.4l7.2-7.2a4.5 4.5 0 0 1 6.4 0Zm-8.8 8.8a1 1 0 0 0 1.4 1.4l6.3-6.3A1 1 0 0 0 13.3 6l-6.3 6.6Z" />
    </svg>
  );
}

function HistoryIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 5a7 7 0 1 1-6.7 9H3l3.2 3.2.1.1L9.6 14H7.1A5 5 0 1 0 12 7v4l3 1.8-.9 1.5L10.5 12V5h1.5Z" />
    </svg>
  );
}

const links = [
  { to: "/", label: "Dashboard", icon: DashboardIcon },
  { to: "/register", label: "Register", icon: RegisterIcon },
  { to: "/prescription", label: "Prescription", icon: PrescriptionIcon },
  { to: "/history", label: "History", icon: HistoryIcon },
];

function SidebarLink({ to, label, icon: Icon }) {
  return (
    <NavLink
      to={to}
      end
      className={({ isActive }) => `navLink ${isActive ? "navLinkActive" : ""}`}
    >
      <span className="navIcon">
        <Icon />
      </span>
      <span className="navTextWrap">
        <span className="navLabel">{label}</span>
      </span>
    </NavLink>
  );
}

export default function App() {
  const location = useLocation();

  return (
    <div className="appShell">
      <div className="ambientGlow ambientGlowLeft" />
      <div className="ambientGlow ambientGlowRight" />

      <aside className="sidebar">
        <div className="brandPanel">
          <div className="brand">
            <div className="brandLogo">M</div>
            <div>
              <div className="brandTitle">MedX Control</div>
              <div className="brandSub">Clinical intelligence cockpit</div>
            </div>
          </div>

          <div className="brandCopy">
            Faster medication review, cleaner patient memory, and a calmer workflow for every consultation.
          </div>
        </div>

        <div className="sidebarLabel">Workspace</div>
        <nav className="nav">
          {links.map((link) => (
            <SidebarLink key={link.to} {...link} />
          ))}
        </nav>

        <div className="sidebarFooter">
          <div className="tinyMuted">Realtime validation, OCR, voice input, and longitudinal history.</div>
          <div className="sidebarMetaRow">
            <span className="miniPill">FastAPI</span>
            <span className="miniPill">MongoDB</span>
            <span className="miniPill">React</span>
          </div>
        </div>
      </aside>

      <main className="main">
        <div className="mainInner">
          <header className="topbar">
            <div>
              <div className="eyebrow">Medication Safety Platform</div>
              <div className="topbarTitle">A sharper frontend for clinical review</div>
              <div className="topbarMeta">
                Premium navigation, smoother transitions, and a more reliable case history view.
              </div>
            </div>

            <div className="topbarActions">
              <span className="badge badgeGhost">Protected Session</span>
              <span className="badge badgeSafe">System Ready</span>
            </div>
          </header>

          <div key={location.pathname} className="pageTransition">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/register" element={<Register />} />
              <Route path="/prescription" element={<Prescription />} />
              <Route path="/history" element={<History />} />
            </Routes>
          </div>
        </div>
      </main>
    </div>
  );
}
