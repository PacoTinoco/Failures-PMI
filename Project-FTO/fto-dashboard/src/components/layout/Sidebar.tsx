'use client';

import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { useState } from 'react';

interface User {
  name: string;
  role: 'ADMIN' | 'COORDINATOR' | 'OPERATOR';
}

interface SidebarProps {
  user?: User;
  onLogout?: () => void;
}

const navigationItems = [
  { href: '/dashboard', label: 'Dashboard', icon: '📊' },
  { href: '/cargar-datos', label: 'Cargar Datos', icon: '📤' },
  { href: '/captura-manual', label: 'Captura Manual', icon: '✏️' },
  { href: '/historico', label: 'Histórico', icon: '📈' },
];

const adminItems = [
  { href: '/admin', label: 'Admin', icon: '⚙️' },
];

const categoryColors: Record<string, string> = {
  S: 'bg-blue-500',
  Q: 'bg-green-500',
  D: 'bg-yellow-500',
  C: 'bg-purple-500',
  M: 'bg-red-500',
};

const roleColors: Record<string, string> = {
  ADMIN: 'bg-red-600',
  COORDINATOR: 'bg-blue-600',
  OPERATOR: 'bg-slate-600',
};

export default function Sidebar({ user, onLogout }: SidebarProps) {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(true);

  const visibleItems =
    user?.role === 'ADMIN'
      ? [...navigationItems, ...adminItems]
      : navigationItems;

  return (
    <aside
      className={`${
        isOpen ? 'w-64' : 'w-20'
      } h-screen bg-slate-900 border-r border-slate-800 flex flex-col transition-all duration-300 shadow-lg`}
    >
      {/* Logo Section */}
      <div className="p-6 border-b border-slate-800">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="w-full flex items-center justify-between mb-4"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center font-bold text-white text-lg">
              F
            </div>
            {isOpen && <span className="font-bold text-lg">FTO</span>}
          </div>
          {isOpen && (
            <span className="text-xs text-slate-400 px-2">☰</span>
          )}
        </button>
        {isOpen && (
          <p className="text-xs text-slate-400 ml-13">Dashboard</p>
        )}
      </div>

      {/* Navigation Items */}
      <nav className="flex-1 px-3 py-6 space-y-2 overflow-y-auto">
        {visibleItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                isActive
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'text-slate-300 hover:bg-slate-800'
              }`}
              title={!isOpen ? item.label : undefined}
            >
              <span className="text-xl flex-shrink-0">{item.icon}</span>
              {isOpen && <span className="text-sm font-medium">{item.label}</span>}
            </Link>
          );
        })}
      </nav>

      {/* User Info Section */}
      {user && (
        <div className="border-t border-slate-800 p-4 space-y-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-slate-600 to-slate-700 flex items-center justify-center text-white font-semibold flex-shrink-0">
              {user.name.charAt(0).toUpperCase()}
            </div>
            {isOpen && (
              <div className="min-w-0">
                <p className="text-sm font-medium truncate">{user.name}</p>
                <span
                  className={`inline-block px-2 py-1 rounded text-xs font-semibold text-white ${
                    roleColors[user.role] || roleColors.USER
                  }`}
                >
                  {user.role}
                </span>
              </div>
            )}
          </div>
          {isOpen && (
            <button
              onClick={onLogout}
              className="w-full bg-slate-800 hover:bg-slate-700 text-slate-200 px-3 py-2 rounded text-sm font-medium transition-colors"
            >
              Logout
            </button>
          )}
        </div>
      )}
    </aside>
  );
}
