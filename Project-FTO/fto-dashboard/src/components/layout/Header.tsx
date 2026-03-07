'use client';

import { useState } from 'react';

interface HeaderProps {
  title: string;
  machines?: { id: string; name: string }[];
  selectedMachine?: string;
  onMachineChange?: (machineId: string) => void;
  userName?: string;
}

export default function Header({
  title,
  machines = [],
  selectedMachine = '',
  onMachineChange = () => {},
  userName = 'User',
}: HeaderProps) {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  return (
    <header className="bg-slate-900 border-b border-slate-800 px-8 py-6 shadow-md">
      <div className="flex items-center justify-between gap-6">
        {/* Left: Title */}
        <div className="flex-1">
          <h1 className="text-3xl font-bold text-slate-100">{title}</h1>
          <p className="text-sm text-slate-400 mt-1">
            {new Date().toLocaleDateString('es-ES', {
              weekday: 'long',
              year: 'numeric',
              month: 'long',
              day: 'numeric',
            })}
          </p>
        </div>

        {/* Center: Machine Selector */}
        {machines.length > 0 && (
          <div className="flex items-center gap-3 min-w-0">
            <label className="text-sm text-slate-300 font-medium whitespace-nowrap">
              Machine:
            </label>
            <div className="relative">
              <button
                onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                className="bg-slate-800 hover:bg-slate-700 text-slate-100 px-4 py-2 rounded-lg flex items-center gap-2 border border-slate-700 min-w-0"
              >
                <span className="truncate">
                  {machines.find((m) => m.id === selectedMachine)?.name ||
                    'Select machine'}
                </span>
                <span className={`transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`}>
                  ▼
                </span>
              </button>

              {isDropdownOpen && (
                <div className="absolute top-full mt-2 right-0 bg-slate-800 border border-slate-700 rounded-lg shadow-lg z-10 min-w-48">
                  {machines.map((machine) => (
                    <button
                      key={machine.id}
                      onClick={() => {
                        onMachineChange(machine.id);
                        setIsDropdownOpen(false);
                      }}
                      className={`w-full text-left px-4 py-2 text-sm transition-colors ${
                        selectedMachine === machine.id
                          ? 'bg-blue-600 text-white'
                          : 'text-slate-300 hover:bg-slate-700'
                      }`}
                    >
                      {machine.name}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Right: User Avatar */}
        <div className="flex items-center gap-3">
          <div className="text-right hidden sm:block">
            <p className="text-sm font-medium text-slate-200">{userName}</p>
            <p className="text-xs text-slate-400">Online</p>
          </div>
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-white font-bold text-lg">
            {userName.charAt(0).toUpperCase()}
          </div>
        </div>
      </div>
    </header>
  );
}
