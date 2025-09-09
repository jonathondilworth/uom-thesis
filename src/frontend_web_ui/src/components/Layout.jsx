import React, { useState, useEffect } from 'react';
import { Link, NavLink, Outlet } from 'react-router-dom';
import {
  SunIcon,
  MoonIcon,
  Bars3Icon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import Footer from './Footer';

export default function Layout() {
  
  const systemPrefersDark = () =>
    window.matchMedia('(prefers-color-scheme: dark)').matches;

  const [darkMode, setDarkMode] = useState(() => {
    const stored = localStorage.getItem('theme');
    return stored ? stored === 'dark' : systemPrefersDark();
  });

  useEffect(() => {
    const root = document.documentElement;
    if (darkMode) {
      root.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      root.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [darkMode]);

  const toggleDarkMode = () => setDarkMode((prev) => !prev);

  const [navOpen, setNavOpen] = useState(false);
  const closeNav = () => setNavOpen(false);

  const linkClasses = ({ isActive }) =>
    isActive
      ? 'text-indigo-600 dark:text-indigo-400 font-semibold'
      : 'text-gray-800 dark:text-gray-200 hover:text-indigo-600 dark:hover:text-indigo-400';

  return (
    <div className="min-h-screen flex flex-col bg-white dark:bg-gray-900">
      
      <nav className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
        <Link
          to="/"
          className="text-2xl font-bold text-indigo-600 dark:text-indigo-400"
        >
          Retrieval Demo
        </Link>

        <div className="hidden md:flex items-center space-x-6">
          <NavLink to="/" className={linkClasses} end>
            Home
          </NavLink>
          <NavLink to="/about" className={linkClasses}>
            About
          </NavLink>
          <NavLink to="/search" className={linkClasses}>
            Search
          </NavLink>
          <NavLink to="/demo" className={linkClasses}>
            Demo
          </NavLink>

          <button
            onClick={toggleDarkMode}
            className="ml-3 p-2 rounded focus:outline-none focus:ring-2 focus:ring-indigo-500"
            aria-label="Toggle dark mode"
          >
            {darkMode ? (
              <SunIcon className="h-6 w-6 text-yellow-400" />
            ) : (
              <MoonIcon className="h-6 w-6 text-gray-800 dark:text-gray-200" />
            )}
          </button>
        </div>

        <button
          className="md:hidden p-2 rounded focus:outline-none focus:ring-2 focus:ring-indigo-500"
          onClick={() => setNavOpen(!navOpen)}
          aria-label="Toggle navigation"
        >
          {navOpen ? (
            <XMarkIcon className="h-6 w-6 text-gray-800 dark:text-gray-200" />
          ) : (
            <Bars3Icon className="h-6 w-6 text-gray-800 dark:text-gray-200" />
          )}
        </button>
      </nav>

      {navOpen && (
        <div className="md:hidden border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
          <NavLink to="/" onClick={closeNav} className="block px-4 py-2" end>
            Home
          </NavLink>
          <NavLink to="/about" onClick={closeNav} className="block px-4 py-2">
            About
          </NavLink>
          <NavLink to="/search" className={closeNav}>
            Search
          </NavLink>
          <NavLink to="/demo" onClick={closeNav} className="block px-4 py-2">
            Demo
          </NavLink>
        </div>
      )}

      <main className="flex-1">
        <Outlet />
      </main>

      <Footer />
    </div>
  );
}
