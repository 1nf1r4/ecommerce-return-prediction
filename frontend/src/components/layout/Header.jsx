import { Link, useNavigate, useLocation } from 'react-router-dom'
import { useState } from 'react'

const Header = ({ user, onLogout }) => {
  const navigate = useNavigate()
  const location = useLocation()
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  const handleLogout = () => {
    onLogout()
    navigate('/')
    setIsMobileMenuOpen(false)
  }

  const isActive = (path) => {
    return location.pathname === path 
      ? 'flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-yellow-400 to-orange-400 text-purple-900 font-semibold shadow-lg transform scale-105 transition-all duration-200' 
      : 'flex items-center gap-2 px-4 py-2 rounded-lg text-white hover:bg-gradient-to-r hover:from-blue-500 hover:to-indigo-500 hover:text-white hover:shadow-md transition-all duration-200'
  }

  const isMobileActive = (path) => {
    return location.pathname === path 
      ? 'flex items-center gap-3 px-4 py-3 bg-gradient-to-r from-indigo-50 to-purple-50 text-indigo-700 font-semibold border-r-4 border-indigo-500 transition-colors' 
      : 'flex items-center gap-3 px-4 py-3 text-gray-700 hover:bg-gray-50 transition-colors'
  }

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen)
  }

  return (
    <header className="bg-gradient-to-r from-indigo-600 to-purple-600 shadow-xl sticky top-0 z-50 backdrop-blur-md">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 text-white font-bold text-xl hover:-translate-y-0.5 transition-transform duration-300">
            <div className="transform hover:rotate-12 transition-transform duration-300">
              <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                <rect width="32" height="32" rx="8" fill="url(#gradient)" />
                <path d="M8 12L16 8L24 12V20C24 22.2091 22.2091 24 20 24H12C9.79086 24 8 22.2091 8 20V12Z" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M16 8V16" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <defs>
                  <linearGradient id="gradient" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
                    <stop stopColor="#667eea" />
                    <stop offset="1" stopColor="#764ba2" />
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <span className="bg-gradient-to-r from-white to-gray-100 bg-clip-text text-transparent">ReturnPredict</span>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-2">
            {user ? (
              <>
                <Link to="/dashboard" className={isActive('/dashboard')}>
                  <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <rect x="3" y="3" width="7" height="7" />
                    <rect x="14" y="3" width="7" height="7" />
                    <rect x="14" y="14" width="7" height="7" />
                    <rect x="3" y="14" width="7" height="7" />
                  </svg>
                  <span>Dashboard</span>
                </Link>
                <Link to="/add-data" className={isActive('/add-data')}>
                  <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <line x1="12" y1="5" x2="12" y2="19" />
                    <line x1="5" y1="12" x2="19" y2="12" />
                  </svg>
                  <span>Add Data</span>
                </Link>
                <Link to="/history" className={isActive('/history')}>
                  <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path d="M3 3v5h5M3.05 13a9 9 0 1 0 1.5-5.5" />
                  </svg>
                  <span>History</span>
                </Link>
                <Link to="/reports" className={isActive('/reports')}>
                  <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <line x1="18" y1="20" x2="18" y2="10" />
                    <line x1="12" y1="20" x2="12" y2="4" />
                    <line x1="6" y1="20" x2="6" y2="14" />
                  </svg>
                  <span>Reports</span>
                </Link>
                <Link to="/profile" className={isActive('/profile')}>
                  <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                    <circle cx="12" cy="7" r="4" />
                  </svg>
                  <span>Profile</span>
                </Link>
                <Link to="/settings" className={isActive('/settings')}>
                  <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <circle cx="12" cy="12" r="3"/>
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1 -1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
                  </svg>
                  <span>Settings</span>
                </Link>
              </>
            ) : (
              <>
                <Link to="/" className={isActive('/')}>
                  <span>Home</span>
                </Link>
                <Link to="/signin" className={isActive('/signin')}>
                  <span>Login</span>
                </Link>
                <Link to="/signup" className={isActive('/signup')}>
                  <span>Sign Up</span>
                </Link>
              </>
            )}
          </nav>

          {/* User Menu */}
          {user && (
            <div className="relative group">
              <div className="w-10 h-10 bg-white bg-opacity-20 rounded-full flex items-center justify-center text-white font-medium cursor-pointer hover:bg-opacity-30 transition-all duration-200">
                <span>{user.name ? user.name.charAt(0).toUpperCase() : 'U'}</span>
              </div>
              <div className="absolute top-12 right-0 w-64 bg-white rounded-lg shadow-lg border border-gray-200 py-2 z-50 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200">
                <div className="px-4 py-3 border-b border-gray-100">
                  <p className="text-sm font-medium text-gray-900">{user.name || 'User'}</p>
                  <p className="text-xs text-gray-500">{user.email}</p>
                </div>
                <Link to="/profile" className="flex items-center gap-3 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors">
                  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                    <circle cx="12" cy="7" r="4" />
                  </svg>
                  Profile & Settings
                </Link>
                <button onClick={handleLogout} className="flex items-center gap-3 px-4 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors w-full text-left">
                  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                    <polyline points="16,17 21,12 16,7" />
                    <line x1="21" y1="12" x2="9" y2="12" />
                  </svg>
                  Logout
                </button>
              </div>
            </div>
          )}

          {/* Mobile Menu Button */}
          <button 
            className="md:hidden p-2 rounded-lg text-white hover:bg-white hover:bg-opacity-10 transition-colors" 
            onClick={toggleMobileMenu}
          >
            <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              {isMobileMenuOpen ? (
                <path d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path d="M3 12h18M3 6h18M3 18h18" />
              )}
            </svg>
          </button>
        </div>
      </div>

        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <div className="md:hidden bg-white border-t border-gray-200 shadow-lg">
            {user ? (
              <>
                <div className="px-4 py-4 border-b border-gray-100">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center text-white text-sm font-medium">
                      <span>{user.name ? user.name.charAt(0).toUpperCase() : 'U'}</span>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-900">{user.name || 'User'}</p>
                      <p className="text-xs text-gray-500">{user.email}</p>
                    </div>
                  </div>
                </div>
                <div className="py-2">
                  <Link to="/dashboard" className={isMobileActive('/dashboard')} onClick={() => setIsMobileMenuOpen(false)}>
                    <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <rect x="3" y="3" width="7" height="7" />
                      <rect x="14" y="3" width="7" height="7" />
                      <rect x="14" y="14" width="7" height="7" />
                      <rect x="3" y="14" width="7" height="7" />
                    </svg>
                    Dashboard
                  </Link>
                  <Link to="/add-data" className={isMobileActive('/add-data')} onClick={() => setIsMobileMenuOpen(false)}>
                    <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <line x1="12" y1="5" x2="12" y2="19" />
                      <line x1="5" y1="12" x2="19" y2="12" />
                    </svg>
                    Add Data
                  </Link>
                  <Link to="/history" className={isMobileActive('/history')} onClick={() => setIsMobileMenuOpen(false)}>
                    <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path d="M3 3v5h5M3.05 13a9 9 0 1 0 1.5-5.5" />
                    </svg>
                    History
                  </Link>
                  <Link to="/reports" className={isMobileActive('/reports')} onClick={() => setIsMobileMenuOpen(false)}>
                    <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <line x1="18" y1="20" x2="18" y2="10" />
                      <line x1="12" y1="20" x2="12" y2="4" />
                      <line x1="6" y1="20" x2="6" y2="14" />
                    </svg>
                    Reports
                  </Link>
                  <Link to="/profile" className="flex items-center gap-3 px-4 py-3 text-gray-700 hover:bg-gray-50 transition-colors" onClick={() => setIsMobileMenuOpen(false)}>
                    <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                      <circle cx="12" cy="7" r="4" />
                    </svg>
                    Profile
                  </Link>
                  <Link to="/settings" className="flex items-center gap-3 px-4 py-3 text-gray-700 hover:bg-gray-50 transition-colors" onClick={() => setIsMobileMenuOpen(false)}>
                    <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <circle cx="12" cy="12" r="3"/>
                      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1 -1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
                    </svg>
                    Settings
                  </Link>
                </div>
                <div className="border-t border-gray-100">
                  <button onClick={handleLogout} className="flex items-center gap-3 px-4 py-3 text-red-600 hover:bg-red-50 transition-colors w-full text-left">
                    <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                      <polyline points="16,17 21,12 16,7" />
                      <line x1="21" y1="12" x2="9" y2="12" />
                    </svg>
                    Logout
                  </button>
                </div>
              </>
            ) : (
              <>
                <div className="py-2">
                  <Link to="/" className="block px-4 py-3 text-gray-700 hover:bg-gray-50 transition-colors" onClick={() => setIsMobileMenuOpen(false)}>Home</Link>
                  <Link to="/signin" className="block px-4 py-3 text-gray-700 hover:bg-gray-50 transition-colors" onClick={() => setIsMobileMenuOpen(false)}>Login</Link>
                  <Link to="/signup" className="block px-4 py-3 text-gray-700 hover:bg-gray-50 transition-colors" onClick={() => setIsMobileMenuOpen(false)}>Sign Up</Link>
                </div>
              </>
            )}
          </div>
        )}
    </header>
  )
}

export default Header