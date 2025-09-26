import { Link } from 'react-router-dom'

const Footer = () => {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="relative bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white overflow-hidden">
      <div className="relative z-10 max-w-6xl mx-auto px-4 py-12">
        {/* Main Footer Content */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-8">
          {/* Company Info */}
          <div className="lg:col-span-1">
            <div className="flex items-center gap-3 mb-4">
              <div className="transform hover:rotate-12 transition-transform duration-300">
                <svg width="28" height="28" viewBox="0 0 32 32" fill="none">
                  <rect width="32" height="32" rx="8" fill="url(#footerGradient)" />
                  <path d="M8 12L16 8L24 12V20C24 22.2091 22.2091 24 20 24H12C9.79086 24 8 22.2091 8 20V12Z" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M16 8V16" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <defs>
                    <linearGradient id="footerGradient" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
                      <stop stopColor="#667eea" />
                      <stop offset="1" stopColor="#764ba2" />
                    </linearGradient>
                  </defs>
                </svg>
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">ReturnPredict</span>
            </div>
            <p className="text-gray-400 text-sm leading-relaxed mb-6">
              Advanced e-commerce return prediction powered by machine learning. 
              Help businesses reduce return rates and optimize customer satisfaction.
            </p>
            <div className="flex gap-4">
              <a href="#" className="w-10 h-10 bg-gray-800 hover:bg-gradient-to-r hover:from-blue-500 hover:to-purple-500 rounded-full flex items-center justify-center text-gray-400 hover:text-white transition-all duration-300 transform hover:scale-110" aria-label="Twitter">
                <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z"/>
                </svg>
              </a>
              <a href="#" className="w-10 h-10 bg-gray-800 hover:bg-gradient-to-r hover:from-blue-500 hover:to-purple-500 rounded-full flex items-center justify-center text-gray-400 hover:text-white transition-all duration-300 transform hover:scale-110" aria-label="LinkedIn">
                <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                </svg>
              </a>
              <a href="#" className="w-10 h-10 bg-gray-800 hover:bg-gradient-to-r hover:from-blue-500 hover:to-purple-500 rounded-full flex items-center justify-center text-gray-400 hover:text-white transition-all duration-300 transform hover:scale-110" aria-label="GitHub">
                <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
              </a>
            </div>
          </div>

          {/* Quick Links */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white mb-4">Quick Links</h3>
            <div className="space-y-3">
              <Link to="/" className="block text-gray-400 hover:text-white hover:pl-2 transition-all duration-200">Home</Link>
              <Link to="/dashboard" className="block text-gray-400 hover:text-white hover:pl-2 transition-all duration-200">Dashboard</Link>
              <Link to="/add-data" className="block text-gray-400 hover:text-white hover:pl-2 transition-all duration-200">Add Data</Link>
              <Link to="/history" className="block text-gray-400 hover:text-white hover:pl-2 transition-all duration-200">Prediction History</Link>
              <Link to="/reports" className="block text-gray-400 hover:text-white hover:pl-2 transition-all duration-200">Reports</Link>
            </div>
          </div>

          {/* Resources */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white mb-4">Resources</h3>
            <div className="space-y-3">
              <a href="#" className="block text-gray-400 hover:text-white hover:pl-2 transition-all duration-200">Documentation</a>
              <a href="#" className="block text-gray-400 hover:text-white hover:pl-2 transition-all duration-200">API Reference</a>
              <a href="#" className="block text-gray-400 hover:text-white hover:pl-2 transition-all duration-200">Tutorials</a>
              <a href="#" className="block text-gray-400 hover:text-white hover:pl-2 transition-all duration-200">Best Practices</a>
              <a href="#" className="block text-gray-400 hover:text-white hover:pl-2 transition-all duration-200">Case Studies</a>
            </div>
          </div>

          {/* Support */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white mb-4">Support</h3>
            <div className="space-y-3">
              <a href="#" className="block text-gray-400 hover:text-white hover:pl-2 transition-all duration-200">Help Center</a>
              <a href="#" className="block text-gray-400 hover:text-white hover:pl-2 transition-all duration-200">Contact Us</a>
              <a href="#" className="block text-gray-400 hover:text-white hover:pl-2 transition-all duration-200">Community</a>
              <a href="#" className="block text-gray-400 hover:text-white hover:pl-2 transition-all duration-200">Status</a>
              <a href="#" className="block text-gray-400 hover:text-white hover:pl-2 transition-all duration-200">Bug Reports</a>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-gray-800 pt-8 mt-12">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div>
              <p className="text-gray-400 text-sm">&copy; {currentYear} ReturnPredict. All rights reserved.</p>
            </div>
            <div className="flex gap-6">
              <a href="#" className="text-gray-400 hover:text-white text-sm transition-colors duration-200">Privacy Policy</a>
              <a href="#" className="text-gray-400 hover:text-white text-sm transition-colors duration-200">Terms of Service</a>
              <a href="#" className="text-gray-400 hover:text-white text-sm transition-colors duration-200">Cookie Policy</a>
            </div>
          </div>
        </div>
      </div>

      {/* Decorative Background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-4 -left-4 w-24 h-24 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-full blur-xl"></div>
        <div className="absolute top-1/2 -right-8 w-32 h-32 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-full blur-xl"></div>
        <div className="absolute -bottom-8 left-1/3 w-20 h-20 bg-gradient-to-br from-indigo-500/20 to-blue-500/20 rounded-full blur-xl"></div>
      </div>
    </footer>
  )
}

export default Footer