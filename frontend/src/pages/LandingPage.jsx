import { Link } from 'react-router-dom'
import { useState, useEffect } from 'react'

const LandingPage = () => {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    setIsVisible(true)
  }, [])

  const features = [
    {
      icon: (
        <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path d="M9 19c-5 0-8-3-8-6s3-6 8-6c1.25 0 2.45.21 3.58.58A5 5 0 0 1 16 6c2.76 0 5 2.24 5 5s-2.24 5-5 5c-1.38 0-2.63-.56-3.54-1.46A13.85 13.85 0 0 1 9 19z"/>
        </svg>
      ),
      title: "AI-Powered Predictions",
      description: "Advanced machine learning algorithms analyze order patterns to predict return likelihood with high accuracy."
    },
    {
      icon: (
        <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <line x1="18" y1="20" x2="18" y2="10"/>
          <line x1="12" y1="20" x2="12" y2="4"/>
          <line x1="6" y1="20" x2="6" y2="14"/>
        </svg>
      ),
      title: "Real-time Analytics",
      description: "Get instant insights on return rates, revenue impact, and customer behavior patterns."
    },
    {
      icon: (
        <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path d="M12 2L2 7L12 12L22 7L12 2Z"/>
          <path d="M2 17L12 22L22 17"/>
          <path d="M2 12L12 17L22 12"/>
        </svg>
      ),
      title: "Easy Integration",
      description: "Seamlessly integrate with your existing e-commerce platform through our simple API."
    },
    {
      icon: (
        <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
          <path d="M7 11V7A5 5 0 0 1 17 7V11"/>
        </svg>
      ),
      title: "Secure & Reliable",
      description: "Enterprise-grade security with 99.9% uptime guarantee and data protection compliance."
    }
  ]

  const stats = [
    { number: "72%", label: "Prediction Accuracy" },
    { number: "30%", label: "Reduction in Returns" },
    { number: "500K+", label: "Orders Analyzed" },
    { number: "24/7", label: "System Uptime" }
  ]

  return (
    <div className={`transition-all duration-1000 ease-out ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
      {/* Hero Section */}
      <section className="min-h-screen flex items-center relative overflow-hidden">
        {/* Animated Background */}
        <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50"></div>
        <div className="absolute inset-0 opacity-30">
          <div className="absolute top-20 left-10 w-72 h-72 bg-purple-300 rounded-full mix-blend-multiply filter blur-xl animate-pulse-glow"></div>
          <div className="absolute top-40 right-10 w-72 h-72 bg-blue-300 rounded-full mix-blend-multiply filter blur-xl animate-pulse-glow animation-delay-2000"></div>
          <div className="absolute bottom-20 left-1/2 w-72 h-72 bg-pink-300 rounded-full mix-blend-multiply filter blur-xl animate-pulse-glow animation-delay-4000"></div>
        </div>
        
        <div className="max-w-7xl w-full mx-auto px-6 sm:px-8 lg:px-12 py-20 relative z-10">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            {/* Left Content */}
            <div className="space-y-10 animate-fade-in">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/80 backdrop-blur-sm shadow-lg border border-purple-100">
                <span className="relative flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-purple-500"></span>
                </span>
                <span className="text-sm font-semibold text-gray-700">AI-Powered Platform</span>
              </div>
              
              <h1 className="text-6xl lg:text-7xl font-extrabold leading-tight">
                <span className="bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 bg-clip-text text-transparent">
                  Predict Returns.
                </span>
                <br/>
                <span className="gradient-text text-shadow">Save Revenue.</span>
              </h1>
              
              <p className="text-xl text-gray-600 leading-relaxed max-w-xl">
                Transform your e-commerce business with AI-powered return prediction. 
                Reduce return rates, optimize inventory, and boost customer satisfaction 
                with our <span className="font-semibold text-purple-600">advanced machine learning platform</span>.
              </p>
              
              <div className="flex flex-col sm:flex-row gap-4">
                <Link to="/signup" className="group inline-flex items-center justify-center px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold rounded-2xl hover:from-blue-700 hover:to-purple-700 transition-all duration-300 transform hover:scale-105 shadow-xl hover-glow">
                  Start Free Trial
                  <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24" className="ml-2 group-hover:translate-x-1 transition-transform duration-300">
                    <line x1="5" y1="12" x2="19" y2="12" strokeWidth={2}/>
                    <polyline points="12,5 19,12 12,19" strokeWidth={2}/>
                  </svg>
                </Link>
                <Link to="/signin" className="inline-flex items-center justify-center px-8 py-4 bg-white border-2 border-gray-200 text-gray-700 font-bold rounded-2xl hover:border-gray-300 hover:shadow-lg transition-all duration-300 transform hover:scale-105">
                  Sign In
                </Link>
              </div>
              
              {/* Trust Indicators */}
              <div className="flex items-center gap-8 mt-8 text-gray-600">
                <div className="flex items-center gap-2">
                  <svg className="h-5 w-5 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  <span className="text-sm font-medium">No credit card required</span>
                </div>
                <div className="flex items-center gap-2">
                  <svg className="h-5 w-5 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                  <span className="text-sm font-medium">14-day free trial</span>
                </div>
              </div>
            </div>
            
            {/* Right Content - Dashboard Preview */}
            <div className="relative hidden lg:block">
              <div className="glass-card rounded-3xl p-8 shadow-2xl transform hover:scale-[1.02] transition-all duration-500 border-2 border-white/20">
                <div className="flex items-center justify-between mb-6 pb-4 border-b border-gray-200/50">
                  <div className="flex gap-2">
                    <div className="w-3 h-3 rounded-full bg-red-400 shadow-sm"></div>
                    <div className="w-3 h-3 rounded-full bg-yellow-400 shadow-sm"></div>
                    <div className="w-3 h-3 rounded-full bg-green-400 shadow-sm"></div>
                  </div>
                  <div className="text-gray-800 font-bold text-lg">ReturnPredict Dashboard</div>
                </div>
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-6 border border-green-200 shadow-sm">
                      <div className="flex items-start gap-4">
                        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center shadow-lg">
                          <svg width="20" height="20" fill="none" stroke="white" viewBox="0 0 24 24" strokeWidth="2">
                            <polyline points="20,6 9,17 4,12"/>
                          </svg>
                        </div>
                        <div className="flex-1">
                          <div className="text-gray-600 text-sm font-medium">Return Rate</div>
                          <div className="text-3xl font-bold text-gray-900 mb-1">12.3%</div>
                          <div className="text-green-600 text-sm font-semibold flex items-center">
                            <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24" className="mr-1">
                              <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>
                              <polyline points="17 6 23 6 23 12"/>
                            </svg>
                            -2.4%
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-200 shadow-sm">
                      <div className="flex items-start gap-4">
                        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg">
                          <svg width="20" height="20" fill="none" stroke="white" viewBox="0 0 24 24" strokeWidth="2">
                            <path d="M12 2L2 7L12 12L22 7L12 2Z"/>
                          </svg>
                        </div>
                        <div className="flex-1">
                          <div className="text-gray-600 text-sm font-medium">Revenue Saved</div>
                          <div className="text-3xl font-bold text-gray-900 mb-1">$45,230</div>
                          <div className="text-blue-600 text-sm font-semibold flex items-center">
                            <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24" className="mr-1">
                              <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>
                              <polyline points="17 6 23 6 23 12"/>
                            </svg>
                            +18.7%
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="bg-gradient-to-br from-gray-50 to-slate-50 rounded-xl p-6 border border-gray-200 shadow-sm">
                    <div className="mb-4">
                      <h4 className="text-gray-800 font-semibold">Performance Analytics</h4>
                      <p className="text-gray-600 text-sm">Monthly return predictions accuracy</p>
                    </div>
                    <div className="flex items-end justify-between gap-3 h-32 bg-white rounded-lg p-4 shadow-inner">
                      <div className="w-8 bg-gradient-to-t from-indigo-400 to-indigo-600 rounded-t shadow-sm" style={{height: '60%'}}></div>
                      <div className="w-8 bg-gradient-to-t from-blue-400 to-blue-600 rounded-t shadow-sm" style={{height: '80%'}}></div>
                      <div className="w-8 bg-gradient-to-t from-purple-400 to-purple-600 rounded-t shadow-sm" style={{height: '45%'}}></div>
                      <div className="w-8 bg-gradient-to-t from-pink-400 to-pink-600 rounded-t shadow-sm" style={{height: '90%'}}></div>
                      <div className="w-8 bg-gradient-to-t from-violet-400 to-violet-600 rounded-t shadow-sm" style={{height: '70%'}}></div>
                      <div className="w-8 bg-gradient-to-t from-cyan-400 to-cyan-600 rounded-t shadow-sm" style={{height: '55%'}}></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Background decoration */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-1/4 left-10 w-72 h-72 bg-white/10 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-1/4 right-10 w-96 h-96 bg-purple-300/10 rounded-full blur-3xl animate-pulse" style={{animationDelay: '2s'}}></div>
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-pink-300/10 rounded-full blur-3xl animate-pulse" style={{animationDelay: '4s'}}></div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-8">
          <div className="text-center mb-16">
            <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-6">Why Choose ReturnPredict?</h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Leverage cutting-edge AI technology to transform your return management strategy
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <div key={index} className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2 group" style={{animationDelay: `${index * 0.1}s`}}>
                <div className="w-16 h-16 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center text-white mb-6 group-hover:scale-110 transition-transform duration-300">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-4">{feature.title}</h3>
                <p className="text-gray-600 leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 bg-gradient-to-r from-indigo-600 to-purple-600 text-white">
        <div className="max-w-7xl mx-auto px-8">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <div key={index} className="text-center group" style={{animationDelay: `${index * 0.1}s`}}>
                <div className="text-4xl lg:text-5xl font-bold mb-2 bg-gradient-to-r from-yellow-300 to-pink-300 bg-clip-text text-transparent group-hover:scale-110 transition-transform duration-300">{stat.number}</div>
                <div className="text-gray-200 text-lg font-medium">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-br from-slate-800 via-slate-700 to-indigo-800 text-white relative overflow-hidden">
        <div className="max-w-4xl mx-auto px-8 text-center relative z-10">
          <div className="space-y-8">
            <h2 className="text-4xl lg:text-5xl font-bold bg-gradient-to-r from-white to-blue-100 bg-clip-text text-transparent">Ready to Reduce Your Return Rates?</h2>
            <p className="text-xl text-blue-100 max-w-2xl mx-auto leading-relaxed">
              Join thousands of businesses already using ReturnPredict to optimize their operations
            </p>
            <div className="flex flex-col items-center gap-6">
              <Link to="/signup" className="inline-flex items-center justify-center px-10 py-5 bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white font-bold rounded-2xl transition-all duration-200 transform hover:scale-105 shadow-2xl hover:shadow-orange-500/25 group">
                Get Started Free
                <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24" className="ml-2 group-hover:translate-x-1 transition-transform duration-200">
                  <line x1="5" y1="12" x2="19" y2="12"/>
                  <polyline points="12,5 19,12 12,19"/>
                </svg>
              </Link>
              <div className="flex items-center gap-2 text-blue-200">
                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24" className="text-green-400">
                  <polyline points="20,6 9,17 4,12"/>
                </svg>
                <span className="font-medium">No credit card required • Free 14-day trial</span>
              </div>
            </div>
          </div>
        </div>
        
        {/* Background decoration */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -top-8 -left-8 w-48 h-48 bg-gradient-to-br from-blue-400/10 to-indigo-500/10 rounded-full blur-3xl"></div>
          <div className="absolute -bottom-8 -right-8 w-64 h-64 bg-gradient-to-br from-purple-400/10 to-pink-500/10 rounded-full blur-3xl"></div>
          <div className="absolute top-1/2 left-1/4 w-32 h-32 bg-gradient-to-br from-cyan-400/10 to-blue-500/10 rounded-full blur-2xl"></div>
        </div>
      </section>
    </div>
  )
}

export default LandingPage