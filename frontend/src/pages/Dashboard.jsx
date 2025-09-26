import { useState } from 'react'
import { useAuth } from '../hooks/useAuth'
import DashboardHeader from '../components/layout/DashboardHeader'

const Dashboard = () => {
  const { user } = useAuth()
  const [showAddOrderModal, setShowAddOrderModal] = useState(false)
  const [newOrder, setNewOrder] = useState({
    productName: '',
    productCategory: '',
    productPrice: '',
    orderQuantity: '',
    userAge: '',
    userGender: '',
    userLocation: '',
    paymentMethod: '',
    shippingMethod: '',
    discountApplied: ''
  })
  const [stats, _setStats] = useState({
    totalOrders: 1234,
    predictedReturns: 156,
    returnRate: 12.6,
    revenueSaved: 45230,
    accuracy: 94.5
  })

  const [recentPredictions, _setRecentPredictions] = useState([
    {
      id: 'ORD-001',
      product: 'Wireless Headphones',
      category: 'Electronics',
      price: 199.99,
      prediction: 'Low Risk',
      probability: 15,
      date: '2024-01-15',
      status: 'processed'
    },
    {
      id: 'ORD-002',
      product: 'Running Shoes',
      category: 'Sports',
      price: 129.99,
      prediction: 'High Risk',
      probability: 78,
      date: '2024-01-15',
      status: 'processed'
    },
    {
      id: 'ORD-003',
      product: 'Laptop Bag',
      category: 'Accessories',
      price: 49.99,
      prediction: 'Medium Risk',
      probability: 45,
      date: '2024-01-14',
      status: 'processed'
    },
    {
      id: 'ORD-004',
      product: 'Smart Watch',
      category: 'Electronics',
      price: 299.99,
      prediction: 'Low Risk',
      probability: 22,
      date: '2024-01-14',
      status: 'processing'
    },
    {
      id: 'ORD-005',
      product: 'Yoga Mat',
      category: 'Sports',
      price: 29.99,
      prediction: 'High Risk',
      probability: 85,
      date: '2024-01-13',
      status: 'processed'
    }
  ])

  const [quickPrediction, setQuickPrediction] = useState({
    productCategory: '',
    productPrice: '',
    orderQuantity: '',
    userAge: '',
    userGender: '',
    userLocation: '',
    paymentMethod: '',
    shippingMethod: '',
    discountApplied: ''
  })

  const [predictionResult, setPredictionResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleAddOrderSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)

    try {
      // Send request to backend API
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          price: parseFloat(newOrder.productPrice),
          quantity: parseInt(newOrder.orderQuantity),
          product_category: newOrder.productCategory,
          user_location: newOrder.userLocation,
          customer_age: parseInt(newOrder.userAge),
          user_gender: newOrder.userGender,
          payment_method: newOrder.paymentMethod,
          shipping_method: newOrder.shippingMethod,
          discount_amount: parseFloat(newOrder.discountApplied) || 0
        })
      })

      if (response.ok) {
        const result = await response.json()
        
        // Add to recent predictions
        const newPrediction = {
          id: `ORD-${Date.now().toString().slice(-3)}`,
          product: newOrder.productName,
          category: newOrder.productCategory,
          price: parseFloat(newOrder.productPrice),
          prediction: result.risk_category,
          probability: Math.round(result.return_probability * 100),
          date: new Date().toISOString().split('T')[0],
          status: 'processed'
        }
        
        _setRecentPredictions(prev => [newPrediction, ...prev.slice(0, 4)])
        
        // Reset form and close modal
        setNewOrder({
          productName: '',
          productCategory: '',
          productPrice: '',
          orderQuantity: '',
          userAge: '',
          userGender: '',
          userLocation: '',
          paymentMethod: '',
          shippingMethod: '',
          discountApplied: ''
        })
        setShowAddOrderModal(false)
        
        alert(`Order added successfully! Return Risk: ${result.risk_category} (${Math.round(result.return_probability * 100)}%)`)
      } else {
        throw new Error('Failed to predict')
      }
    } catch (error) {
      console.error('Error adding order:', error)
      alert('Failed to add order. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleNewOrderChange = (e) => {
    setNewOrder({
      ...newOrder,
      [e.target.name]: e.target.value
    })
  }

  const handleQuickPredictionSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      // Mock prediction result
      const probability = Math.floor(Math.random() * 100)
      const riskLevel = probability > 70 ? 'High Risk' : probability > 40 ? 'Medium Risk' : 'Low Risk'
      
      setPredictionResult({
        probability,
        riskLevel,
        confidence: 94.5,
        factors: [
          { factor: 'Product Category', impact: 'Medium', value: quickPrediction.productCategory },
          { factor: 'Price Range', impact: 'High', value: `$${quickPrediction.productPrice}` },
          { factor: 'User Location', impact: 'Low', value: quickPrediction.userLocation },
          { factor: 'Payment Method', impact: 'Medium', value: quickPrediction.paymentMethod }
        ]
      })
    } catch (error) {
      console.error('Prediction failed:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleInputChange = (e) => {
    setQuickPrediction({
      ...quickPrediction,
      [e.target.name]: e.target.value
    })
  }

  const getPredictionColor = (prediction) => {
    switch (prediction) {
      case 'High Risk': return 'danger'
      case 'Medium Risk': return 'warning'
      case 'Low Risk': return 'success'
      default: return 'info'
    }
  }
  return (
    <>
      <DashboardHeader />
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 py-8">
        <div className="max-w-full mx-auto px-6 xl:px-12 2xl:px-16">
        {/* Header */}
        <div className="flex justify-between items-start mb-8 bg-white/80 backdrop-blur-sm p-8 rounded-2xl shadow-xl border border-white/20 relative overflow-hidden">
          <div className="absolute -top-4 -right-4 w-24 h-24 bg-gradient-to-br from-indigo-200/30 to-purple-200/30 rounded-full blur-xl"></div>
          <div className="relative z-10">
            <h1 className="text-4xl font-bold bg-gradient-to-r from-indigo-700 via-purple-700 to-pink-700 bg-clip-text text-transparent mb-3">
              Welcome back, {user?.user_metadata?.full_name || user?.email?.split('@')[0] || 'User'}!
            </h1>
            <p className="text-slate-600 text-lg font-medium">
              Here's your return prediction overview for today
            </p>
          </div>
          <div className="flex gap-4 relative z-10">
            <button className="flex items-center gap-3 px-6 py-3 bg-white/80 border border-slate-200 text-slate-700 font-semibold rounded-xl hover:bg-white hover:shadow-lg transition-all duration-300 transform hover:-translate-y-0.5 backdrop-blur-sm">
              <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3"/>
              </svg>
              Export Report
            </button>
            <button 
              onClick={() => setShowAddOrderModal(true)}
              className="flex items-center gap-3 px-6 py-3 bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 text-white font-semibold rounded-xl hover:from-indigo-700 hover:via-purple-700 hover:to-pink-700 transition-all duration-300 transform hover:-translate-y-0.5 shadow-xl hover:shadow-2xl"
            >
              <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <line x1="12" y1="5" x2="12" y2="19"/>
                <line x1="5" y1="12" x2="19" y2="12"/>
              </svg>
              Add New Order
            </button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white/80 backdrop-blur-sm p-6 rounded-2xl shadow-xl border border-white/20 transition-all duration-300 hover:-translate-y-2 hover:shadow-2xl relative overflow-hidden group">
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-yellow-600 to-amber-700"></div>
            <div className="absolute -top-6 -right-6 w-16 h-16 bg-gradient-to-br from-yellow-200/20 to-amber-200/20 rounded-full blur-xl group-hover:scale-110 transition-transform duration-300"></div>
            <div className="flex justify-between items-center mb-4 relative z-10">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-yellow-600 to-amber-700 flex items-center justify-center text-white shadow-lg">
                <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path d="M3 3v5h5M3.05 13a9 9 0 1 0 1.5-5.5"/>
                </svg>
              </div>
              <div className="flex items-center gap-1 px-3 py-1 bg-emerald-100 text-emerald-700 text-sm font-semibold rounded-lg shadow-sm">
                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <polyline points="23,6 13.5,15.5 8.5,10.5 1,18"/>
                  <polyline points="17,6 23,6 23,12"/>
                </svg>
                +12%
              </div>
            </div>
            <div className="relative z-10">
              <h3 className="text-sm text-slate-600 font-semibold mb-1">Total Orders</h3>
              <p className="text-3xl font-bold text-slate-800 mb-1">{stats.totalOrders.toLocaleString()}</p>
              <p className="text-sm text-slate-500 font-medium">This month</p>
            </div>
          </div>

          <div className="bg-white/80 backdrop-blur-sm p-6 rounded-2xl shadow-xl border border-white/20 transition-all duration-300 hover:-translate-y-2 hover:shadow-2xl relative overflow-hidden group">
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-orange-500 to-red-500"></div>
            <div className="absolute -top-6 -right-6 w-16 h-16 bg-gradient-to-br from-orange-200/20 to-red-200/20 rounded-full blur-xl group-hover:scale-110 transition-transform duration-300"></div>
            <div className="flex justify-between items-center mb-4 relative z-10">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center text-white shadow-lg">
                <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z"/>
                </svg>
              </div>
              <div className="flex items-center gap-1 px-3 py-1 bg-red-100 text-red-700 text-sm font-semibold rounded-lg shadow-sm">
                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <polyline points="23,18 13.5,8.5 8.5,13.5 1,6"/>
                  <polyline points="17,18 23,18 23,12"/>
                </svg>
                -8%
              </div>
            </div>
            <div className="relative z-10">
              <h3 className="text-sm text-slate-600 font-semibold mb-1">Predicted Returns</h3>
              <p className="text-3xl font-bold text-slate-800 mb-1">{stats.predictedReturns}</p>
              <p className="text-sm text-slate-500 font-medium">High risk orders</p>
            </div>
          </div>

          <div className="bg-white/80 backdrop-blur-sm p-6 rounded-2xl shadow-xl border border-white/20 transition-all duration-300 hover:-translate-y-2 hover:shadow-2xl relative overflow-hidden group">
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 to-cyan-500"></div>
            <div className="absolute -top-6 -right-6 w-16 h-16 bg-gradient-to-br from-blue-200/20 to-cyan-200/20 rounded-full blur-xl group-hover:scale-110 transition-transform duration-300"></div>
            <div className="flex justify-between items-center mb-4 relative z-10">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-white shadow-lg">
                <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z"/>
                </svg>
              </div>
              <div className="flex items-center gap-1 px-3 py-1 bg-emerald-100 text-emerald-700 text-sm font-semibold rounded-lg shadow-sm">
                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <polyline points="23,6 13.5,15.5 8.5,10.5 1,18"/>
                  <polyline points="17,6 23,6 23,12"/>
                </svg>
                +5%
              </div>
            </div>
            <div className="relative z-10">
              <h3 className="text-sm text-slate-600 font-semibold mb-1">Return Rate</h3>
              <p className="text-3xl font-bold text-slate-800 mb-1">{stats.returnRate}%</p>
              <p className="text-sm text-slate-500 font-medium">Overall return rate</p>
            </div>
          </div>

          <div className="bg-white/80 backdrop-blur-sm p-6 rounded-2xl shadow-xl border border-white/20 transition-all duration-300 hover:-translate-y-2 hover:shadow-2xl relative overflow-hidden group">
            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-emerald-500 to-green-500"></div>
            <div className="absolute -top-6 -right-6 w-16 h-16 bg-gradient-to-br from-emerald-200/20 to-green-200/20 rounded-full blur-xl group-hover:scale-110 transition-transform duration-300"></div>
            <div className="flex justify-between items-center mb-4 relative z-10">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-emerald-500 to-green-500 flex items-center justify-center text-white shadow-lg">
                <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
                </svg>
              </div>
              <div className="flex items-center gap-1 px-3 py-1 bg-emerald-100 text-emerald-700 text-sm font-semibold rounded-lg shadow-sm">
                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <polyline points="23,6 13.5,15.5 8.5,10.5 1,18"/>
                  <polyline points="17,6 23,6 23,12"/>
                </svg>
                +18%
              </div>
            </div>
            <div className="relative z-10">
              <h3 className="text-sm text-slate-600 font-semibold mb-1">Revenue Saved</h3>
              <p className="text-3xl font-bold text-slate-800 mb-1">${stats.revenueSaved.toLocaleString()}</p>
              <p className="text-sm text-slate-500 font-medium">From predictions</p>
            </div>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          {/* Quick Prediction */}
          <div className="lg:col-span-2 bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 p-8 relative overflow-hidden">
            <div className="absolute -top-8 -right-8 w-24 h-24 bg-gradient-to-br from-indigo-200/20 to-purple-200/20 rounded-full blur-xl"></div>
            <div className="mb-6 relative z-10">
              <h2 className="text-2xl font-bold bg-gradient-to-r from-indigo-700 to-purple-700 bg-clip-text text-transparent mb-2">Quick Prediction</h2>
              <p className="text-slate-600 font-medium">Get instant return probability for an order</p>
            </div>
            <div>
              <form onSubmit={handleQuickPredictionSubmit} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Product Category</label>
                    <select 
                      name="productCategory" 
                      value={quickPrediction.productCategory}
                      onChange={handleInputChange}
                      className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
                      required
                    >
                      <option value="">Select category</option>
                      <option value="Electronics">Electronics</option>
                      <option value="Clothing">Clothing</option>
                      <option value="Sports">Sports</option>
                      <option value="Home">Home & Garden</option>
                      <option value="Books">Books</option>
                      <option value="Beauty">Beauty</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Product Price ($)</label>
                    <input 
                      type="number" 
                      name="productPrice"
                      value={quickPrediction.productPrice}
                      onChange={handleInputChange}
                      className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
                      placeholder="0.00"
                      step="0.01"
                      required
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Order Quantity</label>
                    <input 
                      type="number" 
                      name="orderQuantity"
                      value={quickPrediction.orderQuantity}
                      onChange={handleInputChange}
                      className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
                      placeholder="1"
                      min="1"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">User Age</label>
                    <input 
                      type="number" 
                      name="userAge"
                      value={quickPrediction.userAge}
                      onChange={handleInputChange}
                      className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
                      placeholder="25"
                      min="18"
                      max="100"
                      required
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">User Gender</label>
                    <select 
                      name="userGender"
                      value={quickPrediction.userGender}
                      onChange={handleInputChange}
                      className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
                      required
                    >
                      <option value="">Select gender</option>
                      <option value="Male">Male</option>
                      <option value="Female">Female</option>
                      <option value="Other">Other</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">User Location</label>
                    <select 
                      name="userLocation"
                      value={quickPrediction.userLocation}
                      onChange={handleInputChange}
                      className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
                      required
                    >
                      <option value="">Select location</option>
                      <option value="Urban">Urban</option>
                      <option value="Suburban">Suburban</option>
                      <option value="Rural">Rural</option>
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Payment Method</label>
                    <select 
                      name="paymentMethod"
                      value={quickPrediction.paymentMethod}
                      onChange={handleInputChange}
                      className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
                      required
                    >
                      <option value="">Select payment</option>
                      <option value="Credit Card">Credit Card</option>
                      <option value="Debit Card">Debit Card</option>
                      <option value="PayPal">PayPal</option>
                      <option value="Cash on Delivery">Cash on Delivery</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Shipping Method</label>
                    <select 
                      name="shippingMethod"
                      value={quickPrediction.shippingMethod}
                      onChange={handleInputChange}
                      className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
                      required
                    >
                      <option value="">Select shipping</option>
                      <option value="Standard">Standard</option>
                      <option value="Express">Express</option>
                      <option value="Next Day">Next Day</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Discount Applied (%)</label>
                  <input 
                    type="number" 
                    name="discountApplied"
                    value={quickPrediction.discountApplied}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
                    placeholder="0"
                    min="0"
                    max="100"
                    step="0.1"
                    required
                  />
                </div>

                <button type="submit" className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white font-semibold py-4 px-6 rounded-xl hover:from-purple-700 hover:to-blue-700 transition-all duration-300 shadow-lg hover:shadow-xl flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed" disabled={loading}>
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path d="M9 19c-5 0-8-3-8-6s3-6 8-6c1.25 0 2.45.21 3.58.58A5 5 0 0 1 16 6c2.76 0 5 2.24 5 5s-2.24 5-5 5c-1.38 0-2.63-.56-3.54-1.46A13.85 13.85 0 0 1 9 19z"/>
                      </svg>
                      Predict Return Risk
                    </>
                  )}
                </button>
              </form>

              {predictionResult && (
                <div className="mt-8 p-6 bg-gray-50 rounded-2xl">
                  <h3 className="text-lg font-bold text-gray-800 mb-6">Prediction Result</h3>
                  <div className="flex items-center justify-center mb-6">
                    <div className={`px-4 py-2 rounded-full text-sm font-semibold ${
                      predictionResult.riskLevel === 'High Risk' ? 'bg-red-100 text-red-700' :
                      predictionResult.riskLevel === 'Medium Risk' ? 'bg-yellow-100 text-yellow-700' :
                      'bg-green-100 text-green-700'
                    }`}>
                      {predictionResult.riskLevel}
                    </div>
                    <div className="ml-8 relative">
                      <svg width="80" height="80" viewBox="0 0 80 80">
                        <circle
                          cx="40"
                          cy="40"
                          r="35"
                          fill="none"
                          stroke="#e2e8f0"
                          strokeWidth="8"
                        />
                        <circle
                          cx="40"
                          cy="40"
                          r="35"
                          fill="none"
                          stroke={predictionResult.riskLevel === 'High Risk' ? '#e53e3e' : 
                                 predictionResult.riskLevel === 'Medium Risk' ? '#ed8936' : '#38a169'}
                          strokeWidth="8"
                          strokeDasharray={`${predictionResult.probability * 2.2} 220`}
                          strokeDashoffset="0"
                          transform="rotate(-90 40 40)"
                          className="transition-all duration-500"
                        />
                      </svg>
                      <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-center">
                        <span className="block text-xl font-bold text-gray-800">{predictionResult.probability}%</span>
                        <span className="text-sm text-gray-600">Return Risk</span>
                      </div>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div className="text-center text-sm text-gray-600">
                      Confidence: {predictionResult.confidence}%
                    </div>
                    <div>
                      <h4 className="text-sm font-semibold text-gray-800 mb-3">Key Factors</h4>
                      <div className="space-y-3">
                        {predictionResult.factors.map((factor, index) => (
                          <div key={index} className="flex items-center justify-between p-3 bg-white rounded-lg">
                            <div>
                              <span className="text-sm font-medium text-gray-800">{factor.factor}</span>
                              <span className="text-sm text-gray-600 ml-2">{factor.value}</span>
                            </div>
                            <div className={`px-2 py-1 rounded-md text-xs font-medium ${
                              factor.impact === 'High' ? 'bg-red-100 text-red-700' :
                              factor.impact === 'Medium' ? 'bg-yellow-100 text-yellow-700' :
                              'bg-green-100 text-green-700'
                            }`}>
                              {factor.impact}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Recent Predictions */}
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 p-6 relative overflow-hidden">
            <div className="absolute -top-6 -right-6 w-20 h-20 bg-gradient-to-br from-purple-200/20 to-pink-200/20 rounded-full blur-xl"></div>
            <div className="flex justify-between items-center mb-6 relative z-10">
              <h2 className="text-xl font-bold bg-gradient-to-r from-purple-700 to-pink-700 bg-clip-text text-transparent">Recent Predictions</h2>
              <button className="px-4 py-2 bg-white/80 border border-slate-200 text-slate-700 text-sm font-semibold rounded-lg hover:bg-white hover:shadow-md transition-all duration-300 transform hover:-translate-y-0.5 backdrop-blur-sm">View All</button>
            </div>
            <div className="overflow-hidden">
              <div className="space-y-4">
                <div className="grid grid-cols-5 gap-4 py-3 px-4 bg-gray-50 rounded-lg text-sm font-medium text-gray-700">
                  <div>Order ID</div>
                  <div>Product</div>
                  <div>Prediction</div>
                  <div>Risk %</div>
                  <div>Status</div>
                </div>
                <div className="space-y-2">
                  {recentPredictions.map((prediction) => (
                    <div key={prediction.id} className="grid grid-cols-5 gap-4 py-4 px-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors duration-200">
                      <div className="flex flex-col">
                        <span className="text-sm font-medium text-gray-800">{prediction.id}</span>
                        <span className="text-xs text-gray-500">{prediction.date}</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-sm font-medium text-gray-800">{prediction.product}</span>
                        <span className="text-xs text-gray-500">{prediction.category}</span>
                      </div>
                      <div>
                        <span className={`inline-block px-2 py-1 rounded-md text-xs font-medium ${
                          prediction.prediction === 'High Risk' ? 'bg-red-100 text-red-700' :
                          prediction.prediction === 'Medium Risk' ? 'bg-yellow-100 text-yellow-700' :
                          'bg-green-100 text-green-700'
                        }`}>
                          {prediction.prediction}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-gray-200 rounded-full h-2 overflow-hidden relative">
                          <div 
                            className={`h-full rounded-full ${
                              prediction.prediction === 'High Risk' ? 'bg-red-500' :
                              prediction.prediction === 'Medium Risk' ? 'bg-yellow-500' :
                              'bg-green-500'
                            }`}
                            style={{width: `${prediction.probability}%`}}
                          ></div>
                        </div>
                        <span className="text-sm font-medium text-gray-600">{prediction.probability}%</span>
                      </div>
                      <div>
                        <span className={`inline-block px-2 py-1 rounded-md text-xs font-medium ${
                          prediction.status === 'processed' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'
                        }`}>
                          {prediction.status === 'processed' ? 'Processed' : 'Processing'}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Performance Overview */}
        <div className="mt-8">
          <div className="bg-white p-6 rounded-2xl shadow-lg">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-bold text-gray-800">Model Performance</h2>
              <div className="flex gap-6">
                <div className="text-center">
                  <span className="block text-sm text-gray-500">Accuracy</span>
                  <span className="text-lg font-bold text-blue-600">{stats.accuracy}%</span>
                </div>
                <div className="text-center">
                  <span className="block text-sm text-gray-500">Predictions Today</span>
                  <span className="text-lg font-bold text-green-600">247</span>
                </div>
              </div>
            </div>
            <div className="p-4 bg-gray-50 rounded-xl">
              <div className="flex items-end justify-center gap-2 h-40 mb-4">
                <div className="flex flex-col items-center">
                  <div className="bg-blue-500 w-8 rounded-t-sm mb-2" style={{height: '60%'}}></div>
                  <span className="text-xs text-gray-600">Mon</span>
                </div>
                <div className="flex flex-col items-center">
                  <div className="bg-blue-500 w-8 rounded-t-sm mb-2" style={{height: '80%'}}></div>
                  <span className="text-xs text-gray-600">Tue</span>
                </div>
                <div className="flex flex-col items-center">
                  <div className="bg-blue-500 w-8 rounded-t-sm mb-2" style={{height: '45%'}}></div>
                  <span className="text-xs text-gray-600">Wed</span>
                </div>
                <div className="flex flex-col items-center">
                  <div className="bg-blue-500 w-8 rounded-t-sm mb-2" style={{height: '90%'}}></div>
                  <span className="text-xs text-gray-600">Thu</span>
                </div>
                <div className="flex flex-col items-center">
                  <div className="bg-blue-500 w-8 rounded-t-sm mb-2" style={{height: '70%'}}></div>
                  <span className="text-xs text-gray-600">Fri</span>
                </div>
                <div className="flex flex-col items-center">
                  <div className="bg-blue-500 w-8 rounded-t-sm mb-2" style={{height: '85%'}}></div>
                  <span className="text-xs text-gray-600">Sat</span>
                </div>
                <div className="flex flex-col items-center">
                  <div className="bg-blue-500 w-8 rounded-t-sm mb-2" style={{height: '65%'}}></div>
                  <span className="text-xs text-gray-600">Sun</span>
                </div>
              </div>
              <div className="flex items-center justify-center">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-blue-500 rounded-sm"></div>
                  <span className="text-sm text-gray-600">Daily Predictions</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    {/* Add New Order Modal */}
    {showAddOrderModal && (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
        <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
          <div className="p-6">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold bg-gradient-to-r from-indigo-700 to-purple-700 bg-clip-text text-transparent">
                Add New Order
              </h2>
              <button 
                onClick={() => setShowAddOrderModal(false)}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            <form onSubmit={handleAddOrderSubmit} className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Product Name</label>
                  <input 
                    type="text" 
                    name="productName"
                    value={newOrder.productName}
                    onChange={handleNewOrderChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
                    placeholder="Enter product name"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Product Category</label>
                  <select 
                    name="productCategory" 
                    value={newOrder.productCategory}
                    onChange={handleNewOrderChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
                    required
                  >
                    <option value="">Select category</option>
                    <option value="Electronics">Electronics</option>
                    <option value="Clothing">Clothing</option>
                    <option value="Sports">Sports</option>
                    <option value="Home">Home & Garden</option>
                    <option value="Books">Books</option>
                    <option value="Beauty">Beauty</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Product Price ($)</label>
                  <input 
                    type="number" 
                    name="productPrice"
                    value={newOrder.productPrice}
                    onChange={handleNewOrderChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
                    placeholder="0.00"
                    step="0.01"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Order Quantity</label>
                  <input 
                    type="number" 
                    name="orderQuantity"
                    value={newOrder.orderQuantity}
                    onChange={handleNewOrderChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
                    placeholder="1"
                    min="1"
                    required
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">User Age</label>
                  <input 
                    type="number" 
                    name="userAge"
                    value={newOrder.userAge}
                    onChange={handleNewOrderChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
                    placeholder="25"
                    min="18"
                    max="100"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">User Gender</label>
                  <select 
                    name="userGender"
                    value={newOrder.userGender}
                    onChange={handleNewOrderChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
                    required
                  >
                    <option value="">Select gender</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">User Location</label>
                  <select 
                    name="userLocation"
                    value={newOrder.userLocation}
                    onChange={handleNewOrderChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
                    required
                  >
                    <option value="">Select location</option>
                    <option value="New York">New York</option>
                    <option value="California">California</option>
                    <option value="Texas">Texas</option>
                    <option value="Florida">Florida</option>
                    <option value="Illinois">Illinois</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Payment Method</label>
                  <select 
                    name="paymentMethod"
                    value={newOrder.paymentMethod}
                    onChange={handleNewOrderChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
                    required
                  >
                    <option value="">Select payment method</option>
                    <option value="credit_card">Credit Card</option>
                    <option value="debit_card">Debit Card</option>
                    <option value="paypal">PayPal</option>
                    <option value="bank_transfer">Bank Transfer</option>
                    <option value="cash_on_delivery">Cash on Delivery</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Shipping Method</label>
                  <select 
                    name="shippingMethod"
                    value={newOrder.shippingMethod}
                    onChange={handleNewOrderChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
                    required
                  >
                    <option value="">Select shipping method</option>
                    <option value="Standard">Standard</option>
                    <option value="Express">Express</option>
                    <option value="Overnight">Overnight</option>
                    <option value="Free">Free Shipping</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Discount Amount ($)</label>
                  <input 
                    type="number" 
                    name="discountApplied"
                    value={newOrder.discountApplied}
                    onChange={handleNewOrderChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
                    placeholder="0.00"
                    min="0"
                    step="0.01"
                  />
                </div>
              </div>

              <div className="flex gap-4 pt-4">
                <button 
                  type="button"
                  onClick={() => setShowAddOrderModal(false)}
                  className="flex-1 py-3 px-6 border border-gray-300 text-gray-700 font-semibold rounded-xl hover:bg-gray-50 transition-all duration-200"
                >
                  Cancel
                </button>
                <button 
                  type="submit" 
                  className="flex-1 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-semibold py-3 px-6 rounded-xl hover:from-purple-700 hover:to-blue-700 transition-all duration-300 shadow-lg hover:shadow-xl flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed" 
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                      Processing...
                    </>
                  ) : (
                    'Add Order & Predict'
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    )}
    </>
  )
}

export default Dashboard
