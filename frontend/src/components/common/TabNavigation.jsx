const TabNavigation = ({ tabs, activeTab, onTabChange, className = "" }) => {
  return (
    <div className={`glass-card rounded-2xl p-2 shadow-lg ${className}`}>
      <nav className="flex space-x-2 overflow-x-auto" aria-label="Tabs">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`
              relative whitespace-nowrap py-3 px-6 rounded-xl font-medium text-sm transition-all duration-300 smooth-transition
              flex items-center gap-2 group
              ${
                activeTab === tab.id
                  ? "bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg hover-glow"
                  : "text-gray-600 hover:bg-gray-100/50 hover:text-gray-900"
              }
            `}
          >
            {tab.icon && (
              <span className={`transition-transform duration-300 ${
                activeTab === tab.id ? "scale-110" : "group-hover:scale-105"
              }`}>
                {tab.icon}
              </span>
            )}
            {tab.label}
            {tab.badge && (
              <span className={`ml-2 py-0.5 px-2.5 rounded-full text-xs font-semibold ${
                activeTab === tab.id
                  ? "bg-white/20 text-white"
                  : "bg-gray-100 text-gray-600"
              }`}>
                {tab.badge}
              </span>
            )}
            {activeTab === tab.id && (
              <span className="absolute bottom-0 left-0 right-0 h-1 bg-white/30 rounded-full"></span>
            )}
          </button>
        ))}
      </nav>
    </div>
  );
};

export default TabNavigation;
