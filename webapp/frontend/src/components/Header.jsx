import { useTheme } from '../hooks/useTheme'

const THEME_ICONS = { light: 'L', dark: 'D', system: 'A' }
const THEME_CYCLE = ['system', 'light', 'dark']

export default function Header({ lastUpdated }) {
  const { theme, setTheme } = useTheme()

  function cycleTheme() {
    const idx = THEME_CYCLE.indexOf(theme)
    setTheme(THEME_CYCLE[(idx + 1) % THEME_CYCLE.length])
  }

  return (
    <header className="sticky top-0 z-50 bg-white/80 dark:bg-gray-950/80 backdrop-blur border-b border-gray-200 dark:border-gray-800 px-4 py-3">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-3">
          <a href="/" className="text-xl font-bold tracking-tight">
            Bar Fly Trading <span className="text-gray-400 dark:text-gray-500 font-normal">&</span> Investing
          </a>
        </div>
        <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
          {lastUpdated && (
            <span className="hidden sm:inline">{lastUpdated}</span>
          )}
          <button
            onClick={cycleTheme}
            className="w-8 h-8 rounded-full border border-gray-300 dark:border-gray-600 flex items-center justify-center hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors text-xs font-mono"
            title={`Theme: ${theme}`}
          >
            {THEME_ICONS[theme]}
          </button>
        </div>
      </div>
    </header>
  )
}
