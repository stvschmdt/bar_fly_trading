import { useWorkbench } from '../hooks/useWorkbench'

export default function WorkbenchBar() {
  const { symbols, remove, clear } = useWorkbench()

  if (!symbols.length) return null

  return (
    <div className="fixed bottom-0 left-0 right-0 z-40 bg-white/90 dark:bg-gray-950/90 backdrop-blur border-t border-gray-200 dark:border-gray-800">
      <div className="max-w-[1600px] mx-auto px-3 py-2 flex items-center gap-2">
        <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 shrink-0">
          Workbench
        </span>

        <div className="flex-1 overflow-x-auto flex gap-1.5 scrollbar-hide">
          {symbols.map((sym) => (
            <span
              key={sym}
              className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-gray-100 dark:bg-gray-800 text-xs font-medium shrink-0"
            >
              {sym}
              <button
                onClick={() => remove(sym)}
                className="ml-0.5 text-gray-400 hover:text-red-500 dark:hover:text-red-400 transition-colors"
                aria-label={`Remove ${sym}`}
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </span>
          ))}
        </div>

        <button
          onClick={clear}
          className="shrink-0 text-[10px] text-gray-400 hover:text-red-500 dark:hover:text-red-400 transition-colors"
        >
          Clear
        </button>
      </div>
    </div>
  )
}
