/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        gain: { light: '#16a34a', dark: '#4ade80' },
        loss: { light: '#dc2626', dark: '#f87171' },
      },
    },
  },
  plugins: [],
}
