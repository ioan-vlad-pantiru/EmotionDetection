/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        emotion: {
          anger: '#ef4444',
          fear: '#8b5cf6',
          anticipation: '#f59e0b',
          trust: '#3b82f6',
          surprise: '#ec4899',
          sadness: '#06b6d4',
          joy: '#10b981',
          disgust: '#84cc16',
          neutral: '#6b7280',
        },
      },
    },
  },
  plugins: [],
}
