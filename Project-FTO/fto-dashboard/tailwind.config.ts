import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'sqdcm-s': '#10B981', // Safety - Emerald Green
        'sqdcm-q': '#3B82F6', // Quality - Blue
        'sqdcm-d': '#F59E0B', // Delivery - Amber
        'sqdcm-c': '#EF4444', // Cost - Red
        'sqdcm-m': '#8B5CF6', // Morale - Violet
      },
    },
  },
  darkMode: 'class',
  plugins: [],
}

export default config
