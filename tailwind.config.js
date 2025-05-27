/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        'primary': {
          50: '#f0f7ff',
          100: '#e0eefe',
          200: '#bae0fd',
          300: '#7ac7fc',
          400: '#38a8f8',
          500: '#0e8eea',
          600: '#026fc8',
          700: '#0559a2',
          800: '#0a4a86',
          900: '#0e3f70',
          950: '#0b2a4d',
        },
        'accent': {
          50: '#fdf3f3',
          100: '#fbe5e5',
          200: '#f9d0d0',
          300: '#f3adad',
          400: '#eb7d7d',
          500: '#dd5151',
          600: '#ca3737',
          700: '#ae2b2b',
          800: '#912626',
          900: '#7b2525',
          950: '#430f0f',
        },
        'neutral': {
          50: '#fcf9f5',
          100: '#f7f1e7',
          200: '#efe1ce',
          300: '#e4cbad',
          400: '#d5ad84',
          500: '#c99466',
          600: '#bc7e53',
          700: '#a36745',
          800: '#86533c',
          900: '#6f4534',
          950: '#3b221a',
        },
        'paper': '#fcf9f2',
        'ink': '#1a1a1a',
      },
      fontFamily: {
        'display': ['"Playfair Display"', 'serif'],
        'sans': ['"Source Sans 3"', 'sans-serif'],
      },
      boxShadow: {
        'paper': '0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.1)',
        'paper-hover': '0 4px 6px rgba(0,0,0,0.05), 0 1px 3px rgba(0,0,0,0.1)',
        'card': '0 4px 12px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.02)',
      },
      backgroundImage: {
        'ruled-paper': 'linear-gradient(0deg, rgba(200,215,240,0.2) 1px, transparent 1px)',
      },
      backgroundSize: {
        'ruled-10': '100% 10px',
      },
    },
  },
  plugins: [],
};