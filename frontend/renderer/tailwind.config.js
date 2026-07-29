const path = require('path');

/** @type {import('tailwindcss').Config} */
module.exports = {
  // Absolute paths: Tailwind resolves relative content globs against
  // the process's cwd, not this file's location, which breaks when
  // `build` is run from the repo root instead of renderer/.
  content: [path.join(__dirname, 'index.html'), path.join(__dirname, 'src/**/*.{ts,tsx}')],
  theme: {
    extend: {
      colors: {
        cream: '#FDFBF7',
        periwinkle: '#CCCCFF',
        plum: '#705553',
        olive: '#3B4430',
        sage: '#A8B89A',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['"TAN Pearl"', 'Georgia', 'serif'],
      },
      boxShadow: {
        pill: '0 8px 30px -6px rgba(59, 68, 48, 0.18), 0 1px 2px rgba(59, 68, 48, 0.06)',
        panel: '0 20px 60px -12px rgba(59, 68, 48, 0.22)',
      },
      backdropBlur: {
        xs: '4px',
      },
    },
  },
  plugins: [],
};
