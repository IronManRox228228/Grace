const path = require('path');

// Resolved explicitly (rather than left to cosmiconfig's cwd-based
// search) so `npm run build` works the same whether it's invoked from
// the project root or from inside renderer/.
module.exports = {
  plugins: {
    tailwindcss: { config: path.join(__dirname, 'tailwind.config.js') },
    autoprefixer: {},
  },
};
