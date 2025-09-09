import forms from '@tailwindcss/forms';
import typography from '@tailwindcss/typography';

export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      transitionProperty: {
        'max-height': 'max-height',
      },
    },
  },
  plugins: [forms, typography],
};
