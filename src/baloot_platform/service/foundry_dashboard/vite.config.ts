import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/foundry-assets/',
  plugins: [react()],
  build: {
    outDir: '../static/foundry',
    emptyOutDir: true,
    sourcemap: false,
    assetsDir: 'assets',
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            {name: 'react', test: /node_modules[\\/]react(?:-dom)?[\\/]/, priority: 2},
            {name: 'tanstack', test: /node_modules[\\/]@tanstack[\\/]/, priority: 1},
          ],
        },
      },
    },
  },
});
