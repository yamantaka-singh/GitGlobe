// `vitest/config` re-exports vite's defineConfig with the `test` key typed.
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    target: 'es2022',
    rollupOptions: {
      output: {
        // three is ~600KB on its own and changes far less often than app code.
        manualChunks: { three: ['three'] },
      },
    },
  },
  server: {
    headers: {
      // Tiles are versioned by layout, but during dev we just want the browser
      // to re-fetch after `npm run gen:tiles`.
      'Cache-Control': 'no-store',
    },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
});
