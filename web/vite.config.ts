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
    // Dev-side counterpart to the `/api` rewrite in vercel.json, so the client
    // talks to one path in both environments and `api.ts` needs no dev branch.
    //
    // Note this proxy resolves Railway from *your* machine, so on a network
    // that blocks up.railway.app (Jio) local dev still fails where production
    // now works — the deployed proxy runs on Vercel's edge, not here. Run the
    // API locally and set VITE_API_URL=http://localhost:8000 to bypass this
    // entirely, or use a resolver that answers for the zone.
    proxy: {
      '/api': {
        target: 'https://gitglobe-api-production.up.railway.app',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
    },
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
