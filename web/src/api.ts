/**
 * Where the API lives.
 *
 * Three components used to hardcode `http://localhost:8000`. That works on a
 * developer's machine and is meaningless once deployed: the browser resolves
 * `localhost` against the *visitor's* computer, so every request failed and the
 * panel rendered its fallbacks — "—", "Unknown", "0" — which read as a
 * repository with no metadata rather than as a broken deployment.
 *
 * Set `VITE_API_URL` in the hosting environment (Vercel: Project Settings →
 * Environment Variables). Vite inlines it at BUILD time, not runtime, so it has
 * to be present when the build runs — adding it afterwards changes nothing
 * until you redeploy.
 */
export const API = (
  (import.meta.env.VITE_API_URL as string | undefined) || 'http://localhost:8000'
).replace(/\/+$/, '');

/**
 * Shout about the two misconfigurations that otherwise fail silently.
 *
 * Both produce exactly the same screen as a repository genuinely having no
 * data, so without this the first sign of trouble is someone asking why every
 * field is blank.
 */
if (typeof window !== 'undefined' && window.location.protocol === 'https:') {
  if (API.includes('localhost') || API.includes('127.0.0.1')) {
    console.error(
      `[gitglobe] API is ${API} on a deployed page — VITE_API_URL was not set ` +
        `at build time, so repository details and search will all fail.`,
    );
  } else if (API.startsWith('http://')) {
    console.error(
      `[gitglobe] API is ${API} (http) on an https page. Browsers block this as ` +
        `mixed content. Use the https:// URL for the API.`,
    );
  }
}
