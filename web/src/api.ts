/**
 * Where the API lives.
 *
 * Same-origin by default, and that is deliberate — it is the fix for an outage
 * that was invisible from any developer machine with working DNS.
 *
 * Pointing the browser straight at `gitglobe-api-production.up.railway.app`
 * made every visitor resolve that hostname themselves, and Jio — India's
 * largest ISP — answers REFUSED for the entire `up.railway.app` zone. Not our
 * subdomain: the zone. Free-hosting wildcard domains get blocklisted wholesale
 * and stay that way, so no redeploy or Railway-side change clears it. Those
 * users saw "Load failed" with no request ever reaching the network.
 *
 * `/api` is proxied to Railway by `vercel.json` in production and by the Vite
 * dev proxy locally, so the only hostname a visitor resolves is the one already
 * serving them the page. Whoever resolves Railway does it somewhere with
 * functioning DNS.
 *
 * `VITE_API_URL` still overrides, which is how you point at a local API:
 * `VITE_API_URL=http://localhost:8000`. Vite inlines it at BUILD time, so it
 * has to be present when the build runs. Setting it to an absolute
 * `up.railway.app` URL re-creates the outage — that is what the check below is
 * watching for.
 */
export const API = (
  (import.meta.env.VITE_API_URL as string | undefined) || '/api'
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
      `[gitglobe] API is ${API} on a deployed page — VITE_API_URL points at the ` +
        `visitor's own machine, so repository details and search will all fail.`,
    );
  } else if (API.includes('up.railway.app')) {
    console.error(
      `[gitglobe] API is ${API} — a hostname whole ISPs refuse to resolve (Jio ` +
        `returns REFUSED for the entire up.railway.app zone). Unset VITE_API_URL ` +
        `so requests go same-origin through the /api proxy in vercel.json.`,
    );
  } else if (API.startsWith('http://')) {
    console.error(
      `[gitglobe] API is ${API} (http) on an https page. Browsers block this as ` +
        `mixed content. Use the https:// URL for the API.`,
    );
  }
}
