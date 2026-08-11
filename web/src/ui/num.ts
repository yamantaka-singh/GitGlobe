/**
 * Number formatting for anything a user reads.
 *
 * Exists because bare `toLocaleString()` follows the *system* locale, so
 * `5000000` renders "50,00,000" on an en-IN machine and "5,000,000" everywhere
 * else — the same corpus described two different ways depending on who opened
 * it, and neither matching the pipeline, whose `f"{n:,}"` is always Western.
 *
 * Pinned rather than localised because the UI is English-only. If that ever
 * changes, this is the one place to change it — and `StarScale.describe` in
 * `global_scale.py` has to change with it or the CLI and the browser will
 * disagree about the same repository.
 */
export function group(n: number): string {
  return Math.round(n).toLocaleString('en-US');
}
