/**
 * Weekend-aware staleness detection for stock data.
 *
 * Weekday:            stale if > 18h old (covers overnight gap)
 * Saturday:           stale if > 42h old (Friday close ~18h + buffer)
 * Sunday:             stale if > 66h old (Friday close ~42h + buffer)
 * Monday pre-market:  stale if > 66h old (Friday close ~66h + buffer)
 *
 * null/missing last_updated => always stale.
 */

const HOUR = 3600 * 1000

// Max age in hours by day-of-week (0=Sun, 6=Sat)
const MAX_AGE_HOURS = {
  0: 66, // Sunday — Friday data is ~48h old + buffer
  1: 66, // Monday — Friday data could be ~72h, generous buffer
  2: 18, // Tuesday
  3: 18, // Wednesday
  4: 18, // Thursday
  5: 18, // Friday
  6: 42, // Saturday — Friday close ~18h ago + buffer
}

/**
 * Returns true if the given ISO timestamp is stale relative to now.
 * @param {string|null|undefined} lastUpdated  ISO 8601 timestamp
 * @param {Date} [now]  override for testing
 * @returns {boolean}
 */
export function isStale(lastUpdated, now = new Date()) {
  if (!lastUpdated) return true

  const updated = new Date(lastUpdated)
  if (isNaN(updated.getTime())) return true

  const ageMs = now.getTime() - updated.getTime()
  const maxAge = (MAX_AGE_HOURS[now.getDay()] || 18) * HOUR

  return ageMs > maxAge
}
