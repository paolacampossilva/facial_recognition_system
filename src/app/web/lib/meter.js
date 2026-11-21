// What the similarity bar shows, with no DOM so it can be tested on its own.

const clamp = (value) => Math.min(1, Math.max(0, value));

export const formatSimilarity = (value) => value.toFixed(3).replace(".", ",");

// `threshold` is null when GET /api/config failed: the bar then has no mark and no limit text.
// `aboveLimit` comes from the API's decision (recognized or unknown), never from comparing here.
export function meterState(value, threshold, aboveLimit) {
  const hasLimit = threshold !== null;
  return {
    fill: clamp(value),
    mark: hasLimit ? clamp(threshold) : null,
    tone: aboveLimit ? "above" : "below",
    limitText: hasLimit ? `${aboveLimit ? "Acima" : "Abaixo"} do limite (${formatSimilarity(threshold)})` : null,
  };
}
