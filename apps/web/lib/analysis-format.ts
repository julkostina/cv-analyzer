export function formatPercent(score: number | null | undefined): string {
  if (score == null || Number.isNaN(score)) return "—";
  return `${Math.round(score * 1000) / 10}%`;
}

/** Whole units 0–100 for a score in 0…1 (no percent symbol). */
export function matchScoreUnits(score: number | null | undefined): number | null {
  if (score == null || Number.isNaN(score)) return null;
  return Math.round(Math.max(0, Math.min(1, score)) * 100);
}

export function triggerBlobDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
