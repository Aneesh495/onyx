export function updateMemoryGauge(elementId, activeBytes, reservedBytes) {
  const elem = document.getElementById(elementId);
  if (!elem) return;

  const pct = reservedBytes > 0 ? ((activeBytes / reservedBytes) * 100).toFixed(1) : 0;
  elem.style.width = `${pct}%`;

  const label = document.getElementById("mem-usage-label");
  if (label) {
    const actMb = (activeBytes / (1024 * 1024)).toFixed(1);
    const resMb = (reservedBytes / (1024 * 1024)).toFixed(1);
    label.textContent = `${actMb} MB / ${resMb} MB (${pct}%)`;
  }
}
