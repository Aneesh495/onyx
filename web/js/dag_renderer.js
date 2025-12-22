export function renderDAG(container, nodes) {
  if (!container) return;
  container.innerHTML = "";

  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "100%");
  svg.setAttribute("viewBox", "0 0 600 200");

  const defs = document.createElementNS(svgNS, "defs");
  const marker = document.createElementNS(svgNS, "marker");
  marker.setAttribute("id", "arrow");
  marker.setAttribute("viewBox", "0 0 10 10");
  marker.setAttribute("refX", "5");
  marker.setAttribute("refY", "5");
  marker.setAttribute("markerWidth", "6");
  marker.setAttribute("markerHeight", "6");
  marker.setAttribute("orient", "auto-start-reverse");

  const path = document.createElementNS(svgNS, "path");
  path.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
  path.setAttribute("fill", "#00d2ff");
  marker.appendChild(path);
  defs.appendChild(marker);
  svg.appendChild(defs);

  const defaultNodes = nodes || [
    { id: "scan", name: "GPU Coalesced Scan", x: 40, y: 70 },
    { id: "filter", name: "Warp Filter", x: 220, y: 70 },
    { id: "agg", name: "Hierarchical Aggregate", x: 400, y: 70 }
  ];

  for (let i = 0; i < defaultNodes.length - 1; ++i) {
    const line = document.createElementNS(svgNS, "line");
    line.setAttribute("x1", defaultNodes[i].x + 120);
    line.setAttribute("y1", defaultNodes[i].y + 20);
    line.setAttribute("x2", defaultNodes[i + 1].x);
    line.setAttribute("y2", defaultNodes[i + 1].y + 20);
    line.setAttribute("stroke", "#00d2ff");
    line.setAttribute("stroke-width", "2");
    line.setAttribute("marker-end", "url(#arrow)");
    svg.appendChild(line);
  }

  defaultNodes.forEach(node => {
    const g = document.createElementNS(svgNS, "g");
    const rect = document.createElementNS(svgNS, "rect");
    rect.setAttribute("x", node.x);
    rect.setAttribute("y", node.y);
    rect.setAttribute("width", 120);
    rect.setAttribute("height", 40);
    rect.setAttribute("rx", "6");
    rect.setAttribute("fill", "#1c222d");
    rect.setAttribute("stroke", "#2f3847");
    rect.setAttribute("stroke-width", "1.5");

    const text = document.createElementNS(svgNS, "text");
    text.setAttribute("x", node.x + 60);
    text.setAttribute("y", node.y + 24);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("fill", "#f0f3f6");
    text.setAttribute("font-size", "10");
    text.setAttribute("font-family", "JetBrains Mono, monospace");
    text.textContent = node.name;

    g.appendChild(rect);
    g.appendChild(text);
    svg.appendChild(g);
  });

  container.appendChild(svg);
}
