import { renderDAG } from "./dag_renderer.js";
import { updateMemoryGauge } from "./charts.js";
import { OnyxStreamClient } from "./streaming_client.js";

const streamClient = new OnyxStreamClient();

async function fetchMetrics() {
  try {
    const res = await fetch("/metrics");
    if (!res.ok) return;
    const data = await res.json();

    if (data.gpu_memory) {
      updateMemoryGauge(
        "mem-bar",
        data.gpu_memory.active_bytes,
        data.gpu_memory.reserved_bytes
      );
      document.getElementById("stat-allocs").textContent = data.gpu_memory.total_allocations;
      document.getElementById("stat-frag").textContent = (data.gpu_memory.fragmentation_ratio * 100).toFixed(1) + "%";
    }

    if (data.engine) {
      document.getElementById("stat-engine-mode").textContent = data.engine.cuda_enabled ? "NVIDIA Native" : "SIMT CPU Emulator";
    }
  } catch (err) {
    // server might be offline during development
  }
}

async function fetchTables() {
  try {
    const res = await fetch("/tables");
    if (!res.ok) return;
    const data = await res.json();
    const container = document.getElementById("table-list-container");
    if (!container || !data.tables) return;

    container.innerHTML = "";
    data.tables.forEach(tbl => {
      const card = document.createElement("div");
      card.className = "table-card";
      card.innerHTML = `
        <div class="table-name">${tbl.name}</div>
        <div class="table-meta">${tbl.rows} rows | ${tbl.batches} chunks</div>
      `;
      card.onclick = () => {
        document.getElementById("query-input").value = `SELECT * FROM ${tbl.name} LIMIT 10;`;
      };
      container.appendChild(card);
    });
  } catch (err) {}
}

async function runQuery() {
  const query = document.getElementById("query-input").value;
  const resultView = document.getElementById("results-output");
  resultView.textContent = "Executing query on GPU kernel...";

  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query })
    });
    const data = await res.json();
    resultView.textContent = JSON.stringify(data, null, 2);

    document.getElementById("stat-latency").textContent = `${data.latency_ms.toFixed(2)} ms`;
    document.getElementById("stat-kernel-time").textContent = `${data.kernel_us.toFixed(1)} us`;
  } catch (err) {
    resultView.textContent = "Failed to communicate with query server: " + err.message;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const dagContainer = document.getElementById("dag-container");
  renderDAG(dagContainer);

  document.getElementById("run-btn").addEventListener("click", runQuery);

  fetchMetrics();
  fetchTables();

  streamClient.subscribe((data) => {
    if (data.gpu_memory) {
      updateMemoryGauge(
        "mem-bar",
        data.gpu_memory.active_bytes,
        data.gpu_memory.reserved_bytes
      );
      document.getElementById("stat-allocs").textContent = data.gpu_memory.total_allocations;
      document.getElementById("stat-frag").textContent = (data.gpu_memory.fragmentation_ratio * 100).toFixed(1) + "%";
    }
  });
  streamClient.connect();
});
