/* StellarClass AI - front-end behaviour (charts and form helpers).
   Every chart is drawn from data served by the Flask app: a sample of the
   real dataset or the metrics saved by the training pipeline. */
(function () {
  "use strict";

  const COLOURS = { STAR: "#e3a948", GALAXY: "#6e9be6", grid: "#24315a", text: "#97a3c0", accent: "#7480f0" };
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (window.Chart) {
    Chart.defaults.color = COLOURS.text;
    Chart.defaults.borderColor = COLOURS.grid;
    Chart.defaults.font.family = '"IBM Plex Sans", system-ui, sans-serif';
    Chart.defaults.animation = reduceMotion ? false : { duration: 500 };
    Chart.defaults.maintainAspectRatio = false;
  }

  const label = (name) => name.replace("_", " − ");

  async function fetchSample(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error("Could not load the data sample (" + res.status + ")");
    const data = await res.json();
    const idx = Object.fromEntries(data.columns.map((c, i) => [c, i]));
    return { idx, rows: data.rows };
  }

  function quantile(sorted, q) {
    const pos = (sorted.length - 1) * q;
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }

  function range(values, lowQ = 0.005, highQ = 0.995) {
    const s = values.slice().sort((a, b) => a - b);
    return [quantile(s, lowQ), quantile(s, highQ)];
  }

  function showChartError(canvasId, message) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const p = document.createElement("p");
    p.className = "note";
    p.textContent = message;
    canvas.parentElement.replaceWith(p);
  }

  function scatterDatasets(sample, xKey, yKey, perClassLimit) {
    return ["GALAXY", "STAR"].map((cls) => {
      const points = [];
      for (const row of sample.rows) {
        if (row[sample.idx["class"]] !== cls) continue;
        points.push({ x: row[sample.idx[xKey]], y: row[sample.idx[yKey]] });
        if (perClassLimit && points.length >= perClassLimit) break;
      }
      return {
        label: cls === "STAR" ? "Star" : "Galaxy",
        data: points,
        backgroundColor: COLOURS[cls] + "99",
        pointRadius: 2,
        pointHoverRadius: 4,
      };
    });
  }

  function scatterOptions(xKey, yKey, xRange, yRange) {
    return {
      parsing: false,
      plugins: {
        legend: { position: "top", align: "end", labels: { boxWidth: 10, usePointStyle: true } },
        tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${label(xKey)} ${c.parsed.x.toFixed(3)}, ${label(yKey)} ${c.parsed.y.toFixed(3)}` } },
      },
      scales: {
        x: { min: xRange[0], max: xRange[1], title: { display: true, text: label(xKey) + (xKey === "redshift" ? "" : " (mag)") } },
        y: { min: yRange[0], max: yRange[1], title: { display: true, text: label(yKey) + (yKey === "redshift" ? "" : " (mag)") } },
      },
    };
  }

  // Overview: colour-colour diagram ------------------------------------
  async function heroChart(canvasId, url) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !window.Chart) return;
    try {
      const sample = await fetchSample(url);
      new Chart(canvas, {
        type: "scatter",
        data: { datasets: scatterDatasets(sample, "u_g", "g_r") },
        options: scatterOptions("u_g", "g_r", [-0.5, 4.5], [-0.5, 2.5]),
      });
    } catch (err) {
      showChartError(canvasId, "The diagram needs the processed dataset. Run python -m src.train, then reload.");
    }
  }

  // Data explorer -------------------------------------------------------
  async function explorer(opts) {
    if (!window.Chart) return;
    new Chart(document.getElementById("class-chart"), {
      type: "bar",
      data: {
        labels: ["Galaxy", "Star"],
        datasets: [{ label: "Objects", data: [opts.counts.GALAXY, opts.counts.STAR], backgroundColor: [COLOURS.GALAXY, COLOURS.STAR] }],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: { y: { title: { display: true, text: "Number of objects" }, beginAtZero: true }, x: { title: { display: true, text: "Class" } } },
      },
    });

    let sample;
    try {
      sample = await fetchSample(opts.sampleUrl);
    } catch (err) {
      showChartError("hist-chart", err.message);
      showChartError("scatter-chart", err.message);
      return;
    }

    const histSelect = document.getElementById("hist-feature");
    const histDesc = document.getElementById("hist-desc");
    let histChart;
    function drawHistogram() {
      const key = histSelect.value;
      const all = sample.rows.map((r) => r[sample.idx[key]]);
      const [lo, hi] = range(all);
      const bins = 40, width = (hi - lo) / bins || 1;
      const edges = Array.from({ length: bins }, (_, i) => lo + i * width);
      const datasets = ["GALAXY", "STAR"].map((cls) => {
        const counts = new Array(bins).fill(0);
        let n = 0;
        for (const row of sample.rows) {
          if (row[sample.idx["class"]] !== cls) continue;
          const v = row[sample.idx[key]];
          if (v < lo || v > hi) continue;
          counts[Math.min(bins - 1, Math.floor((v - lo) / width))] += 1;
          n += 1;
        }
        return {
          label: cls === "STAR" ? "Star" : "Galaxy",
          data: counts.map((c) => (n ? c / n : 0)),
          backgroundColor: COLOURS[cls] + "88",
          borderColor: COLOURS[cls],
          borderWidth: 1,
          barPercentage: 1,
          categoryPercentage: 1,
          grouped: false,
        };
      });
      const config = {
        type: "bar",
        data: { labels: edges.map((e) => (e + width / 2).toFixed(key === "redshift" ? 3 : 2)), datasets },
        options: {
          plugins: { legend: { position: "top", align: "end", labels: { boxWidth: 10 } } },
          scales: {
            x: { title: { display: true, text: label(key) }, ticks: { maxTicksLimit: 8 } },
            y: { title: { display: true, text: "Share of class" } },
          },
        },
      };
      if (histChart) histChart.destroy();
      histChart = new Chart(document.getElementById("hist-chart"), config);
      histDesc.textContent = opts.descriptions[key] || "";
    }
    histSelect.addEventListener("change", drawHistogram);
    drawHistogram();

    const xSel = document.getElementById("x-feature");
    const ySel = document.getElementById("y-feature");
    let scatter;
    function drawScatter() {
      const xKey = xSel.value, yKey = ySel.value;
      const xr = range(sample.rows.map((r) => r[sample.idx[xKey]]));
      const yr = range(sample.rows.map((r) => r[sample.idx[yKey]]));
      if (scatter) scatter.destroy();
      scatter = new Chart(document.getElementById("scatter-chart"), {
        type: "scatter",
        data: { datasets: scatterDatasets(sample, xKey, yKey) },
        options: scatterOptions(xKey, yKey, xr, yr),
      });
    }
    xSel.addEventListener("change", drawScatter);
    ySel.addEventListener("change", drawScatter);
    drawScatter();
  }

  // Model performance ----------------------------------------------------
  function performance(opts) {
    if (!window.Chart) return;
    const toPoints = (roc) => roc.fpr.map((x, i) => ({ x, y: roc.tpr[i] }));
    new Chart(document.getElementById("roc-chart"), {
      type: "line",
      data: {
        datasets: [
          { label: `Random Forest (AUC ${opts.aucRf.toFixed(4)})`, data: toPoints(opts.roc), borderColor: COLOURS.accent, pointRadius: 0, borderWidth: 2 },
          { label: `Logistic Regression (AUC ${opts.aucLr.toFixed(4)})`, data: toPoints(opts.rocLogreg), borderColor: COLOURS.text, pointRadius: 0, borderWidth: 1.5 },
          { label: "Random guess", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: COLOURS.grid, borderDash: [5, 5], pointRadius: 0, borderWidth: 1 },
        ],
      },
      options: {
        parsing: false,
        plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } } },
        scales: {
          x: { type: "linear", min: 0, max: 1, title: { display: true, text: "False positive rate (galaxies called stars)" } },
          y: { min: 0, max: 1, title: { display: true, text: "True positive rate (stars found)" } },
        },
      },
    });

    function bars(canvasId, rows, xTitle) {
      new Chart(document.getElementById(canvasId), {
        type: "bar",
        data: {
          labels: rows.map((r) => label(r.feature)),
          datasets: [{ label: xTitle, data: rows.map((r) => r.importance), backgroundColor: COLOURS.accent }],
        },
        options: {
          indexAxis: "y",
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: (c) => `${c.parsed.x.toFixed(4)} (± ${rows[c.dataIndex].std.toFixed(4)})` } },
          },
          scales: { x: { title: { display: true, text: xTitle }, beginAtZero: true }, y: { title: { display: true, text: "Feature" } } },
        },
      });
    }
    bars("importance-chart", opts.impurity, "Mean decrease in impurity");
    bars("perm-chart", opts.permutation, "Drop in ROC-AUC when shuffled");
  }

  // Classify form: fill with a real example --------------------------------
  document.querySelectorAll("[data-fill]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const values = JSON.parse(btn.dataset.fill);
      for (const [key, value] of Object.entries(values)) {
        const input = document.getElementById("f-" + key);
        if (input) input.value = value;
      }
      document.getElementById("f-u").focus();
    });
  });

  window.StellarClass = { heroChart, explorer, performance };
})();
