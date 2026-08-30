/* Gjenbrukbare render-hjelpere: tall, vektorer, matriser, stolper, disclosure. */
(function () {
  const ML = globalThis.ML;

  // Pent minustegn og fast antall desimaler.
  function fmt(x, dec = 2) {
    if (!isFinite(x)) return "?";
    const v = Math.abs(x) < 0.5 * Math.pow(10, -dec) ? 0 : x;
    return (v < 0 ? "−" : "") + Math.abs(v).toFixed(dec);
  }
  function pct(p, dec = 1) { return (p * 100).toFixed(dec) + " %"; }

  // [0.70, −0.40] med farge etter rolle: data (blå) / param (oransje) / grad (rød)
  function vecHTML(v, cls = "data", dec = 2) {
    const inner = v.map((x) => fmt(x, dec)).join(", ");
    return `<span class="vec ${cls}">[${inner}]</span>`;
  }

  // Matrise med klammer og valgfri etikett under.
  function matHTML(M, cls = "param", label = "", dec = 2) {
    const rows = M.map(
      (r) => `<tr>${r.map((x) => `<td>${fmt(x, dec)}</td>`).join("")}</tr>`
    ).join("");
    const table = `<span class="brack ${cls}"><table class="mat ${cls}">${rows}</table></span>`;
    if (!label) return table;
    return `<span class="matwrap">${table}<span class="matlabel">${label}</span></span>`;
  }

  // Utskrevet prikkprodukt: "0.70·0.31 + (−0.40)·(−0.12) = 0.26"
  function dotCalc(a, b, dec = 2) {
    const terms = a.map((x, i) => {
      const xa = fmt(x, dec), xb = fmt(b[i], dec);
      const wa = x < 0 ? `(${xa})` : xa;
      const wb = b[i] < 0 ? `(${xb})` : xb;
      return `${wa}·${wb}`;
    });
    const res = a.reduce((s, x, i) => s + x * b[i], 0);
    return `${terms.join(" + ")} = <span class="res">${fmt(res, dec)}</span>`;
  }

  // x·W kolonne for kolonne, som liste av mathlines.
  function vecMatCalc(x, W, outName = "resultat", dec = 2) {
    const cols = W[0].length;
    let html = "";
    for (let j = 0; j < cols; j++) {
      const col = W.map((r) => r[j]);
      html += `<div class="mathline">${outName}[${j}] = ${dotCalc(x, col, dec)}</div>`;
    }
    return html;
  }

  function tokenChip(word, id = null, extra = "") {
    const tid = id === null ? "" : `<span class="tid">#${id}</span>`;
    return `<span class="chip ${extra}">${word}${tid}</span>`;
  }

  // Sannsynlighetsstolper for hele vokabularet.
  // opts: {correctIdx, sampledIdx, showLogits, logits}
  function probBars(probs, opts = {}) {
    const rows = ML.VOCAB.map((w, i) => {
      const cls = [
        i === opts.correctIdx ? "correct" : "",
        i === opts.sampledIdx ? "sampled" : "",
      ].join(" ");
      const extra = opts.logits ? `<span class="num" style="color:var(--muted)">logit ${fmt(opts.logits[i])}</span> · ` : "";
      return `<div class="pb-row ${cls}">
        <div class="pb-label">${w}</div>
        <div class="pb-bar"><div class="pb-fill" style="width:${Math.max(probs[i] * 100, 0.5)}%"></div></div>
        <div class="pb-val">${extra}${pct(probs[i])}</div>
      </div>`;
    }).join("");
    return `<div class="probbars">${rows}</div>`;
  }

  // Progressiv avsløring: knapp som åpner en boks. Håndteres av delegert klikk i main.js.
  let discSeq = 0;
  function disclosure(label, bodyHTML) {
    const id = "disc-" + ++discSeq;
    return `<div class="disc">
      <button class="disc-btn" data-disc="${id}">${label}</button>
      <div class="disc-body" id="${id}">${bodyHTML}</div>
    </div>`;
  }

  // Liten SVG-sparkline over historikk (0..1-verdier).
  function sparkline(values, opts = {}) {
    const w = 560, h = 110, pad = 6;
    if (values.length < 2) return `<div class="spark-cap">Tren flere steg for å se utviklingen …</div>`;
    const max = opts.max ?? Math.max(...values, 1e-9);
    const min = opts.min ?? 0;
    const px = (i) => pad + (i / (values.length - 1)) * (w - 2 * pad);
    const py = (v) => h - pad - ((v - min) / (max - min || 1)) * (h - 2 * pad);
    const pts = values.map((v, i) => `${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(" ");
    const color = opts.color || "var(--data)";
    const lastY = py(values[values.length - 1]);
    return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
      <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linejoin="round"/>
      <circle cx="${px(values.length - 1)}" cy="${lastY}" r="4" fill="${color}"/>
    </svg>`;
  }

  // Sammenleggbart ærlighetsnotat – én stille linje i stedet for en boks.
  function honest(text) {
    return `<details class="honest"><summary>🌍 I en ekte LLM …</summary><p>${text}</p></details>`;
  }

  ML.R = { fmt, pct, vecHTML, matHTML, dotCalc, vecMatCalc, tokenChip, probBars, disclosure, sparkline, honest };
})();
