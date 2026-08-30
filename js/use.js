/* Bruksmodus: samme stegformat som treningen – fire små steg,
   der det siste er en lekeplass man blir værende i.
   Parameterne er frosset; data flyter bare fremover. */
(function () {
  const ML = globalThis.ML;
  const R = () => ML.R;
  const W = (i) => ML.VOCAB[i];
  const lv = (n) => ML.state.level >= n;

  function ensureState() {
    if (!ML.state.use)
      ML.state.use = { ctx: [0, 1], temp: 1.0, sampled: null, draw: null, open: {} };
    return ML.state.use;
  }

  function pipeCard(id, title, quick, bodyHTML) {
    const u = ensureState();
    return `<div class="pipe-card ${u.open[id] ? "open" : ""}">
      <button class="pipe-head" data-pipehead="${id}">
        <span class="caret">${u.open[id] ? "▼" : "▶"}</span> ${title}
        <span class="quick">${quick}</span>
      </button>
      <div class="pipe-body">${bodyHTML}</div>
    </div>`;
  }

  function tempRow(idPrefix) {
    const u = ensureState();
    return `<div class="temp-row">
      <span class="note">0.1<br>forsiktig</span>
      <input type="range" id="${idPrefix}-slider" min="0.1" max="3" step="0.05" value="${u.temp}">
      <span class="note" style="text-align:right">3.0<br>vill</span>
      <span class="temp-val" id="${idPrefix}-val">τ = ${u.temp.toFixed(2)}</span>
    </div>`;
  }

  function wireTemp(host, idPrefix, logits, onUpdate) {
    const u = ensureState();
    const slider = host.querySelector(`#${idPrefix}-slider`);
    if (!slider) return;
    slider.addEventListener("input", () => {
      u.temp = Number(slider.value);
      host.querySelector(`#${idPrefix}-val`).textContent = `τ = ${u.temp.toFixed(2)}`;
      onUpdate(ML.softmax(logits, u.temp));
    });
  }

  function tempHint(t) {
    if (t < 0.5) return "Lav temperatur: fordelingen spisses – modellen velger nesten alltid favoritten. Forutsigbart, men kjedelig.";
    if (t <= 1.3) return "Rundt 1.0: fordelingen brukes omtrent som den er.";
    return "Høy temperatur: fordelingen flates ut – også usannsynlige ord får sjansen. Kreativt, men risikabelt.";
  }

  function sampleExplain(probs, r) {
    let acc = 0;
    const rows = probs.map((p, i) => {
      const from = acc; acc += p;
      const hit = r >= from && r < acc;
      return `<div class="mathline" ${hit ? 'style="font-weight:700"' : ""}>${W(i)}: eier intervallet ${R().fmt(from, 3)} – ${R().fmt(acc, 3)} ${hit ? "← r landet her!" : ""}</div>`;
    }).join("");
    return `<p class="note">Hvert ord eier en bit av tallinja 0–1, like stor som sannsynligheten sin.
      Vi trakk r = <b>${R().fmt(r, 3)}</b>:</p>${rows}`;
  }

  /* ---------------- Stegene ---------------- */

  const steps = [
    {
      id: "frozen",
      title: "Modellen er frosset – nå skal den brukes",
      diagram: [],
      render() {
        const m = ML.state.model;
        const trained = ML.state.history.length > 1;
        return `
          <div class="frozen-banner">
            <span class="big">🔒</span>
            <div>PARAMETERE FROSSET<br>
            <span style="font-weight:400;font-size:13.5px">Modellen er ferdig trent (${m.steps} steg).
            Fra nå av leses parameterne bare – ingenting endres, uansett hvor mye vi bruker den.</span></div>
          </div>
          ${trained ? "" : `<p style="padding:10px 14px;border-radius:10px;background:#fffaf0;border:1.5px solid #e7c78a">
            ⚠️ <b>Du har ikke trent modellen ennå</b> – den vil gjette i blinde. Lov å prøve likevel,
            men treningsmodusen er et bedre sted å starte.</p>`}
          <p class="lede">Å <i>bruke</i> modellen – det som skjer når du chatter med en LLM – er
          nøyaktig samme fremoverpass som under treningen. Forskjellen er hva som skjer etterpå:</p>
          <div class="compare">
            <div class="cmp-col train">
              <h4>Trening</h4>
              <ol>
                <li>fremoverpass → prediksjon</li>
                <li>sammenlign med fasit → <b>loss</b></li>
                <li><b>bakoverpass</b> → gradienter</li>
                <li>🟠 <b>parameterne ENDRES</b></li>
              </ol>
            </div>
            <div class="cmp-col use">
              <h4>Bruk</h4>
              <ol>
                <li>fremoverpass → prediksjon</li>
                <li>softmax → sannsynligheter</li>
                <li><b>sampling</b> → velg neste token</li>
                <li>🔒 <b>ingen parametere endres</b></li>
              </ol>
            </div>
          </div>`;
      },
    },

    {
      id: "forward",
      title: "Samme maskin, samme fremoverpass",
      diagram: ["dg-tokens", "dg-emb", "dg-layer", "dg-attn", "dg-ffn", "dg-logits"],
      render() {
        const u = ensureState();
        const m = ML.state.model;
        const tr = m.forward(u.ctx);
        const last = u.ctx.length - 1;
        const A = tr.alphas[last];
        return `
          <p class="lede">Input er <b>«${u.ctx.map((t) => W(t)).join(" ")}»</b> (den endrer du i
          lekeplassen på siste steg). Her er hele rørledningen – klikk deg inn der du vil se tallene:</p>
          ${pipeCard("tok", "1 · Tokens", u.ctx.map((t) => `${W(t)}#${t}`).join(" "), `
            <div class="chiprow">${u.ctx.map((t) => R().tokenChip(W(t), t)).join(" ")}</div>
            <p class="note">Deterministisk oppslag – ingen parametere.</p>`)}
          ${pipeCard("emb", "2 · Embeddinger + posisjon", `${u.ctx.length} × ${ML.D} tall`, `
            ${u.ctx.map((t, i) => `<div class="mathline">x<sub>${W(t)}</sub> = E[${W(t)}] + P[${i}] =
              ${R().vecHTML(m.E[t], "param")} + ${R().vecHTML(m.P[i], "param")} = ${R().vecHTML(tr.x[i], "data")}</div>`).join("")}
            <p class="note">🟠 De oransje verdiene er nå <b>lærte</b> – dette er det treningen skapte.</p>`)}
          ${pipeCard("attn", "3 · Attention", `«${W(u.ctx[last])}» ser mest på «${W(u.ctx[A.indexOf(Math.max(...A))])}»`, `
            <p>Siste token henter informasjon fra hele konteksten:</p>
            <div class="attn-weights">
              ${u.ctx.map((t, j) => `
                <div class="aw-row">
                  <div><b>${W(t)}</b></div>
                  <div class="aw-bar"><div class="aw-fill" style="width:${A[j] * 100}%"></div></div>
                  <div class="aw-pct">${R().pct(A[j])}</div>
                </div>`).join("")}
            </div>
            <div class="mathline">z = ${u.ctx.map((t, j) => `${R().pct(A[j], 0)}·v<sub>${W(t)}</sub>`).join(" + ")} = ${R().vecHTML(tr.z[last], "data")}</div>
            <div class="mathline">h = x + z·W_O = ${R().vecHTML(tr.h[last], "data")}</div>`)}
          ${pipeCard("ffn", "4 · Feed forward", `→ y = ${tr.y[last].map((v) => R().fmt(v)).join(", ")}`, `
            <div class="mathline">f = ReLU(h·W₁)·W₂ = ${R().vecHTML(tr.f[last], "data")}</div>
            <div class="mathline">y = h + f = ${R().vecHTML(tr.y[last], "data")}</div>`)}
          ${pipeCard("logits", "5 · Logits", tr.logits.map((l) => R().fmt(l, 1)).join(" · "), `
            <div class="mathline">logits = y · W_U</div>
            ${ML.VOCAB.map((w, i) => `<div class="mathline">logit(${w}) = ${lv(2) ? R().dotCalc(tr.y[last], m.Wu.map((r) => r[i])) : `<span class="res">${R().fmt(tr.logits[i])}</span>`}</div>`).join("")}`)}
          <p class="note">Legg merke til hva som <i>ikke</i> er her: ingen loss, ingen gradienter,
          ingen oppdatering. Bare regning fremover.</p>`;
      },
      wire(host) {
        const u = ensureState();
        host.querySelectorAll("[data-pipehead]").forEach((b) =>
          b.addEventListener("click", () => {
            u.open[b.dataset.pipehead] = !u.open[b.dataset.pipehead];
            renderUse(false);
          })
        );
      },
    },

    {
      id: "temp",
      title: "Temperatur – skru på personligheten",
      diagram: ["dg-softmax"],
      render() {
        const u = ensureState();
        const tr = ML.state.model.forward(u.ctx);
        const probs = ML.softmax(tr.logits, u.temp);
        return `
          <p class="lede">Før softmax deles alle logitene på <b>temperaturen</b>. Rekkefølgen endres
          aldri – bare hvor <i>bastant</i> fordelingen blir. Dra og se:</p>
          ${tempRow("temp")}
          <div id="temp-bars">${R().probBars(probs)}</div>
          <p class="note" id="temp-hint">${tempHint(u.temp)}</p>
          ${lv(1) ? `<div class="mathline">P(ord) = e^(logit/τ) / sum av alle e^(logit/τ)</div>` : ""}
          ${R().honest(`Dette er nøyaktig samme temperatur-parameter som i API-et til ChatGPT/Claude.
            Lav τ til fakta og kode, høyere τ til idémyldring.`)}`;
      },
      wire(host) {
        const u = ensureState();
        const tr = ML.state.model.forward(u.ctx);
        wireTemp(host, "temp", tr.logits, (probs) => {
          host.querySelector("#temp-bars").innerHTML = R().probBars(probs);
          host.querySelector("#temp-hint").textContent = tempHint(u.temp);
        });
      },
    },

    {
      id: "play",
      title: "Lekeplassen – la modellen skrive 🎲",
      diagram: ["dg-sampling", "dg-next"],
      render() {
        const u = ensureState();
        const tr = ML.state.model.forward(u.ctx);
        const probs = ML.softmax(tr.logits, u.temp);
        return `
          <p class="lede">Modellen velger ikke automatisk det mest sannsynlige ordet – den
          <b>trekker</b> fra fordelingen, som et lykkehjul der hvert ord eier en bit så stor som
          sannsynligheten sin. Derfor kan samme spørsmål gi ulike svar.</p>

          <p><b>Teksten så langt</b> (maks ${ML.MAXPOS} tokens):</p>
          <div class="chiprow">
            ${u.ctx.map((t) => R().tokenChip(W(t), t)).join(" ")}
            ${u.sampled !== null ? `<span class="chip win">${W(u.sampled)}</span>` : ""}
            ${u.ctx.length > 1 ? `<button class="btn ghost" id="ctx-pop" title="Fjern siste token">⌫</button>` : ""}
            <button class="btn ghost" id="ctx-reset">↺ «katten drikker»</button>
          </div>
          <div class="chiprow">
            <span class="note">Bytt ut / bygg selv:</span>
            ${ML.VOCAB.map((w, i) => `<button class="btn" data-add="${i}" ${u.ctx.length >= ML.MAXPOS ? "disabled" : ""}>+ ${w}</button>`).join(" ")}
          </div>

          ${tempRow("play")}
          <div id="play-bars">${R().probBars(probs, { sampledIdx: u.sampled })}</div>

          <div class="sample-stage">
            <button class="btn accent big" id="sample-btn">🎲 Trekk neste token</button>
            ${u.sampled !== null ? `
              <div class="sample-result">
                <span class="note">r = ${R().fmt(u.draw, 3)} →</span>
                <span class="sample-token">${W(u.sampled)}</span>
              </div>` : ""}
          </div>
          ${u.sampled !== null ? `
            <div class="chiprow" style="margin-top:12px">
              ${u.ctx.length < ML.MAXPOS
                ? `<button class="btn primary" id="accept-btn">Legg til i teksten og fortsett →</button>`
                : `<span class="note">Kontekstvinduet på ${ML.MAXPOS} tokens er fullt – ekte modeller har samme grense, bare mye større.</span>`}
            </div>
            ${lv(1) ? sampleExplain(probs, u.draw) : ""}` : ""}`;
      },
      wire(host) {
        const u = ensureState();
        const tr = ML.state.model.forward(u.ctx);
        const probs = () => ML.softmax(tr.logits, u.temp);
        wireTemp(host, "play", tr.logits, (p) => {
          host.querySelector("#play-bars").innerHTML = R().probBars(p, { sampledIdx: u.sampled });
        });
        host.querySelectorAll("[data-add]").forEach((b) =>
          b.addEventListener("click", () => {
            u.ctx.push(Number(b.dataset.add));
            u.sampled = null;
            renderUse(false);
          })
        );
        const pop = host.querySelector("#ctx-pop");
        if (pop) pop.addEventListener("click", () => { u.ctx.pop(); u.sampled = null; renderUse(false); });
        host.querySelector("#ctx-reset").addEventListener("click", () => {
          u.ctx = [0, 1]; u.sampled = null; renderUse(false);
        });
        host.querySelector("#sample-btn").addEventListener("click", () => {
          const p = probs();
          const r = Math.random();
          let acc = 0, pick = p.length - 1;
          for (let i = 0; i < p.length; i++) {
            acc += p[i];
            if (r < acc) { pick = i; break; }
          }
          u.sampled = pick;
          u.draw = r;
          renderUse(false);
        });
        const accept = host.querySelector("#accept-btn");
        if (accept)
          accept.addEventListener("click", () => {
            u.ctx.push(u.sampled);
            u.sampled = null;
            renderUse(false);
          });
      },
    },
  ];

  /* ---------------- Rammeverk rundt stegene ---------------- */

  function renderUse(scroll = true) {
    ensureState();
    const host = document.getElementById("content");
    const i = ML.state.useStepIdx ?? 0;
    const step = steps[i];
    host.innerHTML = `
      <div class="card">
        <div class="step-kicker">Bruksmodus · steg ${i + 1} av ${steps.length}</div>
        <h2>${step.title}</h2>
        ${step.render()}
        <div class="stepnav">
          <button class="btn" id="nav-prev" ${i === 0 ? "disabled" : ""}>← Tilbake</button>
          <div class="spacer"></div>
          <div class="dots">${steps.map((_, j) => `<span class="dot ${j < i ? "done" : j === i ? "now" : ""}"></span>`).join("")}</div>
          <div class="spacer"></div>
          <button class="btn primary" id="nav-next" ${i === steps.length - 1 ? "disabled" : ""}>Neste →</button>
        </div>
      </div>`;
    ML.diagram.highlight(step.diagram);
    ML.diagram.setBackward(false);
    if (step.wire) step.wire(host);
    host.querySelector("#nav-prev").addEventListener("click", () => {
      ML.state.useStepIdx = Math.max(0, i - 1);
      renderUse();
    });
    host.querySelector("#nav-next").addEventListener("click", () => {
      ML.state.useStepIdx = Math.min(steps.length - 1, i + 1);
      renderUse();
    });
    if (scroll) window.scrollTo({ top: 0, behavior: "smooth" });
  }

  ML.renderUse = renderUse;
})();
