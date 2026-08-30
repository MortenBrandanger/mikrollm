/* Bruksmodus: samme rørledning som i treningen, gjennomgått på nytt som
   åtte små steg – ett konsept per skjerm – med frosne parametere, og
   temperatur + sampling (lekeplassen) til slutt.

   Detaljnivået (ML.state.level) styrer hvor mye tall som vises:
     0 = Enkelt, 1 = Med tall, 2 = Full utregning. */
(function () {
  const ML = globalThis.ML;
  const R = () => ML.R;
  const W = (i) => ML.VOCAB[i];
  const lv = (n) => ML.state.level >= n;

  function ensureState() {
    if (!ML.state.use)
      ML.state.use = { ctx: [0, 1], temp: 1.0, sampled: null, draw: null };
    return ML.state.use;
  }

  const ctxText = () => ensureState().ctx.map((t) => W(t)).join(" ");
  const tr = () => ML.state.model.forward(ensureState().ctx);

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
          <p class="lede">Å <i>bruke</i> modellen – det som skjer når du chatter med en LLM – er samme
          maskin som i treningen. Forskjellen er hva som skjer etter prediksjonen:</p>
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
          </div>
          <p class="note">Vi går gjennom rørledningen en gang til, steg for steg – nå med den ferdig
          trente modellen.</p>`;
      },
    },

    {
      id: "tokens",
      title: "Teksten deles i tokens",
      diagram: ["dg-text", "dg-tokens"],
      render() {
        const u = ensureState();
        return `
          <p class="lede">Teksten <b>«${ctxText()}»</b> deles opp i ${u.ctx.length} tokens:</p>
          <div class="chiprow">
            <span class="chip plain">«${ctxText()}»</span>
            <span class="arrow-inline">→</span>
            ${u.ctx.map((t) => R().tokenChip(W(t), t)).join(" ")}
          </div>
          <p>Nøyaktig som i treningen: deterministisk oppslag, ingen parametere involvert.</p>`;
      },
    },

    {
      id: "emb",
      title: "Tokens blir tall – og nå er tallene lærte",
      diagram: ["dg-emb"],
      render() {
        const u = ensureState();
        const m = ML.state.model;
        const t0 = tr();
        return `
          <p class="lede">Hver token slås opp i embedding-tabellen og blir en vektor. Forskjellen fra
          før treningen: verdiene er ikke lenger tilfeldige – <b>dette er det treningen skapte</b>.</p>
          <div style="margin:12px 0">
            ${u.ctx.map((t) => `
              <div class="chiprow">
                ${R().tokenChip(W(t), t)}
                <span class="arrow-inline">→</span>
                ${R().vecHTML(m.E[t], "param")}
              </div>`).join("")}
          </div>
          ${lv(1) ? `
            <p class="note">Pluss posisjonsvektor, som før:</p>
            ${u.ctx.map((t, i) => `<div class="mathline">x<sub>${W(t)}</sub> = E[${W(t)}] + P[${i}] =
              ${R().vecHTML(m.E[t], "param")} + ${R().vecHTML(m.P[i], "param")} = ${R().vecHTML(t0.x[i], "data")}</div>`).join("")}` : ""}`;
      },
    },

    {
      id: "attn",
      title: "Attention – tokens ser på hverandre",
      diagram: ["dg-layer", "dg-attn"],
      render() {
        const u = ensureState();
        const t0 = tr();
        const last = u.ctx.length - 1;
        const A = t0.alphas[last];
        return `
          <p class="lede">Siste token, «${W(u.ctx[last])}», henter informasjon fra hele teksten – med
          de lærte Q-, K- og V-matrisene:</p>
          <div class="attn-weights">
            ${u.ctx.map((t, j) => `
              <div class="aw-row">
                <div><b>${W(t)}</b></div>
                <div class="aw-bar"><div class="aw-fill" style="width:${A[j] * 100}%"></div></div>
                <div class="aw-pct">${R().pct(A[j])}</div>
              </div>`).join("")}
          </div>
          ${lv(1) ? `
            <div class="mathline">z = ${u.ctx.map((t, j) => `${R().pct(A[j], 0)}·v<sub>${W(t)}</sub>`).join(" + ")} = ${R().vecHTML(t0.z[last], "data")}</div>
            <div class="mathline">h = x + z·W_O = ${R().vecHTML(t0.h[last], "data")}</div>` : ""}
          ${lv(2) ? u.ctx.map((t, j) => `
            <div class="mathline note">score(${W(u.ctx[last])} → ${W(t)}) = q·k/√2 = (${R().dotCalc(t0.q[last], t0.k[j])}) / 1.41 = <span class="res">${R().fmt(t0.scores[last][j])}</span></div>`).join("") : ""}`;
      },
    },

    {
      id: "ffn",
      title: "Feed forward – vektoren bearbeides",
      diagram: ["dg-layer", "dg-ffn"],
      render() {
        const u = ensureState();
        const t0 = tr();
        const last = u.ctx.length - 1;
        return `
          <p class="lede">Samme lille nevrale nettverk som før – gang, ReLU, gang – med de lærte
          matrisene W₁ og W₂:</p>
          ${lv(1) ? `
            <div class="mathline">f = ReLU(h·W₁)·W₂ = ${R().vecHTML(t0.f[last], "data")}</div>
            <div class="mathline">y = h + f = ${R().vecHTML(t0.y[last], "data")}</div>`
          : `
            <div class="ffn-flow">
              <div class="shape-box">2 tall</div><span class="op">· W₁</span>
              <div class="shape-box">4 tall</div><span class="op">ReLU</span>
              <div class="shape-box">4 tall</div><span class="op">· W₂</span>
              <div class="shape-box">2 tall</div>
            </div>
            <p class="note">Resultatet legges til h (residual). Ut kommer y – modellens ferdige
            «mening» om hva som bør komme etter «${ctxText()}».</p>`}
          ${lv(2) ? `
            <p class="note"><b>1)</b> h · W₁:</p>${R().vecMatCalc(t0.h[last], ML.state.model.W1, "pre")}
            <p class="note"><b>2)</b> ReLU → ${R().vecHTML(t0.act[last], "data")} &nbsp; <b>3)</b> act · W₂:</p>
            ${R().vecMatCalc(t0.act[last], ML.state.model.W2, "f")}` : ""}`;
      },
    },

    {
      id: "probs",
      title: "Hva tror modellen kommer nå?",
      diagram: ["dg-logits", "dg-softmax"],
      render() {
        const t0 = tr();
        const probs = ML.softmax(t0.logits, 1);
        return `
          <p class="lede">Vektoren ganges med output-matrisen → én logit per ord → softmax →
          sannsynligheter. Modellens gjetning etter <b>«${ctxText()}»</b>:</p>
          ${R().probBars(probs, { logits: lv(1) ? t0.logits : null })}
          <p class="note">Merk forskjellen fra treningen: her finnes ingen fasit og ingen loss –
          gjetningen ER svaret.</p>
          ${lv(2) ? `
            <div class="mathline">logits = y · W_U</div>
            ${ML.VOCAB.map((w, i) => `<div class="mathline">logit(${w}) = ${R().dotCalc(t0.y[t0.T - 1], ML.state.model.Wu.map((r) => r[i]))}</div>`).join("")}` : ""}`;
      },
    },

    {
      id: "temp",
      title: "Temperatur – skru på personligheten",
      diagram: ["dg-softmax"],
      render() {
        const u = ensureState();
        const t0 = tr();
        const probs = ML.softmax(t0.logits, u.temp);
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
        const t0 = tr();
        const u = ensureState();
        wireTemp(host, "temp", t0.logits, (probs) => {
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
        const t0 = tr();
        const probs = ML.softmax(t0.logits, u.temp);
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
        const t0 = tr();
        wireTemp(host, "play", t0.logits, (p) => {
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
          const p = ML.softmax(t0.logits, u.temp);
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
    const i = Math.min(ML.state.useStepIdx ?? 0, steps.length - 1);
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
