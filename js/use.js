/* Bruksmodus: samme rørledning som i treningen, gjennomgått på nytt som
   små steg med frosne parametere – og til slutt temperatur, sampling,
   «det store bildet» og lekeplassen.

   Tekster er betinget av om modellen faktisk er trent (m.steps),
   siden modellen persisteres og kan være i begge tilstander. */
(function () {
  const ML = globalThis.ML;
  const R = () => ML.R;
  const W = (i) => ML.VOCAB[i];
  const lv = (n) => ML.state.level >= n;
  const isTrained = () => ML.state.model.steps > 0;
  const calc = (html) => (lv(2) ? R().disclosure("🧮 Hele utregningen", html) : "");

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

  function sampleFrom(probs) {
    const r = Math.random();
    let acc = 0, pick = probs.length - 1;
    for (let i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (r < acc) { pick = i; break; }
    }
    return { pick, r };
  }

  /* ---------------- Stegene ---------------- */

  const steps = [
    {
      id: "frozen",
      title: "Modellen er frosset",
      levels: false,
      sub: "fra nå av endres ingenting",
      diagram: [],
      render() {
        const m = ML.state.model;
        return `
          <div class="frozen-banner">
            <span class="big">🔒</span>
            <div>PARAMETERE FROSSET<br>
            <span style="font-weight:400;font-size:13.5px">${isTrained()
              ? `Modellen er ferdig trent (${m.steps} steg). `
              : ""}Fra nå av leses parameterne bare – ingenting endres, uansett hvor mye vi bruker
            den.</span></div>
          </div>
          <p class="lede">Å <i>bruke</i> modellen – det som skjer når du chatter med en LLM – er den
          samme maskinen som i treningen, men uten læringen. Vi går gjennom rørledningen en gang til,
          steg for steg${isTrained() ? " – nå med den ferdig trente modellen" : ""}.</p>`;
      },
    },

    {
      id: "tokens",
      title: "Tokenisering",
      sub: "samme oppslag som før",
      diagram: ["dg-tokens"],
      render() {
        const u = ensureState();
        return `
          <p class="lede">Teksten <b>«${ctxText()}»</b> deles opp i ${u.ctx.length} tokens:</p>
          <div class="chiprow">
            <span class="chip plain">«${ctxText()}»</span>
            <span class="arrow-inline">→</span>
            ${u.ctx.map((t) => R().tokenChip(W(t), t)).join(" ")}
          </div>
          <p>Nøyaktig som i treningen: deterministisk oppslag, ingen parametere involvert.</p>
          ${lv(1) ? `
            <p class="note">Hele vokabularet med numrene sine:</p>
            <div class="chiprow">${ML.VOCAB.map((w, i) => R().tokenChip(w, i)).join(" ")}</div>` : ""}`;
      },
    },

    {
      id: "emb",
      title: "Embeddinger",
      sub: () => (isTrained() ? "nå med lærte tall" : "tokens blir tall"),
      diagram: ["dg-emb"],
      render() {
        const u = ensureState();
        const m = ML.state.model;
        const t0 = tr();
        return `
          <p class="lede">Hver token slås opp i embedding-tabellen og blir en vektor.
          ${isTrained()
            ? `Verdiene er ikke lenger tilfeldige – <b>dette er det treningen skapte</b>.`
            : `(Modellen er utrent, så disse er fortsatt de tilfeldige startverdiene.)`}</p>
          <div style="margin:12px 0">
            ${u.ctx.map((t) => `
              <div class="chiprow">
                ${R().tokenChip(W(t), t)}
                <span class="arrow-inline">→</span>
                ${R().vecHTML(m.E[t], "param")}
              </div>`).join("")}
          </div>
          ${lv(1) ? `
            <p class="note">Og som i treningen legges posisjonsvektoren til, så modellen vet
            rekkefølgen:</p>
            ${u.ctx.map((t, i) => R().sumRow([
              { vec: m.E[t], cls: "param", cap: `embedding for «${W(t)}» (lært)` },
              "+",
              { vec: m.P[i], cls: "param", cap: `posisjon: plass ${i} (lært)` },
              "=",
              { vec: t0.x[i], cls: "data", cap: `«${W(t)}» slik den flyter videre` },
            ])).join("")}` : ""}`;
      },
    },

    {
      id: "attn",
      title: "Attention",
      sub: "hvem ser siste token på?",
      diagram: ["dg-attn"],
      render() {
        const u = ensureState();
        const t0 = tr();
        const last = u.ctx.length - 1;
        const A = t0.alphas[last];
        return `
          <p class="lede">Siste token, «${W(u.ctx[last])}», henter informasjon fra hele teksten med
          Q-, K- og V-matrisene${isTrained() ? " – og vektene er nå <b>lærte</b>, ikke tilfeldige" : ""}:</p>
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
          ${calc(u.ctx.map((t, j) => `
            <div class="mathline">score(${W(u.ctx[last])} → ${W(t)}) = q·k/√2 = (${R().dotCalc(t0.q[last], t0.k[j])}) / 1.41 = <span class="res">${R().fmt(t0.scores[last][j])}</span></div>`).join(""))}`;
      },
    },

    {
      id: "ffn",
      title: "Feed forward",
      sub: "vektoren bearbeides",
      diagram: ["dg-ffn"],
      render() {
        const t0 = tr();
        const last = ensureState().ctx.length - 1;
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
            </div>`}
          <p>Ut kommer y – modellens ferdige «mening» om hva som bør komme etter «${ctxText()}».</p>
          ${calc(`
            <p class="note"><b>1)</b> h · W₁:</p>${R().vecMatCalc(t0.h[last], ML.state.model.W1, "pre")}
            <p class="note"><b>2)</b> ReLU → ${R().vecHTML(t0.act[last], "data")} &nbsp; <b>3)</b> act · W₂:</p>
            ${R().vecMatCalc(t0.act[last], ML.state.model.W2, "f")}`)}`;
      },
    },

    {
      id: "probs",
      title: "Sannsynligheter",
      sub: "hva tror modellen kommer nå?",
      diagram: ["dg-logits"],
      render() {
        const t0 = tr();
        const probs = ML.softmax(t0.logits, 1);
        return `
          <p class="lede">Vektoren ganges med output-matrisen → én logit per ord → softmax →
          sannsynligheter. Og her er den store forskjellen fra treningen: <b>det finnes ingen fasit
          og ingen loss – gjetningen ER svaret.</b></p>
          <p style="margin-bottom:2px"><b>Hva tror modellen kommer etter «${ctxText()}»?</b></p>
          ${R().probBars(probs, { logits: lv(1) ? t0.logits : null })}
          ${calc(`
            <div class="mathline">logits = y · W_U</div>
            ${ML.VOCAB.map((w, i) => `<div class="mathline">logit(${w}) = ${R().dotCalc(t0.y[t0.T - 1], ML.state.model.Wu.map((r) => r[i]))}</div>`).join("")}`)}`;
      },
    },

    {
      id: "temp",
      title: "Temperatur",
      sub: "skru på personligheten",
      diagram: ["dg-softmax"],
      render() {
        const u = ensureState();
        const t0 = tr();
        const probs = ML.softmax(t0.logits, u.temp);
        return `
          <p class="lede">Før softmax deles alle logitene på <b>temperaturen</b>. Rekkefølgen endres
          aldri – bare hvor <i>bastant</i> fordelingen blir. Dra og se:</p>
          ${tempRow("temp")}
          <p style="margin-bottom:2px"><b>Hva tror modellen kommer etter «${ctxText()}»?</b></p>
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
      id: "sampling",
      title: "Sampling",
      sub: "terningkastet til slutt",
      diagram: ["dg-sampling"],
      render() {
        const u = ensureState();
        const t0 = tr();
        const probs = ML.softmax(t0.logits, u.temp);
        return `
          <p class="lede">Til slutt velges neste token. Modellen tar <b>ikke</b> automatisk ordet med
          høyest sannsynlighet – den <b>trekker</b> tilfeldig, der hvert ord har så stor sjanse som
          sannsynligheten sin. Som et lykkehjul.</p>
          <p style="margin-bottom:2px"><b>Hva tror modellen kommer etter «${ctxText()}»?</b></p>
          ${R().probBars(probs, { sampledIdx: u.sampled })}
          <div class="sample-stage">
            <button class="btn accent big" id="sample-btn">🎲 Trekk neste token</button>
            ${u.sampled !== null ? `
              <div class="sample-result">
                <span class="note">r = ${R().fmt(u.draw, 3)} →</span>
                <span class="sample-token">${W(u.sampled)}</span>
              </div>` : ""}
          </div>
          ${u.sampled !== null ? `
            <p class="note">Trekk gjerne flere ganger – med nok forsøk kan også de usannsynlige
            ordene dukke opp. Det er derfor en LLM ikke svarer likt hver gang.</p>
            ${lv(1) ? sampleExplain(probs, u.draw) : ""}` : ""}`;
      },
      wire(host) {
        const u = ensureState();
        const t0 = tr();
        host.querySelector("#sample-btn").addEventListener("click", () => {
          const { pick, r } = sampleFrom(ML.softmax(t0.logits, u.temp));
          u.sampled = pick;
          u.draw = r;
          renderUse(false);
        });
      },
    },

    {
      id: "bigpicture",
      title: "Det store bildet",
      levels: false,
      sub: "trening og bruk, side om side",
      diagram: [],
      render() {
        return `
          <p class="lede">Du har nå sett begge sider av maskinen. Samme fremoverpass – men helt
          forskjellige slutter:</p>
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
          <p>Og her er innsikten verdt å ta med seg: <b>å bruke en modell endrer den ikke.</b>
          ChatGPT «lærer» ikke av samtalen din – alt den kan, ble frosset da treningen sluttet. Neste
          ord velges ved å lese de samme parameterne igjen og igjen, pluss et terningkast.</p>`;
      },
    },

    {
      id: "play",
      title: "Lekeplassen",
      sub: "la modellen skrive 🎲",
      diagram: ["dg-next"],
      render() {
        const u = ensureState();
        const t0 = tr();
        const probs = ML.softmax(t0.logits, u.temp);
        return `
          <p class="lede">Du kan alt nå. Bygg din egen tekst, still temperaturen og la modellen
          fortsette den – token for token.</p>

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
          <p style="margin-bottom:2px"><b>Hva tror modellen kommer etter «${ctxText()}»?</b></p>
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
            </div>` : ""}

          <p class="note" style="margin-top:20px">🎓 <b>Du har trent og brukt en ekte transformer.</b>
          Veier videre: ta runden igjen på «Med tall»-nivået · trykk «↺ Nullstill» og se hvor
          tilfeldig alt starter · les ærlighetsnotatet nederst på siden.</p>`;
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
          const { pick, r } = sampleFrom(ML.softmax(t0.logits, u.temp));
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
    const untrained = !isTrained();
    host.innerHTML = `
      ${untrained ? `
        <div class="card" style="border-color:#e7c78a;background:#fffaf0;display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:14px 22px">
          <span style="flex:1;min-width:220px">⚠️ <b>Modellen er ikke trent ennå</b> – alle tall du
          ser her er tilfeldig gjetting. Den kan ikke noe før du har trent den.</span>
          <button class="btn primary" id="goto-train">🎓 Gå til treningen</button>
        </div>` : ""}
      <div class="card">
        <div class="step-kicker">Bruksmodus · steg ${i + 1} av ${steps.length}</div>
        <h2>${step.title} <span class="h2-sub">${typeof step.sub === "function" ? step.sub() : step.sub || ""}</span></h2>
        ${step.render()}
      </div>`;
    ML.diagram.highlight(step.diagram);
    ML.diagram.setBackward(false);
    ML.R.setLevelSwitchEnabled(step.levels !== false);
    const gotoTrain = host.querySelector("#goto-train");
    if (gotoTrain) gotoTrain.addEventListener("click", () => ML.setMode("train"));
    if (step.wire) step.wire(host);
    ML.state.maxUseStep = Math.max(ML.state.maxUseStep || 0, i);
    const goto = (j) => { ML.state.useStepIdx = j; renderUse(); };
    ML.R.renderNav({
      count: steps.length,
      current: i,
      maxVisited: ML.state.maxUseStep,
      titles: steps.map((s) => s.title),
      onPrev() { goto(Math.max(0, i - 1)); },
      onNext() { goto(Math.min(steps.length - 1, i + 1)); },
      onJump(j) { goto(j); },
    });
    if (scroll) window.scrollTo({ top: 0, behavior: "smooth" });
  }

  ML.renderUse = renderUse;
  // Klikk i diagrammet → hopp til steget som forklarer den delen.
  ML.USE_DIAG = {
    "dg-text": 1, "dg-tokens": 1, "dg-emb": 2, "dg-layer": 3, "dg-attn": 3,
    "dg-ffn": 4, "dg-logits": 5, "dg-softmax": 6, "dg-sampling": 7, "dg-next": 9,
  };
})();
