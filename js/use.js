/* Bruksmodus: parameterne er frosset, data flyter bare fremover,
   og til slutt trekkes neste token med temperatur-styrt sampling. */
(function () {
  const ML = globalThis.ML;
  const R = () => ML.R;
  const W = (i) => ML.VOCAB[i];

  function ensureState() {
    if (!ML.state.use)
      ML.state.use = { ctx: [0, 1], temp: 1.0, sampled: null, draw: null, open: {} };
    return ML.state.use;
  }

  function tempProbs(logits, temp) {
    return ML.softmax(logits, temp);
  }

  function pipeCard(id, title, quick, bodyHTML) {
    const u = ensureState();
    return `<div class="pipe-card ${u.open[id] ? "open" : ""}" data-pipe="${id}">
      <button class="pipe-head" data-pipehead="${id}">
        <span class="caret">${u.open[id] ? "▼" : "▶"}</span> ${title}
        <span class="quick">${quick}</span>
      </button>
      <div class="pipe-body">${bodyHTML}</div>
    </div>`;
  }

  function renderUse() {
    const u = ensureState();
    const m = ML.state.model;
    const host = document.getElementById("content");
    const tr = m.forward(u.ctx);
    const last = u.ctx.length - 1;
    const probs = tempProbs(tr.logits, u.temp);
    const A = tr.alphas[last];

    const trained = ML.state.history.length > 1;

    host.innerHTML = `
      <div class="frozen-banner">
        <span class="big">🔒</span>
        <div>PARAMETERE FROSSET<br>
        <span style="font-weight:400;font-size:13.5px">Modellen er ferdig trent (${m.steps} steg).
        Fra nå av leses parameterne bare – ingenting endres, uansett hvor mye vi bruker den.</span></div>
      </div>

      ${trained ? "" : `<div class="card" style="border-color:#e7c78a;background:#fffaf0">
        ⚠️ <b>Du har ikke trent modellen ennå</b> – parameterne er fortsatt tilfeldige, så den vil
        gjette i blinde. Det er lov (og lærerikt!) å prøve, men gå gjerne innom treningsmodusen først.
      </div>`}

      <div class="card">
        <div class="step-kicker">Bruksmodus</div>
        <h2>Samme maskin – men nå bare fremover</h2>
        <p class="lede">Å <i>bruke</i> modellen (det som skjer når du chatter med en LLM) er nøyaktig
        samme fremoverpass som under treningen. Forskjellen er hva som skjer etterpå:</p>
        <div class="compare">
          <div class="cmp-col train">
            <h4>🎓 Trening</h4>
            <ol>
              <li>fremoverpass → prediksjon</li>
              <li>sammenlign med fasit → <b>loss</b></li>
              <li><b>bakoverpass</b> → gradienter</li>
              <li>🟠 <b>parameterne ENDRES</b></li>
            </ol>
          </div>
          <div class="cmp-col use">
            <h4>▶️ Bruk</h4>
            <ol>
              <li>fremoverpass → prediksjon</li>
              <li>softmax → sannsynligheter</li>
              <li><b>sampling</b> → velg neste token</li>
              <li>🔒 <b>ingen parametere endres</b></li>
            </ol>
          </div>
        </div>
      </div>

      <div class="card">
        <h2>Input</h2>
        <p>Bygg en tekst av tokens fra vokabularet (maks ${ML.MAXPOS}):</p>
        <div class="chiprow">
          ${u.ctx.map((t, i) => R().tokenChip(W(t), t)).join(" ")}
          ${u.ctx.length > 1 ? `<button class="btn ghost" id="ctx-pop" title="Fjern siste token">⌫</button>` : ""}
        </div>
        <div class="chiprow">
          <span class="note">Legg til:</span>
          ${ML.VOCAB.map((w, i) => `<button class="btn" data-add="${i}" ${u.ctx.length >= ML.MAXPOS ? "disabled" : ""}>+ ${w}</button>`).join(" ")}
          <button class="btn ghost" id="ctx-reset">↺ «katten drikker»</button>
        </div>
      </div>

      <div class="card">
        <h2>Fremoverpasset, steg for steg</h2>
        <p class="note">Samme rørledning som i treningen – klikk deg inn der du vil se tallene.</p>

        ${pipeCard("tok", "1 · Tokens", u.ctx.map((t) => `${W(t)}#${t}`).join(" "), `
          <div class="chiprow">${u.ctx.map((t) => R().tokenChip(W(t), t)).join(" ")}</div>
          <p class="note">Deterministisk oppslag – ingen parametere.</p>`)}

        ${pipeCard("emb", "2 · Embeddinger + posisjon", `${u.ctx.length} × ${ML.D} tall`, `
          ${u.ctx.map((t, i) => `<div class="mathline">x<sub>${W(t)}</sub> = E[${W(t)}] + P[${i}] =
            ${R().vecHTML(m.E[t], "param")} + ${R().vecHTML(m.P[i], "param")} = ${R().vecHTML(tr.x[i], "data")}</div>`).join("")}
          <p class="note">🟠 De oransje verdiene er nå <b>lærte</b> – sammenlign gjerne med hva de var før treningen.</p>`)}

        ${pipeCard("attn", "3 · Attention", `«${W(u.ctx[last])}» ser mest på «${W(u.ctx[A.indexOf(Math.max(...A))])}»`, `
          <p>Siste token, «${W(u.ctx[last])}», henter informasjon fra hele konteksten:</p>
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
          ${ML.VOCAB.map((w, i) => `<div class="mathline">logit(${w}) = ${R().dotCalc(tr.y[last], m.Wu.map((r) => r[i]))}</div>`).join("")}`)}
      </div>

      <div class="card">
        <h2>Softmax med temperatur 🌡️</h2>
        <p class="lede">Før softmax deles alle logitene på <b>temperaturen</b>. Det endrer ikke
        rekkefølgen – bare hvor <i>bastant</i> fordelingen blir. Dra og se:</p>
        <div class="temp-row">
          <span class="note">0.1<br>forsiktig</span>
          <input type="range" id="temp-slider" min="0.1" max="3" step="0.05" value="${u.temp}">
          <span class="note" style="text-align:right">3.0<br>vill</span>
          <span class="temp-val" id="temp-val">τ = ${u.temp.toFixed(2)}</span>
        </div>
        <div id="temp-bars">${R().probBars(probs, { sampledIdx: u.sampled })}</div>
        <p class="note" id="temp-hint">${tempHint(u.temp)}</p>
      </div>

      <div class="card">
        <h2>Sampling – terningkastet til slutt 🎲</h2>
        <p class="lede">Modellen velger ikke automatisk det mest sannsynlige ordet. Den <b>trekker</b>
        fra fordelingen – som et lykkehjul der hvert ord eier en bit proporsjonal med sannsynligheten
        sin. Derfor kan samme spørsmål gi ulike svar.</p>
        <div class="sample-stage">
          <button class="btn accent big" id="sample-btn">🎲 Trekk neste token</button>
          ${u.sampled !== null ? `
            <div class="sample-result">
              <span class="note">tilfeldig tall r = ${R().fmt(u.draw, 3)} →</span>
              <span class="sample-token">${W(u.sampled)}</span>
            </div>` : ""}
        </div>
        ${u.sampled !== null ? `
          <p class="gen-text">${u.ctx.map((t) => W(t)).join(" ")} <span class="new">${W(u.sampled)}</span></p>
          <div class="chiprow">
            ${u.ctx.length < ML.MAXPOS ? `<button class="btn primary" id="accept-btn">➕ Legg tokenet til teksten og fortsett</button>` : `<span class="note">(kontekstvinduet på ${ML.MAXPOS} tokens er fullt – ekte modeller har samme grense, bare mye større)</span>`}
          </div>
          ${R().disclosure("🔢 Hvordan ble trekket avgjort?", sampleExplain(probs, u.draw))}` : ""}
      </div>`;

    ML.diagram.highlight(["dg-sampling", "dg-next"]);
    ML.diagram.setBackward(false);

    // --- wiring ---
    host.querySelectorAll("[data-pipehead]").forEach((b) =>
      b.addEventListener("click", () => {
        const id = b.dataset.pipehead;
        u.open[id] = !u.open[id];
        renderUse();
      })
    );
    host.querySelectorAll("[data-add]").forEach((b) =>
      b.addEventListener("click", () => {
        u.ctx.push(Number(b.dataset.add));
        u.sampled = null;
        renderUse();
      })
    );
    const pop = host.querySelector("#ctx-pop");
    if (pop) pop.addEventListener("click", () => { u.ctx.pop(); u.sampled = null; renderUse(); });
    host.querySelector("#ctx-reset").addEventListener("click", () => {
      u.ctx = [0, 1]; u.sampled = null; renderUse();
    });

    // Temperatur: oppdater stolpene direkte uten å re-rendre (så slideren ikke mister grepet).
    const slider = host.querySelector("#temp-slider");
    slider.addEventListener("input", () => {
      u.temp = Number(slider.value);
      const p2 = tempProbs(tr.logits, u.temp);
      host.querySelector("#temp-val").textContent = `τ = ${u.temp.toFixed(2)}`;
      host.querySelector("#temp-bars").innerHTML = R().probBars(p2, { sampledIdx: u.sampled });
      host.querySelector("#temp-hint").textContent = tempHint(u.temp);
    });

    host.querySelector("#sample-btn").addEventListener("click", () => {
      const p2 = tempProbs(tr.logits, u.temp);
      const r = Math.random();
      let acc = 0, pick = p2.length - 1;
      for (let i = 0; i < p2.length; i++) {
        acc += p2[i];
        if (r < acc) { pick = i; break; }
      }
      u.sampled = pick;
      u.draw = r;
      renderUse();
    });
    const accept = host.querySelector("#accept-btn");
    if (accept)
      accept.addEventListener("click", () => {
        u.ctx.push(u.sampled);
        u.sampled = null;
        renderUse();
        window.scrollTo({ top: 0, behavior: "smooth" });
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
      return `<div class="mathline" ${hit ? 'style="font-weight:700"' : ""}>${ML.VOCAB[i]}: eier intervallet ${ML.R.fmt(from, 3)} – ${ML.R.fmt(acc, 3)} ${hit ? "← r landet her!" : ""}</div>`;
    }).join("");
    return `<p>Hvert ord eier en bit av tallinja fra 0 til 1, like stor som sannsynligheten sin.
      Vi trakk r = <b>${ML.R.fmt(r, 3)}</b>:</p>${rows}
      <p class="note">Med temperatur nær 0 eier favoritten nesten hele linja – da blir modellen
      deterministisk i praksis. Det er hele hemmeligheten bak «hvorfor svarer ChatGPT forskjellig hver gang».</p>`;
  }

  ML.renderUse = renderUse;
})();
