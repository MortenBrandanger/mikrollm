/* Treningsmodus: ett lite konsept av gangen gjennom én ekte treningsrunde på
   «katten drikker» → «melk», og deretter fri trening.

   Hvor mye som vises styres av det globale detaljnivået (ML.state.level):
     0 = Enkelt (bare konseptet), 1 = Med tall, 2 = Full utregning. */
(function () {
  const ML = globalThis.ML;
  const R = () => ML.R;

  const EX_TOKENS = [0, 1]; // katten drikker
  const TARGET = 2;         // melk
  const LR = 0.2;

  const W = (i) => ML.VOCAB[i];
  const lv = (n) => ML.state.level >= n;

  function trace() { return ML.state.model.forward(EX_TOKENS); }

  function recordEval() {
    const m = ML.state.model;
    const tr = m.forward(EX_TOKENS);
    ML.state.history.push({ step: m.steps, loss: m.loss(tr, TARGET), p: tr.probs[TARGET] });
  }
  ML.recordEval = recordEval;

  /* ---------------- Stegene ---------------- */

  const steps = [
    {
      id: "goal",
      title: "Vi skal lære opp en ekte (bitteliten) språkmodell",
      diagram: ["dg-text"],
      render() {
        return `
          <p class="lede">Modellen skal lære én ting: etter <b>«katten drikker»</b> kommer
          <b>«melk»</b>. Den har bare <b>60 parametere</b>, men er bygget av nøyaktig de samme delene
          som ChatGPT. Ingen juks – bare små matriser.</p>
          <div class="chiprow">
            ${R().tokenChip("katten")} ${R().tokenChip("drikker")}
            <span class="arrow-inline">→</span>
            <span class="chip win">melk</span>
          </div>
          <p>Én ting er viktigere enn alt annet underveis. Hold øye med fargene:</p>
          <div class="legend">
            <div class="legend-item data"><b>🔵 DATA</b> flyter <i>gjennom</i> modellen – skapes av
            teksten din, forsvinner etterpå.</div>
            <div class="legend-item param"><b>🟠 PARAMETERE</b> bor <i>inne i</i> modellen – det den
            har lært. Bare disse endres under trening.</div>
          </div>
          <p class="note">Akkurat nå er alle 60 parameterne tilfeldige tall – modellen kan ingenting.
          💡 Bryteren øverst (<i>Enkelt / Med tall / Utregning</i>) velger hvor dypt du vil gå. Start
          gjerne enkelt og ta en runde til med tall etterpå.</p>`;
      },
    },

    {
      id: "token",
      title: "Tokenisering – teksten deles i biter",
      diagram: ["dg-text", "dg-tokens"],
      render() {
        return `
          <p class="lede">Først deles teksten i <b>tokens</b>. Hos oss er 1 ord = 1 token, og hver
          token har et fast nummer:</p>
          <div class="chiprow">
            <span class="chip plain">«katten drikker»</span>
            <span class="arrow-inline">→</span>
            ${R().tokenChip("katten", 0)} ${R().tokenChip("drikker", 1)}
          </div>
          <p>Dette er ren oppslagstabell – <b>deterministisk</b>, ingen læring, ingen parametere.</p>
          ${lv(1) ? `
            <p class="note">Hele vokabularet – alt modellen noensinne kan si:</p>
            <div class="chiprow">${ML.VOCAB.map((w, i) => R().tokenChip(w, i)).join(" ")}</div>` : ""}
          ${R().honest(`Tokeniseringen deler ord i mindre biter («subord»), så «drikker» kunne blitt
            «drik» + «ker», og vokabularet har ~100 000 biter. Prinsippet er det samme.`)}`;
      },
    },

    {
      id: "emb",
      title: "Embeddinger – hver token blir tall",
      diagram: ["dg-emb"],
      render() {
        const m = ML.state.model;
        return `
          <p class="lede">Modellen kan ikke regne på ord. Hver token slås opp i en tabell og blir en
          <b>vektor</b> – en liste med tall. Hos oss: 2 tall per token.</p>
          <div style="margin:12px 0">
            ${(lv(1) ? [0, 1, 2, 3, 4] : EX_TOKENS).map((i) => `
              <div class="chiprow">
                ${R().tokenChip(W(i), i)}
                <span class="arrow-inline">→</span>
                ${R().vecHTML(m.E[i], "param")}
              </div>`).join("")}
          </div>
          <p>Legg merke til fargen: dette er <b>🟠 parametere</b>. Det finnes ingen fasit for hva
          «katten» skal være som tall – verdiene <b>læres</b>, og siden vi ikke har trent ennå, er de
          bare tilfeldige.</p>
          ${R().honest(`Samme oppslag, men vektorene har f.eks. 4096 tall i stedet for 2.
            Antall tall = «embedding-dimensjoner».`)}`;
      },
    },

    {
      id: "pos",
      title: "Posisjon – modellen må vite rekkefølgen",
      diagram: ["dg-emb"],
      render() {
        const tr = trace();
        const m = ML.state.model;
        return `
          <p class="lede">En transformer ser alle tokens <b>samtidig</b> og aner ikke rekkefølgen av
          seg selv. Derfor får hver plass i setningen sin egen lærte <b>posisjonsvektor</b>, som
          legges til:</p>
          ${lv(1) ? EX_TOKENS.map((t, i) => `
            <div class="mathline">x<sub>${W(t)}</sub> = E[${W(t)}] + P[plass ${i}] =
              ${R().vecHTML(m.E[t], "param")} + ${R().vecHTML(m.P[i], "param")}
              = ${R().vecHTML(tr.x[i], "data")}</div>`).join("")
          : `<div class="mathline" style="text-align:center">x = embedding + posisjonsvektor</div>`}
          ${lv(2) ? EX_TOKENS.map((t, i) => `
            <div class="mathline note">x<sub>${W(t)}</sub>[0] = ${R().fmt(m.E[t][0])} + ${R().fmt(m.P[i][0])} = <span class="res">${R().fmt(tr.x[i][0])}</span>
            &nbsp;&nbsp; x<sub>${W(t)}</sub>[1] = ${R().fmt(m.E[t][1])} + ${R().fmt(m.P[i][1])} = <span class="res">${R().fmt(tr.x[i][1])}</span></div>`).join("") : ""}
          <p>Resultatet x er <b>🔵 data</b> – nå begynner tallene å flyte gjennom modellen.</p>
          ${R().honest(`Posisjon kodes ofte med et triks kalt RoPE i stedet for en lært tabell,
            men jobben er den samme: å skille «katten drikker» fra «drikker katten».`)}`;
      },
    },

    {
      id: "qkv",
      title: "Attention · hver token lager Q, K og V",
      diagram: ["dg-layer", "dg-attn"],
      render() {
        const tr = trace();
        const m = ML.state.model;
        const qkv = ML.state.qkvTab;
        const info = {
          q: { name: "Q – Query", weight: m.Wq, label: "W_Q", out: tr.q,
               what: "«Hva leter jeg etter?» Tokenens <b>spørrevektor</b>." },
          k: { name: "K – Key", weight: m.Wk, label: "W_K", out: tr.k,
               what: "«Hva kan jeg tilby?» Tokenens <b>nøkkelvektor</b> – den andre sammenligner seg mot." },
          v: { name: "V – Value", weight: m.Wv, label: "W_V", out: tr.v,
               what: "«Hva sender jeg videre?» Tokenens <b>verdivektor</b> – det som faktisk blandes til slutt." },
        }[qkv];
        return `
          <p class="lede">Nå skal tokens få snakke sammen – det er <b>attention</b>. Første del:
          hver token lager tre nye vektorer av sin x, ved å gange med tre <b>lærte matriser</b>.
          Klikk på dem:</p>
          <div class="qkv-tabs">
            ${["q", "k", "v"].map((t) => `<button class="qkv-tab ${qkv === t ? "active" : ""}" data-qkv="${t}">${t.toUpperCase()}</button>`).join("")}
          </div>
          <div class="qkv-panel">
            <p style="margin-top:0"><b>${info.name}:</b> ${info.what}</p>
            <div class="mathline">${qkv} = x · ${info.label}</div>
            ${lv(1) ? `
              <div>${R().matHTML(info.weight, "param", `${info.label} (lærte parametere, 2×2)`)}</div>
              ${EX_TOKENS.map((t, i) => `
                <div class="mathline">${qkv}<sub>${W(t)}</sub> =
                  ${R().vecHTML(tr.x[i], "data")} · ${info.label} = ${R().vecHTML(info.out[i], "data")}</div>`).join("")}` : ""}
            ${lv(2) ? `<p class="note">Utregnet for «drikker»:</p>${ML.R.vecMatCalc(tr.x[1], info.weight, qkv)}` : ""}
          </div>
          ${R().honest(`Nøyaktig samme regnestykke – men med mange «attention-hoder» parallelt
            (f.eks. 32), hvert med sine egne W_Q/W_K/W_V.`)}`;
      },
      wire(host) {
        host.querySelectorAll("[data-qkv]").forEach((b) =>
          b.addEventListener("click", () => { ML.state.qkvTab = b.dataset.qkv; renderTrain(); })
        );
      },
    },

    {
      id: "attnw",
      title: "Attention · hvem ser på hvem?",
      diagram: ["dg-layer", "dg-attn"],
      render() {
        const tr = trace();
        const A = tr.alphas[1];
        return `
          <p class="lede">Spørrevektoren til «drikker» sammenlignes med nøkkelen til hver token.
          Resultatet er <b>vekter</b> som summerer til 100 % – så mye oppmerksomhet gir «drikker»
          hver token:</p>
          <div class="attn-weights">
            ${EX_TOKENS.map((t, j) => `
              <div class="aw-row">
                <div><b>${W(t)}</b></div>
                <div class="aw-bar"><div class="aw-fill" style="width:${A[j] * 100}%"></div></div>
                <div class="aw-pct">${R().pct(A[j])}</div>
              </div>`).join("")}
          </div>
          ${lv(1) ? `
            ${EX_TOKENS.map((t, j) => `
              <div class="mathline">score(drikker → ${W(t)}) = q<sub>drikker</sub> · k<sub>${W(t)}</sub> / √2 = <span class="res">${R().fmt(tr.scores[1][j])}</span></div>`).join("")}
            <div class="mathline">softmax(${tr.scores[1].map((s) => R().fmt(s)).join(", ")}) = [${A.map((a) => R().pct(a)).join(", ")}]</div>` : ""}
          ${lv(2) ? `
            <p class="note">Prikkproduktene bak scorene:</p>
            ${EX_TOKENS.map((t, j) => `
              <div class="mathline">q·k for ${W(t)}: ${R().dotCalc(tr.q[1], tr.k[j])}</div>`).join("")}` : ""}
          <p class="note">«katten» får bare se bakover (på seg selv) – ingen token får se fremover i
          setningen. Det kalles <b>kausal</b> attention. Og husk: vektene er ikke magi – de kommer
          rett fra Q- og K-matrisene, som er tilfeldige ennå og læres under trening.</p>
          ${R().honest(`Samme mekanisme, men over tusenvis av tokens samtidig – hver token veier
            hele teksten bak seg.`)}`;
      },
    },

    {
      id: "attnz",
      title: "Attention · informasjonen blandes",
      diagram: ["dg-layer", "dg-attn"],
      render() {
        const tr = trace();
        const m = ML.state.model;
        const A = tr.alphas[1];
        return `
          <p class="lede">Så brukes vektene: den nye representasjonen for «drikker» er et veid
          gjennomsnitt av <b>verdivektorene</b> (V). Slik flyter informasjon fra «katten» inn i
          «drikker»:</p>
          <div class="mathline" style="text-align:center;font-size:15px">
            z = ${R().pct(A[0], 0)} · v<sub>katten</sub> + ${R().pct(A[1], 0)} · v<sub>drikker</sub>
            ${lv(1) ? `= ${R().vecHTML(tr.z[1], "data")}` : ""}
          </div>
          <p>Til slutt <b>legges resultatet til</b> den opprinnelige vektoren – en
          <b>residualkobling</b>: tokenen beholder seg selv og får attention-informasjonen som
          tillegg.</p>
          ${lv(1) ? `<div class="mathline">h<sub>drikker</sub> = x + z·W_O =
            ${R().vecHTML(tr.x[1], "data")} + ${R().vecHTML(tr.attnOut[1], "data")} = ${R().vecHTML(tr.h[1], "data")}</div>` : ""}
          ${lv(2) ? `
            <p class="note">W_O er attentions fjerde lærte matrise:</p>
            <div>${R().matHTML(m.Wo, "param", "W_O (2×2)")}</div>
            ${ML.R.vecMatCalc(tr.z[1], m.Wo, "z·W_O")}` : ""}
          ${R().honest(`Helt likt – residualkoblinger er en av grunnene til at modeller med hundrevis
            av lag i det hele tatt lar seg trene.`)}`;
      },
    },

    {
      id: "ffn",
      title: "Feed forward – et bittelite nevralt nettverk",
      diagram: ["dg-layer", "dg-ffn"],
      render() {
        const tr = trace();
        const m = ML.state.model;
        return `
          <p class="lede">Etter attention bearbeides hver posisjon for seg av et lite nevralt
          nettverk: gang med en lært matrise, klipp negative tall til null (<b>ReLU</b>), gang med en
          lært matrise til.</p>
          ${lv(1) ? `
            <div class="ffn-flow">
              <div class="ffn-stage">${R().vecHTML(tr.h[1], "data")}<div class="stage-cap">h</div></div>
              <span class="op">· W₁ (2×4)</span>
              <div class="ffn-stage">${R().vecHTML(tr.pre[1], "data")}<div class="stage-cap">4 tall</div></div>
              <span class="op">ReLU</span>
              <div class="ffn-stage">${R().vecHTML(tr.act[1], "data")}<div class="stage-cap">negative → 0</div></div>
              <span class="op">· W₂ (4×2)</span>
              <div class="ffn-stage">${R().vecHTML(tr.f[1], "data")}<div class="stage-cap">f</div></div>
            </div>
            <div class="mathline">y = h + f = ${R().vecHTML(tr.y[1], "data")} <span class="note">(residual igjen)</span></div>`
          : `
            <div class="ffn-flow">
              <div class="shape-box">2 tall</div><span class="op">· W₁</span>
              <div class="shape-box">4 tall</div><span class="op">ReLU</span>
              <div class="shape-box">4 tall</div><span class="op">· W₂</span>
              <div class="shape-box">2 tall</div>
            </div>
            <p class="note">Nettverket går <i>opp</i> i bredde og ned igjen – det gir modellen
            arbeidsplass til å kombinere trekk. Resultatet legges til h (residual).</p>`}
          ${lv(2) ? `
            <div>${R().matHTML(m.W1, "param", "W₁ (2×4)")} ${R().matHTML(m.W2, "param", "W₂ (4×2)")}</div>
            <p class="note"><b>1)</b> h · W₁:</p>${R().vecMatCalc(tr.h[1], m.W1, "pre")}
            <p class="note"><b>2)</b> ReLU: [${tr.pre[1].map((v) => `max(0, ${R().fmt(v)})`).join(", ")}] = ${R().vecHTML(tr.act[1], "data")}</p>
            <p class="note"><b>3)</b> act · W₂:</p>${R().vecMatCalc(tr.act[1], m.W2, "f")}` : ""}
          ${R().honest(`Feed-forward er den største delen av modellen: 4096 tall inn, ~14 000 i
            midten. Aktiveringen er gjerne GELU/SiLU – samme idé, mykere knekk.`)}`;
      },
    },

    {
      id: "logits",
      title: "Logits og softmax – hva tror modellen?",
      diagram: ["dg-logits", "dg-softmax"],
      render() {
        const tr = trace();
        const m = ML.state.model;
        return `
          <p class="lede">Den ferdige vektoren for siste token ganges med den lærte
          <b>output-matrisen</b> – da får hvert ord i vokabularet én score (<b>logit</b>), og
          <b>softmax</b> gjør scorene om til sannsynligheter:</p>
          ${R().probBars(tr.probs, { correctIdx: TARGET, logits: lv(1) ? tr.logits : null })}
          ${lv(1) ? `<div class="mathline">logits = y · W_U, &nbsp; y<sub>drikker</sub> = ${R().vecHTML(tr.y[1], "data")}, &nbsp; W_U: ${R().matHTML(m.Wu, "param")}</div>` : ""}
          ${lv(2) ? `
            <div class="mathline">P(ord) = e^logit / sum av alle e^logit</div>
            ${ML.VOCAB.map((w, i) => `<div class="mathline">P(${w}) = e^${R().fmt(tr.logits[i])} / ${R().fmt(tr.logits.reduce((s, l) => s + Math.exp(l), 0))} = <span class="res">${R().pct(tr.probs[i])}</span></div>`).join("")}` : ""}
          <p class="note">Modellen gjetter i blinde – som forventet, alle parameterne er jo
          tilfeldige. Det skal vi gjøre noe med.</p>`;
      },
    },

    {
      id: "loss",
      title: "Loss – hvor feil tok modellen?",
      diagram: ["dg-softmax"],
      render() {
        const tr = trace();
        const m = ML.state.model;
        const p = tr.probs[TARGET];
        const loss = m.loss(tr, TARGET);
        return `
          <p class="lede">Under trening <b>vet vi fasiten</b>: «melk». Da kan vi måle hvor feil
          modellen tok – det tallet kalles <b>loss</b>. Perfekt svar gir loss 0; jo mer feil, jo
          høyere.</p>
          <div class="loss-hero">
            <div>
              <div class="bignum">${R().pct(p)}</div>
              <div class="bignum-cap">modellens tro på «melk»</div>
            </div>
            <div style="font-size:26px;color:var(--muted)">→</div>
            <div>
              <div class="bignum bad">${R().fmt(loss)}</div>
              <div class="bignum-cap">loss (lavere = bedre)</div>
            </div>
          </div>
          ${lv(1) ? R().probBars(tr.probs, { correctIdx: TARGET }) : ""}
          ${lv(2) ? `
            <div class="mathline">loss = −ln( P(riktig token) ) = −ln(${R().fmt(p, 3)}) = <span class="res">${R().fmt(loss, 3)}</span></div>
            <p class="note">−ln(1.0) = 0 (perfekt), −ln(0.5) ≈ 0.69, −ln(0.1) ≈ 2.3 – straffen vokser
            bratt jo sikrere modellen var på feil ting. Kalles cross-entropy.</p>` : ""}
          <p class="note">Nå har vi ett tall for hvor dårlig hele modellen var. Neste steg er selve
          magien: å bruke det til å forbedre alle parameterne samtidig.</p>`;
      },
    },

    {
      id: "backprop",
      title: "Backpropagation – læringen skjer baklengs",
      diagram: ["dg-emb", "dg-layer", "dg-attn", "dg-ffn", "dg-logits", "dg-softmax"],
      backward: true,
      render() {
        const m = ML.state.model;
        const tr = trace();
        const g = m.backward(tr, TARGET);
        const changed = m.countNonzeroGrads(g);
        const total = m.paramCounts().total;
        const maxAbs = (M, G) => {
          let bi = 0, bj = 0;
          for (let i = 0; i < G.length; i++)
            for (let j = 0; j < G[0].length; j++)
              if (Math.abs(G[i][j]) > Math.abs(G[bi][bj])) { bi = i; bj = j; }
          return { val: M[bi][bj], grad: G[bi][bj], i: bi, j: bj };
        };
        const pe = maxAbs(m.E, g.E), pp = maxAbs(m.P, g.P), pu = maxAbs(m.Wu, g.Wu);
        const attn = [["W_Q", m.Wq, g.Wq], ["W_K", m.Wk, g.Wk], ["W_V", m.Wv, g.Wv], ["W_O", m.Wo, g.Wo]]
          .map(([n, M, G]) => ({ n, ...maxAbs(M, G) }))
          .reduce((a, b) => (Math.abs(b.grad) > Math.abs(a.grad) ? b : a));
        const ffn = [["W₁", m.W1, g.W1], ["W₂", m.W2, g.W2]]
          .map(([n, M, G]) => ({ n, ...maxAbs(M, G) }))
          .reduce((a, b) => (Math.abs(b.grad) > Math.abs(a.grad) ? b : a));
        const picks = [
          { name: `E[${W(pe.i)}]`, sub: `embedding, dim ${pe.j}`, val: pe.val, grad: pe.grad },
          { name: `P[pos ${pp.i}]`, sub: `posisjon, dim ${pp.j}`, val: pp.val, grad: pp.grad },
          { name: attn.n, sub: `attention, rad ${attn.i} kol ${attn.j}`, val: attn.val, grad: attn.grad },
          { name: ffn.n, sub: `feed-forward, rad ${ffn.i} kol ${ffn.j}`, val: ffn.val, grad: ffn.grad },
          { name: `W_U (${W(pu.j)}-kolonnen)`, sub: `output, dim ${pu.i}`, val: pu.val, grad: pu.grad },
        ];
        const applied = ML.state.bpJustApplied;
        return `
          <p class="lede">Nå går vi <b>baklengs</b> gjennom modellen – se de røde pilene i
          diagrammet. For hver parameter regnes en <b>gradient</b>: et tall som svarer på</p>
          <p style="text-align:center;font-size:16.5px"><i>«Hvis akkurat denne parameteren var litt
          annerledes – ville svaret blitt bedre eller verre?»</i></p>
          <p>Så flyttes hver parameter et lite hakk i sin beste retning:</p>
          <div class="mathline" style="text-align:center">ny verdi = gammel verdi − ${LR} · gradient</div>
          ${lv(1) ? `
            <table class="bp-table">
              <tr><th>Parameter (utvalg)</th><th>Verdi nå</th><th>Gradient</th><th>Etter oppdatering</th></tr>
              ${picks.map((p) => `
                <tr class="${applied ? "flash" : ""}">
                  <td class="name">${p.name}<br><span class="note">${p.sub}</span></td>
                  <td>${R().fmt(p.val, 3)}</td>
                  <td class="gradval">${R().fmt(p.grad, 3)}</td>
                  <td class="newval">${R().fmt(p.val - LR * p.grad, 3)}</td>
                </tr>`).join("")}
            </table>` : ""}
          <p class="note">Dette er ikke prøving og feiling – gradientene regnes eksakt, med
          kjerneregelen fra matematikk. I dette steget justeres <b>${changed} av ${total}</b>
          parametere <b>samtidig</b> (resten, f.eks. embeddingen til ord som ikke var med, står
          stille). I en ekte modell: milliarder, i hvert eneste steg.</p>
          ${lv(2) ? `
            <p class="note"><b>Hvor starter gradientene?</b> Første ledd er vakkert enkelt:
            gradienten på logitene = sannsynlighetene − fasiten:</p>
            <div class="mathline">${ML.VOCAB.map((w, i) => `d(${w}) = ${R().fmt(tr.probs[i], 2)} − ${i === TARGET ? 1 : 0} = <span class="res">${R().fmt(tr.probs[i] - (i === TARGET ? 1 : 0), 2)}</span>`).join("<br>")}</div>
            <p class="note">Negativt på «melk» = «skulle vært høyere». Derfra sendes tallene baklengs
            gjennom de samme operasjonene som i fremoverpasset – bare derivert.</p>` : ""}
          ${applied
            ? `<p><b style="color:var(--good)">✅ Oppdatert!</b> Alle parameterne har flyttet seg et
               lite hakk. Gå videre og se om modellen ble bedre.</p>`
            : `<button class="btn accent big" id="bp-apply">Kjør oppdateringen (gradient descent)</button>`}`;
      },
      wire(host) {
        const btn = host.querySelector("#bp-apply");
        if (btn)
          btn.addEventListener("click", () => {
            const m = ML.state.model;
            const tr = m.forward(EX_TOKENS);
            const g = m.backward(tr, TARGET);
            m.applyGrads(g, LR);
            recordEval();
            ML.saveModel();
            ML.state.bpJustApplied = true;
            renderTrain();
            ML.inspector.render();
          });
      },
    },

    {
      id: "again",
      title: "Tren igjen og igjen … og se den lære",
      diagram: ["dg-next"],
      render() {
        const m = ML.state.model;
        const tr = trace();
        const p = tr.probs[TARGET];
        const loss = m.loss(tr, TARGET);
        const hist = ML.state.history;
        const learned = p > 0.9;
        return `
          <p class="lede">Ekte læring er bare denne løkka om igjen og om igjen:
          <b>gjett → mål feilen → juster parameterne</b>. Prøv selv:</p>
          <div class="chiprow">
            <button class="btn primary" data-train="1">Tren 1 steg</button>
            <button class="btn primary" data-train="10">Tren 10 steg</button>
            <button class="btn primary" data-train="100">Tren 100 steg</button>
            <span class="note">totalt trent: <b>${m.steps}</b> steg</span>
          </div>
          <div class="loss-hero">
            <div>
              <div class="bignum ${learned ? "good" : ""}">${R().pct(p)}</div>
              <div class="bignum-cap">P(«melk» | «katten drikker»)</div>
            </div>
            <div>
              <div class="bignum ${loss < 0.2 ? "good" : "bad"}">${R().fmt(loss, 3)}</div>
              <div class="bignum-cap">loss</div>
            </div>
          </div>
          ${lv(1) ? R().probBars(tr.probs, { correctIdx: TARGET }) : ""}
          <div class="spark-cap">P(«melk») gjennom treningen:</div>
          ${R().sparkline(hist.map((h) => h.p), { max: 1, color: "var(--good)" })}
          <div class="spark-cap">Loss gjennom treningen:</div>
          ${R().sparkline(hist.map((h) => h.loss), { color: "var(--bad)" })}
          ${learned
            ? `<p><b style="color:var(--good)">🎉 Modellen har lært det!</b> På tide å <b>bruke</b>
               den – der er reglene helt annerledes.</p>
               <button class="btn accent big pulse" id="goto-use">Gå til «Bruk modellen» →</button>`
            : `<p class="note">Tren til «melk» passerer 90 % – da låser vi modellen og tar den i bruk.</p>`}`;
      },
      wire(host) {
        host.querySelectorAll("[data-train]").forEach((b) =>
          b.addEventListener("click", () => {
            const n = Number(b.dataset.train);
            for (let i = 0; i < n; i++) {
              ML.state.model.trainStep(EX_TOKENS, TARGET, LR);
              recordEval();
            }
            ML.saveModel();
            renderTrain();
            ML.inspector.render();
          })
        );
        const go = host.querySelector("#goto-use");
        if (go) go.addEventListener("click", () => ML.setMode("use"));
      },
    },
  ];

  /* ---------------- Rammeverk rundt stegene ---------------- */

  function renderTrain() {
    const host = document.getElementById("content");
    const i = ML.state.stepIdx;
    const step = steps[i];
    const kicker = i === 0 ? "Treningsmodus" : `Treningsmodus · steg ${i} av ${steps.length - 1}`;
    host.innerHTML = `
      <div class="card">
        <div class="step-kicker">${kicker}</div>
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
    ML.diagram.setBackward(!!step.backward);
    if (step.wire) step.wire(host);
    host.querySelector("#nav-prev").addEventListener("click", () => {
      ML.state.stepIdx = Math.max(0, i - 1);
      ML.state.bpJustApplied = false;
      renderTrain();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
    host.querySelector("#nav-next").addEventListener("click", () => {
      ML.state.stepIdx = Math.min(steps.length - 1, i + 1);
      ML.state.bpJustApplied = false;
      renderTrain();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  ML.renderTrain = renderTrain;
  ML.TRAIN = { EX_TOKENS, TARGET, LR };
})();
