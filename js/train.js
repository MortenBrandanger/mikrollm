/* Treningsmodus: ett lite konsept av gangen gjennom én ekte treningsrunde på
   «katten drikker» → «melk», og deretter fri trening.

   Hvor mye som vises styres av det globale detaljnivået (ML.state.level):
     0 = Enkelt (bare konseptet), 1 = Med tall, 2 = Full utregning.
   Tekster om modellens tilstand (tilfeldig/lært) er betinget av m.steps,
   siden modellen persisteres og kan være ferdig trent når man leser. */
(function () {
  const ML = globalThis.ML;
  const R = () => ML.R;

  const EX_TOKENS = [0, 1]; // katten drikker
  const TARGET = 2;         // melk
  const LR = 0.2;

  const W = (i) => ML.VOCAB[i];
  const lv = (n) => ML.state.level >= n;
  const isTrained = () => ML.state.model.steps > 0;
  // Nivå «Utregning» viser regnestykkene bak en lukket knapp, så skjermen
  // holder seg rolig (CNN Explainer-mønsteret: detaljer på forespørsel).
  const calc = (html) => (lv(2) ? R().disclosure("🧮 Hele utregningen", html) : "");

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
      title: "Oppdraget",
      sub: "lær en ekte (bitteliten) språkmodell én ting",
      diagram: ["dg-text"],
      render() {
        const m = ML.state.model;
        return `
          <p class="lede">En språkmodell gjør bare én ting: den <b>gjetter neste token</b>, om og om
          igjen. ChatGPT, Claude – alt bygger på den ene evnen. Vår modell skal lære én eneste
          gjetning: etter <b>«katten drikker»</b> kommer <b>«melk»</b>.</p>
          <div class="chiprow">
            ${R().tokenChip("katten")} ${R().tokenChip("drikker")}
            <span class="arrow-inline">→</span>
            <span class="chip win">melk</span>
          </div>
          <p>Modellen har bare <b>60 parametere</b>, men er bygget av nøyaktig de samme delene som de
          store. Underveis: hold øye med fargene.</p>
          <div class="legend">
            <div class="legend-item data"><b>🔵 DATA</b> flyter <i>gjennom</i> modellen – skapes av
            teksten, forsvinner etterpå.</div>
            <div class="legend-item param"><b>🟠 PARAMETERE</b> bor <i>inne i</i> modellen – det den
            har lært. Bare disse endres under trening.</div>
          </div>
          <p class="note">${isTrained()
            ? `Modellen din har allerede trent <b>${m.steps}</b> steg. Trykk «↺ Nullstill» øverst
               hvis du vil starte fra tilfeldige parametere igjen.`
            : `Akkurat nå er alle 60 parameterne tilfeldige tall – modellen kan ingenting.`}
          💡 Bryteren øverst velger hvor mange tall du vil se.</p>`;
      },
    },

    {
      id: "token",
      title: "Tokenisering",
      sub: "teksten deles i biter",
      diagram: ["dg-tokens"],
      render() {
        return `
          <p class="lede">Først deles teksten i <b>tokens</b> – bitene modellen jobber med. Hos oss
          er 1 ord = 1 token, og hver token har et fast nummer:</p>
          <div class="chiprow">
            <span class="chip plain">«katten drikker»</span>
            <span class="arrow-inline">→</span>
            ${R().tokenChip("katten", 0)} ${R().tokenChip("drikker", 1)}
          </div>
          <p>Samme tekst gir alltid samme tokens – ren oppslagstabell.</p>
          ${lv(1) ? `
            <p class="note">Ingen læring og ingen parametere i dette steget. Hele vokabularet – alt
            modellen noensinne kan si:</p>
            <div class="chiprow">${ML.VOCAB.map((w, i) => R().tokenChip(w, i)).join(" ")}</div>` : ""}
          ${R().honest(`Tokeniseringen deler ord i mindre biter («subord»), så «drikker» kunne blitt
            «drik» + «ker», og vokabularet har ~100 000 biter. Prinsippet er det samme.`)}`;
      },
    },

    {
      id: "emb",
      title: "Embeddinger",
      sub: "hver token blir tall",
      diagram: ["dg-emb"],
      render() {
        const m = ML.state.model;
        return `
          <p class="lede">Modellen kan ikke regne på ord. Hver token slås derfor opp i en tabell og
          blir en <b>vektor</b> – en liste med tall. Hos oss: 2 tall per token.</p>
          <div style="margin:12px 0">
            ${(lv(1) ? [0, 1, 2, 3, 4] : EX_TOKENS).map((i) => `
              <div class="chiprow">
                ${R().tokenChip(W(i), i)}
                <span class="arrow-inline">→</span>
                ${R().vecHTML(m.E[i], "param")}
              </div>`).join("")}
          </div>
          <p>Legg merke til fargen: dette er <b>🟠 parametere</b> – verdiene <b>læres</b>.
          ${isTrained()
            ? `Tallene du ser er allerede formet av treningen din.`
            : `Akkurat nå er de tilfeldige startverdier: det finnes ingen fasit for hva «katten» skal
               være som tall – modellen finner det ut selv.`}</p>
          ${R().honest(`Samme oppslag, men vektorene har f.eks. 4096 tall i stedet for 2.
            Antall tall = «embedding-dimensjoner».`)}`;
      },
    },

    {
      id: "pos",
      title: "Posisjon",
      sub: "modellen må vite rekkefølgen",
      diagram: ["dg-emb"],
      render() {
        const tr = trace();
        const m = ML.state.model;
        return `
          <p class="lede">En transformer ser alle tokens <b>samtidig</b> og aner ikke rekkefølgen av
          seg selv – den må kunne skille «katten drikker» fra «drikker katten». Derfor får hver plass
          i setningen sin egen lærte <b>posisjonsvektor</b>, som legges til:</p>
          ${lv(1) ? EX_TOKENS.map((t, i) => `
            <div class="mathline">x<sub>${W(t)}</sub> = E[${W(t)}] + P[plass ${i}] =
              ${R().vecHTML(m.E[t], "param")} + ${R().vecHTML(m.P[i], "param")}
              = ${R().vecHTML(tr.x[i], "data")}</div>`).join("")
          : `<div class="mathline" style="text-align:center">x = embedding + posisjonsvektor</div>`}
          ${calc(EX_TOKENS.map((t, i) => `
            <div class="mathline note">x<sub>${W(t)}</sub>[0] = ${R().fmt(m.E[t][0])} + ${R().fmt(m.P[i][0])} = <span class="res">${R().fmt(tr.x[i][0])}</span>
            &nbsp;&nbsp; x<sub>${W(t)}</sub>[1] = ${R().fmt(m.E[t][1])} + ${R().fmt(m.P[i][1])} = <span class="res">${R().fmt(tr.x[i][1])}</span></div>`).join(""))}
          <p>Resultatet x er <b>🔵 data</b> – nå begynner tallene å flyte gjennom modellen.</p>
          ${R().honest(`Posisjon kodes ofte med et triks kalt RoPE i stedet for en lært tabell –
            men jobben er den samme.`)}`;
      },
    },

    {
      id: "qkv",
      title: "Attention (1 av 3) · Q, K og V",
      sub: "hver token lager tre nye vektorer",
      diagram: ["dg-attn"],
      render() {
        const tr = trace();
        const m = ML.state.model;
        const qkv = ML.state.qkvTab;
        const info = {
          q: { name: "Q – Query", weight: m.Wq, label: "W_Q", out: tr.q,
               what: "«Hva leter jeg etter?» – lappen du går rundt i biblioteket med." },
          k: { name: "K – Key", weight: m.Wk, label: "W_K", out: tr.k,
               what: "«Hva kan jeg tilby?» – tittelen på bokryggen, det andre sammenligner seg mot." },
          v: { name: "V – Value", weight: m.Wv, label: "W_V", out: tr.v,
               what: "«Hva sender jeg videre?» – selve innholdet i boka, det som faktisk hentes ut." },
        }[qkv];
        return `
          <p class="lede">Nå skal tokens få snakke sammen – det er <b>attention</b>. Tenk på et
          bibliotek: du leter med en lapp (<b>Q</b>), bokryggene har titler (<b>K</b>), og innholdet i
          boka er det du får med deg (<b>V</b>). Hver token lager alle tre, ved å gange sin vektor med
          tre lærte matriser. Klikk på dem:</p>
          <div class="qkv-tabs">
            ${["q", "k", "v"].map((t) => `<button class="qkv-tab ${qkv === t ? "active" : ""}" data-qkv="${t}">${t.toUpperCase()}</button>`).join("")}
          </div>
          <div class="qkv-panel">
            <p style="margin-top:0"><b>${info.name}:</b> ${info.what}</p>
            ${lv(1) ? `
              <div class="mathline">${qkv} = x · ${info.label}</div>
              <div>${R().matHTML(info.weight, "param", `${info.label} (lærte parametere, 2×2)`)}</div>
              ${EX_TOKENS.map((t, i) => `
                <div class="mathline">${qkv}<sub>${W(t)}</sub> =
                  ${R().vecHTML(tr.x[i], "data")} · ${info.label} = ${R().vecHTML(info.out[i], "data")}</div>`).join("")}`
            : `<p class="note">Lages slik: tokenens x ganges med den lærte matrisen ${info.label}.</p>`}
            ${calc(`<p class="note">Utregnet for «drikker»:</p>${ML.R.vecMatCalc(tr.x[1], info.weight, qkv)}`)}
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
      title: "Attention (2 av 3) · hvem ser på hvem?",
      sub: "Q møter K og blir oppmerksomhetsvekter",
      diagram: ["dg-attn"],
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
            <p class="note">Vektene kommer rett fra Q- og K-matrisene – «hvem ser på hvem» er altså
            noe modellen lærer. Score først, så softmax:</p>
            ${EX_TOKENS.map((t, j) => `
              <div class="mathline">score(drikker → ${W(t)}) = q<sub>drikker</sub> · k<sub>${W(t)}</sub> / √2 = <span class="res">${R().fmt(tr.scores[1][j])}</span></div>`).join("")}
            <div class="mathline">softmax(${tr.scores[1].map((s) => R().fmt(s)).join(", ")}) = [${A.map((a) => R().pct(a)).join(", ")}]</div>` : ""}
          ${calc(`
            <p class="note">Prikkproduktene bak scorene:</p>
            ${EX_TOKENS.map((t, j) => `
              <div class="mathline">q·k for ${W(t)}: ${R().dotCalc(tr.q[1], tr.k[j])}</div>`).join("")}`)}
          ${R().honest(`Attention er kausal: ingen token får se fremover i teksten («katten» ser bare
            seg selv). Og i en ekte LLM veier hver token tusenvis av tokens bak seg – samme mekanisme.`)}`;
      },
    },

    {
      id: "attnz",
      title: "Attention (3 av 3) · informasjonen blandes",
      sub: "et veid gjennomsnitt av V-vektorene",
      diagram: ["dg-attn"],
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
          <p>Resultatet <b>legges til</b> den opprinnelige vektoren (en <b>residualkobling</b>) –
          tokenen beholder seg selv og får attention-informasjonen som tillegg.</p>
          ${lv(1) ? `<div class="mathline">h<sub>drikker</sub> = x + z·W_O =
            ${R().vecHTML(tr.x[1], "data")} + ${R().vecHTML(tr.attnOut[1], "data")} = ${R().vecHTML(tr.h[1], "data")}</div>` : ""}
          ${calc(`
            <p class="note">W_O er attentions fjerde lærte matrise:</p>
            <div>${R().matHTML(m.Wo, "param", "W_O (2×2)")}</div>
            ${ML.R.vecMatCalc(tr.z[1], m.Wo, "z·W_O")}`)}
          ${R().honest(`Helt likt – og residualkoblinger er en av grunnene til at modeller med
            hundrevis av lag i det hele tatt lar seg trene.`)}`;
      },
    },

    {
      id: "ffn",
      title: "Feed forward",
      sub: "hver token bearbeides for seg",
      diagram: ["dg-ffn"],
      render() {
        const tr = trace();
        const m = ML.state.model;
        return `
          <p class="lede">Attention flyttet informasjon <b>mellom</b> tokens. <b>Feed forward</b>
          bearbeider hver token <b>for seg</b>: et bittelite nevralt nettverk – gang med en lært
          matrise, klipp negative tall til null (<b>ReLU</b>), gang med en lært matrise til.</p>
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
            <p class="note">Opp i bredde og ned igjen – det gir modellen arbeidsplass til å kombinere
            trekk. Resultatet legges til h (residual, som i attention).</p>`}
          ${calc(`
            <div>${R().matHTML(m.W1, "param", "W₁ (2×4)")} ${R().matHTML(m.W2, "param", "W₂ (4×2)")}</div>
            <p class="note"><b>1)</b> h · W₁:</p>${R().vecMatCalc(tr.h[1], m.W1, "pre")}
            <p class="note"><b>2)</b> ReLU: [${tr.pre[1].map((v) => `max(0, ${R().fmt(v)})`).join(", ")}] = ${R().vecHTML(tr.act[1], "data")}</p>
            <p class="note"><b>3)</b> act · W₂:</p>${R().vecMatCalc(tr.act[1], m.W2, "f")}`)}
          ${R().honest(`Feed-forward er den største delen av modellen: 4096 tall inn, ~14 000 i
            midten. Aktiveringen er gjerne GELU/SiLU – samme idé, mykere knekk.`)}`;
      },
    },

    {
      id: "logits",
      title: "Logits og softmax",
      sub: "hva tror modellen?",
      diagram: ["dg-logits"],
      render() {
        const tr = trace();
        const m = ML.state.model;
        const guessed = ML.state.logitGuess;
        if (guessed == null) {
          // «Gjett først»: forplikt deg før fasiten vises – da husker du svaret.
          return `
            <p class="lede">Den ferdige vektoren for siste token ganges med den lærte
            <b>output-matrisen</b> – da får hvert ord i vokabularet én score (<b>logit</b>), og
            <b>softmax</b> gjør scorene om til sannsynligheter.</p>
            <p><b>Men først – gjett selv:</b> hvilket ord tror du modellen tipper mest på etter
            «katten drikker» ${isTrained() ? "nå" : "– før den har trent i det hele tatt"}?</p>
            <div class="chiprow">
              ${ML.VOCAB.map((w, i) => `<button class="btn" data-guess="${i}">${w}</button>`).join(" ")}
            </div>`;
        }
        const top = tr.probs.indexOf(Math.max(...tr.probs));
        return `
          <p class="lede">Den ferdige vektoren for siste token ganges med den lærte
          <b>output-matrisen</b> – da får hvert ord i vokabularet én score (<b>logit</b>), og
          <b>softmax</b> gjør scorene om til sannsynligheter:</p>
          ${R().probBars(tr.probs, { correctIdx: TARGET, logits: lv(1) ? tr.logits : null })}
          <p>Du gjettet <b>«${W(guessed)}»</b> – modellen tipper mest på <b>«${W(top)}»</b>.
          ${guessed === top ? "Godt gjettet! 🎯" : ""}</p>
          ${lv(1) ? `
            <div class="mathline">logits = y · W_U, &nbsp; der y<sub>drikker</sub> = ${R().vecHTML(tr.y[1], "data")}</div>
            <div class="mathline">W_U (én kolonne per ord): ${R().matHTML(m.Wu, "param")}</div>` : ""}
          ${calc(`
            <div class="mathline">P(ord) = e^logit / sum av alle e^logit</div>
            ${ML.VOCAB.map((w, i) => `<div class="mathline">P(${w}) = e^${R().fmt(tr.logits[i])} / ${R().fmt(tr.logits.reduce((s, l) => s + Math.exp(l), 0))} = <span class="res">${R().pct(tr.probs[i])}</span></div>`).join("")}`)}
          ${isTrained() ? "" : `<p class="note">Modellen gjetter i blinde – som forventet, parameterne
            er jo tilfeldige ennå. Det skal vi gjøre noe med.</p>`}`;
      },
      wire(host) {
        host.querySelectorAll("[data-guess]").forEach((b) =>
          b.addEventListener("click", () => {
            ML.state.logitGuess = Number(b.dataset.guess);
            renderTrain();
          })
        );
      },
    },

    {
      id: "loss",
      title: "Loss",
      sub: "hvor feil tok modellen?",
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
              <div class="bignum ${loss < 0.2 ? "good" : "bad"}">${R().fmt(loss)}</div>
              <div class="bignum-cap">loss (lavere = bedre)</div>
            </div>
          </div>
          ${lv(1) ? R().probBars(tr.probs, { correctIdx: TARGET }) : ""}
          ${calc(`
            <div class="mathline">loss = −ln( P(riktig token) ) = −ln(${R().fmt(p, 3)}) = <span class="res">${R().fmt(loss, 3)}</span></div>
            <p class="note">−ln(1.0) = 0 (perfekt), −ln(0.5) ≈ 0.69, −ln(0.1) ≈ 2.3 – straffen vokser
            bratt jo sikrere modellen var på feil ting. Kalles cross-entropy.</p>`)}`;
      },
    },

    {
      id: "backprop",
      title: "Backpropagation",
      sub: "læringen skjer baklengs",
      diagram: [], // de røde pilene ER markeringen her
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
          ${applied
            ? `<p><b style="color:var(--good)">✅ Oppdatert!</b> ${changed} av ${total} parametere
               flyttet seg et lite hakk – <b>samtidig</b>. Gå videre og se om modellen ble bedre.</p>`
            : `<p style="margin:16px 0"><button class="btn accent big" id="bp-apply">Kjør oppdateringen (gradient descent)</button>
               <span class="note" style="margin-left:10px">justerer ${changed} av ${total} parametere samtidig</span></p>`}
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
            </table>
            <p class="note">Gradientene er eksakte – kjerneregelen fra matematikk, ikke prøving og
            feiling. Parametere som ikke var i bruk (f.eks. embeddingen til ord utenfor setningen)
            får gradient 0 og står stille.</p>` : ""}
          ${calc(`
            <p class="note"><b>Hvor starter gradientene?</b> Første ledd er vakkert enkelt:
            gradienten på logitene = sannsynlighetene − fasiten:</p>
            <div class="mathline">${ML.VOCAB.map((w, i) => `d(${w}) = ${R().fmt(tr.probs[i], 2)} − ${i === TARGET ? 1 : 0} = <span class="res">${R().fmt(tr.probs[i] - (i === TARGET ? 1 : 0), 2)}</span>`).join("<br>")}</div>
            <p class="note">Negativt på «melk» = «skulle vært høyere». Derfra sendes tallene baklengs
            gjennom de samme operasjonene som i fremoverpasset – bare derivert.</p>`)}
          ${R().honest(`I en ekte modell justeres milliarder av parametere i hvert eneste steg –
            nøyaktig samme regnestykke, bare større.`)}`;
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
      title: "Treningsløkka",
      sub: "gjett → mål feilen → juster · om og om igjen",
      diagram: ["dg-next"],
      render() {
        const m = ML.state.model;
        const tr = trace();
        const p = tr.probs[TARGET];
        const loss = m.loss(tr, TARGET);
        const hist = ML.state.history;
        const learned = p > 0.9;
        return `
          <p class="lede">Ekte læring er bare forrige steg om igjen og om igjen:
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
          <div class="spark-cap">P(«melk») gjennom treningen:</div>
          ${R().sparkline(hist.map((h) => h.p), { max: 1, color: "var(--good)" })}
          ${lv(1) ? `
            <div class="spark-cap">Loss gjennom treningen:</div>
            ${R().sparkline(hist.map((h) => h.loss), { color: "var(--bad)" })}
            ${R().probBars(tr.probs, { correctIdx: TARGET })}` : ""}
          ${learned
            ? `<p><b style="color:var(--good)">🎉 Modellen har lært det!</b> Ingen regler ble
               programmert inn – bare 60 tall som ble dyttet litt i riktig retning mange nok ganger.
               På tide å <b>bruke</b> den.</p>
               <button class="btn accent big pulse" id="goto-use">Gå til «Bruk modellen» →</button>`
            : `<p class="note">Tren til «melk» passerer 90 % – da er modellen klar til bruk.</p>`}`;
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
    host.innerHTML = `
      <div class="card">
        <div class="step-kicker">Treningsmodus · steg ${i + 1} av ${steps.length}</div>
        <h2>${step.title} <span class="h2-sub">${step.sub || ""}</span></h2>
        ${step.render()}
      </div>`;
    ML.diagram.highlight(step.diagram);
    ML.diagram.setBackward(!!step.backward);
    if (step.wire) step.wire(host);
    ML.state.maxTrainStep = Math.max(ML.state.maxTrainStep || 0, i);
    const goto = (j) => {
      ML.state.stepIdx = j;
      ML.state.bpJustApplied = false;
      renderTrain();
      window.scrollTo({ top: 0, behavior: "smooth" });
    };
    ML.R.renderNav({
      count: steps.length,
      current: i,
      maxVisited: ML.state.maxTrainStep,
      titles: steps.map((s) => s.title),
      onPrev() { goto(Math.max(0, i - 1)); },
      onNext() { goto(Math.min(steps.length - 1, i + 1)); },
      onJump(j) { goto(j); },
    });
  }

  ML.renderTrain = renderTrain;
  ML.TRAIN = { EX_TOKENS, TARGET, LR };
  // Klikk i diagrammet → hopp til steget som forklarer den delen.
  ML.TRAIN_DIAG = {
    "dg-text": 0, "dg-tokens": 1, "dg-emb": 2, "dg-layer": 4, "dg-attn": 4,
    "dg-ffn": 7, "dg-logits": 8, "dg-softmax": 8, "dg-next": 11,
  };
})();
