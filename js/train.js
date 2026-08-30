/* Treningsmodus: ett steg av gangen gjennom én ekte treningsrunde på
   eksempelet «katten drikker» → «melk», og deretter fri trening. */
(function () {
  const ML = globalThis.ML;
  const R = () => ML.R;

  const EX_TOKENS = [0, 1]; // katten drikker
  const TARGET = 2;         // melk
  const LR = 0.2;

  const W = (i) => ML.VOCAB[i];

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
          <p class="lede">Modellen vår skal lære én ting: at etter <b>«katten drikker»</b> kommer
          <b>«melk»</b>. Den er absurd liten – bare <b>60 parametere</b> – men den er bygget av
          nøyaktig de samme delene som ChatGPT: embeddings, attention med Q/K/V, feed-forward,
          softmax og backpropagation. Ingen juks, bare små matriser.</p>
          <div class="chiprow">
            ${R().tokenChip("katten")} ${R().tokenChip("drikker")}
            <span class="arrow-inline">→</span>
            <span class="chip win">melk</span>
            <span class="note">(fasiten modellen skal lære)</span>
          </div>
          <p>Underveis er det <b>én</b> distinksjon som er viktigere enn alt annet. Hold øye med fargene:</p>
          <div class="legend">
            <div class="legend-item data"><b>🔵 DATA</b> Tall som flyter <i>gjennom</i> modellen akkurat nå
            – de skapes av teksten din og forsvinner etterpå. Eksempel: ${R().vecHTML([0.7, -0.4], "data")}</div>
            <div class="legend-item param"><b>🟠 PARAMETERE</b> Tall som <i>bor inne i</i> modellen
            – de er det modellen har lært, og det er bare disse som endres under trening.</div>
          </div>
          <p class="note">Akkurat nå er alle 60 parameterne tilfeldige tall. Modellen kan ingenting.
          Trykk «Neste» og se hva som skjer i én treningsrunde.</p>`;
      },
    },

    {
      id: "token",
      title: "Steg 1 · Tokenisering",
      diagram: ["dg-text", "dg-tokens"],
      render() {
        return `
          <p class="lede">Først deles teksten i <b>tokens</b> – bitene modellen jobber med.
          Hos oss er 1 ord = 1 token, og hver token har et fast nummer i vokabularet:</p>
          <div class="chiprow">
            <span class="chip plain">«katten drikker»</span>
            <span class="arrow-inline">→</span>
            ${R().tokenChip("katten", 0)} ${R().tokenChip("drikker", 1)}
          </div>
          <p>Dette steget er helt <b>deterministisk</b> – ren oppslagstabell, ingen læring, ingen
          parametere. Samme tekst gir alltid samme tokens.</p>
          ${R().disclosure("📖 Vis hele vokabularet", `
            <div class="chiprow">${ML.VOCAB.map((w, i) => R().tokenChip(w, i)).join(" ")}</div>
            <p class="note">Hele verdenen til modellen vår er disse 5 ordene. Alt den noensinne kan si,
            må være ett av dem.</p>`)}
          <div class="honest">tokeniseringen deler ord i mindre biter («subord»), så «drikker» kunne
          blitt f.eks. «drik» + «ker», og vokabularet har ~100 000 biter. Prinsippet er det samme.</div>`;
      },
    },

    {
      id: "emb",
      title: "Steg 2 · Embeddinger – tokens blir tall",
      diagram: ["dg-emb"],
      render() {
        const tr = trace();
        const m = ML.state.model;
        return `
          <p class="lede">Modellen kan ikke regne på ord. Hver token slås derfor opp i en
          <b>embedding-tabell</b> og blir en <b>vektor</b> – en liste med tall. Hos oss:
          2 tall per token (<b>2 dimensjoner</b>).</p>
          <div style="margin:12px 0">
            ${ML.VOCAB.map((w, i) => `
              <div class="chiprow">
                ${R().tokenChip(w, i)}
                <span class="arrow-inline">→</span>
                ${R().vecHTML(m.E[i], "param")}
                ${EX_TOKENS.includes(i) ? "" : `<span class="note">(ikke i bruk i denne setningen)</span>`}
              </div>`).join("")}
          </div>
          <p>⚠️ Legg merke til fargen: embedding-tabellen er <b>🟠 parametere</b>. Verdiene er
          <b>lærte</b> – og siden vi ikke har trent ennå, er de bare tilfeldige tall. Det finnes ingen
          fasit for hva «katten» skal være som vektor; modellen finner det ut selv under trening.</p>
          ${R().disclosure("🔢 Vis tallene: pluss posisjon", `
            <p>En transformer ser alle tokens samtidig og vet ikke rekkefølgen av seg selv. Derfor legges
            det til en lært <b>posisjonsvektor</b> per plass i setningen:</p>
            ${EX_TOKENS.map((t, i) => `
              <div class="mathline">x<sub>${W(t)}</sub> = E[${W(t)}] + P[pos ${i}] =
                ${R().vecHTML(ML.state.model.E[t], "param")} + ${R().vecHTML(ML.state.model.P[i], "param")}
                = ${R().vecHTML(tr.x[i], "data")}</div>`).join("")}
            <p class="note">Resultatet x er <b>🔵 data</b>: parametere ble kombinert med input og
            begynner nå å flyte gjennom modellen.</p>`)}
          ${R().disclosure("🧮 Vis utregningen", `
            ${EX_TOKENS.map((t, i) => `
              <div class="mathline">x<sub>${W(t)}</sub>[0] = ${R().fmt(ML.state.model.E[t][0])} + ${R().fmt(ML.state.model.P[i][0])} = <span class="res">${R().fmt(tr.x[i][0])}</span></div>
              <div class="mathline">x<sub>${W(t)}</sub>[1] = ${R().fmt(ML.state.model.E[t][1])} + ${R().fmt(ML.state.model.P[i][1])} = <span class="res">${R().fmt(tr.x[i][1])}</span></div>`).join("")}`)}
          <div class="honest">samme oppslag, men vektorene har f.eks. 4096 dimensjoner i stedet for 2,
          og posisjon kodes ofte med et triks kalt RoPE i stedet for en lært tabell.</div>`;
      },
    },

    {
      id: "attn",
      title: "Steg 3 · Attention – tokens ser på hverandre",
      diagram: ["dg-layer", "dg-attn"],
      render() {
        const tr = trace();
        const m = ML.state.model;
        const A = tr.alphas[1];
        const qkv = ML.state.qkvTab;
        const info = {
          q: {
            name: "Q – Query", weight: m.Wq, label: "W_Q",
            what: "«Hva leter jeg etter?» Hver token lager en <b>spørrevektor</b> q ved å gange sin egen vektor x med den lærte matrisen W_Q.",
            out: tr.q,
          },
          k: {
            name: "K – Key", weight: m.Wk, label: "W_K",
            what: "«Hva kan jeg tilby?» Hver token lager en <b>nøkkelvektor</b> k med den lærte matrisen W_K. Query og Key sammenlignes for å avgjøre hvem som er relevant for hvem.",
            out: tr.k,
          },
          v: {
            name: "V – Value", weight: m.Wv, label: "W_V",
            what: "«Hva sender jeg videre hvis noen bryr seg om meg?» Hver token lager en <b>verdivektor</b> v med den lærte matrisen W_V. Det er verdiene som faktisk blandes sammen til slutt.",
            out: tr.v,
          },
        }[qkv];
        return `
          <p class="lede">Hittil har hver token levd i sin egen boble. <b>Attention</b> lar dem utveksle
          informasjon: «drikker» får hente inn hva slags subjekt den hører sammen med. Det skjer ikke ved
          magi – det er tre lærte matriser og litt ganging.</p>

          <p>Hver token lager tre nye vektorer av sin x: <b>Q</b>uery, <b>K</b>ey og <b>V</b>alue.
          Klikk på dem:</p>
          <div class="qkv-tabs">
            ${["q", "k", "v"].map((t) => `<button class="qkv-tab ${qkv === t ? "active" : ""}" data-qkv="${t}">${t.toUpperCase()}</button>`).join("")}
          </div>
          <div class="qkv-panel">
            <p style="margin-top:0"><b>${info.name}:</b> ${info.what}</p>
            <div>
              ${R().matHTML(info.weight, "param", `${info.label} (lærte parametere, 2×2)`)}
              <span class="arrow-inline" style="vertical-align:24px">brukes slik:</span>
            </div>
            ${EX_TOKENS.map((t, i) => `
              <div class="mathline">${qkv}<sub>${W(t)}</sub> = x<sub>${W(t)}</sub> · ${info.label} =
                ${R().vecHTML(tr.x[i], "data")} · ${info.label} = ${R().vecHTML(info.out[i], "data")}</div>`).join("")}
            ${R().disclosure("🧮 Vis utregningen for «drikker»", ML.R.vecMatCalc(tr.x[1], info.weight, qkv))}
          </div>

          <p style="margin-top:18px"><b>Hvor mye ser «drikker» på hver token?</b>
          Spørrevektoren til «drikker» prikkes mot nøkkelen til hver token
          (score = q·k / √2), og softmax gjør scorene om til vekter som summerer til 100 %:</p>
          <div class="attn-weights">
            ${EX_TOKENS.map((t, j) => `
              <div class="aw-row">
                <div><b>${W(t)}</b></div>
                <div class="aw-bar"><div class="aw-fill" style="width:${A[j] * 100}%"></div></div>
                <div class="aw-pct">${R().pct(A[j])}</div>
              </div>`).join("")}
          </div>
          ${R().disclosure("🔢 Vis tallene bak vektene", `
            ${EX_TOKENS.map((t, j) => `
              <div class="mathline">score(drikker → ${W(t)}) = q<sub>drikker</sub>·k<sub>${W(t)}</sub> / √2 =
              (${R().dotCalc(tr.q[1], tr.k[j])}) / 1.41 = <span class="res">${R().fmt(tr.scores[1][j])}</span></div>`).join("")}
            <div class="mathline">softmax(${tr.scores[1].map((s) => R().fmt(s)).join(", ")}) =
              [${A.map((a) => R().pct(a)).join(", ")}]</div>
            <p class="note">«katten» får bare lov å se bakover (på seg selv) – en token får aldri se
            fremover i setningen. Det kalles kausal attention.</p>`)}

          <p><b>Så blandes verdiene.</b> Den nye representasjonen for «drikker» er et veid gjennomsnitt
          av alle V-vektorene – slik flyter informasjon fra «katten» inn i «drikker»:</p>
          <div class="mathline">z<sub>drikker</sub> = ${R().pct(A[0], 0)} · v<sub>katten</sub> + ${R().pct(A[1], 0)} · v<sub>drikker</sub> = ${R().vecHTML(tr.z[1], "data")}</div>
          ${R().disclosure("🔢 Vis siste ledd (W_O og residual)", `
            <p>Blandingen ganges med en fjerde lært matrise ${R().matHTML(m.Wo, "param", "W_O (2×2)")} og
            <b>legges til</b> den opprinnelige vektoren (en «residualkobling» – tokenen beholder seg selv
            og får attention-informasjonen som tillegg):</p>
            <div class="mathline">h<sub>drikker</sub> = x<sub>drikker</sub> + z<sub>drikker</sub>·W_O =
              ${R().vecHTML(tr.x[1], "data")} + ${R().vecHTML(tr.attnOut[1], "data")} = ${R().vecHTML(tr.h[1], "data")}</div>`)}
          <div class="honest">nøyaktig samme regnestykke, men med mange attention-hoder parallelt
          (f.eks. 32) og over tusenvis av tokens samtidig. Vektene her er tilfeldige ennå –
          «hvem ser på hvem» er også noe som læres.</div>`;
      },
      wire(host) {
        host.querySelectorAll("[data-qkv]").forEach((b) =>
          b.addEventListener("click", () => {
            ML.state.qkvTab = b.dataset.qkv;
            renderTrain();
          })
        );
      },
    },

    {
      id: "ffn",
      title: "Steg 4 · Feed forward – et bittelite nevralt nettverk",
      diagram: ["dg-layer", "dg-ffn"],
      render() {
        const tr = trace();
        const m = ML.state.model;
        return `
          <p class="lede">Etter attention bearbeides hver posisjon for seg av et lite
          <b>nevralt nettverk</b>: vektoren ganges med en lært matrise, negative tall klippes til null
          (ReLU), og resultatet ganges med enda en lært matrise.</p>
          <div class="ffn-flow">
            <div class="ffn-stage">${R().vecHTML(tr.h[1], "data")}<div class="stage-cap">h (2 tall)</div></div>
            <span class="op">· W₁ (2×4)</span>
            <div class="ffn-stage">${R().vecHTML(tr.pre[1], "data")}<div class="stage-cap">4 tall</div></div>
            <span class="op">ReLU</span>
            <div class="ffn-stage">${R().vecHTML(tr.act[1], "data")}<div class="stage-cap">negative → 0</div></div>
            <span class="op">· W₂ (4×2)</span>
            <div class="ffn-stage">${R().vecHTML(tr.f[1], "data")}<div class="stage-cap">f (2 tall)</div></div>
          </div>
          <p>Og som i attention-steget: resultatet <b>legges til</b> det som kom inn (residual):</p>
          <div class="mathline">y<sub>drikker</sub> = h + f = ${R().vecHTML(tr.h[1], "data")} + ${R().vecHTML(tr.f[1], "data")} = ${R().vecHTML(tr.y[1], "data")}</div>
          ${R().disclosure("🔢 Vis parameterne", `
            <div>${R().matHTML(m.W1, "param", "W₁ (2×4) – lærte parametere")}
            ${R().matHTML(m.W2, "param", "W₂ (4×2) – lærte parametere")}</div>
            <p class="note">Legg merke til at nettverket først går <i>opp</i> i bredde (2 → 4 tall) og så
            ned igjen (4 → 2). Det gir modellen «arbeidsplass» til å kombinere trekk.</p>`)}
          ${R().disclosure("🧮 Vis utregningen", `
            <p><b>1) h · W₁:</b></p>${R().vecMatCalc(tr.h[1], m.W1, "pre")}
            <p><b>2) ReLU</b> (alt under null blir null):</p>
            <div class="mathline">act = [${tr.pre[1].map((v) => `max(0, ${R().fmt(v)})`).join(", ")}] = ${R().vecHTML(tr.act[1], "data")}</div>
            <p><b>3) act · W₂:</b></p>${R().vecMatCalc(tr.act[1], m.W2, "f")}`)}
          <div class="honest">feed-forward-laget er den største delen av modellen: 4096 tall inn,
          ~14 000 i midten. Aktiveringen er gjerne GELU/SiLU i stedet for ReLU – samme idé, mykere knekk.</div>`;
      },
    },

    {
      id: "logits",
      title: "Steg 5 · Logits og softmax – hva tror modellen?",
      diagram: ["dg-logits", "dg-softmax"],
      render() {
        const tr = trace();
        const m = ML.state.model;
        return `
          <p class="lede">Den ferdige vektoren for siste token, y<sub>drikker</sub> = ${R().vecHTML(tr.y[1], "data")},
          inneholder nå alt modellen «mener» om fortsettelsen. Ved å gange den med den lærte
          output-matrisen W<sub>U</sub> får hvert ord i vokabularet én score – en <b>logit</b>:</p>
          <div class="probbars">
            ${ML.VOCAB.map((w, i) => `
              <div class="pb-row">
                <div class="pb-label">${w}</div>
                <div class="pb-bar"><div class="pb-fill" style="width:${Math.max(2, (tr.logits[i] - Math.min(...tr.logits)) / (Math.max(...tr.logits) - Math.min(...tr.logits) + 1e-9) * 100)}%"></div></div>
                <div class="pb-val num">logit ${R().fmt(tr.logits[i])}</div>
              </div>`).join("")}
          </div>
          <p><b>Softmax</b> gjør scorene om til sannsynligheter som summerer til 100 %:</p>
          ${R().probBars(tr.probs, { correctIdx: TARGET })}
          ${R().disclosure("🔢 Vis output-matrisen", `
            ${R().matHTML(m.Wu, "param", "W_U (2×5) – én kolonne per ord i vokabularet")}
            <p class="note">Kolonne nr. ${TARGET} hører til «melk». Logit(melk) = y·kolonnen =
            ${R().dotCalc(tr.y[1], m.Wu.map((r) => r[TARGET]))}</p>`)}
          ${R().disclosure("🧮 Vis softmax-utregningen", `
            <div class="mathline">P(ord) = e^logit / sum av alle e^logit</div>
            <div class="mathline">sum = ${tr.logits.map((l) => `e^${R().fmt(l)}`).join(" + ")} = ${R().fmt(tr.logits.reduce((s, l) => s + Math.exp(l), 0))}</div>
            ${ML.VOCAB.map((w, i) => `<div class="mathline">P(${w}) = e^${R().fmt(tr.logits[i])} / ${R().fmt(tr.logits.reduce((s, l) => s + Math.exp(l), 0))} = <span class="res">${R().pct(tr.probs[i])}</span></div>`).join("")}`)}
          <p class="note">Modellen gjetter altså i blinde nå – som forventet, siden alle parameterne er
          tilfeldige. Det skal vi gjøre noe med.</p>`;
      },
    },

    {
      id: "loss",
      title: "Steg 6 · Loss – hvor feil tok modellen?",
      diagram: ["dg-softmax"],
      render() {
        const tr = trace();
        const m = ML.state.model;
        const p = tr.probs[TARGET];
        const loss = m.loss(tr, TARGET);
        return `
          <p class="lede">Under trening <b>vet vi fasiten</b>: etter «katten drikker» skal det komme
          «melk». Da kan vi måle hvor feil modellen tok – det tallet kalles <b>loss</b>.</p>
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
          ${R().probBars(tr.probs, { correctIdx: TARGET })}
          <p>Loss er rett og slett «hvor overrasket ble modellen over riktig svar?». Perfekt svar
          (100 % på melk) gir loss 0. Jo mindre sannsynlighet modellen ga «melk», jo høyere loss.</p>
          ${R().disclosure("🧮 Vis utregningen (cross-entropy)", `
            <div class="mathline">loss = −ln( P(riktig token) ) = −ln(${R().fmt(p, 3)}) = <span class="res">${R().fmt(loss, 3)}</span></div>
            <p class="note">−ln fordi: −ln(1.0) = 0 (perfekt), −ln(0.5) ≈ 0.69, −ln(0.1) ≈ 2.3 –
            straffen vokser bratt jo sikrere modellen var på feil ting.</p>`)}
          <p class="note">Nå har vi ett tall som sier hvor dårlig hele modellen var. Neste steg er
          selve magien i maskinlæring: å bruke det tallet til å forbedre alle 60 parameterne samtidig.</p>`;
      },
    },

    {
      id: "backprop",
      title: "Steg 7 · Backpropagation – læringen skjer baklengs",
      diagram: ["dg-emb", "dg-layer", "dg-attn", "dg-ffn", "dg-logits", "dg-softmax"],
      backward: true,
      render() {
        const m = ML.state.model;
        const tr = trace();
        const g = m.backward(tr, TARGET);
        const changed = m.countNonzeroGrads(g);
        const total = m.paramCounts().total;
        // Vis parameteren med størst gradient i hver del av modellen,
        // så tabellen alltid har levende tall.
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
          <p class="lede">Nå går vi <b>baklengs</b> gjennom hele modellen – se de røde pilene i
          diagrammet. For hver eneste parameter regnes en <b>gradient</b>: et tall som svarer på</p>
          <p style="text-align:center;font-size:17px"><i>«Hvis akkurat denne parameteren var litt
          annerledes – ville svaret blitt bedre eller verre?»</i></p>
          <p>Dette er ikke prøving og feiling: gradienten regnes eksakt, med kjerneregelen fra
          matematikk, bakover gjennom softmax → output → feed-forward → attention → embeddinger.
          Så flyttes hver parameter et lite hakk i sin beste retning:</p>
          <div class="mathline" style="text-align:center">ny verdi = gammel verdi − ${LR} · gradient
            <span class="note">(${LR} er «læringsraten» – hvor store skritt vi tar)</span></div>

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
          <p class="note">⚠️ Tabellen viser bare 5 av parameterne. I dette steget får
          <b>${changed} av ${total}</b> parametere en gradient ≠ 0 og justeres <b>samtidig</b>
          (de øvrige, f.eks. embeddingen til ord som ikke var med i setningen, står stille).
          I en ekte modell justeres milliarder av parametere i hvert eneste steg.</p>
          ${applied
            ? `<p><b style="color:var(--good)">✅ Oppdatert!</b> Alle parameterne har flyttet seg et lite hakk.
               Gå til neste steg og se om modellen ble bedre.</p>`
            : `<button class="btn accent big" id="bp-apply">✨ Kjør oppdateringen (gradient descent)</button>`}
          ${R().disclosure("🔢 Hvor kommer gradientene fra?", `
            <p>Første ledd er alltid det samme, og det er vakkert enkelt:
            <b>gradienten på logitene = sannsynlighetene − fasiten.</b></p>
            <div class="mathline">${ML.VOCAB.map((w, i) => `d(${w}) = ${R().fmt(tr.probs[i], 2)} − ${i === TARGET ? 1 : 0} = <span class="res">${R().fmt(tr.probs[i] - (i === TARGET ? 1 : 0), 2)}</span>`).join("<br>")}</div>
            <p class="note">Negativ gradient på «melk» betyr «skulle vært høyere», positiv på de andre
            betyr «skulle vært lavere». Derfra sendes disse tallene baklengs gjennom nøyaktig de samme
            regneoperasjonene som i fremoverpasset – bare derivert.</p>`)}`;
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
            ML.state.bpJustApplied = true;
            renderTrain();
            ML.inspector.render();
          });
      },
    },

    {
      id: "again",
      title: "Steg 8 · Tren igjen og igjen … og se den lære",
      diagram: ["dg-next"],
      render() {
        const m = ML.state.model;
        const tr = trace();
        const p = tr.probs[TARGET];
        const loss = m.loss(tr, TARGET);
        const hist = ML.state.history;
        const learned = p > 0.9;
        return `
          <p class="lede">Én runde = liten forbedring. Ekte læring er bare denne løkka om igjen og om
          igjen: <b>gjett → mål feilen → juster parameterne</b>. Prøv selv:</p>
          <div class="chiprow">
            <button class="btn primary" data-train="1">▶ Tren 1 steg</button>
            <button class="btn primary" data-train="10">⏩ Tren 10 steg</button>
            <button class="btn primary" data-train="100">⏭ Tren 100 steg</button>
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
          ${R().probBars(tr.probs, { correctIdx: TARGET })}
          <div class="spark-cap">P(«melk») gjennom treningen:</div>
          ${R().sparkline(hist.map((h) => h.p), { max: 1, color: "var(--good)" })}
          <div class="spark-cap">Loss gjennom treningen:</div>
          ${R().sparkline(hist.map((h) => h.loss), { color: "var(--bad)" })}
          <p class="note">Alt du ser er de samme 60 parameterne som flyttes bittelitt hver runde –
          ingen regler er programmert inn, ingen ordbok, bare gradient descent.</p>
          ${learned
            ? `<p><b style="color:var(--good)">🎉 Modellen har lært det!</b> Den er ikke lenger i tvil om
               hva katten drikker. På tide å <b>bruke</b> den – der er reglene helt annerledes.</p>
               <button class="btn accent big pulse" id="goto-use">▶️ Gå til «Bruk modellen»</button>`
            : `<p class="note">💡 Tren til «melk» passerer 90 % – da låser vi modellen og tar den i bruk.</p>`}`;
      },
      wire(host) {
        host.querySelectorAll("[data-train]").forEach((b) =>
          b.addEventListener("click", () => {
            const n = Number(b.dataset.train);
            for (let i = 0; i < n; i++) {
              ML.state.model.trainStep(EX_TOKENS, TARGET, LR);
              recordEval();
            }
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
    const kicker = i === 0 ? "Treningsmodus" : `Treningsmodus · ${i} av ${steps.length - 1}`;
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
