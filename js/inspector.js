/* "Hva er inni modellen vår?" – parametertelling som alltid stemmer med
   den faktiske modellen, med klikk-for-å-se-hvor-parameterne-bor. */
(function () {
  const ML = globalThis.ML;

  function rows() {
    const c = ML.state.model.paramCounts();
    return [
      {
        label: "Vokabular", value: `${c.vocab} tokens`, hl: ["dg-tokens"],
        desc: "Ordene modellen kjenner: katten, drikker, melk, vann, sover. Ekte modeller har ~100 000 token-biter.",
      },
      {
        label: "Embedding-dimensjoner", value: c.dims, hl: ["dg-emb"],
        desc: "Hver token blir en vektor med 2 tall. GPT-klassen bruker tusenvis av dimensjoner per token.",
      },
      {
        label: "Embedding-parametere", value: c.embedding, hl: ["dg-emb"],
        desc: `${c.vocab} tokens × ${c.dims} dimensjoner = ${c.embedding} lærte tall i embedding-tabellen E.`,
      },
      {
        label: "Posisjons-parametere", value: c.position, hl: ["dg-emb"],
        desc: `${ML.MAXPOS} posisjoner × ${c.dims} dimensjoner = ${c.position} lærte tall som forteller modellen HVOR i setningen en token står.`,
      },
      {
        label: "Attention-parametere", value: c.attention, hl: ["dg-attn"],
        desc: "Fire 2×2-matriser: W_Q, W_K, W_V og W_O. Det er disse som avgjør hvordan tokens ser på hverandre.",
      },
      {
        label: "Feed-forward-parametere", value: c.ffn, hl: ["dg-ffn"],
        desc: "W₁ (2×4) og W₂ (4×2): et bittelite nevralt nettverk som bearbeider hver posisjon for seg.",
      },
      {
        label: "Output-parametere", value: c.output, hl: ["dg-logits"],
        desc: `W_U (${c.dims}×${c.vocab}) oversetter den ferdige vektoren til én score (logit) per ord i vokabularet.`,
      },
      {
        label: "TOTALT", value: c.total, total: true, hl: ["dg-emb", "dg-attn", "dg-ffn", "dg-logits"],
        desc: `Alle ${c.total} tallene modellen lærer. Når noen sier «7B parametere» mener de nøyaktig dette – bare 7 000 000 000 av dem. Samme byggeklosser, gigantiske matriser.`,
      },
    ];
  }

  let selected = -1;

  function render() {
    const el = document.getElementById("inspector");
    const list = rows();
    el.innerHTML =
      list
        .map(
          (r, i) => `<button class="insp-row ${r.total ? "total" : ""} ${i === selected ? "sel" : ""}" data-insp="${i}">
            <span>${r.label}</span><span class="cnt">${r.value}</span>
          </button>`
        )
        .join("") +
      (selected >= 0 ? `<div class="insp-desc">${list[selected].desc}</div>` : "") +
      `<div class="insp-note">💡 Klikk en rad for å se hvor i arkitekturen parameterne bor. Modellen har trent <b>${ML.state.model.steps}</b> steg.</div>`;

    el.querySelectorAll("[data-insp]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const i = Number(btn.dataset.insp);
        selected = selected === i ? -1 : i;
        ML.diagram.highlightParams(selected >= 0 ? list[selected].hl : []);
        render();
      });
    });
  }

  ML.inspector = { render };
})();
