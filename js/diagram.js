/* Persistent arkitekturdiagram i sidepanelet, med steg-highlighting
   og egen "baklengs"-modus for backpropagation. */
(function () {
  const ML = globalThis.ML;

  function build(el) {
    el.innerHTML = `
      <div class="dg" id="dg-root">
        <div class="dg-box" id="dg-text">Tekst<span class="dg-sub">«katten drikker»</span></div>
        <div class="dg-arrow">↓</div>
        <div class="dg-box" id="dg-tokens">Tokens</div>
        <div class="dg-arrow">↓</div>
        <div class="dg-box" id="dg-emb">Embeddinger<span class="dg-sub">+ posisjon</span></div>
        <div class="dg-arrow">↓</div>
        <div class="dg-layer" id="dg-layer">
          <div class="dg-layer-cap">Transformer-lag × 1</div>
          <div class="dg-box" id="dg-attn">Attention<span class="dg-sub">Q · K · V</span></div>
          <div class="dg-arrow">↓</div>
          <div class="dg-box" id="dg-ffn">Feed forward</div>
        </div>
        <div class="dg-arrow">↓</div>
        <div class="dg-box" id="dg-logits">Logits</div>
        <div class="dg-arrow">↓</div>
        <div class="dg-box" id="dg-softmax">Softmax</div>
        <div class="dg-arrow">↓</div>
        <div class="dg-box" id="dg-sampling">Sampling</div>
        <div class="dg-arrow">↓</div>
        <div class="dg-box" id="dg-next">Neste token</div>
        <div class="dg-bwnote">◀ gradienter flyter baklengs</div>
      </div>`;
  }

  function highlight(ids = []) {
    document.querySelectorAll(".dg-box, .dg-layer").forEach((b) => {
      b.classList.toggle("active", ids.includes(b.id));
      b.classList.remove("hl-param");
    });
  }

  // Oransje markering brukt av parameter-inspektøren.
  function highlightParams(ids = []) {
    document.querySelectorAll(".dg-box, .dg-layer").forEach((b) => {
      b.classList.toggle("hl-param", ids.includes(b.id));
    });
  }

  function setBackward(on) {
    const root = document.getElementById("dg-root");
    if (root) root.classList.toggle("bw", on);
  }

  ML.diagram = { build, highlight, highlightParams, setBackward };
})();
