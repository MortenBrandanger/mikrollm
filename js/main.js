/* Oppstart, modusbytte og globale hendelser. */
(function () {
  const ML = globalThis.ML;

  ML.state = {
    model: new ML.TinyLM(42),
    mode: "train",
    stepIdx: 0,
    useStepIdx: 0,
    maxTrainStep: 0,
    maxUseStep: 0,
    logitGuess: null,
    level: 0, // 0 = Enkelt, 1 = Med tall, 2 = Full utregning
    history: [],
    qkvTab: "q",
    bpJustApplied: false,
    use: null,
  };

  // Treningen skal overleve at siden lastes på nytt: parametere og
  // historikk lagres i localStorage etter hvert treningssteg.
  const STORE_KEY = "mikrollm-v1";
  ML.saveModel = function () {
    try {
      const m = ML.state.model;
      localStorage.setItem(STORE_KEY, JSON.stringify({
        E: m.E, P: m.P, Wq: m.Wq, Wk: m.Wk, Wv: m.Wv, Wo: m.Wo,
        W1: m.W1, W2: m.W2, Wu: m.Wu, steps: m.steps,
        history: ML.state.history,
      }));
    } catch (e) { /* privat modus e.l. – da lever modellen bare i fanen */ }
  };
  function loadModel() {
    try {
      const raw = localStorage.getItem(STORE_KEY);
      if (!raw) return false;
      const d = JSON.parse(raw);
      const m = ML.state.model;
      for (const k of ["E", "P", "Wq", "Wk", "Wv", "Wo", "W1", "W2", "Wu"]) {
        if (!Array.isArray(d[k])) return false;
        m[k] = d[k];
      }
      m.steps = d.steps || 0;
      ML.state.history = Array.isArray(d.history) ? d.history : [];
      return ML.state.history.length > 0;
    } catch (e) { return false; }
  }
  ML.clearSavedModel = function () {
    try { localStorage.removeItem(STORE_KEY); } catch (e) {}
  };

  const LEVELS = ["Enkelt", "Med tall", "Utregning"];
  function renderLevelSwitch() {
    const el = document.getElementById("level-switch");
    el.innerHTML = LEVELS.map(
      (name, i) => `<button class="level-btn ${ML.state.level === i ? "active" : ""}" data-level="${i}">${name}</button>`
    ).join("");
    el.querySelectorAll("[data-level]").forEach((b) =>
      b.addEventListener("click", () => {
        ML.state.level = Number(b.dataset.level);
        renderLevelSwitch();
        render();
      })
    );
  }

  function render() {
    if (ML.state.mode === "train") ML.renderTrain();
    else ML.renderUse();
    ML.inspector.render();
    document.getElementById("tab-train").classList.toggle("active", ML.state.mode === "train");
    document.getElementById("tab-use").classList.toggle("active", ML.state.mode === "use");
    // Vis modellstatus i fanen: ✓ når modellen faktisk har lært oppgaven.
    const learned = ML.state.model.forward([0, 1]).probs[2] > 0.9;
    document.getElementById("tab-use").textContent = learned ? "2 · Bruk modellen ✓" : "2 · Bruk modellen";
  }
  ML.renderAll = render;

  ML.stepForDiagram = function (id) {
    const map = ML.state.mode === "train" ? ML.TRAIN_DIAG : ML.USE_DIAG;
    return map && id in map ? map[id] : -1;
  };

  ML.setMode = function (mode) {
    ML.state.mode = mode;
    render();
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  document.getElementById("tab-train").addEventListener("click", () => ML.setMode("train"));
  document.getElementById("tab-use").addEventListener("click", () => ML.setMode("use"));

  document.getElementById("reset-btn").addEventListener("click", () => {
    ML.clearSavedModel();
    ML.state.model.init(42);
    ML.state.history = [];
    ML.recordEval();
    ML.state.stepIdx = 0;
    ML.state.useStepIdx = 0;
    ML.state.maxTrainStep = 0;
    ML.state.maxUseStep = 0;
    ML.state.logitGuess = null;
    ML.state.bpJustApplied = false;
    ML.state.use = null;
    render();
  });

  // Piltaster for Tilbake/Neste (lytteren settes av renderNav per steg).
  document.addEventListener("keydown", (e) => {
    if (ML._navKeys) ML._navKeys(e);
  });

  // Klikk i arkitekturdiagrammet hopper til tilhørende steg i gjeldende modus.
  document.addEventListener("click", (e) => {
    const box = e.target.closest(".dg-box");
    if (!box) return;
    const idx = ML.stepForDiagram && ML.stepForDiagram(box.id);
    if (idx === undefined || idx < 0) return;
    if (ML.state.mode === "train") { ML.state.stepIdx = idx; ML.renderTrain(); }
    else { ML.state.useStepIdx = idx; ML.renderUse(); }
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  // Progressiv avsløring: delegert klikk for alle "Vis tallene"-knapper.
  document.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-disc]");
    if (!btn) return;
    const body = document.getElementById(btn.dataset.disc);
    if (!body) return;
    const open = body.classList.toggle("open");
    btn.classList.toggle("open", open);
  });

  ML.diagram.build(document.getElementById("diagram"));
  if (!loadModel()) ML.recordEval(); // fersk modell: loggfør utgangspunktet
  renderLevelSwitch();
  render();
})();
