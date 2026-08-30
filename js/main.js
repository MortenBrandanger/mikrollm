/* Oppstart, modusbytte og globale hendelser. */
(function () {
  const ML = globalThis.ML;

  ML.state = {
    model: new ML.TinyLM(42),
    mode: "train",
    stepIdx: 0,
    useStepIdx: 0,
    level: 0, // 0 = Enkelt, 1 = Med tall, 2 = Full utregning
    history: [],
    qkvTab: "q",
    bpJustApplied: false,
    use: null,
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
  }

  ML.setMode = function (mode) {
    ML.state.mode = mode;
    render();
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  document.getElementById("tab-train").addEventListener("click", () => ML.setMode("train"));
  document.getElementById("tab-use").addEventListener("click", () => ML.setMode("use"));

  document.getElementById("reset-btn").addEventListener("click", () => {
    ML.state.model.init(42);
    ML.state.history = [];
    ML.recordEval();
    ML.state.stepIdx = 0;
    ML.state.useStepIdx = 0;
    ML.state.bpJustApplied = false;
    ML.state.use = null;
    render();
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
  ML.recordEval(); // loggfør utgangspunktet (steg 0) i historikken
  renderLevelSwitch();
  render();
})();
