/* MikroLLM – en ekte, bittelille transformer.
 *
 * Arkitektur (60 parametere totalt):
 *   token-embeddings E (5×2), posisjons-embeddings P (4×2),
 *   attention W_Q, W_K, W_V, W_O (alle 2×2),
 *   feed-forward W1 (2×4) + ReLU + W2 (4×2),
 *   output/unembedding W_U (2×5).
 * Residualkoblinger rundt attention og FFN. Ingen LayerNorm og ingen
 * biases (som Llama) – opplyst i UI-et som bevisste utelatelser.
 */
(function () {
  const VOCAB = ["katten", "drikker", "melk", "vann", "sover"];
  const V = VOCAB.length;
  const D = 2;      // embedding-dimensjoner
  const H = 4;      // skjult bredde i feed-forward
  const MAXPOS = 4; // maks kontekstlengde
  const SQRTD = Math.sqrt(D);

  // Seedet PRNG (mulberry32) så modellen starter likt hver gang.
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function makeRandn(rng) {
    return function () {
      let u = 0, w = 0;
      while (u === 0) u = rng();
      while (w === 0) w = rng();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * w);
    };
  }

  function zeros(r, c) { return Array.from({ length: r }, () => new Array(c).fill(0)); }
  function randMat(r, c, rnd, scale) {
    return Array.from({ length: r }, () => Array.from({ length: c }, () => rnd() * scale));
  }
  // radvektor x (lengde r) ganger matrise W (r×c) → vektor (lengde c)
  function vecMat(x, W) {
    const r = W.length, c = W[0].length, out = new Array(c).fill(0);
    for (let j = 0; j < c; j++) {
      let s = 0;
      for (let i = 0; i < r; i++) s += x[i] * W[i][j];
      out[j] = s;
    }
    return out;
  }
  function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function softmax(arr, temp = 1) {
    const m = Math.max(...arr);
    const ex = arr.map((v) => Math.exp((v - m) / temp));
    const s = ex.reduce((a, b) => a + b, 0);
    return ex.map((e) => e / s);
  }

  class TinyLM {
    constructor(seed = 42) { this.init(seed); }

    init(seed) {
      this.seed = seed;
      const rnd = makeRandn(mulberry32(seed));
      const s = 0.5;
      this.E = randMat(V, D, rnd, s);
      this.P = randMat(MAXPOS, D, rnd, s);
      this.Wq = randMat(D, D, rnd, s);
      this.Wk = randMat(D, D, rnd, s);
      this.Wv = randMat(D, D, rnd, s);
      this.Wo = randMat(D, D, rnd, s);
      this.W1 = randMat(D, H, rnd, s);
      this.W2 = randMat(H, D, rnd, s);
      this.Wu = randMat(D, V, rnd, s);
      this.steps = 0;
    }

    // Forward pass. Returnerer ALLE mellomverdier så UI-et kan vise hver utregning.
    forward(tokens) {
      const T = tokens.length;
      const emb = tokens.map((t) => this.E[t].slice());
      const pos = tokens.map((_, i) => this.P[i].slice());
      const x = emb.map((e, i) => e.map((v, d) => v + pos[i][d]));

      const q = x.map((xi) => vecMat(xi, this.Wq));
      const k = x.map((xi) => vecMat(xi, this.Wk));
      const v = x.map((xi) => vecMat(xi, this.Wv));

      // Kausal attention: posisjon i ser bare på j ≤ i.
      const scores = [], alphas = [], z = [], attnOut = [];
      for (let i = 0; i < T; i++) {
        const srow = [];
        for (let j = 0; j <= i; j++) srow.push(dot(q[i], k[j]) / SQRTD);
        const arow = softmax(srow);
        const zi = new Array(D).fill(0);
        for (let j = 0; j <= i; j++)
          for (let d = 0; d < D; d++) zi[d] += arow[j] * v[j][d];
        scores.push(srow);
        alphas.push(arow);
        z.push(zi);
        attnOut.push(vecMat(zi, this.Wo));
      }

      const h = x.map((xi, i) => xi.map((val, d) => val + attnOut[i][d]));
      const pre = h.map((hi) => vecMat(hi, this.W1));
      const act = pre.map((p) => p.map((u) => Math.max(0, u)));
      const f = act.map((a) => vecMat(a, this.W2));
      const y = h.map((hi, i) => hi.map((val, d) => val + f[i][d]));

      const logits = vecMat(y[T - 1], this.Wu);
      const probs = softmax(logits);
      return { tokens, T, emb, pos, x, q, k, v, scores, alphas, z, attnOut, h, pre, act, f, y, logits, probs };
    }

    loss(trace, target) {
      return -Math.log(Math.max(trace.probs[target], 1e-12));
    }

    // Backpropagation: ekte gradienter for alle 60 parametere,
    // regnet baklengs gjennom nøyaktig de samme operasjonene som forward.
    backward(tr, target) {
      const T = tr.T, last = T - 1;
      const g = {
        E: zeros(V, D), P: zeros(MAXPOS, D),
        Wq: zeros(D, D), Wk: zeros(D, D), Wv: zeros(D, D), Wo: zeros(D, D),
        W1: zeros(D, H), W2: zeros(H, D), Wu: zeros(D, V),
      };
      const dx = zeros(T, D);

      // dL/dlogits = softmax − onehot(target)
      const dlog = tr.probs.map((p, i) => p - (i === target ? 1 : 0));

      const dy = new Array(D).fill(0);
      for (let d = 0; d < D; d++) {
        for (let vi = 0; vi < V; vi++) {
          g.Wu[d][vi] += tr.y[last][d] * dlog[vi];
          dy[d] += dlog[vi] * this.Wu[d][vi];
        }
      }

      // y = h + f  (residual)
      const dh = dy.slice();
      const df = dy.slice();

      // f = ReLU(h·W1)·W2
      const dact = new Array(H).fill(0);
      for (let hi = 0; hi < H; hi++) {
        for (let d = 0; d < D; d++) {
          g.W2[hi][d] += tr.act[last][hi] * df[d];
          dact[hi] += df[d] * this.W2[hi][d];
        }
      }
      const dpre = dact.map((v, i) => (tr.pre[last][i] > 0 ? v : 0));
      for (let d = 0; d < D; d++) {
        for (let hi = 0; hi < H; hi++) {
          g.W1[d][hi] += tr.h[last][d] * dpre[hi];
          dh[d] += dpre[hi] * this.W1[d][hi];
        }
      }

      // h = x_last + attnOut_last  (residual)
      for (let d = 0; d < D; d++) dx[last][d] += dh[d];
      const dAttnOut = dh;

      // attnOut = z·Wo
      const dz = new Array(D).fill(0);
      for (let a = 0; a < D; a++) {
        for (let b = 0; b < D; b++) {
          g.Wo[a][b] += tr.z[last][a] * dAttnOut[b];
          dz[a] += dAttnOut[b] * this.Wo[a][b];
        }
      }

      // z = Σ_j α_j·v_j
      const A = tr.alphas[last];
      const dalpha = [], dv = [];
      for (let j = 0; j < T; j++) {
        dalpha.push(dot(dz, tr.v[j]));
        dv.push(dz.map((u) => u * A[j]));
      }
      // softmax-bakover for attention-vektene
      const sumAd = A.reduce((s, a, j) => s + a * dalpha[j], 0);
      const ds = A.map((a, j) => a * (dalpha[j] - sumAd));

      // score_j = q_last·k_j / √d
      const dq = new Array(D).fill(0);
      const dk = zeros(T, D);
      for (let j = 0; j < T; j++) {
        for (let d = 0; d < D; d++) {
          dq[d] += (ds[j] * tr.k[j][d]) / SQRTD;
          dk[j][d] += (ds[j] * tr.q[last][d]) / SQRTD;
        }
      }

      // q = x·Wq, k = x·Wk, v = x·Wv
      for (let a = 0; a < D; a++) {
        for (let b = 0; b < D; b++) {
          g.Wq[a][b] += tr.x[last][a] * dq[b];
          dx[last][a] += dq[b] * this.Wq[a][b];
        }
      }
      for (let j = 0; j < T; j++) {
        for (let a = 0; a < D; a++) {
          for (let b = 0; b < D; b++) {
            g.Wk[a][b] += tr.x[j][a] * dk[j][b];
            dx[j][a] += dk[j][b] * this.Wk[a][b];
            g.Wv[a][b] += tr.x[j][a] * dv[j][b];
            dx[j][a] += dv[j][b] * this.Wv[a][b];
          }
        }
      }

      // x_i = E[token_i] + P[i]
      for (let i = 0; i < T; i++) {
        const t = tr.tokens[i];
        for (let d = 0; d < D; d++) {
          g.E[t][d] += dx[i][d];
          g.P[i][d] += dx[i][d];
        }
      }
      return g;
    }

    applyGrads(g, lr) {
      const upd = (W, G) => {
        for (let i = 0; i < W.length; i++)
          for (let j = 0; j < W[0].length; j++) W[i][j] -= lr * G[i][j];
      };
      upd(this.E, g.E); upd(this.P, g.P);
      upd(this.Wq, g.Wq); upd(this.Wk, g.Wk); upd(this.Wv, g.Wv); upd(this.Wo, g.Wo);
      upd(this.W1, g.W1); upd(this.W2, g.W2); upd(this.Wu, g.Wu);
      this.steps++;
    }

    trainStep(tokens, target, lr) {
      const trace = this.forward(tokens);
      const grads = this.backward(trace, target);
      const loss = this.loss(trace, target);
      this.applyGrads(grads, lr);
      return { trace, grads, loss };
    }

    paramCounts() {
      return {
        vocab: V,
        dims: D,
        embedding: V * D,
        position: MAXPOS * D,
        attention: 4 * D * D,
        ffn: D * H + H * D,
        output: D * V,
        total: V * D + MAXPOS * D + 4 * D * D + (D * H + H * D) + D * V,
      };
    }

    // Hvor mange parametere fikk gradient ≠ 0 i dette steget?
    countNonzeroGrads(g) {
      let n = 0;
      for (const key of Object.keys(g))
        for (const row of g[key])
          for (const val of row) if (val !== 0) n++;
      return n;
    }
  }

  const api = { VOCAB, V, D, H, MAXPOS, SQRTD, TinyLM, softmax, vecMat, dot };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  globalThis.ML = Object.assign(globalThis.ML || {}, api);
})();
