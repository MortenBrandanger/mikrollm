# MikroLLM 🐱

Et interaktivt læringssted som forklarer hvordan en LLM/transformer virker – ved å la deg
**trene** og deretter **bruke** en absurd liten, men *ekte*, språkmodell rett i nettleseren.

Filosofien: **gjør den ekte tingen liten, i stedet for å erstatte den med en analogi.**

## Modellen (60 parametere)

| Del | Form | Parametere |
|---|---|---|
| Token-embeddings `E` | 5 × 2 | 10 |
| Posisjons-embeddings `P` | 4 × 2 | 8 |
| Attention `W_Q, W_K, W_V, W_O` | 4 × (2 × 2) | 16 |
| Feed-forward `W₁, W₂` | 2×4 + 4×2 | 16 |
| Output/unembedding `W_U` | 2 × 5 | 10 |
| **Totalt** | | **60** |

Ekte kausal attention (q·k/√d → softmax → veid sum av V → W_O), residualkoblinger,
ReLU-FFN, cross-entropy-loss og håndskrevet backpropagation (gradientene er verifisert
mot numerisk derivasjon, maks avvik ~1e-10). Utelatt, og opplyst i UI-et: LayerNorm,
biases, dropout, Adam.

Treningseksempelet er «katten drikker» → «melk», vokabular på 5 ord.

## Prøv den

Siden kjører på **https://mortenbrandanger.github.io/mikrollm/**

## Kjøre lokalt

Ingen byggesteg, ingen avhengigheter. Åpne `index.html` direkte, eller:

```sh
python3 -m http.server 8123
# → http://localhost:8123
```

## Struktur

- `js/model.js` – selve transformeren: forward, backward (ekte gradienter), SGD
- `js/render.js` – hjelpere for tall, vektorer, matriser, stolper, «vis utregningen»
- `js/diagram.js` – arkitekturdiagrammet i sidepanelet, med bakoverpass-visning
- `js/inspector.js` – «Hva er inni modellen vår?» (parametertelleren)
- `js/train.js` – treningsmodusen, steg for steg
- `js/use.js` – bruksmodusen: frosne parametere, temperatur og sampling
- `js/main.js` – oppstart og modusbytte

## Bidrag og lisens

Dette er et personlig læringsprosjekt – jeg tar ikke imot bidrag for øyeblikket,
men fork gjerne og lek videre! Koden er [MIT-lisensiert](LICENSE).
