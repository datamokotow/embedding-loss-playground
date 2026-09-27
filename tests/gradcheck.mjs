// Numerical gradient check for every loss in src/engine.js.
//
// 1. Loss level: each loss returns dL/dE for the normalised embeddings E.
//    We nudge every coordinate of E by ±h and compare with the central difference.
// 2. Model level: the full chain (mean pooling -> L2 normalisation -> loss)
//    back-propagated into the token vectors, checked the same way.
//
// Run: node tests/gradcheck.mjs   (exits non-zero on failure)

import { DIM, LOSS_FNS, LOSSES, Lab, backprop, embed, rng } from '../src/engine.js'

const h = 1e-5
const TOL = 1e-4 // max relative error allowed
const r = rng(42)
const randn = () => { let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) }
const unit = () => { const v = Float64Array.from({ length: DIM }, randn); const n = Math.hypot(...v); return v.map((x) => x / n) }
// relative error, with a floor so round-off on near-zero gradients (≈1e-9) is not flagged
const relErr = (a, b) => Math.abs(a - b) / Math.max(1e-5, Math.abs(a) + Math.abs(b))

let failures = 0
function report(name, worst) {
  const ok = worst < TOL
  if (!ok) failures++
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(34)} max rel. error ${worst.toExponential(2)}`)
}

// ---------- 1. loss level ----------
const B = 4
const cases = {
  mnrl: (E) => LOSS_FNS.mnrl(E, [0, 1, 2, 3], [4, 5, 6, 7], null, 10),
  'mnrl + hard negatives': (E) => LOSS_FNS.mnrl(E, [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], 10),
  triplet: (E) => LOSS_FNS.triplet(E, [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], 0.6),
  cosent: (E) => LOSS_FNS.cosent(E, [0, 1, 2, 3], [4, 5, 6, 7], [1, 0.4, 0, 0.4], 20),
  cosine: (E) => LOSS_FNS.cosine(E, [0, 1, 2, 3], [4, 5, 6, 7], [1, 0.4, 0, 0.4]),
  batchall: (E) => LOSS_FNS.batchall(E, [0, 0, 1, 1, 2, 2, 0, 1], 0.6),
  batchhard: (E) => LOSS_FNS.batchhard(E, [0, 0, 1, 1, 2, 2, 0, 1], 0.6)
}
const sizes = { mnrl: 8, 'mnrl + hard negatives': 12, triplet: 12, cosent: 8, cosine: 8, batchall: 8, batchhard: 8 }

console.log('Loss level: dL/dE against central differences')
for (const [name, f] of Object.entries(cases)) {
  const E = Array.from({ length: sizes[name] }, unit)
  const { dE } = f(E)
  let worst = 0
  for (let i = 0; i < E.length; i++) for (let k = 0; k < DIM; k++) {
    const x = E[i][k]
    E[i][k] = x + h; const lp = f(E).loss
    E[i][k] = x - h; const lm = f(E).loss
    E[i][k] = x
    worst = Math.max(worst, relErr((lp - lm) / (2 * h), dE[i][k]))
  }
  report(name, worst)
}

// ---------- 2. model level ----------
console.log('\nModel level: gradients into token vectors (pooling + normalisation + loss)')
const lab = new Lab('retrieval')
for (const lossId of Object.keys(LOSSES)) {
  const cfg = { ...LOSSES[lossId].d, margin: 0.6, sampler: true }
  const probe = lab.probe(lossId, cfg)
  const res = lab.batchLoss(lab.model, probe.c, lossId, probe.batch, cfg, true)
  const grads = new Map()
  res.ems.forEach((em, i) => backprop(em, res.dE[i], grads))
  let worst = 0
  for (const [tok, g] of grads) {
    const w = lab.model.get(tok)
    for (let k = 0; k < DIM; k++) {
      const x = w[k]
      w[k] = x + h; const lp = lab.batchLoss(lab.model, probe.c, lossId, probe.batch, cfg, false).loss
      w[k] = x - h; const lm = lab.batchLoss(lab.model, probe.c, lossId, probe.batch, cfg, false).loss
      w[k] = x
      const num = (lp - lm) / (2 * h)
      worst = Math.max(worst, relErr(num, g[k]))
    }
  }
  report(`${lossId} (${LOSSES[lossId].name})`, worst)
}

// ---------- 3. sanity: training lowers held-out loss ----------
console.log('\nTraining sanity: 40 epochs lowers the held-out loss and does not hurt top-1 accuracy')
for (const lossId of Object.keys(LOSSES)) {
  const lab2 = new Lab('retrieval')
  const cfg = { ...LOSSES[lossId].d, sampler: true }
  const before = lab2.evaluate(lab2.base, lossId, cfg)
  for (let e = 0; e < cfg.epochs; e++) lab2.trainEpoch(lossId, cfg)
  const after = lab2.evaluate(lab2.model, lossId, cfg)
  const ok = after.loss < before.loss && after.recall >= before.recall
  if (!ok) failures++
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${lossId.padEnd(34)} held-out loss ${before.loss.toFixed(3)} -> ${after.loss.toFixed(3)}, top-1 ${Math.round(before.recall * 100)}% -> ${Math.round(after.recall * 100)}%`)
}

console.log(failures ? `\n${failures} check(s) failed` : '\nAll checks passed')
process.exit(failures ? 1 : 0)
