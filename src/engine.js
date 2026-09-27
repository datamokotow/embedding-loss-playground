// @ts-nocheck
// A tiny, exact embedding trainer that runs in the browser.
// Model: bag-of-words mean pooling over 32-d token vectors, L2-normalised.
// Losses: exact forward + backward passes of 7 sentence-transformers losses.
// Optimiser: Adam on the token vectors that appear in each batch.

import { WORLDS, GENERIC, tokenize } from './data.js'

export const DIM = 32

// ---------- deterministic randomness ----------
function hash(s) { let h = 2166136261; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) } return h >>> 0 }
export function rng(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296 } }
function gauss(r) { let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) }
function seededVec(key) { const r = rng(hash(key)); const v = new Float64Array(DIM); for (let i = 0; i < DIM; i++) v[i] = gauss(r); return v }

// The "pretrained" base model: random word vectors, plus shared directions for
// words a general model would already consider related (package ~ parcel ...).
function baseVec(tok) {
  const v = seededVec('w:' + tok)
  GENERIC.forEach((g, gi) => { if (g.includes(tok)) { const gv = seededVec('g:' + gi); for (let i = 0; i < DIM; i++) v[i] += 1.1 * gv[i] } })
  return v
}

export class Model {
  constructor() { this.vecs = new Map(); this.m = new Map(); this.v = new Map(); this.t = 0 }
  get(tok) { let v = this.vecs.get(tok); if (!v) { v = baseVec(tok); this.vecs.set(tok, v) } return v }
  clone() { const c = new Model(); for (const [k, v] of this.vecs) c.vecs.set(k, Float64Array.from(v)); return c }
}

export const dot = (a, b) => { let s = 0; for (let i = 0; i < DIM; i++) s += a[i] * b[i]; return s }
const zeros = (n) => Array.from({ length: n }, () => new Float64Array(DIM))
const axpy = (d, s, g) => { for (let k = 0; k < DIM; k++) d[k] += g * s[k] }
// gradient of g * cos(e_i, e_j) w.r.t. both normalised embeddings
function gcos(E, dE, i, j, g) { axpy(dE[i], E[j], g); axpy(dE[j], E[i], g) }

export function embed(model, toks) {
  const u = new Float64Array(DIM)
  for (const t of toks) { const v = model.get(t); for (let i = 0; i < DIM; i++) u[i] += v[i] }
  if (toks.length) for (let i = 0; i < DIM; i++) u[i] /= toks.length
  let n = 0; for (let i = 0; i < DIM; i++) n += u[i] * u[i]; n = Math.sqrt(n)
  const e = new Float64Array(DIM); if (n > 0) for (let i = 0; i < DIM; i++) e[i] = u[i] / n
  return { toks, u, e, norm: n }
}

// backprop through L2-normalisation and mean pooling into the token vectors
export function backprop(em, de, grads) {
  if (!em.norm) return
  const d = dot(em.e, de); const L = em.toks.length
  for (const t of em.toks) {
    let g = grads.get(t); if (!g) { g = new Float64Array(DIM); grads.set(t, g) }
    for (let i = 0; i < DIM; i++) g[i] += (de[i] - em.e[i] * d) / em.norm / L
  }
}
function adam(model, grads, lr) {
  model.t++; const b1 = 0.9, b2 = 0.999, eps = 1e-8, t = model.t
  for (const [tok, g] of grads) {
    const w = model.get(tok)
    let m = model.m.get(tok), v = model.v.get(tok)
    if (!m) { m = new Float64Array(DIM); v = new Float64Array(DIM); model.m.set(tok, m); model.v.set(tok, v) }
    for (let i = 0; i < DIM; i++) {
      m[i] = b1 * m[i] + (1 - b1) * g[i]; v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i]
      w[i] -= lr * (m[i] / (1 - b1 ** t)) / (Math.sqrt(v[i] / (1 - b2 ** t)) + eps)
    }
  }
}

// ---------- the seven losses: value + gradient w.r.t. normalised embeddings ----------
// Each also returns `info` for the batch inspector.
function L_mnrl(E, A, P, N, scale) {
  const C = P.concat(N || []), B = A.length, dE = zeros(E.length); let loss = 0; const probs = [], cos = []
  for (let i = 0; i < B; i++) {
    const c = C.map((k) => dot(E[A[i]], E[k])); const s = c.map((x) => scale * x)
    const mx = Math.max(...s); const ex = s.map((x) => Math.exp(x - mx)); const Z = ex.reduce((a, b) => a + b, 0)
    loss += Math.log(Z) + mx - s[i]
    C.forEach((k, j) => gcos(E, dE, A[i], k, (ex[j] / Z - (i === j ? 1 : 0)) / B * scale))
    probs.push(ex.map((x) => x / Z)); cos.push(c)
  }
  return { loss: loss / B, dE, info: { probs, cos } }
}
function L_triplet(E, A, P, N, m) {
  const B = A.length, dE = zeros(E.length); let loss = 0; const rows = []
  for (let i = 0; i < B; i++) {
    const cp = dot(E[A[i]], E[P[i]]), cn = dot(E[A[i]], E[N[i]])
    const t = cn - cp + m // = d(a,p) - d(a,n) + m with d = 1 - cos
    if (t > 0) { loss += t / B; gcos(E, dE, A[i], N[i], 1 / B); gcos(E, dE, A[i], P[i], -1 / B) }
    rows.push({ cp, cn, active: t > 0 })
  }
  return { loss, dE, info: { rows } }
}
function L_cosent(E, I, J, Y, scale) {
  const n = I.length, c = I.map((i, k) => dot(E[i], E[J[k]])), terms = []
  for (let k = 0; k < n; k++) for (let l = 0; l < n; l++) if (Y[k] > Y[l]) terms.push([k, l, scale * (c[l] - c[k])])
  const mx = Math.max(0, ...terms.map((t) => t[2])); const Z = Math.exp(-mx) + terms.reduce((a, t) => a + Math.exp(t[2] - mx), 0)
  const dc = new Float64Array(n); terms.forEach(([k, l, v]) => { const w = Math.exp(v - mx) / Z * scale; dc[l] += w; dc[k] -= w })
  const dE = zeros(E.length); for (let k = 0; k < n; k++) gcos(E, dE, I[k], J[k], dc[k])
  const violations = terms.filter((t) => t[2] > 0).length
  return { loss: mx + Math.log(Z), dE, info: { cos: c, y: Y, violations, pairs: terms.length } }
}
function L_cosine(E, I, J, Y) {
  const n = I.length, dE = zeros(E.length); let loss = 0; const c = []
  for (let k = 0; k < n; k++) { const ck = dot(E[I[k]], E[J[k]]), r = ck - Y[k]; loss += r * r / n; gcos(E, dE, I[k], J[k], 2 * r / n); c.push(ck) }
  return { loss, dE, info: { cos: c, y: Y } }
}
function simMatrix(E) { return E.map((a) => E.map((b) => dot(a, b))) }
function L_batchall(E, lab, m) {
  const n = E.length, C = simMatrix(E), act = []; let valid = 0
  for (let a = 0; a < n; a++) for (let p = 0; p < n; p++) {
    if (p === a || lab[p] !== lab[a]) continue
    for (let q = 0; q < n; q++) { if (lab[q] === lab[a]) continue; valid++; const t = C[a][q] - C[a][p] + m; if (t > 1e-12) act.push([a, p, q, t]) }
  }
  const dE = zeros(n); const info = { C, lab, active: act.length, valid }
  if (!act.length) return { loss: 0, dE, info }
  const N = act.length; let loss = 0; act.forEach(([a, p, q, t]) => { loss += t / N; gcos(E, dE, a, q, 1 / N); gcos(E, dE, a, p, -1 / N) })
  return { loss, dE, info }
}
function L_batchhard(E, lab, m) {
  const n = E.length, C = simMatrix(E), dE = zeros(n); let loss = 0; const picks = []
  for (let a = 0; a < n; a++) {
    let p = -1, q = -1
    for (let j = 0; j < n; j++) { if (j === a) continue; if (lab[j] === lab[a]) { if (p < 0 || C[a][j] < C[a][p]) p = j } else { if (q < 0 || C[a][j] > C[a][q]) q = j } }
    picks.push([p, q]); if (p < 0 || q < 0) continue
    const t = C[a][q] - C[a][p] + m; if (t > 0) { loss += t / n; gcos(E, dE, a, q, 1 / n); gcos(E, dE, a, p, -1 / n) }
  }
  return { loss, dE, info: { C, lab, picks } }
}

// ---------- data formats and losses ----------
export const CASES = {
  pairs: { short: 'Pairs', name: '(anchor, positive)', losses: ['mnrl'], cols: ['anchor', 'positive'], sampler: 'nodup' },
  triplets: { short: 'Triplets', name: '(anchor, positive, negative)', losses: ['mnrl_hn', 'triplet'], cols: ['anchor', 'positive', 'negative'], sampler: 'nodup' },
  scored: { short: 'Scored pairs', name: '(text, text, score)', losses: ['cosent', 'cosine'], cols: ['sentence1', 'sentence2', 'score'], sampler: null },
  labels: { short: 'Labels', name: '(text, label)', losses: ['batchall', 'batchhard'], cols: ['text', 'label'], sampler: 'group' }
}
export const LOSSES = {
  mnrl: { name: 'MultipleNegativesRankingLoss', short: 'MNRL', family: 'Ranking', param: 'scale', d: { epochs: 40, lr: 0.02, bs: 6, scale: 10 }, bs: [2, 6],
    how: 'Every anchor is scored against every positive in the batch. Its own positive is the right answer; the other positives act as negatives, so a bigger batch gives a harder, more useful task.' },
  mnrl_hn: { name: 'MultipleNegativesRankingLoss', short: 'MNRL + hard negatives', family: 'Ranking', param: 'scale', d: { epochs: 40, lr: 0.02, bs: 3, scale: 10 }, bs: [2, 6],
    how: 'Same as MNRL, but each row brings its own near-miss negative too, so the anchor competes against the in-batch positives and all the hard negatives.' },
  triplet: { name: 'TripletLoss', short: 'TripletLoss', family: 'Triplet', param: 'margin', d: { epochs: 40, lr: 0.02, bs: 3, margin: 0.3 }, bs: [2, 6],
    how: 'Each triplet is judged on its own. The anchor must sit closer to its positive than to its negative by at least the margin. Once it does, the triplet contributes nothing.' },
  cosent: { name: 'CoSENTLoss', short: 'CoSENT', family: 'Scored similarity', param: 'scale', d: { epochs: 40, lr: 0.02, bs: 16, scale: 20 }, bs: [4, 32],
    how: 'Compares every two pairs in the batch. If pair k has a higher gold score than pair l, its cosine must be higher too. It learns the order of the scores, not their exact values.' },
  cosine: { name: 'CosineSimilarityLoss', short: 'CosineSimilarityLoss', family: 'Scored similarity', param: null, d: { epochs: 40, lr: 0.02, bs: 16 }, bs: [4, 32],
    how: 'Regresses each cosine directly onto its gold score with mean squared error. A pair labelled 0.4 is pushed to a cosine of exactly 0.4.' },
  batchall: { name: 'BatchAllTripletLoss', short: 'BatchAllTriplet', family: 'Label-based triplet', param: 'margin', d: { epochs: 40, lr: 0.02, bs: 12, margin: 0.3 }, bs: [4, 24],
    how: 'Builds every (anchor, same-label, different-label) triplet in the batch and averages the ones that still break the margin.' },
  batchhard: { name: 'BatchHardTripletLoss', short: 'BatchHardTriplet', family: 'Label-based triplet', param: 'margin', d: { epochs: 40, lr: 0.02, bs: 12, margin: 0.3 }, bs: [4, 24],
    how: 'For each anchor, keeps only the hardest positive (the least similar text with the same label) and the hardest negative (the most similar text with a different label).' }
}
export const LOSS_CASE = { mnrl: 'pairs', mnrl_hn: 'triplets', triplet: 'triplets', cosent: 'scored', cosine: 'scored', batchall: 'labels', batchhard: 'labels' }

function shuffle(a, r = Math.random) { a = a.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]] } return a }
const chunk = (a, n) => { const o = []; for (let i = 0; i < a.length; i += n) o.push(a.slice(i, i + n)); return o }

// ---------- a lab bound to one dataset ----------
export class Lab {
  constructor(worldKey = 'retrieval') { this.base = new Model(); this.setWorld(worldKey) }

  setWorld(k) {
    const W = (this.W = WORLDS[k]); this.worldKey = k
    this.DOCS = W.groups; this.DOC = Object.fromEntries(W.groups.map((d) => [d.id, d]))
    this.LABEL = Object.fromEntries(W.groups.map((d, i) => [d.id, i])); this.HARDNEG = W.hardneg; this.RELATED = W.related
    this.TRAIN = W.train.map(([q, doc]) => ({ q, doc })); this.TEST = W.test.map(([q, doc]) => ({ q, doc }))
    this.DOCTOK = Object.fromEntries(W.groups.map((d) => [d.id, tokenize(d.text)]))
    this.scoredCache = null
    this.reset()
  }
  reset() { this.model = this.base.clone(); this.epoch = 0; this.hist = [] }
  related(a, b) { return this.RELATED.some(([x, y]) => (x === a && y === b) || (x === b && y === a)) }
  gold(a, b) { return a === b ? 1 : this.related(a, b) ? 0.4 : 0 }

  genScored() {
    if (this.scoredCache) return this.scoredCache
    const r = rng(12345), rows = [], DOCS = this.DOCS, TRAIN = this.TRAIN
    const others = (d) => DOCS.map((x) => x.id).filter((x) => x !== d)
    TRAIN.forEach((p) => {
      rows.push({ a: p.q, b: this.DOC[p.doc].text, ia: p.doc, ib: p.doc, score: 1 })
      const rel = others(p.doc).filter((x) => this.related(x, p.doc)), un = others(p.doc).filter((x) => !this.related(x, p.doc))
      if (rel.length) { const d = rel[Math.floor(r() * rel.length)]; rows.push({ a: p.q, b: this.DOC[d].text, ia: p.doc, ib: d, score: 0.4 }) }
      const d = un[Math.floor(r() * un.length)]; rows.push({ a: p.q, b: this.DOC[d].text, ia: p.doc, ib: d, score: 0 })
    })
    for (let i = 0; i < TRAIN.length; i++) for (let j = i + 1; j < TRAIN.length; j++) {
      const g = this.gold(TRAIN[i].doc, TRAIN[j].doc)
      if (g === 1 || (g > 0 && r() < 0.35) || (g === 0 && r() < 0.06)) rows.push({ a: TRAIN[i].q, b: TRAIN[j].q, ia: TRAIN[i].doc, ib: TRAIN[j].doc, score: g })
    }
    return (this.scoredCache = rows)
  }
  rows(c) {
    if (c === 'pairs') return this.TRAIN.map((p) => ({ ...p }))
    if (c === 'triplets') return this.TRAIN.map((p) => ({ ...p, neg: this.HARDNEG[p.doc] }))
    if (c === 'scored') return this.genScored()
    return [...this.TRAIN.map((p) => ({ text: p.q, doc: p.doc })), ...this.DOCS.map((d) => ({ text: d.text, doc: d.id, isDoc: true }))]
  }
  makeBatches(c, R, bs, samplerOn, r = Math.random) {
    if (c === 'pairs' || c === 'triplets') {
      let rem = shuffle(R, r); if (!samplerOn) return chunk(rem, bs)
      // NO_DUPLICATES: never put two rows pointing at the same document in one batch
      const out = []
      while (rem.length) {
        const b = [], used = new Set(), left = []
        for (const p of rem) { const keys = c === 'triplets' ? [p.doc, p.neg] : [p.doc]; if (b.length < bs && keys.every((k) => !used.has(k))) { b.push(p); keys.forEach((k) => used.add(k)) } else left.push(p) }
        out.push(b); rem = left
      }
      return out
    }
    if (c === 'labels' && samplerOn) {
      // GROUP_BY_LABEL: batches hold >= 2 texts per label so triplets exist
      const by = {}; R.forEach((x) => (by[x.doc] = by[x.doc] || []).push(x))
      const ch = Object.values(by).flatMap((g) => chunk(shuffle(g, r), 2)).filter((g) => g.length === 2)
      return chunk(shuffle(ch, r).flat(), bs)
    }
    return chunk(shuffle(R, r), bs)
  }
  batchLoss(model, c, lossId, batch, cfg, withEm) {
    let toks
    const B = batch.length, rg = (o, n) => Array.from({ length: n }, (_, i) => o + i)
    if (c === 'pairs' || c === 'triplets') toks = [...batch.map((p) => tokenize(p.q)), ...batch.map((p) => this.DOCTOK[p.doc]), ...(c === 'triplets' ? batch.map((p) => this.DOCTOK[p.neg]) : [])]
    else if (c === 'scored') toks = [...batch.map((p) => tokenize(p.a)), ...batch.map((p) => tokenize(p.b))]
    else toks = batch.map((p) => tokenize(p.text))
    const ems = toks.map((t) => embed(model, t)), E = ems.map((m) => m.e)
    const A = rg(0, B), P = rg(B, B), N = rg(2 * B, B); let res
    switch (lossId) {
      case 'mnrl': res = L_mnrl(E, A, P, null, cfg.scale); break
      case 'mnrl_hn': res = L_mnrl(E, A, P, N, cfg.scale); break
      case 'triplet': res = L_triplet(E, A, P, N, cfg.margin); break
      case 'cosent': res = L_cosent(E, A, P, batch.map((p) => p.score), cfg.scale); break
      case 'cosine': res = L_cosine(E, A, P, batch.map((p) => p.score)); break
      case 'batchall': res = L_batchall(E, batch.map((p) => this.LABEL[p.doc]), cfg.margin); break
      case 'batchhard': res = L_batchhard(E, batch.map((p) => this.LABEL[p.doc]), cfg.margin); break
    }
    if (withEm) res.ems = ems
    return res
  }
  trainEpoch(lossId, cfg) {
    const c = LOSS_CASE[lossId]; const R = this.rows(c)
    const batches = this.makeBatches(c, R, cfg.bs, cfg.sampler)
    let tot = 0
    for (const b of batches) {
      const r = this.batchLoss(this.model, c, lossId, b, cfg, true); const grads = new Map()
      r.ems.forEach((em, i) => backprop(em, r.dE[i], grads)); adam(this.model, grads, cfg.lr); tot += r.loss
    }
    this.epoch++
    return tot / batches.length
  }
  // A fixed batch (same rows every epoch) so the inspector shows the same examples moving.
  probe(lossId, cfg) {
    const c = LOSS_CASE[lossId]; const R = this.rows(c)
    const b = this.makeBatches(c, R, cfg.bs, cfg.sampler, rng(7))[0]
    const res = this.batchLoss(this.model, c, lossId, b, cfg, false)
    return { batch: b, c, ...res }
  }
  heldOut(c) {
    if (c === 'pairs') return this.TEST.map((t) => ({ q: t.q, doc: t.doc }))
    if (c === 'triplets') return this.TEST.map((t) => ({ q: t.q, doc: t.doc, neg: this.HARDNEG[t.doc] }))
    if (c === 'scored') return this.TEST.flatMap((t) => this.DOCS.map((d) => ({ a: t.q, b: d.text, ia: t.doc, ib: d.id, score: this.gold(t.doc, d.id) })))
    return [...this.TEST.map((t) => ({ text: t.q, doc: t.doc })), ...this.DOCS.map((d) => ({ text: d.text, doc: d.id }))]
  }
  evaluate(model, lossId, cfg) {
    const c = LOSS_CASE[lossId]
    const D = this.DOCS.map((d) => embed(model, this.DOCTOK[d.id]).e)
    const T = this.TEST.map((t) => embed(model, tokenize(t.q)).e)
    const sims = T.map((e) => D.map((d) => dot(e, d)))
    let hit = 0, pos = 0, neg = 0
    sims.forEach((row, i) => { const gi = this.LABEL[this.TEST[i].doc]; if (row.indexOf(Math.max(...row)) === gi) hit++; pos += row[gi]; neg += (row.reduce((a, b) => a + b, 0) - row[gi]) / (row.length - 1) })
    const n = this.TEST.length
    const loss = lossId === 'mnrl_hn' ? this.batchLoss(model, 'pairs', 'mnrl', this.heldOut('pairs'), cfg, false).loss : this.batchLoss(model, c, lossId, this.heldOut(c), cfg, false).loss
    const g = [], s = []; this.TEST.forEach((t, i) => this.DOCS.forEach((d, j) => { g.push(this.gold(t.doc, d.id)); s.push(sims[i][j]) }))
    return { loss, recall: hit / n, pos: pos / n, neg: neg / n, gap: (pos - neg) / n, rho: spearman(g, s) }
  }
  search(model, text) {
    const q = embed(model, tokenize(text)).e
    return this.DOCS.map((d) => ({ d, s: dot(q, embed(model, this.DOCTOK[d.id]).e) })).sort((a, b) => b.s - a.s)
  }
  // All points for the map: documents, training texts, held-out texts
  points() {
    return [
      ...this.DOCS.map((d) => ({ kind: 'doc', text: d.text, doc: d.id, toks: this.DOCTOK[d.id] })),
      ...this.TRAIN.map((t) => ({ kind: 'train', text: t.q, doc: t.doc, toks: tokenize(t.q) })),
      ...this.TEST.map((t) => ({ kind: 'test', text: t.q, doc: t.doc, toks: tokenize(t.q) }))
    ]
  }
}

export function spearman(x, y) {
  const rk = (a) => { const idx = a.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]); const r = new Array(a.length)
    for (let i = 0; i < idx.length;) { let j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++; for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2; i = j + 1 } return r }
  const rx = rk(x), ry = rk(y), n = x.length, mx = rx.reduce((a, b) => a + b) / n, my = ry.reduce((a, b) => a + b) / n
  let sxy = 0, sx = 0, sy = 0; for (let i = 0; i < n; i++) { sxy += (rx[i] - mx) * (ry[i] - my); sx += (rx[i] - mx) ** 2; sy += (ry[i] - my) ** 2 }
  return sxy / Math.sqrt(sx * sy || 1)
}

// ---------- 2-D projection: PCA, then rotate/flip to match the previous frame ----------
export function pca2(X) {
  const n = X.length, d = X[0].length, mu = new Float64Array(d)
  X.forEach((x) => { for (let i = 0; i < d; i++) mu[i] += x[i] / n })
  const Xc = X.map((x) => x.map((v, i) => v - mu[i]))
  const C = Array.from({ length: d }, () => new Float64Array(d))
  Xc.forEach((x) => { for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] += x[i] * x[j] })
  const comps = []
  for (let k = 0; k < 2; k++) {
    let v = new Float64Array(d).map((_, i) => Math.sin(i * 1.7 + k + 1))
    for (let it = 0; it < 80; it++) {
      const w = new Float64Array(d); for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) w[i] += C[i][j] * v[j]
      comps.forEach((c) => { const p = dot2(w, c); for (let i = 0; i < d; i++) w[i] -= p * c[i] })
      const nn = Math.sqrt(dot2(w, w)) || 1; v = w.map((x) => x / nn)
    }
    comps.push(v)
  }
  return Xc.map((x) => [dot2(x, comps[0]), dot2(x, comps[1])])
}
const dot2 = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s }
// orthogonal Procrustes in 2-D (rotation + optional reflection) onto the previous layout
export function align(Y, prev) {
  if (!prev) return Y
  let best = null
  for (const flip of [1, -1]) {
    const Z = Y.map(([x, y]) => [x, flip * y])
    let a = 0, b = 0; Z.forEach(([x, y], i) => { const [px, py] = prev[i]; a += x * px + y * py; b += x * py - y * px })
    const th = Math.atan2(b, a), c = Math.cos(th), s = Math.sin(th)
    const R = Z.map(([x, y]) => [c * x - s * y, s * x + c * y])
    const err = R.reduce((e, [x, y], i) => e + (x - prev[i][0]) ** 2 + (y - prev[i][1]) ** 2, 0)
    if (!best || err < best.err) best = { R, err }
  }
  return best.R
}

// Exposed for tests/gradcheck.mjs
export const LOSS_FNS = { mnrl: L_mnrl, triplet: L_triplet, cosent: L_cosent, cosine: L_cosine, batchall: L_batchall, batchhard: L_batchhard }
