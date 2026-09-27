// @ts-nocheck
// UI for the embedding loss playground. No framework, no dependencies.
import { Lab, LOSSES, CASES, LOSS_CASE, embed, pca2, align } from './engine.js'
import { WORLDS, tokenize } from './data.js'

export const MARKUP =
  "<div class='ll ll-box ll-pg' id='ll-playground' data-ll-playground>\n  <div class='ll-pg-head'>\n    <div>\n      <p class='ll-label'>Dataset</p>\n      <div class='ll-chips' data-ref='worlds'></div>\n    </div>\n    <div>\n      <p class='ll-label'>Loss <span class='ll-hint'>(the data format comes with it)</span></p>\n      <div class='ll-chips' data-ref='losses'></div>\n    </div>\n  </div>\n\n  <div class='ll-pg-how'>\n    <div class='ll-pg-howtext'>\n      <p class='ll-pg-lossname' data-ref='lossName'></p>\n      <p data-ref='lossHow'></p>\n    </div>\n    <div class='ll-pg-rows'>\n      <p class='ll-label'>One batch of training rows: <span data-ref='fmtName' class='ll-mono'></span></p>\n      <div data-ref='rows'></div>\n    </div>\n  </div>\n\n  <div class='ll-pg-controls'>\n    <label>Epochs<input type='range' min='5' max='100' step='5' data-ref='epochs' /></label>\n    <label>Learning rate<input type='range' min='0' max='5' step='1' data-ref='lr' /></label>\n    <label>Batch size<input type='range' min='2' max='6' step='1' data-ref='bs' /></label>\n    <label data-ref='scaleWrap'>Scale<input type='range' min='1' max='40' step='1' data-ref='scale' /></label>\n    <label data-ref='marginWrap'>Margin<input type='range' min='0' max='1' step='0.05' data-ref='margin' /></label>\n    <label class='ll-check' data-ref='samplerWrap'><input type='checkbox' data-ref='sampler' checked /> <span data-ref='samplerName'>No-duplicates batches</span></label>\n  </div>\n\n  <div class='ll-pg-actions'>\n    <button class='ll-btn' data-ref='run'>\u25b6 Train</button>\n    <button class='ll-btn ghost' data-ref='step'>+1 epoch</button>\n    <button class='ll-btn ghost' data-ref='reset'>Reset</button>\n    <span class='ll-pg-status' data-ref='status' aria-live='polite'></span>\n  </div>\n\n  <div class='ll-pg-grid'>\n    <figure class='ll-fig ll-fig-map'>\n      <figcaption>\n        <b>Embedding space</b>\n        <span>32 dimensions projected to 2</span>\n      </figcaption>\n      <svg data-ref='map' viewBox='0 0 400 300' role='img' aria-label='2-D projection of the embeddings'></svg>\n      <div class='ll-legend' data-ref='mapLegend'></div>\n    </figure>\n\n    <figure class='ll-fig'>\n      <figcaption>\n        <b>Inside one batch</b>\n        <span data-ref='inspTitle'></span>\n      </figcaption>\n      <div data-ref='insp' class='ll-insp'></div>\n      <p class='ll-note' data-ref='inspNote'></p>\n    </figure>\n\n    <figure class='ll-fig'>\n      <figcaption>\n        <b>Loss per epoch</b>\n        <span class='ll-legend ll-legend-inline'>\n          <span><i class='ll-sw' style='background:var(--c0)'></i>train</span>\n          <span><i class='ll-sw' style='background:var(--c1)'></i>held-out</span>\n        </span>\n      </figcaption>\n      <svg data-ref='curve' viewBox='0 0 400 200' role='img' aria-label='Loss per epoch'></svg>\n    </figure>\n\n    <figure class='ll-fig'>\n      <figcaption>\n        <b>Held-out quality</b>\n        <span>base model \u2192 fine-tuned</span>\n      </figcaption>\n      <div class='ll-tiles' data-ref='tiles'></div>\n    </figure>\n  </div>\n\n  <div class='ll-pg-search'>\n    <p class='ll-label'>Try a new query against the six documents</p>\n    <input type='text' data-ref='q' class='ll-input' autocomplete='off' />\n    <div class='ll-chips ll-qchips' data-ref='qchips'></div>\n    <div class='ll-res2'>\n      <div><p class='ll-label'>Base model</p><div data-ref='resBase'></div></div>\n      <div><p class='ll-label'>Fine-tuned</p><div data-ref='resFt'></div></div>\n    </div>\n  </div>\n\n  <details class='ll-code'>\n    <summary>The same run in sentence-transformers</summary>\n    <pre><code data-ref='code'></code></pre>\n  </details>\n  <div class='ll-tip' data-ref='tip' hidden></div>\n</div>"

/**
 * Render the playground into `container` and start it.
 * Any element on the page with data-ll-loss="<loss id>" (and optional
 * data-ll-world, data-ll-cfg) becomes a button that loads that loss and trains.
 */
export function mountPlayground(container) {
  container.innerHTML = MARKUP
  const root = container.firstElementChild
  init(root)
  return root
}

function init(root) {
  const $ = (k) => root.querySelector(`[data-ref="${k}"]`)
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c])
  const fmt = (x, d = 2) => (x < 0 ? '−' : '') + Math.abs(x).toFixed(d)
  const LRS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
  const col = (i) => `var(--c${i})`
  const lab = new Lab('retrieval')
  const S = { loss: 'mnrl', running: false, hist: [], baseEval: null, prev: null, trails: [] }

  // ---------- controls ----------
  function renderWorlds() {
    $('worlds').innerHTML = Object.entries(WORLDS)
      .map(
        ([k, w]) =>
          `<button class="ll-chip" data-k="${k}" aria-pressed="${lab.worldKey === k}">${esc(w.domain)}</button>`,
      )
      .join('')
  }
  function renderLosses() {
    $('losses').innerHTML = Object.entries(LOSSES)
      .map(([k, l]) => `<button class="ll-chip" data-k="${k}" aria-pressed="${S.loss === k}">${esc(l.short)}</button>`)
      .join('')
  }
  $('worlds').addEventListener('click', (e) => {
    const b = e.target.closest('[data-k]')
    if (!b) return
    stop()
    lab.setWorld(b.dataset.k)
    S.prev = null
    renderWorlds()
    resetRun()
    renderSearchChips()
  })
  $('losses').addEventListener('click', (e) => {
    const b = e.target.closest('[data-k]')
    if (!b) return
    setLoss(b.dataset.k)
  })

  function cfg() {
    return {
      epochs: +$('epochs').value,
      lr: LRS[+$('lr').value],
      bs: +$('bs').value,
      scale: +$('scale').value,
      margin: +$('margin').value,
      sampler: $('sampler').checked,
    }
  }
  function syncOuts() {
    const c = cfg()
    labelOut('epochs', c.epochs)
    labelOut('lr', c.lr)
    labelOut('bs', c.bs)
    labelOut('scale', c.scale)
    labelOut('margin', c.margin.toFixed(2))
  }
  // put the value next to the label text
  function labelOut(k, v) {
    const inp = $(k)
    const lab_ = inp.closest('label')
    let span = lab_.querySelector('.ll-lbl')
    if (!span) {
      span = document.createElement('span')
      span.className = 'll-lbl'
      const txt = [...lab_.childNodes].find((n) => n.nodeType === 3 && n.textContent.trim())
      span.innerHTML = `<span>${txt.textContent.trim()}</span><b class="ll-v"></b>`
      txt.remove()
      lab_.prepend(span)
    }
    span.querySelector('.ll-v').textContent = v
    span.querySelector('.ll-v').style.color = 'var(--ll-fg)'
  }

  function setLoss(k, opts = {}) {
    stop()
    S.loss = k
    const L = LOSSES[k],
      c = CASES[LOSS_CASE[k]]
    $('epochs').value = L.d.epochs
    $('lr').value = LRS.indexOf(L.d.lr)
    $('bs').min = L.bs[0]
    $('bs').max = L.bs[1]
    $('bs').step = L.bs[1] > 10 ? 2 : 1
    $('bs').value = L.d.bs
    if (L.d.scale) $('scale').value = L.d.scale
    if (L.d.margin != null) $('margin').value = L.d.margin
    if (opts.cfg)
      Object.entries(opts.cfg).forEach(([kk, v]) => {
        if (kk === 'sampler') $('sampler').checked = v
        else if (kk === 'lr') $('lr').value = LRS.indexOf(v)
        else $(kk).value = v
      })
    $('scaleWrap').hidden = L.param !== 'scale'
    $('marginWrap').hidden = L.param !== 'margin'
    $('samplerWrap').hidden = !c.sampler
    $('samplerName').textContent = c.sampler === 'group' ? 'Group-by-label batches' : 'No-duplicates batches'
    if (!opts.cfg || opts.cfg.sampler === undefined) $('sampler').checked = true
    renderLosses()
    syncOuts()
    resetRun()
  }
  ;['epochs', 'lr', 'bs', 'scale', 'margin', 'sampler'].forEach((k) =>
    $(k).addEventListener('input', () => {
      syncOuts()
      if (['bs', 'sampler', 'scale', 'margin'].includes(k)) {
        renderRows()
        renderInspector()
        renderCode()
      }
      if (k !== 'epochs' && lab.epoch === 0) {
        S.baseEval = lab.evaluate(lab.base, S.loss, cfg())
        renderTiles()
        renderCurve()
      }
    }),
  )

  // ---------- run loop ----------
  function resetRun() {
    stop()
    lab.reset()
    S.hist = []
    S.trails = []
    S.baseEval = lab.evaluate(lab.base, S.loss, cfg())
    S.hist.push({ epoch: 0, train: null, test: S.baseEval.loss })
    renderAll()
  }
  function epoch() {
    const c = cfg()
    const tl = lab.trainEpoch(S.loss, c)
    const ev = lab.evaluate(lab.model, S.loss, c)
    S.hist.push({ epoch: lab.epoch, train: tl, test: ev.loss })
    S.lastEval = ev
  }
  function loop() {
    if (!S.running) return
    if (lab.epoch >= cfg().epochs) {
      stop()
      return
    }
    epoch()
    renderLive()
    setTimeout(loop, 70)
  }
  function start() {
    if (lab.epoch >= cfg().epochs) resetRun()
    S.running = true
    $('run').textContent = '❚❚ Pause'
    loop()
  }
  function stop() {
    S.running = false
    if ($('run')) $('run').textContent = lab && lab.epoch > 0 && lab.epoch < cfg().epochs ? '▶ Resume' : '▶ Train'
  }
  $('run').addEventListener('click', () => (S.running ? stop() : start()))
  $('step').addEventListener('click', () => {
    stop()
    epoch()
    renderLive()
  })
  $('reset').addEventListener('click', () => {
    resetRun()
  })

  // ---------- rendering ----------
  function renderAll() {
    renderHow()
    renderRows()
    renderLive()
    renderCode()
    runSearch()
  }
  function renderLive() {
    renderStatus()
    renderMap()
    renderInspector()
    renderCurve()
    renderTiles()
    runSearch()
  }

  function renderHow() {
    const L = LOSSES[S.loss],
      c = CASES[LOSS_CASE[S.loss]]
    $('lossName').innerHTML = `${esc(L.name)} <small>· ${esc(L.family)}</small>`
    $('lossHow').textContent = L.how
    $('fmtName').textContent = c.name
  }
  function docName(id) {
    return lab.DOC[id].short
  }
  function sw(id) {
    return `<i class="ll-sw" style="background:${col(lab.LABEL[id])}"></i>`
  }
  function renderRows() {
    const pr = lab.probe(S.loss, cfg())
    const b = pr.batch.slice(0, 4)
    const c = pr.c
    let h = '<table class="ll-rowtbl"><thead><tr>'
    if (c === 'pairs')
      h +=
        '<th>anchor</th><th>positive</th></tr></thead><tbody>' +
        b.map((r) => `<tr><td>${esc(r.q)}</td><td>${sw(r.doc)} ${esc(lab.DOC[r.doc].text)}</td></tr>`).join('')
    if (c === 'triplets')
      h +=
        '<th>anchor</th><th>positive</th><th>negative</th></tr></thead><tbody>' +
        b
          .map(
            (r) =>
              `<tr><td>${esc(r.q)}</td><td>${sw(r.doc)} ${esc(docName(r.doc))}</td><td>${sw(r.neg)} ${esc(docName(r.neg))}</td></tr>`,
          )
          .join('')
    if (c === 'scored')
      h +=
        '<th>sentence1</th><th>sentence2</th><th class="num">score</th></tr></thead><tbody>' +
        b
          .map((r) => `<tr><td>${esc(r.a)}</td><td>${esc(r.b)}</td><td class="num">${r.score.toFixed(1)}</td></tr>`)
          .join('')
    if (c === 'labels')
      h +=
        '<th>text</th><th>label</th></tr></thead><tbody>' +
        b.map((r) => `<tr><td>${esc(r.text)}</td><td>${sw(r.doc)} ${esc(docName(r.doc))}</td></tr>`).join('')
    h += '</tbody></table>'
    if (pr.batch.length > 4) h += `<p class="ll-note">+ ${pr.batch.length - 4} more rows in this batch</p>`
    $('rows').innerHTML = h
  }
  function renderStatus() {
    const last = S.hist[S.hist.length - 1]
    $('status').textContent =
      `epoch ${lab.epoch} / ${cfg().epochs}` +
      (last.train != null ? ` · train loss ${fmt(last.train, 3)}` : '') +
      ` · held-out ${fmt(last.test, 3)}`
    if (!S.running) stop()
  }

  // --- map ---
  function renderMap() {
    const pts = lab.points()
    const model = lab.model
    let Y = pca2(pts.map((p) => Array.from(embed(model, p.toks).e)))
    Y = align(Y, S.prev)
    S.prev = Y
    const W = 400,
      H = 300,
      pad = 22
    const xs = Y.map((p) => p[0]),
      ys = Y.map((p) => p[1])
    const x0 = Math.min(...xs),
      x1 = Math.max(...xs),
      y0 = Math.min(...ys),
      y1 = Math.max(...ys)
    const sc = Math.min((W - 2 * pad) / (x1 - x0 || 1), (H - 2 * pad) / (y1 - y0 || 1))
    const cx = (W - (x1 - x0) * sc) / 2,
      cy = (H - (y1 - y0) * sc) / 2
    const P = Y.map(([x, y]) => [cx + (x - x0) * sc, cy + (y1 - y) * sc])
    S.trails.push(P)
    if (S.trails.length > 40) S.trails.shift()
    const svg = $('map')
    let h = ''
    // faint lines from each text to its document
    const docIdx = Object.fromEntries(pts.map((p, i) => [p.kind === 'doc' ? p.doc : '_', i]))
    pts.forEach((p, i) => {
      if (p.kind !== 'doc') {
        const j = docIdx[p.doc]
        h += `<line x1="${P[i][0].toFixed(1)}" y1="${P[i][1].toFixed(1)}" x2="${P[j][0].toFixed(1)}" y2="${P[j][1].toFixed(1)}" stroke="${col(lab.LABEL[p.doc])}" stroke-opacity=".18" stroke-width="1"/>`
      }
    })
    pts.forEach((p, i) => {
      const c = col(lab.LABEL[p.doc])
      const [x, y] = P[i]
      const tip = esc(
        `${p.kind === 'doc' ? 'Document' : p.kind === 'train' ? 'Training text' : 'Held-out text'}: ${p.text}`,
      )
      if (p.kind === 'doc')
        h += `<g data-tip="${tip}"><rect x="${(x - 6).toFixed(1)}" y="${(y - 6).toFixed(1)}" width="12" height="12" rx="2" fill="${c}" class="ring"/><text x="${(x + 9).toFixed(1)}" y="${(y + 3.5).toFixed(1)}" style="fill:var(--ll-fg);font-weight:600">${esc(lab.DOC[p.doc].short)}</text></g>`
      else if (p.kind === 'train')
        h += `<circle data-tip="${tip}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4.5" fill="${c}" class="ring"/>`
      else
        h += `<circle data-tip="${tip}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4.5" fill="var(--ll-bg)" stroke="${c}" stroke-width="2"/>`
    })
    svg.innerHTML = h
    $('mapLegend').innerHTML =
      `<span><svg width="10" height="10"><rect width="10" height="10" rx="2" fill="var(--ll-muted)"/></svg>document</span><span><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="var(--ll-muted)"/></svg>training text</span><span><svg width="10" height="10"><circle cx="5" cy="5" r="3.5" fill="none" stroke="var(--ll-muted)" stroke-width="1.6"/></svg>held-out text</span><span>colour = the document it belongs to</span>`
  }

  // --- loss curve ---
  function renderCurve() {
    const W = 400,
      H = 200,
      l = 34,
      r = 10,
      t = 10,
      b = 24
    const E = Math.max(cfg().epochs, 1)
    const vals = S.hist.flatMap((h) => [h.train, h.test]).filter((v) => v != null)
    const raw = Math.max(...vals, 1e-6)
    const step = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10].find((q) => raw / q <= 5) || 20
    const maxY = Math.ceil((raw * 1.02) / step) * step
    const X = (e) => l + (e / E) * (W - l - r),
      Yp = (v) => t + (1 - v / maxY) * (H - t - b)
    let h = ''
    for (let v = 0; v <= maxY + 1e-9; v += step) {
      h += `<line class="grid" x1="${l}" x2="${W - r}" y1="${Yp(v)}" y2="${Yp(v)}"/><text x="${l - 5}" y="${Yp(v) + 3}" text-anchor="end">${+v.toFixed(2)}</text>`
    }
    h += `<line class="ax" x1="${l}" x2="${W - r}" y1="${H - b}" y2="${H - b}"/>`
    ;[0, Math.round(E / 2), E].forEach(
      (e) => (h += `<text x="${X(e)}" y="${H - b + 14}" text-anchor="middle">${e}</text>`),
    )
    h += `<text x="${W - r}" y="${H - 2}" text-anchor="end">epoch</text>`
    const line = (key, c) => {
      const p = S.hist.filter((q) => q[key] != null)
      if (!p.length) return ''
      return (
        `<polyline fill="none" stroke="${c}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="${p.map((q) => `${X(q.epoch).toFixed(1)},${Yp(q[key]).toFixed(1)}`).join(' ')}"/>` +
        `<circle cx="${X(p[p.length - 1].epoch)}" cy="${Yp(p[p.length - 1][key])}" r="3.5" fill="${c}" class="ring"/>`
      )
    }
    h += line('train', col(0)) + line('test', col(1))
    // hover targets
    S.hist.forEach((q) => {
      h += `<rect x="${X(q.epoch) - (W - l - r) / E / 2}" y="${t}" width="${(W - l - r) / E}" height="${H - t - b}" fill="transparent" data-tip="epoch ${q.epoch}${q.train != null ? ` · train ${fmt(q.train, 3)}` : ''} · held-out ${fmt(q.test, 3)}"/>`
    })
    $('curve').innerHTML = h
  }

  // --- tiles ---
  function renderTiles() {
    const b = S.baseEval,
      n = lab.epoch ? S.lastEval || lab.evaluate(lab.model, S.loss, cfg()) : b
    const tile = (k, bv, nv, d, pct, hint) => {
      const f = (v) => (pct ? Math.round(v * 100) + '%' : fmt(v))
      const dv = nv - bv
      return `<div class="ll-tile" data-tip="${esc(hint)}"><div class="k">${k}</div><div class="v">${f(nv)} <small>from ${f(bv)}</small></div><div class="d">${lab.epoch ? (dv >= 0 ? '▲ ' : '▼ ') + (pct ? Math.round(Math.abs(dv) * 100) + ' pts' : fmt(Math.abs(dv))) : 'train to compare'} ${d}</div></div>`
    }
    $('tiles').innerHTML =
      tile(
        'Top-1 accuracy',
        b.recall,
        n.recall,
        '',
        true,
        'Share of the 6 held-out texts whose most similar document is the right one.',
      ) +
      tile(
        'Right vs wrong gap',
        b.gap,
        n.gap,
        'cosine',
        false,
        'Average cosine to the right document minus average cosine to the wrong ones. Bigger is a cleaner separation.',
      ) +
      tile(
        'Score correlation ρ',
        b.rho,
        n.rho,
        '',
        false,
        'Spearman correlation between model cosine and gold scores (1 same, 0.4 related, 0 unrelated). Rewards getting the related-but-different cases right.',
      ) +
      tile(
        'Held-out loss',
        b.loss,
        n.loss,
        '',
        false,
        'The training loss, computed on held-out texts the model never trained on.',
      )
  }

  // --- batch inspector ---
  function trunc(s, n = 22) {
    return s.length > n ? s.slice(0, n - 1) + '…' : s
  }
  function renderInspector() {
    const pr = lab.probe(S.loss, cfg())
    const k = S.loss
    let svg = '',
      title = '',
      note = ''
    if (k === 'mnrl' || k === 'mnrl_hn') {
      const B = pr.batch.length,
        cands = [...pr.batch.map((r) => r.doc), ...(k === 'mnrl_hn' ? pr.batch.map((r) => r.neg) : [])]
      const nC = cands.length,
        lw = 118,
        top = 34,
        cw = Math.min(40, (400 - lw) / nC),
        ch = Math.min(30, 220 / B)
      const Hh = top + B * ch + 6
      svg += cands
        .map(
          (d, j) =>
            `<rect x="${lw + j * cw + cw / 2 - 5}" y="4" width="10" height="10" rx="2" fill="${col(lab.LABEL[d])}"/><text x="${lw + j * cw + cw / 2}" y="26" text-anchor="middle" font-size="9">${j < B ? 'p' : 'n'}${(j % B) + 1}</text>`,
        )
        .join('')
      pr.info.probs.forEach((row, i) => {
        svg += `<text x="${lw - 6}" y="${top + i * ch + ch / 2 + 3}" text-anchor="end">${esc(trunc(pr.batch[i].q, 20))}</text>`
        row.forEach((p, j) => {
          svg += `<rect x="${lw + j * cw + 1}" y="${top + i * ch + 1}" width="${cw - 2}" height="${ch - 2}" rx="3" fill="var(--ll-seq)" fill-opacity="${(0.06 + 0.94 * p).toFixed(3)}" ${i === j ? 'stroke="var(--ll-fg)" stroke-width="2"' : ''} data-tip="${esc(`${pr.batch[i].q} → ${lab.DOC[cands[j]].text}: cos ${fmt(pr.info.cos[i][j])}, softmax ${Math.round(p * 100)}%`)}"/>`
          if (cw > 26)
            svg += `<text x="${lw + j * cw + cw / 2}" y="${top + i * ch + ch / 2 + 3}" text-anchor="middle" style="fill:${p > 0.55 ? '#fff' : 'var(--ll-fg)'}" font-size="9" pointer-events="none">${Math.round(p * 100)}</text>`
        })
      })
      svg = `<svg viewBox="0 0 400 ${Hh}">${svg}</svg>`
      title = `softmax over ${nC} candidates, per anchor`
      note = `Each row is a cross-entropy problem: the outlined cell (its own positive) should take all the probability. Every other column is a free negative${k === 'mnrl_hn' ? ', including the hard negatives n1…n' + B : ''}. Scale ${cfg().scale} sharpens the softmax.`
    } else if (k === 'triplet') {
      const m = cfg().margin,
        lw = 118,
        x0 = lw + 8,
        x1 = 392,
        X = (c) => x0 + ((c + 0.5) / 1.5) * (x1 - x0),
        rh = 44
      svg += [-0.5, 0, 0.5, 1]
        .map(
          (v) =>
            `<line class="grid" x1="${X(v)}" x2="${X(v)}" y1="14" y2="${16 + pr.info.rows.length * rh}"/><text x="${X(v)}" y="10" text-anchor="middle">${v}</text>`,
        )
        .join('')
      pr.info.rows.forEach((r, i) => {
        const y = 28 + i * rh,
          row = pr.batch[i]
        svg += `<text x="${lw - 6}" y="${y + 3}" text-anchor="end">${esc(trunc(row.q, 20))}</text>`
        const need = Math.min(r.cn + m, 1)
        svg += `<rect x="${X(r.cn)}" y="${y - 5}" width="${Math.max(0, X(need) - X(r.cn))}" height="10" rx="2" fill="var(--ll-fg)" fill-opacity=".1" data-tip="margin zone: the positive must be to the right of ${fmt(r.cn + m)}"/>`
        svg += `<line x1="${X(Math.min(r.cp, r.cn))}" x2="${X(Math.max(r.cp, r.cn))}" y1="${y}" y2="${y}" stroke="var(--ll-muted)" stroke-width="1"/>`
        svg += `<g data-tip="${esc(`negative: ${lab.DOC[row.neg].text} (cos ${fmt(r.cn)})`)}"><path d="M${X(r.cn) - 4},${y - 4}l8,8m0,-8l-8,8" stroke="${col(lab.LABEL[row.neg])}" stroke-width="2.5" stroke-linecap="round"/></g>`
        svg += `<circle cx="${X(r.cp)}" cy="${y}" r="5" fill="${col(lab.LABEL[row.doc])}" class="ring" data-tip="${esc(`positive: ${lab.DOC[row.doc].text} (cos ${fmt(r.cp)})`)}"/>`
        svg += `<text x="${x1}" y="${y + 14}" text-anchor="end" font-size="9">${r.active ? 'still pushing' : 'margin met, gradient 0'}</text>`
      })
      svg = `<svg viewBox="0 0 400 ${24 + pr.info.rows.length * rh + 8}">${svg}</svg>`
      title = 'cosine to positive (●) and negative (✕)'
      note = `The shaded band is the margin (${m.toFixed(2)}). While the positive sits inside or left of it, the triplet pushes. Once it clears the band the loss is exactly 0 and that triplet stops teaching.`
    } else if (k === 'cosent' || k === 'cosine') {
      const lw = 60,
        x0 = lw + 6,
        x1 = 392,
        X = (c) => x0 + ((c + 0.5) / 1.5) * (x1 - x0),
        groups = [1, 0.4, 0],
        rh = 44
      svg += [-0.5, 0, 0.5, 1]
        .map(
          (v) =>
            `<line class="grid" x1="${X(v)}" x2="${X(v)}" y1="14" y2="${16 + groups.length * rh}"/><text x="${X(v)}" y="10" text-anchor="middle">${v}</text>`,
        )
        .join('')
      groups.forEach((g, gi) => {
        const y = 36 + gi * rh
        svg += `<text x="${lw - 6}" y="${y + 3}" text-anchor="end">gold ${g.toFixed(1)}</text>`
        if (k === 'cosine')
          svg += `<line x1="${X(g)}" x2="${X(g)}" y1="${y - 14}" y2="${y + 14}" stroke="var(--ll-fg)" stroke-width="2" data-tip="target cosine ${g}"/>`
        const idx = pr.info.y
          .map((y_, i) => [y_, i])
          .filter(([y_]) => y_ === g)
          .map(([, i]) => i)
        idx.forEach((i, n) => {
          const c = pr.info.cos[i],
            jit = ((n % 5) - 2) * 4
          svg += `<circle cx="${X(Math.max(-0.5, Math.min(1, c)))}" cy="${y + jit}" r="4.5" fill="${col(gi === 0 ? 0 : gi === 1 ? 3 : 1)}" class="ring" data-tip="${esc(`${pr.batch[i].a} ↔ ${pr.batch[i].b}: gold ${g}, cos ${fmt(c)}`)}"/>`
        })
      })
      svg = `<svg viewBox="0 0 400 ${22 + groups.length * rh + 8}">${svg}</svg>`
      title = `${pr.batch.length} scored pairs, by gold score`
      note =
        k === 'cosine'
          ? 'Mean squared error pulls each dot onto its target line: 1.0, 0.4 and 0.0 exactly. It spends effort on precise values even when the order is already right.'
          : `CoSENT only asks for order: every gold-1.0 dot right of every gold-0.4 dot, and so on. ${pr.info.violations} of ${pr.info.pairs} orderings in this batch are still wrong. Where the dots land doesn't matter.`
    } else {
      const lab_ = pr.info.lab,
        n = lab_.length,
        C = pr.info.C,
        s = Math.min(22, 250 / n),
        off = 18,
        Wd = off + n * s
      const order = lab_
        .map((l, i) => [l, i])
        .sort((a, b) => a[0] - b[0])
        .map(([, i]) => i)
      order.forEach((i, r) => {
        svg += `<rect x="${off + r * s + 1}" y="2" width="${s - 2}" height="10" rx="2" fill="${col(lab_[i])}" data-tip="${esc(pr.batch[i].text)}"/>`
        svg += `<rect x="2" y="${off + r * s + 1}" width="10" height="${s - 2}" rx="2" fill="${col(lab_[i])}" data-tip="${esc(pr.batch[i].text)}"/>`
        order.forEach((j, c) => {
          if (i === j) return
          const same = lab_[i] === lab_[j],
            v = Math.max(0, Math.min(1, (C[i][j] + 0.2) / 1.2))
          svg += `<rect x="${off + c * s + 1}" y="${off + r * s + 1}" width="${s - 2}" height="${s - 2}" rx="2" fill="var(--ll-seq)" fill-opacity="${(0.05 + 0.9 * v).toFixed(3)}" ${same ? 'stroke="var(--ll-fg)" stroke-width="1.5"' : ''} data-tip="${esc(`${pr.batch[i].text} ↔ ${pr.batch[j].text}: cos ${fmt(C[i][j])} (${same ? 'same label' : 'different label'})`)}"/>`
        })
        if (k === 'batchhard') {
          const [p, q] = pr.info.picks[i]
          if (p >= 0) {
            const c = order.indexOf(p)
            svg += `<circle cx="${off + c * s + s / 2}" cy="${off + r * s + s / 2}" r="${s / 4}" fill="none" stroke="var(--ll-fg)" stroke-width="2" pointer-events="none"/>`
          }
          if (q >= 0) {
            const c = order.indexOf(q),
              cx = off + c * s + s / 2,
              cy = off + r * s + s / 2,
              d = s / 4
            svg += `<path d="M${cx - d},${cy - d}L${cx + d},${cy + d}M${cx + d},${cy - d}L${cx - d},${cy + d}" stroke="var(--ll-fg)" stroke-width="2" pointer-events="none"/>`
          }
        }
      })
      svg = `<svg viewBox="0 0 ${Math.max(Wd, 200)} ${Wd + 4}" style="max-width:min(100%, 290px);margin:auto">${svg}</svg>`
      title = `cosine between all ${n} texts, sorted by label`
      note =
        k === 'batchall'
          ? `Outlined cells share a label and should be dark; the rest should be light. ${pr.info.active} of ${pr.info.valid} possible triplets in this batch still break the margin, and the loss averages over those.`
          : 'For each row, ○ marks the hardest positive (least similar same-label text) and ✕ the hardest negative (most similar other-label text). Only those two cells get a gradient.'
    }
    $('insp').innerHTML = svg
    $('inspTitle').textContent = title
    $('inspNote').textContent = note
  }

  // --- search ---
  function renderSearchChips() {
    $('qchips').innerHTML = lab.W.chips
      .map((c) => `<button class="ll-chip" data-q="${esc(c)}">${esc(c)}</button>`)
      .join('')
    $('q').placeholder = lab.W.placeholder
    $('q').value = lab.W.chips[0]
    runSearch()
  }
  $('qchips').addEventListener('click', (e) => {
    const b = e.target.closest('[data-q]')
    if (!b) return
    $('q').value = b.dataset.q
    runSearch()
  })
  $('q').addEventListener('input', runSearch)
  function resHTML(rs) {
    return rs
      .slice(0, 4)
      .map(
        (r, i) =>
          `<div class="ll-res${i === 0 ? ' top' : ''}"><span class="n">${sw(r.d.id)}${esc(r.d.short)}</span><span class="bar"><i style="width:${Math.max(0, r.s) * 100}%;background:${col(lab.LABEL[r.d.id])}"></i></span><span class="s">${fmt(r.s)}</span></div>`,
      )
      .join('')
  }
  function runSearch() {
    const q = $('q').value.trim()
    if (!q || !tokenize(q).length) {
      $('resBase').innerHTML = $('resFt').innerHTML = '<p class="ll-note">Type a few words.</p>'
      return
    }
    $('resBase').innerHTML = resHTML(lab.search(lab.base, q))
    $('resFt').innerHTML = lab.epoch
      ? resHTML(lab.search(lab.model, q))
      : '<p class="ll-note">Train first, then compare.</p>'
  }

  // --- code ---
  function renderCode() {
    const k = S.loss,
      L = LOSSES[k],
      c = cfg(),
      cs = CASES[LOSS_CASE[k]]
    const s = (x) => `<span class="s">"${esc(x)}"</span>`,
      cm = (x) => `<span class="c"># ${esc(x)}</span>`,
      kw = (x) => `<span class="k">${x}</span>`
    const pr = lab.probe(k, c).batch.slice(0, 2)
    let cols = ''
    if (cs.cols[0] === 'anchor')
      cols =
        `    ${s('anchor')}: [${pr.map((r) => s(r.q)).join(', ')}, ...],\n    ${s('positive')}: [${pr.map((r) => s(lab.DOC[r.doc].text)).join(', ')}, ...],` +
        (cs.cols[2] ? `\n    ${s('negative')}: [${pr.map((r) => s(lab.DOC[r.neg].text)).join(', ')}, ...],` : '')
    else if (cs.cols[0] === 'sentence1')
      cols = `    ${s('sentence1')}: [${pr.map((r) => s(r.a)).join(', ')}, ...],\n    ${s('sentence2')}: [${pr.map((r) => s(r.b)).join(', ')}, ...],\n    ${s('score')}: [${pr.map((r) => r.score.toFixed(1)).join(', ')}, ...],`
    else
      cols = `    ${s('text')}: [${pr.map((r) => s(r.text)).join(', ')}, ...],\n    ${s('label')}: [${pr.map((r) => lab.LABEL[r.doc]).join(', ')}, ...],  ${cm('integer class ids')}`
    const lossLine = {
      mnrl: `losses.MultipleNegativesRankingLoss(model, scale=${c.scale})`,
      mnrl_hn: `losses.MultipleNegativesRankingLoss(model, scale=${c.scale})`,
      triplet: `losses.TripletLoss(\n    model,\n    distance_metric=losses.TripletDistanceMetric.COSINE,\n    triplet_margin=${c.margin},\n)`,
      cosent: `losses.CoSENTLoss(model, scale=${c.scale})`,
      cosine: `losses.CosineSimilarityLoss(model)`,
      batchall: `losses.BatchAllTripletLoss(\n    model,\n    distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance,\n    margin=${c.margin},\n)`,
      batchhard: `losses.BatchHardTripletLoss(\n    model,\n    distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance,\n    margin=${c.margin},\n)`,
    }[k]
    const sampler =
      cs.sampler && c.sampler
        ? `\n    batch_sampler=BatchSamplers.${cs.sampler === 'group' ? 'GROUP_BY_LABEL' : 'NO_DUPLICATES'},`
        : ''
    $('code').innerHTML = `${kw('from')} datasets ${kw('import')} Dataset
${kw('from')} sentence_transformers ${kw('import')} (
    SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments,
)
${kw('from')} sentence_transformers.sentence_transformer ${kw('import')} losses
${kw('from')} sentence_transformers.base.sampler ${kw('import')} BatchSamplers
${cm('sentence-transformers < 5.4: from sentence_transformers import losses')}
${cm('and from sentence_transformers.training_args import BatchSamplers')}

model = SentenceTransformer(${s('sentence-transformers/all-MiniLM-L6-v2')})

train_dataset = Dataset.from_dict({   ${cm(cs.name)}
${cols}
})

loss = ${lossLine.replace(/\n/g, '\n')}

args = SentenceTransformerTrainingArguments(
    output_dir=${s('out/' + lab.W.slug + '-' + k)},
    num_train_epochs=${Math.max(1, Math.round(c.epochs / 20))},   ${cm('real models need far fewer epochs than this toy')}
    per_device_train_batch_size=${k.startsWith('batch') || k.startsWith('cos') ? 32 : 64},
    learning_rate=2e-5,${sampler}
)
SentenceTransformerTrainer(model=model, args=args, train_dataset=train_dataset, loss=loss).train()`
  }

  // ---------- tooltip ----------
  const tip = $('tip')
  root.addEventListener('pointermove', (e) => {
    const el = e.target.closest && e.target.closest('[data-tip]')
    if (!el || !root.contains(el)) {
      tip.hidden = true
      return
    }
    tip.textContent = el.getAttribute('data-tip')
    tip.hidden = false
    const r = tip.getBoundingClientRect()
    let x = e.clientX + 12,
      y = e.clientY + 14
    if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - 12
    if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - 14
    tip.style.left = x + 'px'
    tip.style.top = y + 'px'
  })
  root.addEventListener('pointerleave', () => (tip.hidden = true))

  // ---------- external "train this" buttons in the article ----------
  document.addEventListener('click', (e) => {
    const b = e.target.closest && e.target.closest('[data-ll-loss]')
    if (!b) return
    e.preventDefault()
    if (b.dataset.llWorld && b.dataset.llWorld !== lab.worldKey) {
      lab.setWorld(b.dataset.llWorld)
      S.prev = null
      renderWorlds()
      renderSearchChips()
    }
    let extra
    try {
      extra = b.dataset.llCfg ? JSON.parse(b.dataset.llCfg) : undefined
    } catch {
      extra = undefined
    }
    setLoss(b.dataset.llLoss, { cfg: extra })
    root.scrollIntoView({ behavior: 'smooth', block: 'start' })
    setTimeout(start, 500)
  })

  // ---------- boot ----------
  renderWorlds()
  renderSearchChips()
  setLoss('mnrl')
}
