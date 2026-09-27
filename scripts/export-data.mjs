// Writes python/datasets.json from src/data.js so the Python scripts train on
// exactly the same four datasets as the playground.
//
// Run: npm run export-data

import { writeFileSync } from 'node:fs'
import { WORLDS } from '../src/data.js'

const out = Object.fromEntries(
  Object.entries(WORLDS).map(([key, w]) => [
    key,
    {
      domain: w.domain,
      documents: Object.fromEntries(w.groups.map((g) => [g.id, g.text])),
      train: w.train.map(([text, doc]) => ({ text, doc })),
      test: w.test.map(([text, doc]) => ({ text, doc })),
      hard_negative: w.hardneg,
      related: w.related
    }
  ])
)

writeFileSync(new URL('../python/datasets.json', import.meta.url), JSON.stringify(out, null, 2) + '\n')
console.log('python/datasets.json written:', Object.keys(out).join(', '))
