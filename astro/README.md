# Using the playground in Astro

Copy `src/` and `astro/` into your project, then in any `.astro` or `.mdx` page:

```mdx
import LossPlayground from '../astro/LossPlayground.astro'
import TryIt from '../astro/TryIt.astro'

<TryIt loss="triplet" cfg={{ margin: 0 }} label="TripletLoss with margin 0" />

<LossPlayground />
```

`TryIt` renders a button that loads a loss (and optional dataset/settings) into the playground and starts training. Loss ids: `mnrl`, `mnrl_hn`, `triplet`, `cosent`, `cosine`, `batchall`, `batchhard`. Datasets: `retrieval`, `similarity`, `clustering`, `dedup`.

## Theme variables

The styles read your site's colours from these CSS variables, as HSL triples (the shadcn/ui convention):

```css
:root {
  --primary: 200 29% 45%;
  --foreground: 240 10% 3.9%;
  --muted-foreground: 240 3.8% 32%;
  --background: 0 0% 100%;
  --border: 240 5.9% 88%;
}
.dark { /* dark values */ }
```

Dark mode switches on a `.dark` class on an ancestor (usually `<html>`).
