// @ts-nocheck
// Datasets, tokenizer and loss catalog for the embedding loss playground.
// Four small datasets, a tokenizer with a few generic synonym groups, and the loss catalog.

export const WORLDS={
  retrieval:{
    task:'Search / RAG retrieval',short:'Retrieval',slug:'support-search',domain:'Help-center search',
    q:'question',qs:'questions',t:'article',ts:'articles',who:'Customer writes',tlabel:'Help-center article',
    title:'Help-center search: six articles, 18 customer questions',
    lede:'Customers write “I want my money back”; the article is titled “Refund request process”. A general-purpose model doesn\'t know those belong together in <i>your</i> domain.',
    hn:'a look-alike wrong article (money words: refund vs. cancel; shipping words: tracking vs. address)',
    groups:[
      {id:'refund',short:'Refund',text:'Refund request process for returned orders'},
      {id:'ship',short:'Tracking',text:'Track shipment delivery status'},
      {id:'pass',short:'Password',text:'Reset account password'},
      {id:'cancel',short:'Cancel plan',text:'Cancel subscription plan'},
      {id:'damage',short:'Damaged',text:'Report damaged or defective product'},
      {id:'address',short:'Address',text:'Change shipping address'}],
    train:[['i want my money back','refund'],['how do i get reimbursed','refund'],['give me a refund for this','refund'],
      ['where is my package','ship'],['parcel has not arrived','ship'],['order taking forever to come','ship'],
      ['cant log in','pass'],['forgot my login details','pass'],['locked out and cant sign in','pass'],
      ['stop charging me every month','cancel'],['end my membership','cancel'],['quit monthly billing','cancel'],
      ['item arrived broken','damage'],['screen is cracked','damage'],['box was smashed in transit','damage'],
      ['i moved to a new house','address'],['wrong place to deliver','address'],['update where you send it','address']],
    test:[['get my money reimbursed','refund'],['package still not arrived','ship'],['forgot how to sign in','pass'],
      ['end monthly charging','cancel'],['item cracked and broken','damage'],['send it to my new house','address']],
    hardneg:{refund:'cancel',ship:'address',pass:'cancel',cancel:'refund',damage:'refund',address:'ship'},
    related:[['refund','damage'],['refund','cancel'],['ship','address'],['ship','damage']],
    presets:[['money back ↔ refund','i want my money back','Refund request process for returned orders'],['moved house ↔ address','i moved to a new house','Change shipping address'],['parcel ↔ tracking','where is my parcel','Track shipment delivery status'],['unrelated','forgot my login details','Report damaged or defective product']],
    chips:['stop charging my card every month','my parcel is late','screen arrived smashed','reimbursed for broken item','i moved, update my address','cant sign in to account'],
    placeholder:'e.g. i was charged twice'},
  similarity:{
    task:'Similarity scoring',short:'Similarity',slug:'review-similarity',domain:'Product-review similarity',
    q:'review snippet',qs:'review snippets',t:'reference opinion',ts:'reference opinions',who:'Review snippet',tlabel:'Reference opinion',
    title:'Headphone reviews: six opinions, 18 review snippets',
    lede:'Reviews that say opposite things about the same feature share most of their words (“battery goes for days” vs. “battery drains so fast”). A generic model scores them as near-duplicates. A similarity model has to learn that the <i>opinion</i> matters, not just the topic.',
    hn:'the opposite opinion about the same feature (battery good vs. battery bad)',
    groups:[
      {id:'bgood',short:'Battery +',text:'Battery life is excellent'},
      {id:'bbad',short:'Battery −',text:'Battery drains too quickly'},
      {id:'sgood',short:'Sound +',text:'Sound quality is excellent'},
      {id:'sbad',short:'Sound −',text:'Sound quality is poor'},
      {id:'fgood',short:'Comfort +',text:'Very comfortable to wear'},
      {id:'fbad',short:'Comfort −',text:'Uncomfortable to wear'}],
    train:[['charge lasts all week','bgood'],['battery goes for days','bgood'],['rarely need to recharge','bgood'],
      ['dies after two hours','bbad'],['battery drains so fast','bbad'],['constantly need to recharge','bbad'],
      ['crisp highs and deep bass','sgood'],['audio is amazing','sgood'],['music sounds rich','sgood'],
      ['muddy and tinny audio','sbad'],['bass is weak','sbad'],['music sounds flat','sbad'],
      ['feels like nothing on my head','fgood'],['soft cushions all day','fgood'],['can wear for hours','fgood'],
      ['hurts after an hour','fbad'],['pinches my ears','fbad'],['too tight and heavy','fbad']],
    test:[['lasts days on one charge','bgood'],['drains after two hours','bbad'],['amazing deep bass','sgood'],
      ['weak and tinny sound','sbad'],['soft and light for hours','fgood'],['tight and hurts my ears','fbad']],
    hardneg:{bgood:'bbad',bbad:'bgood',sgood:'sbad',sbad:'sgood',fgood:'fbad',fbad:'fgood'},
    related:[['bgood','bbad'],['sgood','sbad'],['fgood','fbad']],
    presets:[['battery: good vs. bad','battery goes for days','Battery drains too quickly'],['bass ↔ sound','crisp highs and deep bass','Sound quality is excellent'],['cushions ↔ comfort','soft cushions all day','Very comfortable to wear'],['unrelated','pinches my ears','Battery life is excellent']],
    chips:['battery barely lasts a day','rich bass and clear vocals','ears hurt after a while','lasts all week without charging','sounds muddy','light and comfy'],
    placeholder:'e.g. goes a whole week between charges'},
  clustering:{
    task:'Classification / clustering',short:'Classification',slug:'ticket-router',domain:'IT ticket routing',
    q:'ticket',qs:'tickets',t:'category',ts:'categories',who:'Helpdesk ticket',tlabel:'Routing category',
    title:'IT helpdesk: six routing categories, 18 tickets',
    lede:'Every ticket must land with the right team. To a generic model, “outlook not syncing” and “vpn wont connect” both sound like connection trouble. Your routing categories draw the line somewhere else.',
    hn:'the team a ticket is most often mis-routed to (email vs. access, hardware vs. printers)',
    groups:[
      {id:'net',short:'Network',text:'Network and VPN issues'},
      {id:'hw',short:'Hardware',text:'Laptop hardware repair'},
      {id:'acc',short:'Access',text:'Account access and permissions'},
      {id:'mail',short:'Email',text:'Email and calendar support'},
      {id:'print',short:'Printing',text:'Printer problems'},
      {id:'sw',short:'Software',text:'Software installs and licenses'}],
    train:[['wifi keeps dropping','net'],['vpn wont connect from home','net'],['internet is really slow','net'],
      ['laptop screen flickers','hw'],['keyboard keys stuck','hw'],['battery swollen on my laptop','hw'],
      ['need permissions for the shared drive','acc'],['account locked after password change','acc'],['request admin rights','acc'],
      ['outlook not syncing','mail'],['meeting invites not arriving','mail'],['inbox full cannot receive mail','mail'],
      ['printer jammed on floor 3','print'],['cant find the printer','print'],['prints come out blank','print'],
      ['install python on my machine','sw'],['excel crashes on startup','sw'],['need a license for figma','sw']],
    test:[['vpn drops every few minutes','net'],['laptop keyboard stopped working','hw'],['locked out and need drive access','acc'],
      ['outlook invites not syncing','mail'],['floor 3 printer jammed again','print'],['need a license to install excel','sw']],
    hardneg:{net:'mail',hw:'print',acc:'mail',mail:'acc',print:'hw',sw:'hw'},
    related:[['net','mail'],['acc','mail'],['hw','print'],['sw','hw']],
    presets:[['wifi ↔ network','wifi keeps dropping','Network and VPN issues'],['permissions ↔ access','need permissions for the shared drive','Account access and permissions'],['blank prints ↔ printer','prints come out blank','Printer problems'],['unrelated','outlook not syncing','Laptop hardware repair']],
    chips:['cant connect to wifi in the office','my screen is cracked','outlook calendar is empty','printer out of toner','need admin rights to install','excel keeps freezing'],
    placeholder:'e.g. teams calls keep dropping'},
  dedup:{
    task:'Duplicate detection',short:'Duplicates',slug:'forum-dedup',domain:'Duplicate-question detection',
    q:'question',qs:'questions',t:'canonical question',ts:'canonical questions',who:'New forum question',tlabel:'Canonical question',
    title:'Developer forum: six canonical questions, 18 duplicates',
    lede:'Forum users ask the same thing in endless ways. “Take back a commit I just made” duplicates “How to undo the last git commit”. “How to delete a git branch” uses the same git vocabulary but is a different question.',
    hn:'a different question from the same topic (undo commit vs. delete branch)',
    groups:[
      {id:'rev',short:'Reverse list',text:'How to reverse a list in Python'},
      {id:'sortd',short:'Sort dict',text:'How to sort a dictionary by value'},
      {id:'undo',short:'Undo commit',text:'How to undo the last git commit'},
      {id:'delb',short:'Delete branch',text:'How to delete a git branch'},
      {id:'center',short:'Center div',text:'How to center a div in CSS'},
      {id:'font',short:'Font size',text:'How to change font size in CSS'}],
    train:[['flip order of python list','rev'],['python list backwards','rev'],['invert list elements order','rev'],
      ['order dict by its values','sortd'],['sort python dict values ascending','sortd'],['rank dictionary entries by value','sortd'],
      ['revert my most recent commit','undo'],['git remove last commit','undo'],['take back a commit i just made','undo'],
      ['remove local branch git','delb'],['get rid of old branch','delb'],['git branch cleanup delete','delb'],
      ['align div in the middle','center'],['css horizontally and vertically center element','center'],['put box in center of page','center'],
      ['make text bigger css','font'],['css text size larger','font'],['increase heading font','font']],
    test:[['flip list backwards','rev'],['rank dict entries ascending','sortd'],['take back my recent commit','undo'],
      ['get rid of old local one','delb'],['align box in the middle','center'],['larger heading text','font']],
    hardneg:{rev:'sortd',sortd:'rev',undo:'delb',delb:'undo',center:'font',font:'center'},
    related:[['rev','sortd'],['undo','delb'],['center','font']],
    presets:[['take back ↔ undo','take back a commit i just made','How to undo the last git commit'],['same topic, different question','python list backwards','How to sort a dictionary by value'],['middle ↔ center','align div in the middle','How to center a div in CSS'],['unrelated','make text bigger css','How to delete a git branch']],
    chips:['reverse array order python','sort dict descending by value','git reset last commit','delete remote branch','vertically center text css','bigger font css'],
    placeholder:'e.g. discard my last git commit'},
};

export const STOP = new Set('i my me a an the is to of for this it was and how do has have in on at or you your be where with'.split(' '));
export const stem = w => (w.length>3 && w.endsWith('s') && !w.endsWith('ss')) ? w.slice(0,-1) : w;
export const tokenize = s => s.toLowerCase().replace(/[^a-z0-9\s]/g,' ').split(/\s+/).filter(w=>w.length>1 && !STOP.has(w)).map(stem);
export const GENERIC = [
  ['package','parcel','shipment','delivery','shipping','deliver','arrived','transit'],
  ['broken','cracked','smashed','damaged','defective'],
  ['login','log','sign'],
  ['money','refund','reimbursed','billing','charging','card'],
  ['month','monthly'],
  ['battery','charge','recharge'],
  ['audio','sound','music','bass','highs','vocal'],
  ['head','ear','wear','cushion'],
  ['hour','day','week'],
  ['wifi','internet','network','vpn','connect'],
  ['laptop','keyboard','screen'],
  ['mail','email','inbox','outlook','sync','syncing'],
  ['install','software','license'],
  ['python','list','dict','dictionary'],
  ['git','commit','branch'],
  ['css','div','element','font','text','box'],
].map(g=>g.map(stem));

export const CATALOG = [
 {
  "id": "mnrl",
  "name": "MultipleNegativesRankingLoss",
  "alias": "MNRL",
  "cat": "ranking",
  "desc": "The go-to loss for semantic search. Uses other in-batch samples as negatives, so you only need (anchor, positive) pairs. Bigger batch = more negatives = stronger signal.",
  "data": "(anchor, positive) pairs\n(anchor, positive, negative) triplets also accepted",
  "labels": "None",
  "best": [
   "Semantic search",
   "FAQ matching",
   "Document retrieval"
  ],
  "pros": [
   "Simplest data requirement",
   "In-batch negatives scale with batch size",
   "Strong out-of-the-box performance"
  ],
  "cons": [
   "May learn false negatives from batch",
   "Quality depends heavily on batch size"
  ],
  "params": {
   "scale": "20.0 (softmax temperature)",
   "similarity_fct": "cos_sim (or dot product)"
  },
  "starred": true
 },
 {
  "id": "cached-mnrl",
  "name": "CachedMultipleNegativesRankingLoss",
  "alias": "CachedMNRL",
  "cat": "ranking",
  "desc": "Memory-efficient MNRL that uses gradient caching to simulate massive batch sizes (4000+) on limited GPU memory. ~20% slower per step but each step sees far more negatives.",
  "data": "(anchor, positive) pairs",
  "labels": "None",
  "best": [
   "Large-scale training (100k+ pairs)",
   "Memory-constrained GPUs"
  ],
  "pros": [
   "Effective batch of thousands",
   "Same data format as MNRL",
   "Better gradients per step"
  ],
  "cons": [
   "~20% slower per step",
   "More complex memory management"
  ],
  "params": {
   "mini_batch_size": "32 (GPU encoding batch)",
   "scale": "20.0"
  },
  "starred": true
 },
 {
  "id": "mnrl-sym",
  "name": "MultipleNegativesSymmetricRankingLoss",
  "alias": "SymMNRL",
  "cat": "ranking",
  "desc": "Symmetric variant of MNRL that treats both directions equally: A->B and B->A. Useful when the similarity relationship is bidirectional rather than query->document.",
  "data": "(anchor, positive) pairs",
  "labels": "None",
  "best": [
   "Paraphrase detection",
   "Duplicate detection",
   "Symmetric similarity"
  ],
  "pros": [
   "Better for bidirectional tasks",
   "Same simple data format"
  ],
  "cons": [
   "Slower than vanilla MNRL",
   "Not ideal for asymmetric search"
  ],
  "params": {
   "scale": "20.0"
  }
 },
 {
  "id": "cached-mnrl-sym",
  "name": "CachedMultipleNegativesSymmetricRankingLoss",
  "alias": "CachedSymMNRL",
  "cat": "ranking",
  "desc": "Cached version of symmetric ranking loss. Combines gradient caching with bidirectional similarity optimization.",
  "data": "(anchor, positive) pairs",
  "labels": "None",
  "best": [
   "Large-scale symmetric similarity"
  ],
  "pros": [
   "Huge effective batch",
   "Bidirectional"
  ],
  "cons": [
   "Slower",
   "Complex setup"
  ],
  "params": {
   "mini_batch_size": "32"
  }
 },
 {
  "id": "triplet",
  "name": "TripletLoss",
  "alias": "TripletLoss",
  "cat": "triplet",
  "desc": "Classic triplet loss: explicitly push anchor closer to positive and away from negative by a margin. Best when you have carefully curated hard negatives.",
  "data": "(anchor, positive, negative) triplets",
  "labels": "None (implicit in triplet structure)",
  "best": [
   "Ranking with hard negatives",
   "Re-ranking pipelines"
  ],
  "pros": [
   "Explicit control over negatives",
   "Works well with hard negatives",
   "Interpretable margin"
  ],
  "cons": [
   "Need to mine/curate negatives",
   "3x more data annotation effort",
   "Margin tuning required"
  ],
  "params": {
   "distance_metric": "COSINE",
   "triplet_margin": "0.5 (tune lower if can't converge)"
  }
 },
 {
  "id": "batch-all-triplet",
  "name": "BatchAllTripletLoss",
  "alias": "BatchAllTrip",
  "cat": "triplet",
  "desc": "Generates all valid triplets from class-labeled inputs within a batch. No explicit triplet construction needed -- just label your data.",
  "data": "Single sentences + class labels",
  "labels": "Class labels (int)",
  "best": [
   "Classification-based clustering",
   "When you have class labels, not pairs"
  ],
  "pros": [
   "Automatic triplet mining",
   "Uses all valid combinations",
   "Simple label-based data"
  ],
  "cons": [
   "Many easy triplets dilute signal",
   "Requires class labels"
  ],
  "params": {
   "distance_metric": "COSINE",
   "margin": "5"
  }
 },
 {
  "id": "batch-hard-triplet",
  "name": "BatchHardTripletLoss",
  "alias": "BatchHardTrip",
  "cat": "triplet",
  "desc": "Selects only the hardest positive (furthest same-class) and hardest negative (closest different-class) within each batch. Much stronger training signal.",
  "data": "Single sentences + class labels",
  "labels": "Class labels (int)",
  "best": [
   "Hard negative mining without explicit curation"
  ],
  "pros": [
   "Strongest signal per batch",
   "No manual negative selection"
  ],
  "cons": [
   "Can be unstable early in training",
   "Noisy with small batches"
  ],
  "params": {
   "distance_metric": "COSINE",
   "margin": "5"
  }
 },
 {
  "id": "batch-hard-soft",
  "name": "BatchHardSoftMarginTripletLoss",
  "alias": "BatchHardSoft",
  "cat": "triplet",
  "desc": "Like BatchHardTripletLoss but uses a soft margin (log-sum-exp) instead of a fixed margin. No margin hyperparameter to tune.",
  "data": "Single sentences + class labels",
  "labels": "Class labels (int)",
  "best": [
   "When margin tuning is impractical"
  ],
  "pros": [
   "No margin to tune",
   "Harder examples get more gradient"
  ],
  "cons": [
   "Can be less stable",
   "Requires class labels"
  ],
  "params": {
   "distance_metric": "COSINE"
  }
 },
 {
  "id": "batch-semi-hard",
  "name": "BatchSemiHardTripletLoss",
  "alias": "BatchSemiHard",
  "cat": "triplet",
  "desc": "Uses semi-hard negatives: closer than the positive but still on the correct side of the margin. Balances learning signal with stability.",
  "data": "Single sentences + class labels",
  "labels": "Class labels (int)",
  "best": [
   "Stable training with automatic mining"
  ],
  "pros": [
   "More stable than full hard mining",
   "Good learning signal"
  ],
  "cons": [
   "May plateau if batch too small",
   "Requires class labels"
  ],
  "params": {
   "distance_metric": "COSINE",
   "margin": "5"
  }
 },
 {
  "id": "cosent",
  "name": "CoSENTLoss",
  "alias": "CoSENT",
  "cat": "similarity",
  "desc": "Optimizes cosine similarity to match human-annotated scores. Uses a ranking-based objective that's more robust than direct regression (unlike CosineSimilarityLoss).",
  "data": "(sentence_A, sentence_B, score) triples",
  "labels": "Float similarity (0-1)",
  "best": [
   "Semantic textual similarity (STS)",
   "Paraphrase scoring",
   "Graded relevance"
  ],
  "pros": [
   "Learns nuanced similarity",
   "Robust ranking objective",
   "Works with continuous scores"
  ],
  "cons": [
   "Needs scored annotations",
   "Slower to converge than MNRL"
  ],
  "params": {
   "scale": "20.0"
  }
 },
 {
  "id": "angle",
  "name": "AnglELoss",
  "alias": "AnglE",
  "cat": "similarity",
  "desc": "Angular-based loss that optimizes embedding angles rather than distances. A modern alternative to CoSENTLoss with better gradient properties on hard examples.",
  "data": "(sentence_A, sentence_B, score) triples",
  "labels": "Float similarity (0-1)",
  "best": [
   "STS tasks",
   "When CoSENT plateaus"
  ],
  "pros": [
   "Better gradients than CoSENT",
   "Angle-aware optimization"
  ],
  "cons": [
   "Needs scored data",
   "Newer, less battle-tested"
  ],
  "params": {
   "scale": "20.0"
  }
 },
 {
  "id": "cosine-sim",
  "name": "CosineSimilarityLoss",
  "alias": "CosSim",
  "cat": "similarity",
  "desc": "Direct regression: minimizes MSE between predicted cosine similarity and target score. Simple but less effective than CoSENT/AnglE for most tasks.",
  "data": "(sentence_A, sentence_B, score) triples",
  "labels": "Float similarity (0-1)",
  "best": [
   "Quick baselines",
   "Simple similarity tasks"
  ],
  "pros": [
   "Very simple",
   "Easy to understand"
  ],
  "cons": [
   "Weaker than CoSENT/AnglE",
   "Sensitive to score distribution"
  ],
  "params": {
   "loss_fct": "nn.MSELoss()"
  }
 },
 {
  "id": "contrastive",
  "name": "ContrastiveLoss",
  "alias": "Contrastive",
  "cat": "contrastive",
  "desc": "Pulls similar pairs together and pushes dissimilar pairs apart. Requires binary labels indicating whether each pair is similar or dissimilar.",
  "data": "(sentence_A, sentence_B, label) triples",
  "labels": "Binary (0 or 1)",
  "best": [
   "Binary similarity judgments",
   "Verification tasks"
  ],
  "pros": [
   "Works with binary labels",
   "Intuitive objective"
  ],
  "cons": [
   "Needs both positive and negative pairs",
   "No in-batch negatives"
  ],
  "params": {
   "distance_metric": "COSINE",
   "margin": "0.5"
  }
 },
 {
  "id": "online-contrastive",
  "name": "OnlineContrastiveLoss",
  "alias": "OnlineContrastive",
  "cat": "contrastive",
  "desc": "Contrastive loss that dynamically focuses on hard positives and hard negatives within each batch, ignoring easy examples for a stronger signal.",
  "data": "(sentence_A, sentence_B, label) triples",
  "labels": "Binary (0 or 1)",
  "best": [
   "Hard-example mining with binary labels"
  ],
  "pros": [
   "Online hard mining",
   "Stronger signal than vanilla contrastive"
  ],
  "cons": [
   "Still needs binary labels",
   "Can be unstable"
  ],
  "params": {
   "distance_metric": "COSINE",
   "margin": "0.5"
  }
 },
 {
  "id": "ct",
  "name": "ContrastiveTensionLoss",
  "alias": "CT",
  "cat": "contrastive",
  "desc": "Self-supervised contrastive learning from single sentences. Generates positive pairs by passing the same sentence through two independent encoders.",
  "data": "Single sentences",
  "labels": "None",
  "best": [
   "Unsupervised domain adaptation",
   "No labeled data available"
  ],
  "pros": [
   "No labels needed",
   "Domain adaptation"
  ],
  "cons": [
   "Requires two encoders",
   "Weaker than supervised losses"
  ],
  "params": {}
 },
 {
  "id": "ct-ibn",
  "name": "ContrastiveTensionLossInBatchNegatives",
  "alias": "CT-IBN",
  "cat": "contrastive",
  "desc": "Contrastive tension with in-batch negatives for stronger self-supervised learning. Uses (anchor, anchor) pairs and treats other batch items as negatives.",
  "data": "(anchor, anchor) duplicate pairs",
  "labels": "None",
  "best": [
   "Self-supervised with better negatives"
  ],
  "pros": [
   "Better than vanilla CT",
   "Still no labels"
  ],
  "cons": [
   "Two encoders needed",
   "Complex setup"
  ],
  "params": {}
 },
 {
  "id": "softmax",
  "name": "SoftmaxLoss",
  "alias": "Softmax",
  "cat": "classification",
  "desc": "Classic cross-entropy classification on top of [CLS] token or pooled embeddings. Maps pairs to discrete classes (entailment/contradiction/neutral for NLI).",
  "data": "(sentence_A, sentence_B) pairs + class label",
  "labels": "Class labels (int)",
  "best": [
   "NLI-based training",
   "Multi-class similarity"
  ],
  "pros": [
   "Standard classification setup",
   "Works with NLI datasets"
  ],
  "cons": [
   "Adds classification head",
   "Less direct for similarity"
  ],
  "params": {
   "num_labels": "3 (for NLI)"
  }
 },
 {
  "id": "mse",
  "name": "MSELoss",
  "alias": "MSE",
  "cat": "distillation",
  "desc": "Minimizes mean squared error between student embeddings and teacher embeddings. Used for knowledge distillation from a larger model to a smaller one.",
  "data": "Sentences (teacher generates targets)",
  "labels": "Teacher embeddings (auto-generated)",
  "best": [
   "Model compression",
   "Distilling large models"
  ],
  "pros": [
   "Simple distillation",
   "Flexible teacher choice"
  ],
  "cons": [
   "Needs teacher model",
   "Bounded by teacher quality"
  ],
  "params": {}
 },
 {
  "id": "margin-mse",
  "name": "MarginMSELoss",
  "alias": "MarginMSE",
  "cat": "distillation",
  "desc": "Distillation loss that preserves relative score margins from a teacher (e.g., cross-encoder). Learns that the score gap between pairs should match the teacher's gaps.",
  "data": "(query, doc_pos, doc_neg) + teacher scores",
  "labels": "Teacher similarity scores",
  "best": [
   "Distilling cross-encoder rerankers into bi-encoders"
  ],
  "pros": [
   "Preserves ranking margins",
   "Strong for search"
  ],
  "cons": [
   "Complex data pipeline",
   "Needs teacher scores"
  ],
  "params": {
   "similarity_fct": "pairwise_dot_score"
  }
 },
 {
  "id": "distill-kl",
  "name": "DistillKLDivLoss",
  "alias": "KLDistill",
  "cat": "distillation",
  "desc": "KL-divergence distillation: matches the student's score distribution to the teacher's. More nuanced than MSE as it captures the full distribution shape.",
  "data": "(query, positive, negative) + teacher scores",
  "labels": "Teacher score distributions",
  "best": [
   "High-fidelity distillation"
  ],
  "pros": [
   "Captures distribution shape",
   "Better than MSE for ranking"
  ],
  "cons": [
   "Complex setup",
   "Needs teacher scores"
  ],
  "params": {}
 },
 {
  "id": "embed-distill",
  "name": "EmbedDistillLoss",
  "alias": "EmbedDistill",
  "cat": "distillation",
  "desc": "Distills knowledge from teacher embeddings to student model with optional projection. Can distill across different embedding dimensions.",
  "data": "Sentences + teacher embeddings",
  "labels": "Teacher embeddings",
  "best": [
   "Cross-dimensional distillation"
  ],
  "pros": [
   "Works across dimensions",
   "Simple objective"
  ],
  "cons": [
   "Needs pre-computed teacher embeddings"
  ],
  "params": {}
 },
 {
  "id": "matryoshka",
  "name": "MatryoshkaLoss",
  "alias": "Matryoshka",
  "cat": "modifier",
  "desc": "Not a standalone loss -- wraps any other loss to train embeddings that work at multiple dimensions. Truncate 384d to 64d and keep ~97% quality. Massive storage/speed savings.",
  "data": "Same as the wrapped loss",
  "labels": "Same as the wrapped loss",
  "best": [
   "Production deployment",
   "Storage/speed optimization"
  ],
  "pros": [
   "Near-free compression (3-6x)",
   "Drop-in wrapper",
   "No quality cliff"
  ],
  "cons": [
   "Slightly longer training",
   "Must choose dimension list"
  ],
  "params": {
   "matryoshka_dims": "[256, 128, 64, 32]",
   "matryoshka_weights": "[1, 1, 1, 1]"
  }
 },
 {
  "id": "matryoshka-2d",
  "name": "Matryoshka2dLoss",
  "alias": "Matryoshka2D",
  "cat": "modifier",
  "desc": "Two-dimensional Matryoshka: compresses both embedding dimensions and model layers. Enables early exit from transformer layers for even faster inference.",
  "data": "Same as the wrapped loss",
  "labels": "Same as the wrapped loss",
  "best": [
   "Maximum inference speed",
   "Edge deployment"
  ],
  "pros": [
   "Layer + dimension compression",
   "Flexible speed/quality trade-off"
  ],
  "cons": [
   "Complex training",
   "Needs layer-aware model"
  ],
  "params": {
   "matryoshka_dims": "[256, 128, 64]",
   "n_layers_per_step": "-1"
  }
 },
 {
  "id": "adaptive-layer",
  "name": "AdaptiveLayerLoss",
  "alias": "AdaptiveLayer",
  "cat": "modifier",
  "desc": "Trains adaptive projection layers on top of each transformer layer, so you can get usable embeddings from any intermediate layer for faster inference.",
  "data": "Same as the wrapped loss",
  "labels": "Same as the wrapped loss",
  "best": [
   "Variable-speed inference",
   "Layer-wise early exit"
  ],
  "pros": [
   "Trade speed for quality at inference",
   "Works with any base loss"
  ],
  "cons": [
   "More parameters to train",
   "Moderate quality drop at shallow layers"
  ],
  "params": {
   "n_layers_per_step": "-1"
  }
 },
 {
  "id": "gist",
  "name": "GISTEmbedLoss",
  "alias": "GIST",
  "cat": "ranking",
  "desc": "Guided In-Sample Triplets: uses a guide model to identify true negatives among in-batch samples, solving MNRL's false-negative problem.",
  "data": "(anchor, positive) pairs",
  "labels": "None (guide model provides signal)",
  "best": [
   "High-precision retrieval",
   "When false negatives hurt"
  ],
  "pros": [
   "Eliminates false negatives",
   "Better than vanilla MNRL"
  ],
  "cons": [
   "Needs guide model (slow)",
   "Higher GPU memory"
  ],
  "params": {
   "guide": "SentenceTransformer('model')"
  }
 },
 {
  "id": "cached-gist",
  "name": "CachedGISTEmbedLoss",
  "alias": "CachedGIST",
  "cat": "ranking",
  "desc": "Memory-efficient variant of GIST that caches guide model embeddings. Combines the false-negative filtering with gradient caching.",
  "data": "(anchor, positive) pairs",
  "labels": "None",
  "best": [
   "Large-scale guided training"
  ],
  "pros": [
   "GIST quality at scale",
   "Memory efficient"
  ],
  "cons": [
   "Complex pipeline",
   "Two models in memory"
  ],
  "params": {
   "mini_batch_size": "32"
  }
 },
 {
  "id": "dae",
  "name": "DenoisingAutoEncoderLoss",
  "alias": "DAE",
  "cat": "unsupervised",
  "desc": "Unsupervised: corrupts input sentences (deletion, swapping, insertion) and trains the model to reconstruct the original. Great for domain-adaptive pretraining.",
  "data": "Single sentences (or damaged, original pairs)",
  "labels": "None",
  "best": [
   "Domain adaptation",
   "Pretraining on unlabeled corpus"
  ],
  "pros": [
   "No labels at all",
   "Good for domain shift"
  ],
  "cons": [
   "Weak signal per example",
   "Needs large corpus"
  ],
  "params": {
   "decoder_name_or_path": "model name",
   "noise_fn": "default (delete/swap/insert)"
  }
 },
 {
  "id": "mega-batch",
  "name": "MegaBatchMarginLoss",
  "alias": "MegaBatch",
  "cat": "ranking",
  "desc": "Margin-based ranking loss optimized for very large batches. Uses a margin between positive and most-similar negative within a mega-batch for hard negative mining.",
  "data": "(anchor, positive) pairs",
  "labels": "None",
  "best": [
   "Large batch training",
   "Margin-based ranking"
  ],
  "pros": [
   "Effective hard negative mining",
   "Works at scale"
  ],
  "cons": [
   "Needs large batches",
   "More memory"
  ],
  "params": {
   "positive_margin": "0.8",
   "negative_margin": "0.3"
  }
 },
 {
  "id": "gor",
  "name": "GlobalOrthogonalRegularizationLoss",
  "alias": "GOR",
  "cat": "modifier",
  "desc": "Regularization loss that encourages orthogonal embedding dimensions, reducing redundancy. Used as an auxiliary loss alongside a primary training objective.",
  "data": "Any (regularizes embeddings)",
  "labels": "None",
  "best": [
   "Reducing dimension redundancy",
   "Improving downstream compression"
  ],
  "pros": [
   "Reduces dimension correlation",
   "Pairs with any loss"
  ],
  "cons": [
   "Auxiliary only",
   "May hurt if over-weighted"
  ],
  "params": {
   "weight": "balance with primary loss"
  }
 }
];
