# Runway Search — Evaluation Protocol (PRE-REGISTERED)

**Status: written and committed before any image was embedded, any index was
built, or any retrieval output was observed.** The commit that adds this file
contains no embedding, indexing, or retrieval code.

This document specifies, in advance:
1. What the system is asked to do (§1)
2. What corpus it searches (§2)
3. How relevance is defined and who judges it (§3)
4. The exact metrics and the headline number (§4)
5. The system configuration being evaluated, and its baseline (§5)
6. The 20 test queries and their per-query relevance criteria (§6)
7. Failure modes anticipated *before* seeing results (§7)
8. The rules for amending this document (§8)

Results, error analysis, and interpretation can be found in `README.md`.

---

## 1. The task

Given a natural-language query string, return a ranked list of images from the
indexed corpus, optionally narrowed by a metadata filter on garment category or
attribute. The retrieval unit is a whole image, not a garment crop
(see §5 for why, and §7.1 for the failure mode this choice is expected to
cause).

An image is a candidate match if it *contains* a garment satisfying the query.
A query for "a cardigan" is satisfied by a full-body outfit photo in which a
cardigan appears; it does not require the cardigan to be the only or the
dominant garment. This is a deliberate, pre-registered choice, and §7.1 records
the specific way it is expected to hurt.

## 2. Corpus

**Fashionpedia** ([Jia et al., ECCV 2020](https://fashionpedia.github.io/home/)).
No scraped commercial fashion photography is used anywhere in this project. The
README's "Data and licensing" section covers why, including the parts of
Fashionpedia's terms that are stricter than they look at first.

- **Indexed set:** every image in `instances_attributes_train2020.json`
  (n = 45,623 in the file) that carries at least one instance annotation in one
  of the 27 *main apparel* categories. Images annotated only with apparel
  *parts*, like a lone `sleeve` mask, are left out, since a query like "a
  cardigan" has no sensible answer in an image like that. Applying that rule
  leaves **n = 45,622** indexed images. Exactly one image is dropped (id 1870,
  annotated only with `neckline`, `sleeve`, `sleeve`).
- **Held out and not indexed:** `val2020` (n = 1,158). Nothing is trained here,
  CLIP is used zero-shot, so this is not a leakage control. I'm keeping it as a
  clean set for a later crop-level or fine-tuned variant, so that variant can be
  compared without reusing a corpus I have already looked through.
- **Not used:** `test2020`. Its attribute annotations aren't public, so metadata
  ground truth (§3.1) can't be built for it.

**Attribute vocabulary, resolved 2026-09-20.** The §6 predicates were originally
written against the documented ontology, with the attribute names marked
`[VERIFY]` because they're only visible inside the annotation JSON. They have now
been checked against the real label set in
`instances_attributes_val2020.json`, before any embedding or retrieval, under §8
rule 1. What I found:

- All 46 categories exist as documented, and the 27 main apparel / 19 parts split
  holds. Every category name used in §6 is confirmed, including
  `shirt, blouse` (0), `top, t-shirt, sweatshirt` (1), `jumpsuit` (11) and
  `shoe` (23).
- The 294 attributes sit in **11** super-categories, not 9 as the paper
  describes: nickname (153), silhouette (25), neckline type (25), textile
  finishing (21), textile pattern (18), length (15), opening type (10),
  non-textile material type (10), waistline (7), animal (6), leather (4).
- **No color attribute exists anywhere in the 294.** That confirms the Tier D
  design (§3.3) and C2's known gap rather than changing either.
- Three predicates needed real changes, logged in §9: B4 (no `leather` label
  exists at all), C1 (no skinny silhouette and no shoe nicknames), and B1 (the
  right label sits under a different super-category than I assumed).

Attributes were read from the val file because it loads in seconds. The train
file should carry the identical ontology, and that gets confirmed in
`pipeline/02_build_dataset.py` rather than assumed.

## 3. Relevance

### 3.1 Primary: metadata-defined relevance (objective, computable)

For each Tier A/B/C query, §6 specifies a **predicate** over Fashionpedia's
category and attribute annotations ahead of time. The relevant set `R(q)` is
every indexed image containing at least one garment instance that satisfies that
predicate.

This is objective, reproducible, and fixed before I see any output. Its weakness
is worth stating up front rather than burying it: it measures how well the
retrieval model *agrees with Fashionpedia's labels*, which is a proxy for visual
relevance and not the thing itself. Where Fashionpedia's attribute labels are
sparse or missing, a retrieval that is actually correct gets scored as a miss.
That is what §3.2 is for.

### 3.2 Secondary: human judgment (catches label sparsity)

For every query, I judge the top 10 returned images by hand against the written
prose relevance description in §6. That produces **manual precision@10**.

Judging procedure, fixed in advance:
- The 10 images are shuffled into random order and judged **without rank
  information visible**, so I can't unconsciously grade the top result more
  generously.
- Each image is scored **2 = clearly relevant, 1 = arguably relevant,
  0 = not relevant**, against the prose description only.
- Precision@10 is computed twice, **strict** (only 2s count) and **lenient**
  (2s and 1s count). Both get reported. Reporting only the flattering one is the
  failure this rule is here to stop.
- A judgment log with query, image id, score, and a one-line reason is written to
  `docs/relevance_judgments.csv`, so the grading can be audited instead of taken
  on my word.

**Single-judge limitation.** I make all the judgments
myself and know what the system is supposed to do. There is no second
annotator, so there is no inter-rater agreement statistic. It is important to
note that this is a real weakness of the secondary metric, and it carries into
the README's limitations section. The
mitigations are the ones a single judge has available: the criteria were written
before any output existed, the rubric is fixed, and the judging is blindly ranked.

### 3.3 Tier D queries have no metadata ground truth

Tier D (§6) covers color, style, and occasion, and Fashionpedia labels none of
them. `R(q)` is undefined for these, so they are evaluated by **manual
precision@10 only** and are **excluded from every aggregate recall number**.
Mixing an ungradeable query into a headline average would leave the average
meaning nothing. They still get run and reported one by one, and they
are the main evidence behind the "no style classification in v1" scope decision.

## 4. Metrics

Let `R(q)` be the metadata-defined relevant set and `topk(q)` the first k
results.

**Primary headline metric, capped recall@10:**

```
recall@k(q) = |R(q) ∩ topk(q)| / min(|R(q)|, k)
```

The `min(|R|, k)` denominator is a choice that has to be understood to read
these numbers honestly. Plain recall@k is `|R ∩ topk| / |R|`, which can never
exceed `k/|R|`. For a query like "a cardigan", `|R|` is plausibly in the
thousands, so plain recall@10 can't get above roughly 0.005 even if the system
is perfect. At that point the number measures corpus size instead of retrieval
quality. The capped version equals ordinary precision@k whenever `|R| ≥ k`, and
ordinary recall@k whenever `|R| < k`, so it stays readable in both cases.

To keep transparency:
- **`|R(q)|` is reported for every query**, so it is visible which case each
  query falls into.
- **Plain, uncapped recall@k is reported next to it** for every query. The capped
  version is the headline, and the uncapped version goes in the same table
  rather than getting left out.

**Reported at k ∈ {1, 5, 10, 20}.** Headline = capped recall@10, averaged over
Tier A/B/C queries (n = 16).

**Secondary metrics:**
- Manual precision@10, strict and lenient (§3.2), across all 20 queries.
- **Rank of first relevant result** by the §3.1 predicate, plus MRR over
  Tier A/B/C. It's cheap to compute, and it separates "found nothing" from
  "found it at rank 7."
- Per-tier breakdown. One grand average across easy and hard queries hides the
  exact structure the error analysis is about.

**Pre-registered adequacy thresholds.** Committed now so they can't be moved
afterward:
- Tier A (single category): **capped recall@10 ≥ 0.80**. These are the floor.
  Falling below it means something is wrong with the pipeline, not with the hard
  queries.
- Tier B (category + one attribute): **≥ 0.50**.
- Tier C (compositional): **≥ 0.30**.
- Any individual query with capped recall@10 **below 0.30 is designated a
  failure case** and goes into the error analysis, with its retrieved images
  inspected and failure mode categorized (§7). This is a commitment to *look
  at* bad results, not a bar the project needs to clear.

**The metric set is frozen here.** Metrics will not be added after results are shown. If some other metric turns out to be genuinely informative, it gets reported in a
clearly marked post-hoc section, kept separate from these numbers.

## 5. System under evaluation

Fixed in advance so configuration choices aren't tuned against the eval set.

- **Primary model:** FashionCLIP (`patrickjohncyh/fashion-clip`), a CLIP
  checkpoint adapted to fashion imagery.
- **Baseline comparator:** OpenAI CLIP ViT-B/32 (`openai/clip-vit-base-patch32`),
  zero-shot. Both models load through Hugging Face `transformers` and run through
  the identical loading, preprocessing and embedding code, so any difference in
  results comes from the weights rather than from two libraries preparing images
  differently. Both run against the identical corpus, queries, and predicates. Whether the
  fashion-specific model actually beats general CLIP is a real finding either
  way, and committing to report it now keeps the comparison from quietly
  disappearing if the domain-specific model loses.
- **Index:** FAISS `IndexFlatIP` over L2-normalized embeddings, which is exact
  inner product, so exact cosine similarity. At roughly 46k vectors, an
  approximate index (IVF/HNSW) buys speed I don't need and adds approximation
  error that would get confounded with model quality. Exact search keeps the
  eval measuring the embeddings.
- **Text prompt template:** the raw query string, with **no** `"a photo of ..."`
  wrapper. Chosen in advance and applied the same way everywhere. Trying both
  and keeping whichever scored better would be tuning on the eval set. If I test
  the alternative later, it gets reported as a clearly labeled post-hoc
  comparison.
- **Metadata filtering** is evaluated as a *separate* condition instead of being
  folded into the headline. Each Tier B query also runs with a hard pre-filter
  on its category, and the filtered versus unfiltered difference is reported.
  Reporting only the filtered numbers would overstate what the embeddings are
  doing on their own.
- **Determinism:** fixed random seed, recorded checkpoint hashes, one fixed
  corpus snapshot.

## 6. Test queries

20 queries across four difficulty tiers. The tiers exist so one average can't
hide the difference between what works and what doesn't, and Tier D is in the
set, specifically, because it is expected to fail. A query set containing only
queries the system handles well is a demo, not an evaluation.

The predicates below were written against the documented Fashionpedia ontology
before the annotation JSON was opened, and every one was marked `[VERIFY]` at
that point. They were resolved against the real label set on 2026-09-20, before
any embedding or retrieval, and the corrections are logged in §9. Category and
attribute IDs shown in parentheses are the verified ones.

---

### Tier A: single garment category (n = 5)
*Directly in the 27-category ontology. These establish the floor. If they fail,
nothing further down is interpretable.*

**A1. `cardigan`**
- **Relevant:** any image showing a cardigan, meaning an open-front knitted
  garment that fastens down the front. Buttoned or open, worn or flat-lay, any
  color.
- **Predicate:** `category == "cardigan"` (3)
- **Fails if:** capped recall@10 < 0.30, meaning fewer than 3 of the top 10
  contain a cardigan by Fashionpedia's own labeling.

**A2. `jumpsuit`**
- **Relevant:** a one-piece garment joining a top and full-length trousers.
  Rompers and playsuits (short-legged) count as *arguably* relevant (score 1)
  under §3.2, since the ontology doesn't separate them.
- **Predicate:** `category == "jumpsuit"` (11)

**A3. `shorts`**
- **Relevant:** any image showing shorts, meaning a bifurcated leg garment
  ending above the knee. Distinct from a skirt (not bifurcated) and from cropped
  trousers (below the knee). Those score 0.
- **Predicate:** `category == "shorts"` (7)

**A4. `scarf`**
- **Relevant:** a scarf worn around the neck, shoulders or head, or shown on its
  own. Distinct from `headband, head covering, hair accessory`. A headscarf is
  genuinely ambiguous between the two and scores 1, not 2.
- **Predicate:** `category == "scarf"` (25)

**A5. `umbrella`**
- **Relevant:** an umbrella, open or closed, held or otherwise visible.
- **Predicate:** `category == "umbrella"` (26)
- **Note, recorded in advance:** this is the rarest Tier A category and should
  have a small `|R|`, which makes it the one Tier A query where plain and capped
  recall may come apart noticeably. I included it on purpose as a low-frequency
  probe.

---

### Tier B: category + one attribute (n = 7)
*Tests whether attribute-level language reaches the embedding, and, by comparing
§3.1 against §3.2, how much Fashionpedia's attribute sparsity distorts the
score.*

**B1. `sleeveless dress`**
- **Relevant:** a dress with no sleeves, so shoulders or arms are bare.
  Strapless, spaghetti-strap, tank and halter all count. Short sleeves score 0,
  and a sleeveless *top* that isn't a dress scores 0.
- **Predicate:** image contains an instance with `category == "dress"` (10),
  AND an instance with `category == "sleeve"` (31) carrying length attribute
  `156:sleeveless`.
- **Note, recorded in advance:** `sleeveless` sits under the *length*
  super-category in Fashionpedia, not under any sleeve-type group. Same target,
  and the `[VERIFY]` wording I originally wrote for this predicate named the
  wrong group.
- **Known gap (amended 2026-10-04, see §9):** Fashionpedia attaches `156` to the
  `sleeve` *part* instance, not to the garment, so the predicate is image-level
  like Tier C. It can't confirm that the sleeveless sleeve belongs to the dress
  rather than to another garment in the same photo. Manual precision@10 (§3.2)
  grades that.

**B2. `striped shirt`**
- **Relevant:** a shirt or blouse with a stripe pattern, in any stripe
  direction, width or color. A striped *dress* or *sweater* scores 0 (wrong
  category), and check, gingham, plaid or other patterns score 0.
- **Predicate:** `category == "shirt, blouse"` (0) AND textile-pattern attribute
  `328:stripe`. The label is "stripe", not "striped". `331:chevron`,
  `330:herringbone (pattern)` and `332:argyle` are excluded, matching the prose.

**B3. `high waisted pants`**
- **Relevant:** trousers whose waistband sits at or above the natural waist.
- **Predicate:** `category == "pants"` (6) AND waistline attribute
  `141:high waist`.
- **Note, recorded in advance:** waistline is a fine visual distinction, and it's
  often hidden by a tucked or untucked top. I expect this to be one of the
  weaker Tier B queries, and to get hit hard by label sparsity.

**B4. `fur coat`**
- **Relevant:** a coat whose outer surface is fur, or reads as fur in the photo,
  meaning visible pile or hair rather than a woven or knitted face. Full-fur
  coats and coats with fur across most of the body both count. A cloth coat with
  fur only on the collar, cuffs or hood scores 1, since the trim is fur but the
  coat isn't. The judgment is made from a photograph, so faux fur counts as fully
  relevant (score 2). I can't tell real from fake by eye, and pretending I can
  would make the rubric unusable. Shearling, where the pile faces in and the
  leather faces out, scores 1 rather than 2, since it reads as fur only at the
  edges.
- **Predicate:** `category == "coat"` (9) AND non-textile-material attribute
  `289:fur`.
- **Note, recorded in advance:** this query replaced `leather jacket` during
  `[VERIFY]` resolution. Fashionpedia has no `leather` attribute at all, and its
  "leather" super-category holds only `suede`, `shearling`, `crocodile` and
  `snakeskin`, so a plain leather jacket carries none of them and the predicate
  couldn't be built. `fur` tests the same thing, whether material language
  reaches the embedding, with a label that exists.

**B5. `floral print dress`**
- **Relevant:** a dress with a flower-motif print. Note the collision hazard,
  flagged in advance: `flower` is also one of the 19 *apparel parts*, meaning a
  3D applied flower ornament. A dress with an applied fabric flower but no
  floral print scores 0 under the prose criterion, and the predicate has to
  target the textile *pattern* attribute instead of the `flower` part category.
- **Predicate:** `category == "dress"` (10) AND textile-pattern attribute
  `325:floral`. Explicitly **not** the `flower` decoration category (39), which
  is confirmed to exist and is the collision named above. `340:plant` is a
  separate pattern label and is excluded.

**B6. `distressed denim jeans`**
- **Relevant:** denim trousers with visible distressing, meaning rips, tears,
  fraying, abrasion or heavy fading. Clean, undamaged jeans score 0.
- **Predicate:** `category == "pants"` (6) AND nickname `36:jeans` AND
  textile-finishing attribute in {`297:distressed`, `300:frayed`, `298:washed`}.
- **Note, recorded in advance:** Fashionpedia has no denim material attribute, so
  denim comes from the nickname `36:jeans` instead. I'm counting `frayed` and
  `washed` alongside `distressed` because my prose criterion already names
  fraying and heavy fading, so limiting the predicate to `distressed` alone would
  make it stricter than the criterion it's supposed to formalize. This predicate
  is still a three-way conjunction, so I expect a small `|R|` driven as much by
  how completely things were annotated as by how many such garments are actually
  in the corpus.

**B7. `baggy jeans`**
- **Relevant:** denim trousers cut deliberately loose and wide through the leg,
  with volume from the hip down and no close contact with the calf.
  Straight-leg jeans that merely aren't tight score 0. The cut has to read as
  intentionally oversized.
- **Predicate:** `category == "pants"` (6) AND nickname `36:jeans` AND silhouette
  attribute in {`131:baggy`, `132:wide leg`, `137:loose (fit)`, `138:oversized`}.
- **Note, recorded in advance:** this is the opposite pole of C1's `skinny
  jeans`, and I included the two as a pair on purpose. If both score well, the
  embedding is resolving fit language. If both retrieve the same images, it is
  matching "jeans" and throwing the fit term away, which is a cleaner diagnostic
  than either query gives on its own. It shares B6's three-way conjunction
  sparsity risk.

---

### Tier C: compositional / multi-garment (n = 4)
*CLIP-family models are known to handle composition and binding poorly. I expect
this to be the hardest gradeable tier. The pre-registered threshold (0.30)
reflects that, and it's a prediction to be checked rather than a bar picked
because it's easy to clear.*

**C1. `skinny jeans and boots`**
- **Relevant:** one image showing **both** close-fitting denim trousers **and**
  boots, on the same person. Only one of the two present scores 1 (arguably
  relevant), which is exactly the partial-match behavior this query is here to
  measure. Neither scores 0.
- **Predicate:** image contains an instance with `category == "pants"` (6) AND
  nickname `36:jeans` AND silhouette `135:tight (fit)`, AND an instance with
  `category == "shoe"` (23).
- **Known gap, stated in advance:** neither "skinny" nor "boots" is enforceable.
  Fashionpedia has no skinny or slim silhouette label, so `135:tight (fit)` is
  the closest one that exists, and there are no shoe nicknames at all in the 153
  nicknames, so boots can't be told apart from sneakers or heels. The predicate
  relaxes to "tight jeans plus any shoe," which most full-body outfit photos
  satisfy, so `|R|` will be large and metadata recall will overstate performance
  here in the same way it does on C2. Manual precision@10 (§3.2) is what actually
  grades "skinny" and "boots" on this query. I chose to relax rather than swap
  the query so the B7 baggy versus C1 skinny pairing stays intact, and the
  overstatement is measured instead of hidden.

**C2. `blazer over a white t-shirt`**
- **Relevant:** a blazer-style tailored jacket worn open over a visibly white
  t-shirt. Both are required for a score of 2, and one only scores 1.
- **Predicate:** instance with `category == "jacket"` (4) AND nickname
  `17:blazer`, AND instance with `category == "top, t-shirt, sweatshirt"` (1).
- **Known gap, stated in advance:** Fashionpedia doesn't label color, so the
  predicate **can't enforce "white"**. The metadata ground truth is therefore
  strictly weaker than the prose criterion, and metadata recall will overstate
  performance on this query while manual precision won't. That divergence is a
  pre-registered measurement, not something discovered later.

**C3. `long coat with a scarf`**
- **Relevant:** a coat reaching at least mid-calf, worn with a visible scarf.
- **Predicate:** instance with `category == "coat"` (9) AND a length attribute
  in {`152:below the knee (length)`, `154:maxi (length)`, `155:floor (length)`},
  AND an instance with `category == "scarf"` (25).

**C4. `oversized blazer`**
- **Relevant:** a blazer cut deliberately loose and large, with dropped
  shoulders, long sleeves and boxy volume. A well-fitted blazer scores 0,
  including one that only looks slightly large.
- **Predicate:** `category == "jacket"` (4) AND nickname `17:blazer` AND
  silhouette attribute in {`138:oversized`, `137:loose (fit)`}.
- **Note, recorded in advance:** "oversized" is a judgment about intended cut
  versus accidental fit, and I expect it to be both sparsely labeled and
  genuinely ambiguous to me as the judge. I'm flagging it now so that ambiguity
  doesn't get mistaken later for a model failure.

---

### Tier D: out-of-ontology (n = 4), EXPECTED TO FAIL
*Color, style and occasion. Fashionpedia labels none of these, so `R(q)` is
undefined and they're judged by manual precision@10 only (§3.3) and excluded
from all aggregate recall figures.*

*These queries are the pre-registered evidence base for the v1 scope decision to
leave style and genre classification out. The decision is recorded as a
prediction here, before the data is in. If Tier D performs poorly, the write-up
reports that as the measured reason style search is out of scope, instead of
asserting it as an untested assumption. If Tier D performs well, that gets
reported too, and the scope decision is re-examined in the write-up rather than
defended.*

**D1. `monochrome black outfit`**
- **Relevant:** an outfit where every visible garment is black or near-black.
  One clearly non-black garment scores 0.
- **Predicate:** none possible, since Fashionpedia's 294 attributes don't include
  color. Manual judgment only.

**D2. `business casual office outfit`**
- **Relevant:** an outfit plausibly appropriate for a business-casual workplace,
  such as tailored trousers or a knee-length skirt with a shirt, blouse, knit or
  blazer. Excludes both full formal suiting and obvious athleisure or
  streetwear.
- **Predicate:** none, since occasion is unlabeled. Manual judgment only.
- **Noted in advance:** the prose criterion itself is culturally loaded and my
  boundary as the judge is subjective. I'm recording that so it reads as a
  limitation of the evaluation and not only of the system.

**D3. `something you would wear to the beach`**
- **Relevant:** swimwear, cover-ups, light summer dresses, shorts with sandals,
  or a clear beach setting.
- **Predicate:** none. Manual judgment only.
- **Noted in advance:** this query also probes whether retrieval leans on the
  *scene* (sand, sea) instead of the *garment*, which is a different failure
  mode from the others (§7.4).

**D4. `goth outfit`**
- **Relevant:** an outfit legible as goth style, so predominantly black, with
  subculturally specific signals like heavy boots, leather, lace or mesh, silver
  hardware and dramatic layering. Plain all-black minimal dressing scores 1 at
  most.
- **Predicate:** none. This is exactly the style and genre labeling that
  Fashionpedia doesn't provide and that v1 doesn't attempt.
- **Noted in advance:** I expect this to be the weakest query in the set. Its job
  is to make the out-of-scope decision an empirical finding instead of an
  assertion.

---

## 7. Anticipated failure modes (written before any results)

Recorded now so the error analysis in §5 of the README can tell *predicted*
failures apart from *discovered* ones. Predicting a failure and then observing
it is a stronger result than explaining it after the fact, and a failure listed
here that never materializes is just as worth reporting.

**7.1 Whole-image embedding dilutes small garments.** Fashionpedia images are
mostly full-outfit photographs. A single CLIP embedding of an entire image is
dominated by whatever is visually largest. Queries for small or peripheral items
(A4 `scarf`, A5 `umbrella`, and the accessory half of C1 and C3) should suffer
most. The alternative is crop-level indexing, meaning embedding each garment
mask's bounding box instead of the whole image. **I'm naming that alternative
here, in advance, as a secondary condition**, so if I run it later it reads as a
planned comparison and not as a rescue attempt after the primary configuration
disappointed.

**7.2 Attribute label sparsity depresses metadata recall.** Fashionpedia's
attributes are long-tailed and aren't applied exhaustively to every instance. A
garment that gets retrieved correctly but whose relevant attribute simply wasn't
annotated counts as a miss under §3.1. The diagnostic is specified ahead of
time: **wherever manual precision@10 (§3.2) is substantially higher than capped
recall@10 (§3.1) on the same query, label sparsity is the leading explanation
rather than retrieval quality.** I expect that gap to be widest on B3 and B6.

**7.3 Confusion between visually similar categories.** Specific pairs I'm
predicting will be confused, named now instead of after looking at a confusion
matrix: `cardigan` vs `sweater` vs `jacket` (A1), `shorts` vs `skirt` (A3),
`jacket` vs `coat` (C4 vs C3), and `pants` vs `tights, stockings` (B3).

**7.4 Scene and context override garment.** CLIP embeddings encode setting, pose
and photographic style, not just clothing. Queries with strong scene
associations, D3 especially, may retrieve images that match the *context* while
missing the garment.

**7.5 Compositional queries collapse to one term.** For C1 through C3, I expect
retrieval to be dominated by whichever term is more visually salient or better
represented in the corpus, returning images that match one conjunct only. It
gets diagnosed through the score-1 (partial match) rate in §3.2.

**7.6 Corpus coverage gaps.** Some queries may have few true instances in
Fashionpedia no matter how good the model is. `|R(q)|` is reported per query
(§4) so a low score caused by an empty corpus is distinguishable from one caused
by bad retrieval. Those are different findings and can't be reported as the same
one.

## 8. Amendment rules

The value of this document is entirely in the fact that it was written first.
These rules exist so necessary corrections don't quietly turn into
results-driven edits.

1. **Predicate corrections (`[VERIFY]` items).** The §6 predicates were written
   against the documented ontology before the annotation JSON was opened.
   Attribute names and IDs may be corrected **only** to match what actually
   exists in `instances_attributes_train2020.json`, and **only before any
   retrieval is run**. Each correction gets logged in §9 with the date, the
   original text, the corrected text, and the reason.
2. **No post-hoc predicate edits.** Once any retrieval output has been observed,
   the predicates are frozen. A predicate that turns out to be a poor
   formalization of its prose criterion gets reported as a *limitation of the
   evaluation* in the README instead of being rewritten to improve the score.
3. **No query removal.** No query gets dropped from the set after results are
   seen, for any reason. A badly designed query gets reported as a badly
   designed query.
4. **Query additions** are allowed only as an explicitly labeled post-hoc set,
   reported separately, and never merged into the pre-registered averages.
5. **Thresholds (§4) and metrics (§4) are frozen** as of this commit.
6. If any rule here gets broken, the write-up says so directly, the same way
   Project Bioenergetics' H5 section documents its own post-hoc status and the
   methodological risk that carries.

## 9. Amendment log

Every entry has to record the date, what changed, and why.

| Date | Section | Change | Reason |
|---|---|---|---|
| 2026-09-20 | §6 B4 | Query changed from `leather jacket` to `fur coat`. Predicate is now `category == "coat"` (9) AND `289:fur`. | Fashionpedia has no `leather` attribute. Its "leather" super-category holds only suede, shearling, crocodile and snakeskin, so a plain leather jacket carries none of them and no predicate could be built. `fur` tests the same material-language question with a label that exists. Resolved before any embedding or retrieval. |
| 2026-09-20 | §6 C1 | Predicate relaxed to `pants` (6) AND `36:jeans` AND `135:tight (fit)`, AND `shoe` (23). Prose criterion unchanged. | No skinny or slim silhouette label exists, and there are no shoe nicknames in the 153, so "boots" can't be enforced. Relaxed rather than swapped so the B7/C1 baggy-versus-skinny pairing survives. The resulting overstatement is documented in the query's known-gap note and graded by manual precision. |
| 2026-09-20 | §6 B1 | Predicate now names length attribute `156:sleeveless` instead of a "sleeve-type attribute". | `sleeveless` sits under the length super-category. Same target, wrong group named in the original wording. |
| 2026-09-20 | §6 B6, B7 | Denim now comes from nickname `36:jeans`. B6 accepts `297:distressed`, `300:frayed`, `298:washed`. B7 accepts `131:baggy`, `132:wide leg`, `137:loose (fit)`, `138:oversized`. | No denim material attribute exists. The finishing set matches the prose criterion, which already names fraying and heavy fading, so `distressed` alone would have been stricter than the criterion it formalizes. |
| 2026-09-20 | §6 (all others) | `[VERIFY]` markers removed and replaced with confirmed category and attribute IDs. | Names matched the real label set with no change in meaning. |
| 2026-09-28 | §2 | Added the indexed-set count the rule actually produces, n = 45,622, and named the one dropped image. The "n = 45,623" is now labeled as the count in the file. | Clarification only. The inclusion rule is unchanged. The original text gave the file's image count next to the rule, which read as if every image were indexed. Recorded after `pipeline/02_build_dataset.py` ran, and before any embedding or retrieval. |
| 2026-09-28 | §5 | The baseline now loads through `transformers` (`openai/clip-vit-base-patch32`) instead of `open_clip`. Same model and weights. | FashionCLIP is only distributed in `transformers` format, with no `open_clip` config. Loading both models through one library means they share identical preprocessing and embedding code, so the comparison isolates the weights. Recorded before any embedding or retrieval. |
| 2026-10-04 | §6 B1 | Predicate changed from `dress` (10) AND `156:sleeveless` on the same instance to an image-level conjunction: a `dress` (10) instance AND a `sleeve` (31) instance carrying `156:sleeveless`. `\|R\|` goes from 1 to 112. Prose criterion unchanged. | Counting `\|R\|` for every predicate showed `156` sits on the `sleeve` part instance (298 instances) and almost never on the garment (1 dress instance), so the original predicate matched a single image and could not grade the query. This corrects which instance the label is attached to, under the spirit of rule 1. Stated plainly per rule 6: rule 1's "before any retrieval is run" is not literally met, since a smoke test had already ranked images for four non-eval queries (`glasses`, `hat`, `tie`, `glove`). No retrieval for any §6 query had been run or observed. The image-level gap is recorded in B1's known-gap note. |
