# Training, development and evaluation disclosure

We generate antimicrobial peptide candidates using the supplied AMP-Diffusion checkpoint with a deterministic 250-step DDIM sampler. A fixed seed produces 120,000 raw candidates. Canonical, unique sequences without exact matches to the challenge reference form the candidate pool. A 50,000-member library preserves empirical-tempered length allocation and matches embedding-cluster proportions to the training reference using deterministic quality ordering. The Top-100 ranker averages candidate-pool percentiles from species-equal APEX activity votes and five supervised MIC predictors using physicochemical features or ESM2 representations. Selection applies the documented reference, within-library diversity and physicochemical constraints without manual sequence curation. Model selection uses computational public-development comparisons across activity, weak-species performance, developability, distribution and diversity. This approach improves the combined registered scenario ranking, with explicit losses in APEX and HC50 proxies relative to the historical control. Independent external MIC validation is unavailable; generated peptides have not been experimentally validated. Two prior local independent default-seed generations produced byte-identical library, Top-100 and ranking files, and both passed local and vendored official validators. Effective source, model and configuration hashes accompany those records; the additional isolated-clone two-run validation passed, and fully distributed final-revision verification remains pending.

The five deployed supervised refits share 5,036 exact BATTLE-AMP observations from public
revision `72e26f705badbd83ddcd347ba96cff16ae4278f6`; their training matrix hash is
`5fd5b9d7680ececc7bf82849c3e463c424d3a84cff4ac40b1fb31ad358018258`.
Public training, development and evaluation partitions were reused for development and model
selection. QMAP consensus targets are development/evaluation material, not raw MIC labels or
an independent final holdout. Original-paper additions did not enter these deployed refits.
The record-level chemistry is not completely resolved; sequence-only representations do not
establish that every training measurement concerns a linear, free-terminal molecule.
The subsequent corrected MIC/HC50 diagnostic fits did not replace the deployed weights.
The generated sequences follow the challenge's linear/free-terminal contract, but their
predicted activity cannot be certified for that chemistry from unresolved training records.

The supplied APEX weights have recorded public MIT provenance; their complete checkpoint-linked
training list and overlap remain unresolved. These limits are disclosed alongside the official
starter provenance; no additional permission request is required merely to use the supplied
model as intended. This does not claim complete overlap knowledge or organizer adjudication. No independent external MIC validation or successful nested joint
MIC/HC50 validation is claimed. The reviewed joint cohort has only four molecules, eight endpoint
pairs and one paper/homology component. HC50 proxies and all generated sequences lack new
experimental validation. In the registered comparison, the adopted policy worsens APEX activity,
weak-species activity, HC50 proxy and Top cluster count relative to the historical control while
improving distribution/developability and combined scenario rank.
All final sequences and their order are produced automatically; no manual curation, allowlist,
post-generation sequence edits or hemolysis oracle enters the deployed selector.
