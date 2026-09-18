# Data and asset provenance

| Asset | Source | Use and terms |
| --- | --- | --- |
| Generator and training reference | [Official starter kit](https://github.com/szczurek-lab/ampdiffusion-starter-kit/tree/1a862af9078e6b55c87d1fa576f3da81851ba94b) | Supplied model and exact reference FASTA; upstream MIT notice retained. Training-source databases: DRAMP 3.0, APD3 and DBAASP. |
| Challenge reference | [Official challenge](https://github.com/szczurek-lab/amp-challenge-2027) | Exact reference for duplicate and similarity compliance; BSD-3-Clause notice retained. |
| APEX | [APEX-pathogen](https://gitlab.com/machine-biology-group-public/apex-pathogen/-/tree/417a4441a1e6ef8b10d2352e1c059622d5259f3a) | Eight unchanged pretrained models; MIT notice in apex/LICENSE. Complete checkpoint-specific training disclosure not established. |
| ESM2 | [FAIR ESM](https://github.com/facebookresearch/esm) | MIT; ESM2-8M and 650M, UR50/D 2021_04 pretraining. Fixed URLs/hashes in configs/inference_encoders.json. |
| Five MIC predictors | Public BATTLE-AMP-derived training | Source revision, observation identifiers, parameters and hashes in training_disclosure.json and training_observation_ids.txt. |

The two reference FASTAs are byte-identical to the supplied inputs and required for length
allocation, library matching and similarity checks. Their presence in the official starter kit
is provenance and intended-use evidence, not a blanket license for the underlying databases.
This release does not add raw BATTLE-AMP, QMAP, APD or DBAASP measurement tables.
Dataset/weight provenance and code licenses are separate from Full eligibility.

The APEX paper links study supplementary data, which we have not established as a full training
release for these weights: https://www.nature.com/articles/s41564-025-02061-0 and
https://data.mendeley.com/datasets/d8yzgtdrcp/3. The known information and remaining limits are disclosed. Use of the official supplied model
does not require a new permission request in this workflow; final eligibility is not self-certified.

Public training/development/evaluation reuse and unresolved chemical forms are disclosed in
disclosure.md. QMAP targets were used in development, not as independent raw-MIC holdout labels.
The deployed bundles precede subsequent chemistry-corrected diagnostic refits.
Observation identifiers and source hashes support audit without mirroring raw tables.
Local research paths in training_disclosure.json identify historical provenance only; inference
never reads those paths. source_provenance.json lists unchanged source/model files.
