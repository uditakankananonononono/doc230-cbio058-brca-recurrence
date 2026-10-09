# CBIO058 results (PREREG.md and src/run058.py committed together at e0be7d3; in the commit chain they precede this results commit, dates are author-set). Raw: results/results058.json, results/run058.log.
Cohort: METABRIC via cBioPortal datahub, 1979 patients with expression + copy number + ER status + recurrence status; 803 recurred. Model = my AIME-STYLE reimplementation from the abstract's description (confounder-conditioned cross-modal autoencoder + RF), not the authors' AIME; not the student's data.

## Verdict (D2/D3 gates)
Did NOT reach the preregistered bar. D2 pooled out-of-fold AUROC 0.592 (bootstrap 95% CI 0.566-0.616; fold sd 0.015), gate >= 0.70 FAILED. The embedding does not beat simple baselines: RF on raw top-variance expression+CNA features 0.611, RF on clinical-only variables 0.616; AIME-style minus best baseline = -0.024 (gate "adds value" FAILED). Recurrence is only weakly predictable from these data in this setup.

## Other directions
- D1: 1505 ER+ (recurrence rate 0.395) vs 474 ER- (0.439); non-recurred median follow-up 141 months, 144 non-recurred with < 60 months; cohorts 1-5 only in this file.
- D4 ER: ER alone gives AUROC 0.505 (gate 0.55 not met: ER is not a material confounder here). Without the confounder input AUROC 0.600 vs 0.592 with it (|diff| < 0.02: adjustment did not matter). Within ER+ 0.606, within ER- 0.606.
- D5 label validity: dropping non-recurred with < 60 months follow-up (n=1835): AIME-style 0.597 (change 0.005), clinical 0.628; not censoring-sensitive by the gate. Follow-up length alone (a post-outcome variable, never a feature) gives AUROC 0.747, far above any omics model: recurrence labels are strongly tied to observation time, so the labels are a coarse proxy (not a clean "disease-free" definition).
- D6 treatment: chemotherapy/hormone/radio/surgery alone give AUROC 0.571 (gate 0.55 met: treatment is a material confound, i.e. as predictive as the embedding). Within hormone-therapy YES 0.593, NO 0.621.
- D7 leakage: fitting variance filter and autoencoder on all samples gives 0.586 (-0.006 vs 0.592); no leakage effect.
- D8 cohort: cohort identity predictable from the embedding at 0.468 vs majority 0.385; leave-one-cohort-out recurrence AUROC by held-out cohort 1: 0.612, 2: 0.574, 3: 0.580, 4: 0.663, 5: 0.570, mean 0.600; not cohort-dependent by the gate (it is as weak as the pooled result).
- D9 null/biomarkers: full-pipeline permutation AUROC 0.501 (5 perms; sane). "Top-25 genes" (my definition in the prereg): held-out AUROC 0.568 vs 0.546 for 25 random genes from the same 2,000 (diff 0.022 < 0.05: NOT specific); mean pairwise Jaccard across folds 0.008 (gate 0.30: NOT stable). No support for a 25-gene biomarker list in this cohort and no external validation was attempted.
- D10 modality ablation (descriptive): expression-only 0.606, CNA-only 0.588, both 0.592.

## Net claim
In METABRIC (n=1,979), an AIME-style confounder-conditioned multi-omics embedding with a random forest predicts recurrence status only weakly (AUROC about 0.59), no better than clinical-only or raw-feature random forests, and the 25-gene list is neither specific nor stable. This does not show the student's model fails on their data; it is a re-test of the idea on a different public cohort with my own implementation. Recurrence status here is also tied to follow-up length (AUROC 0.75 from follow-up alone) and treatment (0.57), so even the weak signal cannot be attributed to tumor biology without more careful time-to-event modelling (not done).

## Limits
One cohort; binary label ignores time-to-event and censoring; hyperparameters fixed, no tuning (the autoencoder could be improved; not tried); 60-epoch AE, latent 32, RF 300 trees, single CV seed; bootstrap CI treats out-of-fold predictions as independent; the student's data and code unavailable. Disclosures: a crash-test run on permuted labels (400 patients, 1 epoch) preceded the prereg commit (in PREREG); the full run completed in one pass.
