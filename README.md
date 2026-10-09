partial re-test, negative: on public METABRIC (n=1,979) an AIME-style confounder-conditioned multi-omics autoencoder + random forest predicts breast cancer recurrence only weakly (pooled AUROC 0.592, CI 0.566-0.616), below the preregistered 0.70 gate and no better than clinical-only (0.616) or raw-feature (0.611) baselines; the 25-gene list is neither specific nor stable; treatment alone is as predictive as the embedding; follow-up length alone gives 0.75.

Parent CBIO058 "Deep-Learning to Predict Breast Cancer Recurrence" (2023). Own AIME-style reimplementation (not the authors' code), different cohort from the student's. 10 directions (D1-D10): PREREG.md + src/run058.py committed together (e0be7d3); in the commit chain they precede the results (dates are author-set). Results: RESULTS.md. Raw: results/.
Reproduce: download https://datahub.assets.cbioportal.org/brca_metabric.tar.gz (sha256 in data_manifest.txt) and untar into data/, then python src/run058.py (python 3.10, torch CPU 2.14, numpy 1.26, scikit-learn 1.7.2, pandas 2.3; about 13 minutes on 2 CPU, under 2 GB RAM).
Constitution: real public data, CPU only, no stubs.


Corrections after gate review: see RESULTS.md "Corrections" (D9a permutation check was invalid as run; one corrected permutation gives 0.4955).