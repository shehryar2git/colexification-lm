# Appendix E · Everything marked PENDING USER RUN

Nothing in this report is estimated or imputed. Each row below is a result that the build environment could not produce, with the command that produces it.

| Item | What is missing | Command | Source |
| --- | --- | --- | --- |
| Model condition `bert_base` | embeddings and every statistic that depends on them (H1, H2, H3, robustness) | pip install -r requirements.txt; python -m src.embeddings.contextual | results/embedding_coverage.csv |
| Model condition `xlmr_base` | embeddings and every statistic that depends on them (H1, H2, H3, robustness) | pip install -r requirements.txt; python -m src.embeddings.contextual | results/embedding_coverage.csv |
| Model condition `labse` | embeddings and every statistic that depends on them (H1, H2, H3, robustness) | pip install -r requirements.txt; python -m src.embeddings.contextual | results/embedding_coverage.csv |
| Model condition `xlmr_translated` | embeddings and every statistic that depends on them (H1, H2, H3, robustness) | pip install -r requirements.txt; python -m src.embeddings.translated_input | results/embedding_coverage.csv |
| Model condition `labse_translated` | embeddings and every statistic that depends on them (H1, H2, H3, robustness) | pip install -r requirements.txt; python -m src.embeddings.translated_input | results/embedding_coverage.csv |
| Robustness check `h2_human_audited_only` | English gold standard restricted to rows a person has decided (D-55) | no human decisions in audit/english_colexification_manual_check.csv; fill the `decision` column, then: python -m src.analysis.robustness | results/robustness.csv |
| Robustness check `h3_gradient_wikipedia_log10_articles_full` | Family gradient: full AUC vs Wikipedia (log10 articles, largest edition) | PENDING USER RUN: `python run.py analyse` with internet access (query.wikidata.org, meta.wikimedia.org, *.wikipedia.org) | results/robustness.csv |
| Robustness check `h3_gradient_wikipedia_log10_articles_rarefied` | Family gradient: rarefied AUC vs Wikipedia (log10 articles, largest edition) | PENDING USER RUN: `python run.py analyse` with internet access (query.wikidata.org, meta.wikimedia.org, *.wikipedia.org) | results/robustness.csv |
| Robustness check `h1_template_set` | Template sensitivity: each of the 10 templates alone, and leave-one-template-out (D-13). Needs the contextual models. | python run.py embeddings  # then: python -m src.analysis.robustness | results/robustness.csv |
| Robustness check `h1_layer_sweep` | Layer sweep: hidden layers -1, -2, -4 and the mean of the last four (D-13). Needs the contextual models. | python run.py embeddings  # then: python -m src.analysis.robustness | results/robustness.csv |
| H3 gradient vs wikipedia_log10_articles (rarefied) | PENDING USER RUN: `python run.py analyse` with internet access (query.wikidata.org, meta.wikimedia.org, *.wikipedia.org) | python run.py analyse   (needs access to query.wikidata.org and *.wikipedia.org) | results/h3_typology.csv |
| H3 gradient vs wikipedia_log10_articles (full) | PENDING USER RUN: `python run.py analyse` with internet access (query.wikidata.org, meta.wikimedia.org, *.wikipedia.org) | python run.py analyse   (needs access to query.wikidata.org and *.wikipedia.org) | results/h3_typology.csv |
| Infomap communities | the second community algorithm (Louvain is computed; the reference partition does not depend on Infomap) | `pip install infomap==2.8.0`, then `python run.py network` | results/community_stability.json |
| UMAP 2-D map | one of the four dimensionality reductions; the display map (mds_metric) is chosen from those that ran and is re-chosen automatically | pip install umap-learn==0.5.6; python run.py network | results/dr_choice.json |
| Human audit of the English gold standard | 432 of 432 rows still carry Claude's suggested decision; every H2 number is labelled PROVISIONAL | edit audit/english_colexification_manual_check.csv, then python run.py analyse report | audit/english_colexification_manual_check.csv |
