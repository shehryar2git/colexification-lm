# Appendix C · Every robustness check

Source: `results/robustness.csv`. A check *survives* when it reproduces the primary analysis's conclusion (sign and significance at α = 0.05), not when it reproduces the number.

| hypothesis | check | label | description | statistic | value | ci_low | ci_high | p | verdict | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1 | typ_minfam1 | word2vec (EN, static) | Edge rule >= 1 family instead of >= 2 (D-07, check i) | Spearman rho (RSA) | 0.130 | 0.102 | 0.160 | 0.0005 | survives |  |
| H1 | typ_minfam3 | word2vec (EN, static) | Edge rule >= 3 families instead of >= 2 (D-07, check i) | Spearman rho (RSA) | 0.118 | 0.088 | 0.149 | 0.0005 | survives |  |
| H1 | typ_binary_profile | word2vec (EN, static) | Binary (unweighted) colexification profiles (check i/ii) | Spearman rho (RSA) | 0.139 | 0.114 | 0.165 | 0.0005 | survives |  |
| H1 | typ_rate | word2vec (EN, static) | First-order distance 1 - family rate instead of shortest path (D-08, check ii) | Spearman rho (RSA) | 0.136 | 0.117 | 0.158 | 0.0005 | survives |  |
| H1 | typ_profile_cosine | word2vec (EN, static) | Profile cosine on family rates (D-39, check ii) | Spearman rho (RSA) | 0.138 | 0.115 | 0.165 | 0.0005 | survives |  |
| H1 | typ_language_profile | word2vec (EN, static) | Language-weighted profiles instead of family-weighted (D-04, check ii) | Spearman rho (RSA) | 0.140 | 0.114 | 0.169 | 0.0005 | survives |  |
| H1 | typ_evidence_all | word2vec (EN, static) | All CLICS4 entries, including forms whose source entry maps to several concepts (D-37) | Spearman rho (RSA) | 0.139 | 0.112 | 0.169 | 0.0005 | survives |  |
| H1 | typ_identity_form | word2vec (EN, static) | Colexification identity on the exact form string instead of segments (D-03) | Spearman rho (RSA) | 0.135 | 0.107 | 0.166 | 0.0005 | survives |  |
| H1 | typ_no_johansson | word2vec (EN, static) | Without the single most family-diverse source dataset (D-38) | Spearman rho (RSA) | 0.117 | 0.087 | 0.148 | 0.0005 | survives |  |
| H1 | concepts_threshold_100 | word2vec (EN, static) | Concept set at the robustness threshold t = 100 families instead of the Kneedle knee (D-05, check iv) | Spearman rho (RSA) | 0.147 | 0.114 | 0.182 | 0.0005 | survives |  |
| H1 | concepts_no_anchor_additions | word2vec (EN, static) | Without the concepts kept only by the anchor rule (D-05b) | Spearman rho (RSA) | 0.128 | 0.100 | 0.160 | 0.0005 | survives |  |
| H1 | concepts_connected_only | word2vec (EN, static) | Only concepts connected in the colexification network; the isolated ones enter the primary through the max + 1 convention (D-42) | Spearman rho (RSA) | 0.141 | 0.114 | 0.169 | 0.0005 | survives |  |
| H1 | concepts_model_maximal | word2vec (EN, static) | This model's maximal concept set instead of the six-condition intersection (D-16, check iv) | Spearman rho (RSA) | 0.118 | 0.096 | 0.140 | 0.0005 | survives |  |
| H1 | model_uncentred | word2vec (EN, static) | Uncentred cosine similarities instead of mean-centred vectors (D-47) | Spearman rho (RSA) | 0.142 | 0.107 | 0.178 | 0.0005 | survives |  |
| H1 | data_clics3 | word2vec (EN, static) | CLICS3's published family-weighted network (>= 3 families) instead of our CLICS4 recount (D-01, check v-b); compare with typ_minfam3 | Spearman rho (RSA) | 0.103 | 0.076 | 0.134 | 0.0005 | survives | clics3_nodes=2919; clics3_edges=4228; working_concepts_in_clics3=400; edges_among_working_concepts=886; share_pairs_disconnected=0.2814042726303174 |
| H1 | lofo_austronesian | word2vec (EN, static) | Leave-one-family-out: all Austronesian languages removed (check v) | Spearman rho (RSA) | 0.128 | 0.099 | 0.155 | 0.0005 | survives |  |
| H1 | lofo_atlantic-congo | word2vec (EN, static) | Leave-one-family-out: all Atlantic-Congo languages removed (check v) | Spearman rho (RSA) | 0.119 | 0.093 | 0.150 | 0.0005 | survives |  |
| H1 | lofo_sino-tibetan | word2vec (EN, static) | Leave-one-family-out: all Sino-Tibetan languages removed (check v) | Spearman rho (RSA) | 0.124 | 0.096 | 0.154 | 0.0005 | survives |  |
| H1 | lofo_nuclear_trans_new_guinea | word2vec (EN, static) | Leave-one-family-out: all Nuclear Trans New Guinea languages removed (check v) | Spearman rho (RSA) | 0.126 | 0.097 | 0.155 | 0.0005 | survives |  |
| H1 | lofo_pama-nyungan | word2vec (EN, static) | Leave-one-family-out: all Pama-Nyungan languages removed (check v) | Spearman rho (RSA) | 0.129 | 0.101 | 0.159 | 0.0005 | survives |  |
| H1 | lofo_afro-asiatic | word2vec (EN, static) | Leave-one-family-out: all Afro-Asiatic languages removed (check v) | Spearman rho (RSA) | 0.128 | 0.099 | 0.158 | 0.0005 | survives |  |
| H1 | lofo_indo-european | word2vec (EN, static) | Leave-one-family-out: all Indo-European languages removed (check v) | Spearman rho (RSA) | 0.126 | 0.097 | 0.159 | 0.0005 | survives |  |
| H1 | lofo_tupian | word2vec (EN, static) | Leave-one-family-out: all Tupian languages removed (check v) | Spearman rho (RSA) | 0.130 | 0.100 | 0.162 | 0.0005 | survives |  |
| H1 | lofo_austroasiatic | word2vec (EN, static) | Leave-one-family-out: all Austroasiatic languages removed (check v) | Spearman rho (RSA) | 0.127 | 0.100 | 0.158 | 0.0005 | survives |  |
| H1 | lofo_nakh-daghestanian | word2vec (EN, static) | Leave-one-family-out: all Nakh-Daghestanian languages removed (check v) | Spearman rho (RSA) | 0.124 | 0.096 | 0.153 | 0.0005 | survives |  |
| H1 | lofo_turkic | word2vec (EN, static) | Leave-one-family-out: all Turkic languages removed (check v) | Spearman rho (RSA) | 0.128 | 0.100 | 0.156 | 0.0005 | survives |  |
| H1 | lofo_uralic | word2vec (EN, static) | Leave-one-family-out: all Uralic languages removed (check v) | Spearman rho (RSA) | 0.127 | 0.099 | 0.160 | 0.0005 | survives |  |
| H1 | lofo_tai-kadai | word2vec (EN, static) | Leave-one-family-out: all Tai-Kadai languages removed (check v) | Spearman rho (RSA) | 0.126 | 0.096 | 0.156 | 0.0005 | survives |  |
| H1 | lofo_hmong-mien | word2vec (EN, static) | Leave-one-family-out: all Hmong-Mien languages removed (check v) | Spearman rho (RSA) | 0.129 | 0.098 | 0.158 | 0.0005 | survives |  |
| H1 | lofo_tucanoan | word2vec (EN, static) | Leave-one-family-out: all Tucanoan languages removed (check v) | Spearman rho (RSA) | 0.128 | 0.099 | 0.158 | 0.0005 | survives |  |
| H1 | lofo_arawakan | word2vec (EN, static) | Leave-one-family-out: all Arawakan languages removed (check v) | Spearman rho (RSA) | 0.130 | 0.100 | 0.158 | 0.0005 | survives |  |
| H1 | lofo_summary | word2vec (EN, static) | Leave-one-family-out over 16 families: range of rho | Spearman rho (RSA) | 0.119 | 0.119 | 0.130 | 0.0005 | survives | min rho 0.119, max rho 0.130 |
| H2 | h2_unsure_as_present | word2vec (EN, static) | Audited *unsure* pairs counted as English-present instead of excluded (D-55) | Cliff's delta | 0.142 | -0.005 | 0.279 | 0.0390 | does not survive |  |
| H2 | h2_unsure_as_absent | word2vec (EN, static) | Audited *unsure* pairs counted as English-absent instead of excluded (D-55) | Cliff's delta | 0.221 | 0.004 | 0.429 | 0.0246 | does not survive |  |
| H2 | h2_no_input_colexified | word2vec (EN, static) | Without pairs that an input language itself colexifies, for every condition (D-49) | Cliff's delta | 0.147 | -0.093 | 0.375 | 0.1253 | survives |  |
| H2 | h2_matching_1to1 | word2vec (EN, static) | 1:1 matching instead of 1:3 (D-53) | Cliff's delta | 0.166 | -0.119 | 0.409 | 0.1080 | survives |  |
| H2 | h2_caliper_010 | word2vec (EN, static) | Caliper 0.10 SD instead of 0.20 SD (D-53) | Cliff's delta | 0.212 | 0.009 | 0.405 | 0.0320 | does not survive |  |
| H2 | h2_outcome_cosine | word2vec (EN, static) | Raw cosine as the outcome instead of the mutual neighbour rank (D-24) | Cliff's delta | 0.185 | -0.047 | 0.388 | 0.0498 | does not survive |  |
| H2 | h2_minfam1 | word2vec (EN, static) | Edge rule >= 1 family instead of >= 2, population rebuilt (D-07, check i) | Cliff's delta | 0.239 | 0.026 | 0.447 | 0.0180 | does not survive |  |
| H2 | h2_minfam3 | word2vec (EN, static) | Edge rule >= 3 family instead of >= 2, population rebuilt (D-07, check i) | Cliff's delta | 0.197 | -0.023 | 0.406 | 0.0468 | does not survive |  |
| H2 | h2_threshold_100 | word2vec (EN, static) | Concept set at the robustness threshold t = 100 (280 of 330 analysis concepts) (D-05, check iv) | Cliff's delta | 0.047 | -0.224 | 0.302 | 0.3740 | survives |  |
| H2 | h2_lofo_austronesian | word2vec (EN, static) | Leave-one-family-out: all Austronesian languages removed, population rebuilt (check v) | Cliff's delta | 0.114 | -0.091 | 0.338 | 0.1569 | survives |  |
| H2 | h2_lofo_atlantic-congo | word2vec (EN, static) | Leave-one-family-out: all Atlantic-Congo languages removed, population rebuilt (check v) | Cliff's delta | 0.196 | -0.027 | 0.391 | 0.0384 | does not survive |  |
| H2 | h2_lofo_sino-tibetan | word2vec (EN, static) | Leave-one-family-out: all Sino-Tibetan languages removed, population rebuilt (check v) | Cliff's delta | 0.169 | -0.035 | 0.375 | 0.0657 | survives |  |
| H2 | h2_lofo_nuclear_trans_new_guinea | word2vec (EN, static) | Leave-one-family-out: all Nuclear Trans New Guinea languages removed, population rebuilt (check v) | Cliff's delta | 0.102 | -0.117 | 0.313 | 0.1844 | survives |  |
| H2 | h2_lofo_pama-nyungan | word2vec (EN, static) | Leave-one-family-out: all Pama-Nyungan languages removed, population rebuilt (check v) | Cliff's delta | 0.118 | -0.105 | 0.323 | 0.1471 | survives |  |
| H2 | h2_lofo_afro-asiatic | word2vec (EN, static) | Leave-one-family-out: all Afro-Asiatic languages removed, population rebuilt (check v) | Cliff's delta | 0.187 | -0.031 | 0.386 | 0.0455 | does not survive |  |
| H2 | h2_lofo_indo-european | word2vec (EN, static) | Leave-one-family-out: all Indo-European languages removed, population rebuilt (check v) | Cliff's delta | 0.130 | -0.078 | 0.333 | 0.1276 | survives |  |
| H2 | h2_lofo_tupian | word2vec (EN, static) | Leave-one-family-out: all Tupian languages removed, population rebuilt (check v) | Cliff's delta | 0.158 | -0.075 | 0.356 | 0.0793 | survives |  |
| H2 | h2_lofo_austroasiatic | word2vec (EN, static) | Leave-one-family-out: all Austroasiatic languages removed, population rebuilt (check v) | Cliff's delta | 0.169 | -0.038 | 0.361 | 0.0658 | survives |  |
| H2 | h2_lofo_nakh-daghestanian | word2vec (EN, static) | Leave-one-family-out: all Nakh-Daghestanian languages removed, population rebuilt (check v) | Cliff's delta | 0.109 | -0.103 | 0.308 | 0.1663 | survives |  |
| H2 | h2_lofo_turkic | word2vec (EN, static) | Leave-one-family-out: all Turkic languages removed, population rebuilt (check v) | Cliff's delta | 0.153 | -0.072 | 0.356 | 0.0869 | survives |  |
| H2 | h2_lofo_uralic | word2vec (EN, static) | Leave-one-family-out: all Uralic languages removed, population rebuilt (check v) | Cliff's delta | 0.161 | -0.062 | 0.357 | 0.0762 | survives |  |
| H2 | h2_lofo_tai-kadai | word2vec (EN, static) | Leave-one-family-out: all Tai-Kadai languages removed, population rebuilt (check v) | Cliff's delta | 0.119 | -0.084 | 0.311 | 0.1424 | survives |  |
| H2 | h2_lofo_hmong-mien | word2vec (EN, static) | Leave-one-family-out: all Hmong-Mien languages removed, population rebuilt (check v) | Cliff's delta | 0.119 | -0.099 | 0.321 | 0.1473 | survives |  |
| H2 | h2_lofo_tucanoan | word2vec (EN, static) | Leave-one-family-out: all Tucanoan languages removed, population rebuilt (check v) | Cliff's delta | 0.127 | -0.083 | 0.342 | 0.1291 | survives |  |
| H2 | h2_lofo_arawakan | word2vec (EN, static) | Leave-one-family-out: all Arawakan languages removed, population rebuilt (check v) | Cliff's delta | 0.095 | -0.130 | 0.293 | 0.1985 | survives |  |
| H2 | h2_lofo_summary | word2vec (EN, static) | Leave-one-family-out over 16 families: range of Cliff's delta | Cliff's delta | 0.095 | 0.095 | 0.196 | 0.1985 | does not survive | delta 0.09-0.20 |
| H2 | h2_unmatched | word2vec (EN, static) | No matching at all (confounded; reported for comparison, D-24) | Cliff's delta | 0.590 | 0.451 | 0.695 | 0.0000 | does not survive |  |
| H2 | h2_human_audited_only | word2vec (EN, static) | English gold standard restricted to rows a person has decided (D-55) | Cliff's delta |  |  |  |  | PENDING USER RUN | no human decisions in audit/english_colexification_manual_check.csv; fill the `decision` column, then: python -m src.analysis.robustness |
| H3 | h3_all_families_above_chance | word2vec (EN, static) | Every qualifying family's fit, full-family AUC with BH-adjusted permutation p (D-09 amendment) | AUC range | 0.658 | 0.658 | 0.919 | 0.0007 | survives | 16 families, AUC 0.66-0.92 |
| H3 | h3_spearman_instead_of_auc | word2vec (EN, static) | Spearman correlation with the within-family rate matrix instead of the pair-level AUC, as the proposal asks (D-09 amendment) | Spearman rho | 0.015 | 0.015 | 0.071 | 0.0007 | survives | the matrix is >99% ties, so rho is small by construction |
| H3 | h3_gradient_wikipedia_log10_articles_full | word2vec (EN, static) | Family gradient: full AUC vs Wikipedia (log10 articles, largest edition) | Spearman rho |  |  |  |  | PENDING USER RUN | PENDING USER RUN: `python run.py analyse` with internet access (query.wikidata.org, meta.wikimedia.org, *.wikipedia.org) |
| H3 | h3_gradient_joshi_class_full | word2vec (EN, static) | Family gradient: full AUC vs Joshi et al. (2020) class, max over family | Spearman rho | 0.352 |  |  | 0.1566 | does not survive | PROVISIONAL: 37 languages whose class the paper states (lower bounds per family); for the full taxonomy place lang2tax.t |
| H3 | h3_gradient_log10_n_languages_clics_full | word2vec (EN, static) | Control: family fit must *not* track how densely CLICS4 samples the family (full AUC) | Spearman rho | -0.179 |  |  | 0.7508 | survives (control) | done |
| H3 | h3_gradient_wikipedia_log10_articles_rarefied | word2vec (EN, static) | Family gradient: rarefied AUC vs Wikipedia (log10 articles, largest edition) | Spearman rho |  |  |  |  | PENDING USER RUN | PENDING USER RUN: `python run.py analyse` with internet access (query.wikidata.org, meta.wikimedia.org, *.wikipedia.org) |
| H3 | h3_gradient_joshi_class_rarefied | word2vec (EN, static) | Family gradient: rarefied AUC vs Joshi et al. (2020) class, max over family | Spearman rho | -0.050 |  |  | 0.5590 | does not survive | PROVISIONAL: 37 languages whose class the paper states (lower bounds per family); for the full taxonomy place lang2tax.t |
| H3 | h3_gradient_log10_n_languages_clics_rarefied | word2vec (EN, static) | Control: family fit must *not* track how densely CLICS4 samples the family (rarefied AUC) | Spearman rho | 0.321 |  |  | 0.1130 | survives (control) | done |
| H1 | h1_template_set |  | Template sensitivity: each of the 10 templates alone, and leave-one-template-out (D-13). Needs the contextual models. |  |  |  |  |  | PENDING USER RUN | python run.py embeddings  # then: python -m src.analysis.robustness |
| H1 | h1_layer_sweep |  | Layer sweep: hidden layers -1, -2, -4 and the mean of the last four (D-13). Needs the contextual models. |  |  |  |  |  | PENDING USER RUN | python run.py embeddings  # then: python -m src.analysis.robustness |
| H1/H2/H3 | condition_pending | BERT-base (EN) | Every check above, for this condition |  |  |  |  |  | PENDING USER RUN | python run.py embeddings |
| H1/H2/H3 | condition_pending | XLM-R base (multi) | Every check above, for this condition |  |  |  |  |  | PENDING USER RUN | python run.py embeddings |
| H1/H2/H3 | condition_pending | LaBSE (multi) | Every check above, for this condition |  |  |  |  |  | PENDING USER RUN | python run.py embeddings |
| H1/H2/H3 | condition_pending | XLM-R (multi input) | Every check above, for this condition |  |  |  |  |  | PENDING USER RUN | python run.py embeddings |
| H1/H2/H3 | condition_pending | LaBSE (multi input) | Every check above, for this condition |  |  |  |  |  | PENDING USER RUN | python run.py embeddings |
