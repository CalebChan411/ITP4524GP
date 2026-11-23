# GameLens course project

- The main project is `part_b/game_recommendation`: a Python modelling and evaluation project, following the structure of Part A2. A website is not required.
- Keep the former website under `archive/website_prototype` as a historical prototype. Do not continue adding website features unless the user asks.
- Use English for code, reports, slides, presentation scripts and saved results. Discussion with the user may be Chinese.
- Base the main recommendation methods on ITP4524 Lecture 9 and the supplied Python Foundations. Identify necessary supplementary encoding and ranking evaluation methods clearly.
- Use real Steam metadata and historical user-game interactions with recorded sources and retrieval dates. Never fabricate players, observations, model results or contributions.
- Keep raw/source data, prepared data, split assignments, fitted models and evaluation outputs separate and reproducible.
- Fit transformations and any interaction-derived statistics on training data only. Keep validation/test interactions out of model profiles, popularity counts and similarity matrices.
- Compare methods using the same held-out users, candidate games and ranking cutoffs. State the sampling limitations and do not treat unobserved interactions as confirmed dislikes.
- Include only necessary code, datasets, models, report, slides and results in the submission archive. Exclude README and AGENTS files.
- Keep exactly four numbered model files in `models/`. Record the selected model filename in `results/model_manifest.json`, without saving a duplicate selected-model binary.
- Name the two test scenarios Test 01 (full-history query) and Test 02 (single-game query). Save their metrics in separate clearly named CSVs; Top 5 and Top 10 are cutoffs within each scenario.
- In the report and presentation, explain the design and implementation, analyse comparable results, and finish with one justified model recommendation. Use validation for selection and state the limits of the final test evidence.
