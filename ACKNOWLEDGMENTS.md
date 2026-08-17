# Acknowledgments and contributor roles

## Research context

Miguel Camacho Horvitz contributed research software and data engineering to a
historical job-advertisement research project led by Joan Martínez and Ellora
Derenoncourt. His work included corpus processing and software for extracting
offered wages and candidate employer locations from OCR, geocoding candidate
addresses, and running the workflow in batches on a research computing cluster.

Miguel is **not a coauthor of the paper**. This repository is a portfolio and
software-audit artifact, not an official paper repository or replication package.
The researchers' names describe the project context and do not imply their
endorsement of this public release.

## AI-assisted audit and public release

OpenAI Codex and Anthropic Claude were used as coding and review assistants in
2026. Their assistance included:

- inspecting the existing pipeline for correctness, reproducibility, data-loss,
  and security risks;
- proposing and implementing refactors and regression tests;
- constructing a synthetic demonstration corpus;
- checking documentation against executable commands; and
- reviewing public-release provenance, attribution, and licensing boundaries.

Miguel reviewed the resulting changes and documentation, decided which changes
to accept, and accepts responsibility for the repository's current contents and
claims. AI assistance is not independent validation, authorship of the research
paper, or a substitute for empirical validation on the intended data.

## External data and software

The project depends on U.S. Census Bureau geography, GeoNames postal-code data,
Ubikuity state adjacency, the SymSpell ecosystem, Geoapify, and other Python
packages listed in `requirements.txt`. Detailed data notices appear in `NOTICE`.

The production advertisement corpus and paper datasets are not redistributed.
