# Spoken script for the 10-slide HTML showcase

This script follows the ten slides in presentation mode. The three speakers have roughly equal speaking time: Ali covers slides 1–3, Abdallah slides 4–6, and Abbas slides 7–10. Use each numbered heading to stay in sync with the right-arrow navigation.

## Speaker 1 — Ali Omar

### Slide 1 — Conditional hybrid retrieval

Good morning. We are Ali Omar, Abdallah Mestrah, and Abbas Yazbek from Lebanese University, supervised by Dr. Ahmad Abboud. Our project studies a practical question: can we avoid running every retrieval method on every query and reduce response time while keeping most of the retrieval quality?

### Slide 2 — The paper that motivates this project

We chose *Blended RAG* by Sawarkar, Mangal, and Solanki. It combines keyword retrieval with dense and learned sparse retrieval in Elasticsearch, including ELSER. The motivation is that different retrieval methods can find different relevant documents. The paper evaluates retrieval quality and generated answers.

Its reported TREC-COVID nDCG@10 and SQuAD exact-match and F1 scores are useful context, but they are not direct comparison targets for our results. We did not run the same engine, models, document processing, or full evaluation setup.

### Slide 3 — Same idea, different system

Our recreation follows the paper’s broad idea of combining complementary retrieval signals. We use Python BM25, MiniLM dense retrieval, SPLADE learned sparse retrieval, and reciprocal rank fusion. We evaluated all 50 TREC-COVID queries and a declared SQuAD retrieval setup. For SQuAD, our indexed follow-up uses 256 fixed questions against the full corpus.

This is a methodological recreation, not a reproduction of the paper’s Elasticsearch and ELSER system. We also use retrieval metrics such as Recall@100, recall retention, nDCG@10, and MAP@10. We do not report generated-answer EM or F1, so the published answer scores cannot be compared with ours. Abdallah will explain what our first experiments showed and how they led to the implementation changes.

## Speaker 2 — Abdallah Mestrah

### Slide 4 — The initial cascade had little time to save

In our first TREC-COVID cascade, BM25 used about 1,410 milliseconds of the 1,493 milliseconds required by full fusion. Since the lexical first stage was already taking most of the time, skipping neural retrieval could save only a little. This showed us that the first stage itself had to become cheaper before conditional routing could make a larger difference.

These timings describe our Python setup with prepared indexes. They exclude index construction, model loading, corpus encoding, and answer generation. Our retrieval engine, models, document construction, metrics, and timing protocol also differ from the paper’s. We did not run matched ablations to isolate each difference, so we do not attribute score gaps to a single cause.

### Slide 5 — First fix the lexical bottleneck

We replaced repeated lexical scoring with an indexed BM25 implementation. It reuses term contributions stored in a compatible postings index. We apply the same indexed BM25 backend to fusion, standalone BM25, and every cascade policy, while keeping the saved retrieval outputs identical.

In separate TREC component runs, indexed BM25 reduced mean time from roughly 1,410 milliseconds to about 8 milliseconds. That component result is not the end-to-end cascade speedup. To measure the additional effect of routing fairly, we compare the cascade against fusion that uses the same indexed BM25 backend. Abbas will present those results and the limits of what they support.

### Slide 6 — BM25 first, then a learned gate

Each query first runs BM25. Eleven features from the query and its BM25 ranking go into a standardized Ridge regression model trained on NFCorpus training queries. The model estimates how much recall dense and SPLADE may add. If predicted gain is below the selected threshold, the system returns BM25; otherwise it runs both neural retrievers and fuses their rankings with BM25.

The gate does not use relevance labels at inference time. It is a learned prediction model, but the prediction is used only to choose whether to pay for the neural retrieval stages. Training and transfer results also showed that a threshold calibrated on one collection may miss its recall target on another, so we treat cross-collection reliability as an open question.

## Speaker 3 — Abbas Yazbek

### Slide 7 — TREC-COVID: indexing helps, routing adds more

On all 50 TREC-COVID queries, our selected r99 cascade averaged 73.90 milliseconds, compared with 91.28 milliseconds for equally indexed fusion. That is a 1.24-times speedup over indexed fusion, while retaining 96.68 percent of its mean Recall@100.

Compared with our earlier non-indexed fusion run, the cascade is 20.21 times faster across separate runs. That larger ratio includes both the indexing change and conditional routing. Indexing alone showed a 16.36-times ratio in the reported comparison. The TREC primary setting was selected after inspecting quality results, so this follow-up demonstrates execution performance on that setup, not independent quality generalization.

### Slide 8 — SQuAD: a declared query sample

For SQuAD, we used a fixed sample of 256 questions and the full 20,958-document corpus. The r90 cascade averaged 18.59 milliseconds; equally indexed fusion averaged 29.55 milliseconds. That is a 1.59-times speedup over indexed fusion and 99.22 percent retained mean Recall@100 on this sample.

Its 7.91-times speedup against earlier fusion includes indexing and routing. The query sample and corpus construction are part of the result, so we should not present this as a full SQuAD evaluation or as a direct reproduction of the paper’s answer-generation experiment.

### Slide 9 — The cascade is not always the best choice

We also compared standalone BM25, Dense, and SPLADE. On TREC-COVID, SPLADE alone was faster than indexed fusion and retained more mean Recall@100; the cascade primary had a higher MAP value. On the SQuAD sample, BM25 was very fast and stayed above the stated 90 percent recall floor, while SPLADE led the ranking measures.

These comparisons show that the cascade is one option, not a universal winner. The best choice depends on the collection and the quality target. Earlier NFCorpus and SCIDOCS transfer results are also relevant: some frozen thresholds missed their intended recall targets, and Dense was a stronger speed-recall option in those settings. We keep those outcomes visible when describing the contribution.

### Slide 10 — What the evidence supports

Our results support a bounded engineering conclusion: indexing made our lexical first stage much cheaper, and conditional routing then reduced mean query time relative to equally indexed fusion on the tested settings. They do not show the same speedup in the original Elasticsearch and ELSER system, preserve answer quality, or prove overall superiority over standalone retrieval.

Future work should implement the changes in Elasticsearch and ELSER, set the protocol before evaluation, and test more collections, including NQ, HotpotQA, and CoQA. It should measure generated-answer exact match and F1, compare strong standalone and adaptive baselines, and repeat timing across hardware and under load. Index memory and update costs also need measurement.

Our contribution is therefore an evaluated implementation result in our Python setup, with clear limits on transfer. The final references on this slide point to the base paper and the supporting retrieval, benchmark, and model work. Thank you.
