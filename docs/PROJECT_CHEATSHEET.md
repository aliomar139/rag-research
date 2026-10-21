# Project cheat sheet: how to explain our retrieval research

Updated 22 October 2026.

This guide follows the project story in plain language. It starts with the search concepts, explains what we changed, then summarizes what the experiments show and where the evidence lives.

## 1. What is the project about?

When someone asks a question, a search system has to find useful documents from a large collection. Our project studies whether it is necessary to run every search method for every question.

The regular hybrid system runs BM25, Dense, and SPLADE, then combines their rankings. We call that **static fusion**. Our idea was to run BM25 first and use a small decision model to decide whether the other two searches are likely to help.

That two-stage approach is a **cascade**. The decision model is the **gate**.

> Run the inexpensive first search. If it looks sufficient, return it. If it may be missing useful documents, run the more expensive searches too.

This project measures retrieval quality and retrieval time. It does not measure the quality of answers written by a language model.

## 2. The basic concepts

### What is RAG?

**Retrieval-augmented generation (RAG)** is a system with two main steps:

1. A retriever finds documents related to a question.
2. A language model uses those documents to help write an answer.

Retrieval quality and answer quality are different things. Our experiments study the first step. We did not test generated-answer Exact Match or F1.

### What are the corpus, query, ranking, and qrels?

- **Corpus:** the collection of documents being searched.
- **Query:** the question or search text.
- **Ranking:** documents ordered from most to least relevant according to a retriever.
- **qrels:** relevance judgments used as the evaluation answer key. They say which documents are judged relevant to which queries.

The retriever does not see qrels when it makes its live decision. Training qrels help create the examples used to teach the gate. Evaluation qrels score its results after the rankings and decisions are fixed.

### What is a dense vector?

A **dense vector** is a list of numbers that represents text. A neural model encodes a query and each document into vectors. Text with similar meaning tends to have vectors that are close to each other.

Dense search can match a question to a document even when they use different words. For example, it may connect "heart attack" and "myocardial infarction." It can be less reliable for exact names, numbers, and rare terms. In our project, MiniLM creates the vectors and FAISS searches them.

### What does "sparse" mean?

A sparse vector has many possible positions, but most are zero. SPLADE uses a neural language model to assign weights to vocabulary terms. It can give weight to related terms as well as terms written in the text, which lets it combine learned expansion with term-based search.

### What is BM25?

**BM25** is a widely used word-matching method. It scores documents using how often query words occur, how rare those words are across the corpus, and document length. Rare, matching terms usually count more than common terms.

Our **Best Fields BM25** scores the title and abstract separately, then keeps the higher of the two scores. This lets a strong title match count without requiring the same match to appear in the abstract.

## 3. Why use three retrievers?

Each retriever catches a different kind of match:

| Retriever | In simple terms | Useful when |
|---|---|---|
| BM25 | Matches words and weighs their frequency and rarity | The query and document share important terms |
| Dense MiniLM | Matches similar meanings using vectors | The relevant document uses different wording |
| SPLADE | Uses learned term weights and expansion | Related terms may help, while term-level matching is useful |

The methods can help one another, but running all three costs more than running one. That cost is the reason to study conditional retrieval.

## 4. How do we combine the rankings? RRF

BM25, Dense, and SPLADE produce scores on different scales. Adding their raw scores directly would let the scale of one method distort the combination.

**Reciprocal rank fusion (RRF)** avoids that problem. It uses each document's position in each ranking, rather than comparing the raw scores:

```text
RRF score for a document = sum over retrievers of 1 / (60 + its rank)
```

For example, a document ranked near the top by several retrievers receives a strong combined score. A document that appears in only one list gets credit from that list alone. The constant 60 softens the effect of small rank differences.

In our **static fusion** baseline, all three retrievers run for every query and RRF combines their top results. On the cascade's early-exit path, only BM25 ran, so there is nothing to fuse. On the escalated path, the system reuses the existing BM25 ranking, runs Dense and SPLADE, and applies RRF to all three rankings.

## 5. What are we measuring?

### Recall@100

**Recall@100** measures how many of the judged relevant documents appear in the first 100 results. If 20 documents are judged relevant and 10 appear in the top 100, recall is 10/20, or 50%.

### Hit@k, or top-k retrieval accuracy

**Hit@k** asks whether at least one relevant document appears in the first k results. Each query gets a hit or a miss; averaging those outcomes gives the hit rate.

If a query has 20 relevant documents and the top 10 contains one, it gets a hit at 10, but its Recall@10 is only 1/20, or 5%. This is why top-k retrieval accuracy and recall can give different numbers for the same ranking. Neither measures the correctness of a generated answer.

### Recall retention

We compare a method's average Recall@100 with static fusion's average:

```text
recall retention = method's mean Recall@100 / fusion's mean Recall@100
```

If fusion has mean recall of 50% and the cascade has 45%, the cascade retains 90% of fusion recall.

That does **not** mean 90% of the document lists match, 90% of all relevant documents were found, or 90% of answers are correct. Retention can be above 100% if a standalone retriever has higher average recall than fusion.

### nDCG@10 and MAP@10

These metrics focus on the ordering of the first 10 results:

- **nDCG@10** gives more credit when relevant documents appear near the top and can distinguish different relevance grades.
- **MAP@10** rewards relevant documents appearing early and penalizes irrelevant results ahead of them.

Recall@100, nDCG@10, and MAP@10 measure different aspects of retrieval. A method can lead on one and not the others.

### Latency, speedup, and time saved

**Retrieval latency** is how long the search system takes to return its ranking for a query. Speedup is a ratio: reference time divided by the new system's time. If one takes 100 ms and the other takes 50 ms, the second is 2x faster and saves 50% of the time.

Our main timing results report **mean query latency**, the average across the measured queries. **p95 latency** is the time within which approximately 95% of measured queries finish; roughly 5% take longer. A lower mean does not prove that the slowest queries improved. The project does not show a p95 latency improvement.

### What happens before timing, and what happens during it?

| Stage | Work included | Part of reported query latency? |
|---|---|---|
| Offline preparation | Download data, load models, encode corpus documents, build indexes | No |
| Live retrieval | Process the query, run the selected retrievers, compute gate features and its decision, and fuse results when needed | Yes |

Document representations are prepared beforehand, but neural query encoding runs when a query uses Dense or SPLADE. The gate is already trained; it does not retrain for each question. Generating an LLM answer is outside the experiment.

The indexed runs use a T4 GPU, one CPU thread, warmups, randomized query/method order, and GPU synchronization around timed queries. Warmups let the system get ready before measurement. Repeated timing passes rerun queries to measure runtime variation; they are not additional training rounds. Each timed method performs its own retrieval without borrowing another method's neural query results.

These measurements describe search with prepared indexes. They do not measure total setup time, index memory, or production performance under many simultaneous users.

### What is a confidence interval?

A reported percentage comes from a particular set of queries. A **confidence interval** expresses uncertainty in that estimate. We calculate 95% query-bootstrap intervals by resampling the measured queries 1,000 times, keeping each query's method and baseline results paired.

For example, TREC r95 retains 91.95% at the measured point, but its retention interval is 86.88%-95.76%. Because the interval crosses below 90%, the result gives weaker evidence for clearing that floor than the point estimate alone suggests. TREC r99's interval, 93.58%-98.75%, stays above the floor.

An interval does not guarantee performance on a future dataset or mean that 95% of individual queries achieve that retention. These intervals describe query-sampling uncertainty; they do not capture all GPU, hardware, or session variation.

### Why do the charts show two different speedups?

Both charts show the same methods, but divide by different reference times:

| Comparison | Reference | What the ratio includes |
|---|---|---|
| Versus our original fusion | Earlier Python fusion with the slower BM25 implementation | Indexing plus routing improvements across separate runs |
| Versus equally indexed fusion | Fusion that already uses the efficient BM25 scorer | Additional routing benefit with the same indexed first stage |

The versions form this progression:

```text
Original fusion -> make BM25 faster -> indexed fusion -> add the gate -> indexed cascade
```

For the same 256 SQuAD questions:

| Version | Mean time | Speedup vs original fusion | Speedup vs indexed fusion |
|---|---:|---:|---:|
| Original fusion | 147.01 ms | 1.00x | Not this comparison's reference |
| Indexed fusion | 29.55 ms | 4.98x | 1.00x |
| Indexed cascade r90, secondary | 18.59 ms | 7.91x | 1.59x |

The 7.91x ratio is the combined improvement over our starting implementation. The 1.59x ratio, equivalent to 37.08% less mean time, is the extra gain over already indexed fusion. This r90 setting retained 99.22% of fusion's mean Recall@100 on the sample.

"Original fusion" always means our earlier Python implementation, not the published paper's Elasticsearch/ELSER system. Cross-run comparisons include session variation. The growing bars show different gate settings, rather than performance as corpus size increases.

## 6. What did the original paper do, and what did we change?

The project is inspired by *Blended RAG*. The paper combines keyword, dense-vector, and learned-sparse retrieval in Elasticsearch, using ELSER for learned sparse search.

Our implementation uses Python BM25, MiniLM, SPLADE, and RRF. We then added the BM25-first cascade and an indexed BM25 scorer. This is an adaptation of the broad hybrid-retrieval idea, not an exact reproduction of the original Elasticsearch/ELSER system. The paper's published numbers are context, not matched baselines for our speedup claims.

## 7. How the cascade decides

The final gate is a small **Ridge regression** model. In plain terms, it learns how query and BM25-result clues relate to the additional recall that fusion might provide.

The final model uses 11 features. They describe query length, term rarity, and the shape of the BM25 score distribution. Here are the features in plain language:

| Feature | What it tells the gate |
|---|---|
| `query_token_count` | Number of query words after tokenization |
| `query_unique_token_count` | Number of distinct query words |
| `bm25_top1` | Score of BM25's highest-ranked document |
| `bm25_top2` | Score of BM25's second-ranked document |
| `bm25_gap12` | Difference between the first and second scores |
| `bm25_top10_mean` | Average BM25 score among the top 10 results |
| `bm25_top10_std` | How much the top 10 scores vary |
| `bm25_top1_per_query_token` | Top score divided by the number of query words |
| `max_idf` | Rarity of the rarest query term in the corpus |
| `mean_idf` | Average rarity of the distinct query terms |
| `idf_std` | How much the query terms differ in rarity |

IDF means *inverse document frequency*: a word that appears in few documents has a higher IDF than a common word. A large BM25 score gap can mean one result stands out, but the gate does not use that clue alone; it combines all 11 features.

During training, we run BM25 and fusion on training questions and score both against training qrels. Their measured recalls give the **training target**, the extra recall we want the gate to learn to estimate:

```text
training target = max(actual fusion Recall@100 - actual BM25 Recall@100, 0)
```

At query time, the trained gate sees the query and BM25-result features. It calculates a prediction without knowing the true recall or running the neural branches first:

```text
predicted gain = max(Ridge(standardized query and BM25 features), 0)
```

The training target comes from known relevance labels. The live prediction comes from the fitted model and can be wrong. The decision is:

```text
if predicted gain <= selected threshold:
    return BM25's top 100 results (early exit)
else:
    run Dense and SPLADE, then fuse all three rankings with RRF
```

For example, if the predicted gain is 0.01 and the threshold is 0.02, the gate exits because 0.01 is below the threshold. If the threshold is 0.005, it escalates because 0.01 is above the threshold. The real gain is known only afterward, when the ranking is scored against qrels.

Ridge is a linear model with regularization. It multiplies each standardized feature by a learned weight, adds the results and an intercept, then floors negative predictions at zero. Standardization puts features on comparable scales. Regularization discourages overly large weights, which can help when features overlap or are correlated. The model uses `alpha = 10`; alpha controls the strength of that regularization.

### What are training, development, testing, and transfer?

| Term | In simple terms | Role in our project |
|---|---|---|
| Training | Examples used to teach the model | NFCorpus train supplies the gate's weights and calibration thresholds |
| Development, or dev | Data inspected while investigating design choices | NFCorpus dev supplies development evidence; it is not an untouched final test |
| Evaluation or testing | Measure a fixed system on a declared set of queries | Later studies evaluate the frozen policies, with each study's inspection status reported |
| Transfer | Use a trained model on a different collection | Apply the NFCorpus-trained gate to SCIDOCS, TREC-COVID, or SQuAD without refitting it |

**Data leakage** happens when evaluation information influences learning or selection in a way the claimed protocol excludes. In this project, the final weights and thresholds use NFCorpus train labels only. NFCorpus test had already been inspected earlier and is excluded from this calibration and confirmation sequence.

Freezing the gate means keeping its weights and thresholds fixed during later evaluation. It prevents retuning those values to suit each outcome. It does not make every later study independent of previously inspected evidence; the indexed TREC follow-up has that limitation.

### What do r90, r92, r95, and r99 mean?

These names refer to **training calibration targets** of 90%, 92%, 95%, and 99% recall retention. Five-fold out-of-fold predictions were generated on 2,590 NFCorpus training queries using seed 42. This means each training query's calibration prediction came from a fold model that did not train on that query. After selecting the thresholds, the final Ridge model was fitted on all 2,590 training queries.

| Policy | Frozen predicted-gain threshold | Training early exits | How it behaves |
|---|---:|---:|---|
| r90 | 0.08477 | 41.7% | Most aggressive; exits most often |
| r92 | 0.08109 | 34.6% | Allows fewer early exits |
| r95 | 0.07567 | 23.3% | More conservative |
| r99 | 0.06300 | 6.4% | Most conservative; escalates most often |

The threshold is the predicted-gain cutoff, not a recall score. A *higher* cutoff lets more predictions count as small enough to exit. During calibration, out-of-fold predictions determine which training queries would exit. The cost of each selected exit is its measured clipped training loss, `max(actual fusion recall - actual BM25 recall, 0)`, calculated from training qrels. The calibration procedure adds those measured losses and selects a cutoff so their total fits the policy's loss budget. Losses on one query are not canceled by BM25 doing better on another query.

For a target of 90%, the budget allows total clipped loss of up to 10% of the sum of fusion's training-query recalls. This is an aggregate budget, so an individual query can lose more than 10% even when the calibration target is met.

They are not guarantees on other datasets. For example, an r99 setting can retain less than 99% on TREC-COVID. The project reports the measured retention separately from the target.

The final evaluation gate is the compact **11-feature Ridge** model. The repository also contains an earlier hand-written threshold router and exploratory gate versions; those are not the final reported policy.

## 8. What happened as we developed the project?

### First, we tried to skip neural searches

The initial concept was to let BM25 answer easy queries and call Dense and SPLADE only when BM25 might miss useful documents. Early experiments helped explore the idea, but some used cached rankings, restricted candidate sets, or estimated timing. Those results could not establish the timing of real conditional execution.

### Then we learned that the gate and first stage both matter

A gate can skip neural work and still fail to save time if BM25 is already slow. In the original TREC-COVID implementation, BM25 averaged about 1,410 ms while full fusion averaged about 1,493 ms. The first stage consumed most of the total time before the gate could make its decision.

### We made BM25 faster

The project added a postings-based BM25 scorer. It precomputes term contributions during index preparation and stores them by term, so a query can accumulate scores from matching postings instead of repeating the original whole-corpus work.

This changed how the existing BM25 scores were computed, not the BM25 formula. Checks found identical measured rankings and scores. We applied the indexed scorer to both fusion and the cascade, so the indexed-fusion comparison isolates the additional gain from routing.

## 9. What the final measurements show

### What do primary and secondary results mean?

The **primary** setting is the main comparison declared before its evaluation outcomes. **Secondary** settings show other operating points, including faster but less reliable choices. Keeping this distinction avoids choosing a favorable result afterward and presenting it as the original plan.

For the indexed follow-up, r99 was the preselected primary evaluated against the user's 90% retention floor. Its name still refers to a 99% training target. The r90, r92, and r95 settings remain secondary even when one is faster. Earlier studies keep their originally declared primaries.

### Indexed TREC-COVID

The indexed follow-up covers all **50 queries** and all **171,332 documents**, with three timing passes. The preselected primary is the r99 policy:

| Method | Recall retained | Mean time | Compared with indexed fusion |
|---|---:|---:|---:|
| Indexed static fusion | 100.00% | 91.28 ms | Reference |
| Cascade r99, primary | 96.68% | 73.90 ms | 1.24x faster |
| Cascade r95, secondary | 91.95% | 55.66 ms | 1.64x faster |
| Cascade r90, secondary | 84.53% | 40.41 ms | 2.26x faster |
| SPLADE alone | 104.69% | 52.77 ms | 1.73x faster |

The primary saved 19.04% of mean time versus indexed fusion. Its recall-retention interval was 93.58%-98.75%, above the study's 90% floor, but it missed the nominal 99% target.

The r95 point was faster and retained 91.95%, but its recall interval crossed below 90%. SPLADE alone had higher observed recall and nDCG than the primary cascade, and was faster. The cascade had higher MAP@10. This is a trade-off, not an overall win.

The primary cascade was also 20.21x faster than the earlier, non-indexed fusion run. That comparison combines the BM25 indexing improvement and routing, and it spans separate runs. The cascade-only comparison is 1.24x against equally indexed fusion.

TREC quality had already been inspected when the indexed follow-up was planned. The gate stayed frozen, and r99 was selected before the new indexed outcomes, but this follow-up measures faster execution on known quality. It is not a fresh independent test of whether the gate generalizes to an unseen collection.

### Indexed SQuAD

The indexed follow-up uses a fixed, hash-selected sample of **256 questions** and the full **20,958-paragraph corpus**, with three timing passes.

The fastest cascade, r90, was 1.59x faster than indexed fusion and retained 99.22% of fusion recall. The preselected primary r99 retained 100% on this sample and was 1.04x faster.

BM25 alone retained 95.70% and took 1.03 ms, making it the fastest listed method above a 90% recall floor for this sample. The indexed SQuAD numbers apply to these 256 questions, not all 10,570 questions.

### Which original-paper datasets did we actually evaluate?

| Dataset | Completed scope in this project |
|---|---|
| TREC-COVID | All 50 queries and the full corpus; three timing passes for both original and indexed studies |
| SQuAD | Original study: all 10,570 questions, one complete timing pass. Indexed study: 256 fixed questions, three passes. Both use the full 20,958-paragraph corpus |
| Natural Questions, or NQ | No completed current full-corpus live benchmark; earlier restricted or qrel-informed exploratory results do not count as that evaluation |
| HotpotQA | Preparation code exists, but no completed current run |
| CoQA | No completed current run |

NFCorpus and SCIDOCS are additional development and transfer collections used in this project. They do not replace the uncompleted datasets above. Original SQuAD's extra partial timing records do not establish three complete passes, and the indexed sample's results cannot be presented as results for all 10,570 questions.

### What these results do and do not establish

The evidence supports a measured execution improvement over full fusion at selected recall trade-offs. It does not establish that the cascade is always the best choice: a standalone method can be faster and have stronger quality on a dataset.

The study also does not establish a new routing principle, preserved generated-answer quality, or speedup on the original Elasticsearch/ELSER system. Full NQ and HotpotQA evaluations and any CoQA evaluation remain pending.

## 10. The contribution, in one answer

If asked "What did your project contribute?", you can say:

> We built a BM25-first retrieval cascade and a faster indexed BM25 implementation that preserved the measured rankings and scores. We evaluated the cascade against equally indexed fusion and standalone retrievers, measuring how latency and retrieval quality changed at several frozen recall-retention settings. The cascade reduced mean latency compared with fusion in the reported settings, but its usefulness depended on the dataset and competing standalone methods.

## 11. Questions a professor may ask

**Why combine different retrievers?**  They find different kinds of matches: exact terms, semantic similarity, and learned term expansion.

**Why use RRF?**  The retrievers' raw scores are on different scales. RRF combines their ranks instead.

**What does the gate predict?**  It estimates the recall gain that fusion may add over BM25, using query and BM25-result features.

**How can it predict recall without seeing the correct documents?**  During training, known relevance labels teach it how the features relate to measured recall gain. During live use, it applies that learned relationship without seeing qrels. The estimate can be wrong.

**What does r99 guarantee?**  Nothing about a new dataset. It names a 99% training calibration target. We report measured retention separately.

**Why not headline whichever setting was fastest?**  The main comparison was selected before its outcomes. We report faster secondary settings too, along with their quality losses and uncertainty.

**Why was indexed BM25 necessary?**  BM25 was so slow in the original implementation that it consumed most of the query time. Making the first stage faster gave the cascade room to save time by skipping neural searches.

**Does 7.91x mean the gate alone made SQuAD nearly eight times faster?**  No. That ratio combines indexing and routing against our earlier fusion. The gate's additional gain over already indexed fusion is 1.59x at that sampled secondary point.

**Did the cascade beat fusion?**  The primary indexed cascade was faster on average than equally indexed fusion while retaining most of its Recall@100 in the measured TREC-COVID and SQuAD settings. That does not mean it beat every standalone method.

**Did you reproduce Blended RAG?**  We reproduced the broad idea of combining retrieval methods, but changed the engine and models. Our results are not a direct reproduction of the original system or its speed.

**Did you test answer quality?**  No. We evaluated retrieval rankings, not LLM-generated answers.

## 12. Where to find the work in the repository

| If you want to inspect... | Open... |
|---|---|
| Short project overview | [README.md](../README.md) |
| Full research method, protocol, results, and limitations | [FINAL_MANUSCRIPT.md](FINAL_MANUSCRIPT.md) |
| Both speedup references and every measured operating point | [SPEEDUP_SUMMARY.md](SPEEDUP_SUMMARY.md) |
| Project story and experiment history | [adhd-explanation.md](adhd-explanation.md) |
| BM25 and Best Fields scoring | [bm25_retriever.py](../cascaded-blended-rag/src/bm25_retriever.py) |
| Dense vectors and FAISS | [dense_retriever.py](../cascaded-blended-rag/src/dense_retriever.py) |
| SPLADE representations and retrieval | [splade_retriever.py](../cascaded-blended-rag/src/splade_retriever.py) |
| Early baseline RRF implementation | [04_baseline_blended_rag.py](../cascaded-blended-rag/experiments/04_baseline_blended_rag.py) |
| Final cascade and Ridge inference | [retention_tradeoff.py](../cascaded-blended-rag/src/retention_tradeoff.py) |
| Gate features, training, and calibration | [12_calibrate_nfcorpus_train_check_dev.py](../cascaded-blended-rag/experiments/12_calibrate_nfcorpus_train_check_dev.py) |
| Final four-policy training and calibration | [16_prepare_retention_tradeoff.py](../cascaded-blended-rag/experiments/16_prepare_retention_tradeoff.py) |
| Earlier heuristic router prototype | [cascaded_router.py](../cascaded-blended-rag/src/cascaded_router.py) |
| Indexed BM25 scorer | [indexed_bm25.py](../cascaded-blended-rag/src/indexed_bm25.py) |
| Indexed benchmark runner | [25_benchmark_indexed_cascade.py](../cascaded-blended-rag/experiments/25_benchmark_indexed_cascade.py) |
| TREC-COVID indexed results and caveats | [TREC_COVID_INDEXED_FINDINGS.md](findings/TREC_COVID_INDEXED_FINDINGS.md) |
| SQuAD indexed results and caveats | [SQUAD_INDEXED_FINDINGS.md](findings/SQUAD_INDEXED_FINDINGS.md) |
| Index of all experiment findings | [findings/README.md](findings/README.md) |
| Numbered experiment map | [experiments/README.md](../cascaded-blended-rag/experiments/README.md) |

## 13. Short version to remember

> BM25 matches words, Dense matches meaning, and SPLADE learns useful term expansions. Static fusion runs all three and combines their rankings with RRF. Our cascade runs BM25 first and uses a trained gate to decide whether to run the neural retrievers. We also made BM25 much faster with a postings-based implementation. The experiments show a speed-quality trade-off against fusion, while standalone baselines and dataset transfer limit broader claims.
