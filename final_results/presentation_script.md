# Presentation speaking script

Matches `conditional_retrieval.html`. Estimated speaking time: 10–12 minutes. Short paragraphs are speaking pauses; source notes are for preparation.

## Slide 1 — Does every query need every retriever?

Our project asks a simple question: does every search need all the retrieval methods?

We started from Blended RAG, which uses different search methods to improve the information given to a language model.

We wanted to reduce retrieval time. Our idea was to run BM25 first, then decide whether the question also needs Dense and SPLADE.

## Slide 2 — The original Blended RAG paper

The 2024 Blended RAG paper is by Kunal Sawarkar, Abhilasha Mangal, and Shivam Raj Solanki. It studies how better retrieval improves the context given to a generator.

It uses Elasticsearch with BM25, dense vectors, and ELSER, a learned sparse encoder. These provide word matching and semantic search.

Its key idea combines search indexes with field-query strategies. Best Fields rewards the strongest field match; Cross Fields lets terms match across fields; Most Fields combines evidence from multiple fields. It also tests Phrase Prefix.

The authors select six promising index-and-query combinations using retrieval accuracy, then evaluate the generator with FLAN-T5-XXL.

They report top-k retrieval accuracy and nDCG@10, followed by Exact Match and F1 for generated answers. Their SQuAD answer results include 57.63 percent Exact Match and 68.4 percent F1.

*Preparation source: [Blended RAG, sections III–V](https://arxiv.org/html/2404.07220v2). These are the paper’s results, not our cascade’s results.*

## Slide 3 — The efficiency question

The paper reports limitations: field-query gains depend on available metadata, and large indexes create storage and processing costs. In its CoQA experiment, missing metadata limited the benefit of hybrid field queries.

Our study asks a further question: when is the extra search work useful for an individual question?

Some questions contain exact names or technical terms that BM25 can already match well. Other questions use different wording and may benefit from neural search.

In our static baseline, every question runs through all three retrievers. We wanted to choose the work needed for each question while measuring the recall we might lose.

That led us to a gate with a recall-loss budget. We measure its speed against our own full-fusion baseline.

*Preparation source for the paper’s stated limits: [Blended RAG, section VI](https://arxiv.org/html/2404.07220v2). The recall-budget gate is our project’s approach.*

## Slide 4 — What we implemented

We built our own Python version inspired by the paper.

Our system uses BM25, MiniLM for Dense search, and SPLADE for learned sparse search. We combine their rankings using Reciprocal Rank Fusion, or RRF.

For Best Fields BM25, we search title and abstract separately and keep the higher field score for each document. RRF combines ranking positions: each retriever contributes one divided by 60 plus the document’s rank.

We then added two changes: a faster BM25 implementation and a small learned gate.

The original paper uses Elasticsearch and ELSER. Our speed results compare versions of our own system.

## Slide 5 — How the cascade works

When a question arrives, BM25 runs first and returns its top documents.

We calculate 11 simple features from the question and BM25 results. These include the question’s length, the top scores, and the gap between those scores.

Ridge uses these features to predict how much full fusion might improve recall.

Ridge is a small regression model. It adjusts the feature scales, multiplies them by learned weights, and adds the results to produce a predicted gain. We keep MiniLM and SPLADE pretrained; we train the gate.

If the predicted improvement is small, we return BM25’s results. Otherwise, we run Dense and SPLADE and fuse all three rankings.

## Slide 6 — Training and recall budgets

We trained the gate on 2,590 NFCorpus questions with known relevant documents.

For each training question, we ran BM25 and full fusion. We measured their recalls and calculated how much fusion helped. Ridge learned to predict that improvement from the 11 features.

The training target is fusion Recall@100 minus BM25 Recall@100, with negative values set to zero. Recall@100 measures the share of known relevant documents found in the first 100 results.

We created four settings: r90, r92, r95, and r99. For example, r95 aims to keep at least 95 percent of fusion’s recall during training.

To choose its threshold, we sorted questions by predicted gain and added the actual losses of possible early exits. We kept the largest cutoff that stayed within the allowed loss.

We use five-fold out-of-fold predictions for this step. Each training question receives a calibration prediction from a model fitted without that question. After choosing the thresholds, we fit the final Ridge model on all training questions.

The r95 cutoff selected 604 early exits and retained about 95.42 percent during calibration. For new questions, we simply compare the prediction with that saved cutoff.

## Slide 7 — Why indexing mattered

We found a major problem in our first implementation: BM25 itself took most of the time.

On TREC-COVID, it took about 1,410 milliseconds, while full fusion took about 1,493 milliseconds. Even if we skipped the neural methods, we had already paid most of the cost.

We moved repeated scoring work into index preparation. BM25 then took about eight milliseconds, with matching scores and rankings.

The index stores each term’s contribution in the documents where it occurs. At query time, we add those stored contributions and select the top scores. We preserve the original scoring formula and tie order.

We gave this improvement to both fusion and the cascade. This let us measure the extra saving from routing fairly.

## Slide 8 — How we tested it

We trained and selected the thresholds using NFCorpus training data. We then kept the model and thresholds fixed when evaluating other datasets.

During testing, early exits really skipped Dense and SPLADE. We measured the full query path and counted the neural calls.

The final indexed tests covered all 50 TREC-COVID questions and a fixed sample of 256 SQuAD questions. Both used their full document collections.

We measured three warmed timing passes and checked the saved rankings and quality scores separately.

We synchronized GPU timing and randomized method order to reduce timing bias. We report mean latency and slower-query measures such as p95. These timings exclude index construction and answer generation.

## Slide 9 — Measured quality and speed

The results show a trade-off: more early exits can save more time, but may lose more recall.

On TREC-COVID, indexed fusion took about 91 milliseconds. The conservative r99 cascade took about 74 milliseconds and retained 96.68 percent of fusion’s recall.

Recall retention is the cascade’s average Recall@100 divided by fusion’s average Recall@100. So 96.68 percent retention describes a ratio between retrieval scores; it does not mean 96.68 percent answer accuracy.

The faster r95 setting took about 56 milliseconds and retained 91.95 percent. However, its uncertainty range went below our 90 percent floor.

The r90 setting was faster again, but retained only 84.53 percent, so it missed that floor.

On the SQuAD sample, the fastest cascade was 1.59 times faster than indexed fusion and retained 99.22 percent of its recall.

## Slide 10 — What the project establishes

Our main finding is that a cascade becomes more useful when its first stage is cheap.

We built a working system that skips neural calls, improved BM25 execution, and measured the balance between speed and recall.

The cascade was faster than full fusion, but some single retrievers remained better choices. SPLADE performed strongly on TREC, and BM25 alone was very fast on SQuAD.

Our results support an improvement within our system. A next step is to test the changes in the original Elasticsearch and ELSER setup and measure generated-answer quality.
