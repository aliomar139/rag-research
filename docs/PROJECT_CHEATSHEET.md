# Project cheat sheet: how to explain our retrieval research

Updated 22 October 2026.

This guide follows the project story in plain language. It starts with the search concepts, explains what we changed, then summarizes what the experiments show and where the evidence lives.

For a model or algorithm name, use **Ctrl+F**. The explanations distinguish the final evaluated pipeline, the original paper's system, and historical experiments so that a name appearing in the repository does not imply it was part of the final benchmark.

## 1. What is the project about?

When someone asks a question, a search system has to find useful documents from a large collection. Our project studies whether it is necessary to run every search method for every question.

The regular hybrid system runs BM25, Dense, and SPLADE, then combines their rankings. We call that **static fusion**. Our idea was to run BM25 first and use a small decision model to decide whether the other two searches are likely to help.

That two-stage approach is a **cascade**. The decision model is the **gate**.

> Run the inexpensive first search. If it looks sufficient, return it. If it may be missing useful documents, run the more expensive searches too.

The final study measures retrieval quality and retrieval time. Generated-answer quality is outside the final cascade evaluation; Section 6 explains the separate historical generation work.

## 2. The basic concepts

### What is RAG?

**Retrieval-augmented generation (RAG)** is a system with two main steps:

1. A retriever finds documents related to a question.
2. A language model uses those documents to help write an answer.

Retrieval quality and answer quality are different things. Our final experiments study the first step. Generated-answer Exact Match and F1 are outside the completed cascade evaluation.

### What are the corpus, query, ranking, and qrels?

- **Corpus:** the collection of documents being searched.
- **Query:** the question or search text.
- **Ranking:** documents ordered from most to least relevant according to a retriever.
- **qrels:** relevance judgments used as the evaluation answer key. They say which documents are judged relevant to which queries.

The retriever does not see qrels when it makes its live decision. Training qrels help create the examples used to teach the gate. Evaluation qrels score its results after the rankings and decisions are fixed.

### What are dense retrieval and embeddings?

A **dense vector**, also called a dense embedding, is a list of numbers that represents text. A neural model encodes a query and each document into vectors. Text with similar meaning tends to receive similar vectors. **Dense retrieval** is the search approach that finds documents by comparing those vectors.

Dense search can match a question to a document even when they use different words. For example, it may connect "heart attack" and "myocardial infarction." It can be less reliable for exact names, numbers, and rare terms. In our project, MiniLM creates the vectors and FAISS searches them.

### What are MiniLM and Sentence Transformers?

**MiniLM** is a family of compact Transformer neural models. Our specific pretrained sentence-embedding model is `sentence-transformers/all-MiniLM-L6-v2`. It converts each text into a vector containing 384 numbers. **Sentence Transformers** is the software library we use to load that model and produce its embeddings. The [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) describes its embedding output.

We load the existing model weights and encode our corpus; we do not fine-tune MiniLM in this study. The part we train for our contribution is the Ridge gate.

**L6** identifies its six Transformer layers, as recorded in the [model configuration](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/config.json). **v2** identifies the model release. The 384 numbers are the length of each output embedding; they are different from the model's total number of learned parameters.

The MiniLM family uses **knowledge distillation**: a smaller student learns from a larger teacher's behavior. See the [MiniLM paper](https://arxiv.org/abs/2002.10957). The downloaded sentence-embedding checkpoint has already undergone its published training; our experiment does not repeat that training.

### What are Transformers, BERT, and Sentence-BERT?

| Name | In simple terms | Why it appears here |
|---|---|---|
| Transformer | A neural-network architecture that uses attention to combine information across tokens | The neural encoders and FLAN-T5 are based on Transformer architectures |
| Attention | A learned calculation that lets a token's representation depend on other relevant tokens | Helps the model represent a word using its context |
| BERT | A Transformer encoder pretrained to represent text using context on both sides | The loaded MiniLM configuration uses a BERT-style encoder; the SPLADE checkpoint is also BERT-based |
| Sentence-BERT, or SBERT | An approach for learning sentence embeddings that can be compared efficiently | Related work underlying sentence-embedding retrieval; our actual checkpoint is the named MiniLM model |
| Encoder-decoder | A model with an encoder that represents the input and a decoder that generates output tokens | FLAN-T5 uses this structure to produce text answers |

Primary references: the [Transformer paper](https://arxiv.org/abs/1706.03762), [BERT paper](https://arxiv.org/abs/1810.04805), and [Sentence-BERT paper](https://arxiv.org/abs/1908.10084). These describe model designs, rather than additional retrieval branches in our final system.

### What do pretrained, fine-tuned, inference, and checkpoint mean?

| Term | Meaning in this project |
|---|---|
| Pretrained | Model weights were learned before our experiment; we download existing MiniLM and SPLADE weights |
| Fine-tuning | Further training that changes an existing model's weights for a task or dataset; we do not fine-tune the neural retrievers here |
| Inference | Apply a trained model to new input, such as encoding a question or predicting its escalation gain |
| Checkpoint / model revision | A saved set of weights and its version; the frozen protocol records the neural-model revisions |
| Parameter | A learned number inside a model; this differs from an input feature such as query length |

### What is FAISS, and what does it do here?

**FAISS** is a library for searching collections of vectors. After MiniLM creates document embeddings, FAISS stores them in an index. For a new query, MiniLM creates its embedding and FAISS finds the highest-scoring document vectors.

```text
Document text -> MiniLM -> document vectors -> FAISS index
Query text    -> MiniLM -> query vector    -> search that index -> ranked documents
```

Our code uses `faiss.IndexFlatIP`, which compares the query with every stored vector using inner products. It performs exact, exhaustive vector search. We normalize query and document vectors to length one, so their inner product equals cosine similarity. See the official [FAISS index descriptions](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) and [similarity documentation](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances).

**Cosine similarity** compares the directions of vectors. A higher value means the query and document embeddings point in more similar directions. MiniLM supplies the learned text representation; FAISS performs the vector search.

### What are kNN, top-k, dot products, and normalization?

| Term | Meaning | Our implementation |
|---|---|---|
| kNN, or k-nearest-neighbor search | Find the k vectors most similar to the query under a chosen comparison | The dense branch returns nearest document embeddings using FAISS |
| Top-k | Keep the highest-ranked k results | The final retrieval output contains 100 documents; ranking metrics may inspect only the first 10 |
| Dot product / inner product | Multiply corresponding vector entries and add the products | Scores dense vectors and SPLADE's sparse representations |
| Vector normalization | Rescale a vector to a chosen length, here length one | Makes the dense branch's inner products equal cosine similarity |
| Approximate nearest-neighbor search, or ANN | Trade exact neighbor recovery for a potentially faster search | Useful background when reading vector-search papers; our `IndexFlatIP` performs exhaustive exact search |

Vector normalization is different from **StandardScaler**, which rescales the gate's feature columns using their training means and standard deviations. Our SPLADE branch uses sparse dot products without the dense branch's cosine-normalization step.

### What are sparse representations and SPLADE?

A sparse vector has many possible positions, but most are zero. SPLADE uses a neural language model to assign weights to vocabulary terms. It can give weight to related terms as well as terms written in the text, which lets it combine learned expansion with term-based search.

Our pretrained model is `naver/splade-cocondenser-ensembledistil`, a learned sparse encoder for passage retrieval. Its [model card](https://huggingface.co/naver/splade-cocondenser-ensembledistil) identifies the model and its dot-product similarity. We use its existing weights rather than fine-tuning it on the evaluation collections.

In our implementation, SPLADE encodes documents into a SciPy sparse matrix. A query receives its own sparse representation, which is scored against that matrix. The dense branch uses FAISS; the SPLADE branch uses sparse-matrix scoring. Their search implementations are visible in `dense_retriever.py` and `splade_retriever.py`.

The words **CoCondenser** and **ensembledistil** identify the initialization and distillation variant of our published SPLADE checkpoint. Distillation transfers information from teacher models during the checkpoint's training. We load one completed checkpoint, rather than running an extra teacher ensemble during retrieval. See the checkpoint's [training-method paper](https://arxiv.org/abs/2205.04733).

### What is BM25?

**BM25** is a word-matching scoring algorithm. It scores documents using how often query words occur, how rare those words are across the corpus, and document length. Rare, matching terms usually count more than common terms. It calculates scores from corpus statistics and a scoring formula.

Our **Best Fields BM25** scores the title and abstract separately, then keeps the higher of the two scores. This lets a strong title match count without requiring the same match to appear in the abstract.

### What are TF, IDF, TF-IDF, postings, and BM25S?

| Name | In simple terms | Status in this project |
|---|---|---|
| Term frequency, or TF | How often a term occurs in a document | One input to BM25 scoring |
| Inverse document frequency, or IDF | A measure of how rare a term is across the corpus | Used by BM25 and several gate features |
| TF-IDF | A classical weighting scheme combining term frequency and rarity | Background for understanding lexical retrieval; it is not a separate final benchmark method |
| Inverted index / postings | A mapping from a term to documents containing it and associated scoring information | Our indexed BM25 precomputes per-term field contributions to avoid repeating the original scoring work |
| BM25S | Existing work on efficient BM25 using eager sparse scoring | Related work cited in the manuscript; our benchmark uses the project's own `IndexedBestFieldsBM25` implementation |

BM25S is described in its [paper](https://arxiv.org/abs/2407.03618). The fact that we implement efficient BM25 does not establish that the general indexing idea is new.

### Which names are models, algorithms, or tools?

The retrieval names describe different parts of the system:

| Name | What kind of thing is it? | Its job in our project |
|---|---|---|
| BM25 | Lexical scoring algorithm | Rank documents from matching words and corpus statistics |
| Dense retrieval | Retrieval approach | Find documents using embedding similarity |
| MiniLM | Pretrained neural model | Create the dense query and document embeddings |
| FAISS | Vector-search library | Search our dense embeddings using `IndexFlatIP` |
| SPLADE | Learned sparse retrieval model | Create weighted term representations for the sparse branch |

The remaining names describe the gate, fusion, and the original paper's setup:

| Name | What kind of thing is it? | Where it fits |
|---|---|---|
| Ridge regression | Learning algorithm that produces a fitted linear model | Our trained gate predicts the gain from running fusion |
| StandardScaler | Feature preprocessing step | Center and scale the gate's input features using training statistics |
| RRF | Ranking-combination algorithm | Combine BM25, Dense, and SPLADE rankings |
| ELSER | Elastic's learned sparse retrieval model | Used in the original paper's system; explained in Section 6 |
| Elasticsearch | Search engine | Runs the original paper's retrieval infrastructure; our reported pipeline runs in Python |

### What software runs those models and algorithms?

| Name | What it does in this implementation |
|---|---|
| Sentence Transformers / Hugging Face Transformers | Load the pretrained dense model and the SPLADE tokenizer/model, respectively |
| PyTorch / CUDA | Execute neural-model calculations; CUDA enables supported work on the NVIDIA GPU |
| scikit-learn | Fit the StandardScaler + Ridge gate and generate cross-validation predictions |
| NumPy / SciPy | Handle numerical arrays and sparse matrices used in scoring |
| rank-bm25 | Supply the original `BM25Okapi` scorer that our compatible indexed implementation preserves |

An **encoder** converts text into a numerical representation. A **tokenizer** splits text into the input pieces a model or scorer expects. Our BM25 tokenizer uses lowercase word tokens, while the neural models use their own tokenizers. These software libraries provide the operations used by the retrieval methods and gate.

Additional tool names appear in the implementation and historical notebook:

| Name | What it does | Where it fits |
|---|---|---|
| Hugging Face Hub | Hosts downloadable model checkpoints, configurations, and tokenizers | Supplies the pretrained models used by our Python libraries |
| ranx | Computes ranking-evaluation metrics | Used in the earlier baseline evaluation scripts |
| Accelerate / `device_map="auto"` | Help place a large neural model across available devices | Appear in the legacy FLAN-T5-XXL loading code |
| FP16 / FP32 | Store floating-point numbers using 16 or 32 bits | The legacy XXL notebook explicitly loads FP16 weights; numerical precision affects memory and computation |
| bitsandbytes / quantization | A library and techniques for lower-bit model storage or computation | The legacy notebook installs bitsandbytes, but its displayed XXL load uses FP16; installation alone does not establish a 4-bit or 8-bit run |

The **CPU** handles general processing, while the **GPU** can accelerate neural calculations. **T4** names the NVIDIA GPU used for the reported Kaggle timings; **VRAM** is memory on a GPU. A hardware name is separate from a model name such as MiniLM or FLAN-T5-XXL.

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

### What are precision, MRR, Exact Match, and answer F1?

| Metric | What it measures | Status here |
|---|---|---|
| Precision@k | The fraction of the first k returned documents judged relevant | A retrieval concept; different from recall's fraction of all judged-relevant documents |
| MRR@k | Average reciprocal rank of the first relevant result, with zero if none occurs in the first k | Appears in earlier baseline scripts; it is different from MAP, which considers multiple relevant results |
| Exact Match, or EM | The fraction of generated answers that exactly match an accepted reference under the evaluation's normalization rules | An answer-generation metric reported by the original paper and historical work, outside our final cascade evaluation |
| Answer token F1 | Combines the precision and recall of answer tokens relative to a reference answer | An answer-generation metric; its recall concerns answer words, rather than retrieved documents |

For answer F1, precision asks how many predicted answer tokens are supported by the reference, and recall asks how many reference tokens the answer recovers. Their harmonic mean gives F1. Keeping the same retrieval retention does not establish that generated answers keep the same EM or F1.

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

### What are Elasticsearch and ELSER?

**Elasticsearch** is a search engine that stores indexed documents and executes search queries. A search engine provides the infrastructure in which retrieval methods run.

**ELSER** stands for **Elastic Learned Sparse EncodeR**. It is Elastic's model for semantic sparse retrieval. It expands text into weighted terms using learned relevance associations, helping retrieve related content beyond literal word matches. The [official ELSER documentation](https://www.elastic.co/docs/explore-analyze/machine-learning/nlp/ml-nlp-elser) explains how Elasticsearch stores and searches its weighted terms.

SPLADE and ELSER both serve the learned sparse retrieval role, but they are different models with different weights and implementations. We use SPLADE in our Python experiments. ELSER belongs to the original paper's setup, so there is no ELSER inference step in our final benchmark. We have not demonstrated that our speedups carry over to the Elasticsearch/ELSER system.

### What is XXL? What are T5 and FLAN-T5-XXL?

In this project context, **XXL means the size variant in `google/flan-t5-xxl`**, the large answer-generation model. Its role is to write an answer using a question and retrieved context.

| Part of the name | Meaning |
|---|---|
| T5 | Text-to-Text Transfer Transformer: an encoder-decoder model that accepts text and generates text |
| FLAN | Identifies the instruction-tuned version, trained on tasks expressed through instructions |
| XXL | The extra-extra-large size variant of this model family |
| 11B parameters | Approximately 11 billion learned model weights; B means billion |

The [T5 paper](https://arxiv.org/abs/1910.10683) explains the text-to-text design, and the [FLAN-T5-XXL model card](https://huggingface.co/google/flan-t5-xxl) describes the instruction-tuned checkpoint and model size.

### How is MiniLM different from FLAN-T5-XXL?

| Model | Input and output | Role | Status |
|---|---|---|---|
| `all-MiniLM-L6-v2` | Text in, 384-number embedding out | Represent queries and documents for dense retrieval | Used in our final evaluated pipeline |
| `flan-t5-xxl` | Instructions/question/context in, generated text out | Write an answer after retrieval | Used in the original paper's RAG evaluation and present in historical project generation work; outside the final cascade evaluation |

```text
Retrieval stage: question -> find documents using BM25 / MiniLM + FAISS / SPLADE
Generation stage: question + retrieved passages -> FLAN-T5-XXL -> written answer
```

Section V of [Blended RAG](https://arxiv.org/html/2404.07220v2) specifies FLAN-T5-XXL for its generation experiments. Our legacy [reproduction notebook](../notebooks/legacy/reproduction.ipynb) contains XXL generation code, and [kaggle_nq_and_xxl_results.json](../cascaded-blended-rag/results/kaggle_nq_and_xxl_results.json) contains a historical summary. Those records do not constitute a matched evaluation of answer quality under our final cascade versus fusion. The current contribution therefore reports retrieval quality and query retrieval time.

### What other model names appear in the original paper's comparisons?

These are context from the paper's comparison tables. They are not methods we ran in the final indexed benchmark.

| Name | What it is | Where it belongs |
|---|---|---|
| monoT5-3B | A T5-based reranker that reads a query/document pair and scores relevance; 3B indicates the model scale | A prior retrieval-ranking comparison; see the [monoT5 paper](https://arxiv.org/abs/2003.06713) |
| COCO-DR Large | A dense retriever trained to cope with distribution shifts using contrastive and distributionally robust learning | A prior retrieval comparison; see the [COCO-DR paper](https://arxiv.org/abs/2210.15212) |
| PaLM 540B | Google's Pathways Language Model at a 540-billion-parameter scale | A prior answer-generation comparison; see the [PaLM paper](https://arxiv.org/abs/2204.02311) |
| GLaM | Generalist Language Model, using a mixture of expert networks with selective activation | A prior answer-generation comparison; see the [GLaM paper](https://arxiv.org/abs/2112.06905) |
| RAG-original / RAG-end2end | RAG pipeline variants compared in the original paper, including a variant further adapted to SQuAD | Published pipeline comparisons, rather than names for our indexed cascade |

A **reranker** scores an initial candidate list after retrieval. Our final system combines rankings with RRF rather than adding a monoT5 reranking stage. **Zero-shot** and **one-shot** in prompting describe using no task examples or one example in the prompt; they do not mean the underlying language model was never trained.

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

### What are the other gate algorithms in the repository?

| Name | How it makes a prediction or decision | Status |
|---|---|---|
| Heuristic / hand-written threshold router | Apply manually chosen rules to clues such as BM25 scores | Early prototype in `cascaded_router.py` |
| Decision tree | Follow learned feature-threshold branches and return a prediction at a leaf | Development candidate in experiment 10 |
| Random Forest | Average predictions from a collection of randomized decision trees | Earlier learned-router attempt and development reference |
| ExtraTrees | Average trees with additional randomness in the candidate split thresholds | Development candidate, including small ensembles |
| Ridge69 / coverage-feature gate | Apply Ridge to an expanded 69-feature input set | Feature-development ablation; the final indexed confirmations use Ridge11 |

The tree methods are explained in [scikit-learn's ensemble guide](https://scikit-learn.org/stable/modules/ensemble.html). The local [model-comparison script](../cascaded-blended-rag/experiments/10_compare_router_models.py) shows which candidates were compared. Their presence is part of the development record, rather than evidence that the final runtime runs all these models.

A **regressor** predicts a number, as Ridge does for gain. A **classifier** predicts a category. Our gain regressor produces a number first; the calibrated cutoff turns it into an early-exit or escalation decision.

An **oracle policy** uses known relevance outcomes to estimate what hindsight routing could achieve. It is an offline diagnostic, not a deployable gate. The **random-routing reference** chooses the same number of exits without the learned gain ranking; its reported recall is an offline comparison, not a separately timed live system.

An **ablation** changes a component to investigate its contribution, such as comparing different feature sets. The original fusion, indexed fusion, and indexed cascade comparisons also separate the observed indexing improvement from the additional routing gain.

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

**What is the difference between Dense, MiniLM, and FAISS?**  Dense is the retrieval approach. MiniLM is our model for turning text into dense vectors. FAISS is the library that searches those vectors.

**Did you use ELSER or SPLADE?**  We used SPLADE. ELSER is the learned sparse model used in the original paper's Elasticsearch setup. They fill a similar role but are different models.

**Did MiniLM or XXL write the answers in the final experiment?**  MiniLM produced retrieval embeddings. The final experiment did not include answer generation. FLAN-T5-XXL belongs to the original paper's generation stage and the project's separate historical generation work.

**Why use RRF?**  The retrievers' raw scores are on different scales. RRF combines their ranks instead.

**What does the gate predict?**  It estimates the recall gain that fusion may add over BM25, using query and BM25-result features.

**How can it predict recall without seeing the correct documents?**  During training, known relevance labels teach it how the features relate to measured recall gain. During live use, it applies that learned relationship without seeing qrels. The estimate can be wrong.

**What does r99 guarantee?**  Nothing about a new dataset. It names a 99% training calibration target. We report measured retention separately.

**Why not headline whichever setting was fastest?**  The main comparison was selected before its outcomes. We report faster secondary settings too, along with their quality losses and uncertainty.

**Why was indexed BM25 necessary?**  BM25 was so slow in the original implementation that it consumed most of the query time. Making the first stage faster gave the cascade room to save time by skipping neural searches.

**Does 7.91x mean the gate alone made SQuAD nearly eight times faster?**  No. That ratio combines indexing and routing against our earlier fusion. The gate's additional gain over already indexed fusion is 1.59x at that sampled secondary point.

**Did the cascade beat fusion?**  The primary indexed cascade was faster on average than equally indexed fusion while retaining most of its Recall@100 in the measured TREC-COVID and SQuAD settings. That does not mean it beat every standalone method.

**Did you reproduce Blended RAG?**  We reproduced the broad idea of combining retrieval methods, but changed the engine and models. Our results are not a direct reproduction of the original system or its speed.

**Did you test answer quality?**  The final cascade study evaluated retrieval rankings. Historical FLAN-T5-XXL generation records are separate; they do not establish preserved answer quality under the final cascade.

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

## 14. If I am asked about an implementation step, where do I find it?

Use this section during questions about the code. Every folder below is relative to the repository root, `C:\Users\user\Desktop\rag-research-alone`. Click the file link, or press **Ctrl+P** in VS Code and type its filename. Use **Ctrl+F** to find the function named in the explanation.

### Preparing the data and running the retrievers

| If I am asked... | Folder | Open this file and show... |
|---|---|---|
| "Where do the datasets come from, and how is the corpus prepared?" | `cascaded-blended-rag/src/` | [paper_datasets.py](../cascaded-blended-rag/src/paper_datasets.py): `prepare_dataset`, `prepare_beir`, and `prepare_squad`. These prepare the declared collections; the SQuAD function builds the paragraph corpus and relevance labels. |
| "Where do you read queries, splits, and relevance judgments?" | `cascaded-blended-rag/experiments/` | [12_calibrate_nfcorpus_train_check_dev.py](../cascaded-blended-rag/experiments/12_calibrate_nfcorpus_train_check_dev.py): `load_corpus`, `load_split`, and `load_qrels`. The final benchmark reuses these helpers. |
| "Where are tokenization and the original BM25/Best Fields scores?" | `cascaded-blended-rag/src/` | [bm25_retriever.py](../cascaded-blended-rag/src/bm25_retriever.py): `simple_tokenize` and `BM25Retriever.retrieve_best_fields`. Show how title and abstract scores are combined. |
| "Where are the dense vectors and FAISS search?" | `cascaded-blended-rag/src/` | [dense_retriever.py](../cascaded-blended-rag/src/dense_retriever.py): `DenseRetriever.__init__` prepares or loads document embeddings and the search index; `retrieve` encodes and searches a query. |
| "Where does SPLADE encode text and retrieve documents?" | `cascaded-blended-rag/src/` | [splade_retriever.py](../cascaded-blended-rag/src/splade_retriever.py): `_encode_documents` prepares sparse document representations; `retrieve` handles a query. |

### Making BM25 faster and teaching the gate

| If I am asked... | Folder | Open this file and show... |
|---|---|---|
| "What code made BM25 faster?" | `cascaded-blended-rag/src/` | [indexed_bm25.py](../cascaded-blended-rag/src/indexed_bm25.py): `FieldPostings` stores term contributions, `IndexedBestFieldsBM25` scores queries, and `stable_top_k` selects results with the declared tie rule. |
| "Where are the 11 gate features calculated?" | `cascaded-blended-rag/experiments/` | [12_calibrate_nfcorpus_train_check_dev.py](../cascaded-blended-rag/experiments/12_calibrate_nfcorpus_train_check_dev.py): `features` calculates the query-length, term-rarity, and BM25-score features. |
| "Where do you train Ridge and produce out-of-fold predictions?" | `cascaded-blended-rag/experiments/` | [16_prepare_retention_tradeoff.py](../cascaded-blended-rag/experiments/16_prepare_retention_tradeoff.py): `model` creates StandardScaler + Ridge; `main` prepares the final out-of-fold predictions and fits the exported model. `nested_curve` belongs to the development comparison. |
| "Where are the four retention thresholds calibrated?" | `cascaded-blended-rag/src/` | [retention_tradeoff.py](../cascaded-blended-rag/src/retention_tradeoff.py): `calibrate_targets` applies the clipped training-loss budgets. It calls `select_threshold` in [router_diagnostics.py](../cascaded-blended-rag/src/router_diagnostics.py). |
| "Where are the actual frozen weights and r90/r92/r95/r99 cutoffs?" | `cascaded-blended-rag/results/` | [nfcorpus_frozen_retention_routers.json](../cascaded-blended-rag/results/nfcorpus_frozen_retention_routers.json): inspect `routers`, then `ridge11`, its coefficients/scaling values, and `policies`. The final confirmation uses `ridge11`; other saved families are development artifacts. |

### Following one live query through the system

| If I am asked... | Folder | Open this file and show... |
|---|---|---|
| "Where does the gate predict a gain and choose whether to stop?" | `cascaded-blended-rag/src/` | [retention_tradeoff.py](../cascaded-blended-rag/src/retention_tradeoff.py): `predict_gain` performs the fitted Ridge calculation; `ConditionalCascade.run` returns BM25 when the prediction is at or below the threshold. |
| "Where do static fusion and cascade run different branches?" | `cascaded-blended-rag/src/` | [retention_tradeoff.py](../cascaded-blended-rag/src/retention_tradeoff.py): `ConditionalCascade.run` handles BM25/Dense/SPLADE baselines, static fusion, and the cascade. Show the early return and the later neural calls; escalated queries reuse their BM25 ranking. |
| "Where are the rankings combined with RRF?" | `cascaded-blended-rag/experiments/` | [12_calibrate_nfcorpus_train_check_dev.py](../cascaded-blended-rag/experiments/12_calibrate_nfcorpus_train_check_dev.py): `fuse` implements the rank-based combination. [25_benchmark_indexed_cascade.py](../cascaded-blended-rag/experiments/25_benchmark_indexed_cascade.py) passes this helper into the final engines. |
| "Where is an identical query/document ID excluded?" | `cascaded-blended-rag/src/` | [query_self_exclusion.py](../cascaded-blended-rag/src/query_self_exclusion.py): the BM25 and neural wrappers apply the declared self-match rule. The reported confirmations had no actual ID overlap. |
| "Where are the fixed query sample and benchmark configuration enforced?" | `cascaded-blended-rag/experiments/` and `cascaded-blended-rag/results/` | [25_benchmark_indexed_cascade.py](../cascaded-blended-rag/experiments/25_benchmark_indexed_cascade.py): `benchmark` checks the frozen configuration and selects limited queries by SHA256 ordering. [indexed_cascade_protocol.json](../cascaded-blended-rag/results/indexed_cascade_protocol.json) records the declared settings and fingerprints. |

### Measuring performance and checking the evidence

| If I am asked... | Folder | Open this file and show... |
|---|---|---|
| "Where are warmups, repeated timings, and GPU synchronization?" | `cascaded-blended-rag/experiments/` | [25_benchmark_indexed_cascade.py](../cascaded-blended-rag/experiments/25_benchmark_indexed_cascade.py): `benchmark` prepares the engines, warms them up, randomizes query/method order, and times each retrieval with synchronization. It also records neural branch counts. |
| "Where do you calculate recall, ranking metrics, retention, and uncertainty?" | `cascaded-blended-rag/experiments/` and `cascaded-blended-rag/src/` | [12_calibrate_nfcorpus_train_check_dev.py](../cascaded-blended-rag/experiments/12_calibrate_nfcorpus_train_check_dev.py): `recall` and `score_run`. [25_benchmark_indexed_cascade.py](../cascaded-blended-rag/experiments/25_benchmark_indexed_cascade.py) aggregates method ratios; [tradeoff_analysis.py](../cascaded-blended-rag/src/tradeoff_analysis.py) contains `latency_summary` and `paired_intervals`. |
| "Where are the final numbers I can inspect?" | `cascaded-blended-rag/results/indexed_trec-covid_q0/` and `cascaded-blended-rag/results/indexed_squad_q256/` | Open the [TREC benchmark report](../cascaded-blended-rag/results/indexed_trec-covid_q0/indexed_cascade_benchmark.json) or [SQuAD benchmark report](../cascaded-blended-rag/results/indexed_squad_q256/indexed_cascade_benchmark.json). Each folder also has `quality_speed.csv` and verification records. |
| "How did you check that the retrieval metrics were scored correctly?" | `cascaded-blended-rag/experiments/` | [29_verify_squad_metrics.py](../cascaded-blended-rag/experiments/29_verify_squad_metrics.py) independently scores SQuAD rankings; [31_verify_beir_metrics.py](../cascaded-blended-rag/experiments/31_verify_beir_metrics.py) checks BEIR metrics against raw labels. Saved findings are in each indexed result folder's `independent_metric_verification.json`. |
| "How did you check that indexing preserved rankings and scores?" | `cascaded-blended-rag/experiments/` | [30_compare_indexed_execution.py](../cascaded-blended-rag/experiments/30_compare_indexed_execution.py) compares saved original/indexed outputs. [24_benchmark_indexed_bm25_cpu.py](../cascaded-blended-rag/experiments/24_benchmark_indexed_bm25_cpu.py) checks the scorer on CPU. Read `paired_backend_comparison.json` in the indexed result folders for the saved comparison. |

### Finding Kaggle workflows, caches, and earlier results

| If I am asked... | Folder | Open this file and show... |
|---|---|---|
| "What was uploaded and run in Kaggle?" | `notebooks/kaggle_fast_finish_v1/` | [START_HERE.md](../notebooks/kaggle_fast_finish_v1/START_HERE.md) explains the final workflow. `PASTE_CELL.py`, `RUN_IN_KAGGLE.py`, and `RUN_TREC_INDEXED.py` contain the notebook-side steps. The [notebook index](../notebooks/README.md) distinguishes older workflows. |
| "How do you prepare a source ZIP for those runs?" | `cascaded-blended-rag/experiments/` | [26_package_indexed_cascade.py](../cascaded-blended-rag/experiments/26_package_indexed_cascade.py) builds the indexed package; [28_package_fast_finish.py](../cascaded-blended-rag/experiments/28_package_fast_finish.py) builds the combined workflow package. |
| "Where do downloaded datasets, indexes, and model caches live?" | `cascaded-blended-rag/data/` | Start with [data/README.md](../cascaded-blended-rag/data/README.md): raw inputs use `raw/`, and generated indexes use `processed*` folders. [cache_utils.py](../cascaded-blended-rag/src/cache_utils.py) checks corpus fingerprints and cache metadata. Large local inputs are excluded from the sharing ZIP. |
| "Where are the earlier fusion benchmark and the one-pass SQuAD procedure?" | `cascaded-blended-rag/experiments/` | [20_benchmark_paper_datasets.py](../cascaded-blended-rag/experiments/20_benchmark_paper_datasets.py) is the original-backend benchmark. [27_finalize_first_completed_pass.py](../cascaded-blended-rag/experiments/27_finalize_first_completed_pass.py) finalizes the completed original pass without claiming three complete passes. |
| "Where do you import downloaded outputs and document reproduction?" | `cascaded-blended-rag/experiments/` and `docs/reproducibility/` | [22_import_paper_benchmark.py](../cascaded-blended-rag/experiments/22_import_paper_benchmark.py) imports matching result exports. The [reproduction guide](reproducibility/README.md) explains local artifact restoration and the checks; original downloaded ZIPs are retained locally under `artifacts/downloads/`. |

### Finding figures, written results, and submission materials

| If I am asked... | Folder | Open this file and show... |
|---|---|---|
| "Where are the speedup charts and their underlying values made?" | `cascaded-blended-rag/experiments/`, `tools/`, and `docs/figures/` | [32_prepare_report_assets.py](../cascaded-blended-rag/experiments/32_prepare_report_assets.py) generates figures/tables from saved reports. [build_publication_figures.py](../tools/build_publication_figures.py) makes publication assets. [all_measured_speedups.csv](figures/all_measured_speedups.csv) contains the recorded comparisons. |
| "Where is the paper source and the explanation of the results?" | `docs/` and `docs/findings/` | [FINAL_MANUSCRIPT.md](FINAL_MANUSCRIPT.md) is the manuscript; [FINAL_RESULTS.md](FINAL_RESULTS.md) summarizes results. The [TREC findings](findings/TREC_COVID_INDEXED_FINDINGS.md) and [SQuAD findings](findings/SQUAD_INDEXED_FINDINGS.md) explain evidence and caveats. |
| "Where do the HTML, PDFs, and presentation come from?" | `tools/` and `final_results/` | [build_research_materials.py](../tools/build_research_materials.py) builds paper/HTML material using [showcase.template.html](../tools/showcase.template.html). [build_research_presentation.mjs](../tools/build_research_presentation.mjs) authors slides. The [final-results guide](../final_results/README.md) links deliverables and rebuild instructions. |
| "Where is the editable LaTeX paper for Overleaf?" | `final_results/overleaf_ieee/` | [main.tex](../final_results/overleaf_ieee/main.tex) is the LaTeX source; the adjacent [README.md](../final_results/overleaf_ieee/README.md) gives usage instructions. |
| "How is the repository checked and packaged for sharing?" | `tools/` | [check_repository.py](../tools/check_repository.py) audits sharing hygiene and frozen fingerprints. [build_submission.py](../tools/build_submission.py) packages shareable files. [write_release_manifest.py](../tools/write_release_manifest.py) records final-material fingerprints. |

### Which files should I show first?

For a walkthrough of the final system, open [25_benchmark_indexed_cascade.py](../cascaded-blended-rag/experiments/25_benchmark_indexed_cascade.py) and find `benchmark`. It connects dataset preparation, indexed BM25, the frozen gate, neural retrieval, timing, and evaluation. Follow its calls into `src/` when asked how an individual step works.

The earlier [cascaded_router.py](../cascaded-blended-rag/src/cascaded_router.py) is a heuristic prototype, not the final Ridge gate. Historical loader scripts and cached exploratory experiments are also different from the final benchmark. Use the files above when explaining the reported implementation. Frozen scripts and protocol files should be read as evidence, not edited during a demonstration.

If asked about XXL, open [notebooks/legacy/reproduction.ipynb](../notebooks/legacy/reproduction.ipynb) and find `google/flan-t5-xxl`. If asked about MiniLM, open [src/dense_retriever.py](../cascaded-blended-rag/src/dense_retriever.py) and find `SentenceTransformer`. State which experiment each file belongs to: the former is historical generation code, while the latter supplies the dense branch of the final evaluated retrieval pipeline.
