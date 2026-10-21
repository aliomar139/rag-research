# Spoken script for the 10-slide HTML showcase

This script follows the ten slides in presentation mode. The three speakers have roughly equal speaking time: Ali covers slides 1–3, Abdallah slides 4–6, and Abbas slides 7–10. Use each numbered heading to stay in sync with the right-arrow navigation.

## Speaker 1 — Ali Omar

### Slide 1 — Conditional hybrid retrieval

Good morning. We are Ali Omar, Abdallah Mestrah, and Abbas Yazbek from Lebanese University, supervised by Dr. Ahmad Abboud. Our project studies a practical question: can we avoid running every retrieval method on every query and reduce response time while keeping most of the retrieval quality?

### Slide 2 — The paper that motivates this project

We chose a paper called Blended RAG. It combines three ways of searching: keyword search, dense search, and learned sparse search. Each method can find useful documents that the others may miss. The paper uses Elasticsearch and ELSER, and measures both search results and generated answers.

The scores on this slide come from that paper. They are background information, not our results. Our project asks a different question: do we need to run all three search methods for every question?

We tested TREC-COVID and SQuAD. We used NFCorpus to train and develop our gate, and SCIDOCS to check how it works on another dataset.

### Slide 3 — Same idea, different system

We followed the same general idea, but used different tools. Our system uses Python, BM25, MiniLM, and SPLADE. We combine their ranked results using a method called reciprocal rank fusion. We also added a gate that decides when the extra search methods are needed.

Because our tools and measurements are different, we cannot directly compare our scores with the paper’s scores. We measured search quality and search time. We did not measure the quality of generated answers.

One important measure is recall retention. It compares our method’s average recall with the average recall of full fusion, which runs all three methods. For example, 96.68 percent retention means keeping 96.68 percent of fusion’s average recall. It does not mean finding 96.68 percent of all relevant documents.

For SQuAD, we used 20,958 documents. Each question’s original paragraph is treated as the relevant document. Other paragraphs that might also contain an answer were not judged. Abdallah will now explain our first results.

## Speaker 2 — Abdallah Mestrah

### Slide 4 — The initial cascade had little time to save

At first, our cascade saved very little time. On TREC-COVID, it reduced average search time from about 1,493 to 1,440 milliseconds. That is only about 3.6 percent less time. It also kept only 84.53 percent of fusion’s recall, below our 90 percent minimum.

On the full SQuAD test, the first cascade saved about 6.1 percent of the time and kept 99.38 percent of fusion’s recall.

The main problem was BM25. On TREC-COVID, BM25 alone took about 1,410 milliseconds out of the total 1,493. Since we always run BM25 first, skipping the other methods could not save much time.

We also found that a setting chosen on one dataset did not always meet its recall target on another. This meant we needed to check both speed and quality carefully.

### Slide 5 — First fix the lexical bottleneck

Our first step was to make BM25 faster. We built an index that stores useful information in advance, so BM25 does less repeated work for each question.

In the TREC follow-up, BM25 took about 8 milliseconds instead of about 1,410. Our checks showed that the saved search results stayed the same.

We used this faster BM25 in every method we compared. This makes the comparison fair: both full fusion and the cascade benefit from the same improvement.

The old and new timings came from separate runs, so some variation is possible. The main point is that fixing the slow first step gave our gate more room to save time.

### Slide 6 — BM25 first, then a learned gate

Here is how the system works. First, every question runs through BM25. Then a small prediction model looks at eleven pieces of information about the question and its search results.

The model estimates whether Dense and SPLADE are likely to improve recall. If the expected gain is small, we return the BM25 results. If it is larger, we run both extra methods and combine all three result lists.

We trained the model on 2,590 NFCorpus questions and chose four decision settings using training data. During a real search, the gate does not know the correct answers or relevance labels.

We kept the same trained model and decision settings when testing the faster BM25. Checks on the saved TREC and SQuAD records showed matching rankings, scores, decisions, and recall across the old and new versions.

Our timing includes processing the question, searching, and making the gate decision. It excludes building indexes, loading models, downloading data, and generating answers. Abbas will now explain the results.

## Speaker 3 — Abbas Yazbek

### Slide 7 — TREC-COVID: indexing helps, routing adds more

We tested all 50 TREC-COVID questions against about 171,000 documents. We repeated the timing three times on a T4 GPU.

With faster BM25, full fusion took about 91 milliseconds per question. Our main cascade setting, called r99, took about 74 milliseconds. That is about 19 percent less time, or a 1.24-times speedup. It kept 96.68 percent of fusion’s average recall.

There are two comparisons on this slide. Against equally indexed fusion, the gain is 1.24 times. Against our earlier, slower fusion, the combined gain is 20.21 times. That larger number includes both faster BM25 and the gate. It is not a comparison with the original paper’s system.

Faster settings lose more recall. The r95 setting kept 91.95 percent on average, but its uncertainty range went below our 90 percent minimum. The fastest setting reached 36.96 times the speed of our earlier fusion, but kept only 84.53 percent of recall. It therefore failed the minimum.

### Slide 8 — SQuAD: a declared query sample

For this follow-up, we tested 256 fixed SQuAD questions against all 20,958 documents. We repeated the timing three times. The comparison with the earlier version uses those same questions.

Full fusion took about 30 milliseconds per question. The fastest cascade setting, r90, took about 19 milliseconds and kept 99.22 percent of fusion’s average recall. That is about 37 percent less time, or a 1.59-times speedup over equally indexed fusion.

Against our earlier fusion, the speedup is 7.91 times. Again, this includes both indexing and the gate. The other cascade settings kept all of fusion’s measured recall on this sample, but saved less time.

These results apply to the 256 tested questions. They do not prove the same result for every SQuAD question. Also, a lower average time does not mean every search is faster. The slowest searches did not improve in the same way.

### Slide 9 — The cascade is not always the best choice

We also tested each search method on its own. This is important because full fusion is not always the best starting point for comparison.

On TREC-COVID, SPLADE alone was faster than our main cascade and found more relevant documents than fusion. The cascade did better on MAP, one measure of ranking quality. So the best choice depends on which quality measure matters.

On the SQuAD sample, BM25 alone took about one millisecond and kept 95.70 percent of fusion’s recall. That already passed our 90 percent minimum. SPLADE had the strongest ranking scores on this sample.

Our earlier tests also showed that settings trained on NFCorpus could miss their recall targets on other datasets. The lesson is simple: the cascade can help, but it is not always the fastest or best option. We must compare it with strong single methods too.

### Slide 10 — What the evidence supports

Our main finding is that we had to fix the slow BM25 step first. After that, the gate could save more average search time by skipping extra work on some questions.

On TREC-COVID, our main result was a combined speedup of 20.21 times over our earlier fusion, while keeping 96.68 percent of fusion’s recall. Only 1.24 times of that comparison comes from routing against equally indexed fusion. The larger speedup includes indexing too, across separate runs.

The fastest setting lost too much recall. So our goal is not simply to choose the biggest speedup. We need a useful balance between speed and search quality.

There is still more to test. We need more datasets, generated-answer quality, and comparisons with other adaptive methods. We also need to test many users at once and measure energy use, index size, and update costs.

These results describe our Python system. More testing is needed to know how well they carry over to other systems and datasets. The references on this slide show the paper and supporting research. Thank you.
