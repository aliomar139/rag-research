from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'final_results' / 'research_showcase_4page.pdf'
FONTS = ROOT / 'final_results' / 'assets' / 'fonts'
for name, file in [('Plex', 'IBMPlexSans-Regular.ttf'), ('PlexSemi', 'IBMPlexSans-SemiBold.ttf'), ('Newsreader', 'Newsreader-600.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONTS / file)))

INK = colors.HexColor('#252B29')
MUTED = colors.HexColor('#53605C')
TEAL = colors.HexColor('#176C70')
RED = colors.HexColor('#B84934')
PAPER = colors.HexColor('#F7F4ED')
LINE = colors.HexColor('#D8D1C6')
PALE = colors.HexColor('#E9EFEB')
WHITE = colors.white

STYLES = {
    'eyebrow': ParagraphStyle('eyebrow', fontName='PlexSemi', fontSize=8.3, leading=11, textColor=RED, spaceAfter=7),
    'title': ParagraphStyle('title', fontName='Newsreader', fontSize=28, leading=31, textColor=INK, spaceAfter=5),
    'subtitle': ParagraphStyle('subtitle', fontName='Plex', fontSize=12.2, leading=16, textColor=MUTED, spaceAfter=10),
    'byline': ParagraphStyle('byline', fontName='PlexSemi', fontSize=8.8, leading=12, textColor=TEAL, spaceAfter=13),
    'h1': ParagraphStyle('h1', fontName='Newsreader', fontSize=20, leading=24, textColor=INK, spaceBefore=1, spaceAfter=8),
    'h2': ParagraphStyle('h2', fontName='PlexSemi', fontSize=11.2, leading=14, textColor=TEAL, spaceBefore=5, spaceAfter=5),
    'body': ParagraphStyle('body', fontName='Plex', fontSize=9.45, leading=13.4, textColor=INK, spaceAfter=6),
    'small': ParagraphStyle('small', fontName='Plex', fontSize=8.05, leading=10.5, textColor=MUTED),
    'tablehead': ParagraphStyle('tablehead', fontName='PlexSemi', fontSize=8, leading=10, textColor=WHITE),
    'cell': ParagraphStyle('cell', fontName='Plex', fontSize=8.1, leading=10.5, textColor=INK),
    'cellbold': ParagraphStyle('cellbold', fontName='PlexSemi', fontSize=8.1, leading=10.5, textColor=INK),
    'callout': ParagraphStyle('callout', fontName='Plex', fontSize=9.1, leading=12.4, textColor=INK),
    'metric': ParagraphStyle('metric', fontName='Newsreader', fontSize=21, leading=23, textColor=TEAL),
    'metriclabel': ParagraphStyle('metriclabel', fontName='PlexSemi', fontSize=8.6, leading=11, textColor=INK),
    'metricnote': ParagraphStyle('metricnote', fontName='Plex', fontSize=8, leading=10.2, textColor=MUTED),
    'metricsecondary': ParagraphStyle('metricsecondary', fontName='Newsreader', fontSize=16, leading=19, textColor=TEAL, spaceBefore=3),
    'tinyred': ParagraphStyle('tinyred', fontName='PlexSemi', fontSize=7.8, leading=10, textColor=RED),
    'ref': ParagraphStyle('ref', fontName='Plex', fontSize=7.15, leading=9.1, textColor=MUTED),
    'caption': ParagraphStyle('caption', fontName='Plex', fontSize=7.4, leading=9, textColor=MUTED, alignment=1),
}

def P(text, style='body'):
    return Paragraph(text, STYLES[style])

def box(text):
    table = Table([[P(text, 'callout')]], colWidths=[170 * mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PALE), ('BOX', (0, 0), (-1, -1), .5, LINE),
        ('LINEBEFORE', (0, 0), (0, -1), 3, TEAL), ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10), ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    return table

def grid(data, widths, header=True):
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign='LEFT')
    commands = [('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 7),
                ('RIGHTPADDING', (0, 0), (-1, -1), 7), ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6), ('LINEBELOW', (0, 0), (-1, -1), .45, LINE)]
    if header:
        commands.append(('BACKGROUND', (0, 0), (-1, 0), TEAL))
    for row in range(1 if header else 0, len(data)):
        if row % 2 == 0:
            commands.append(('BACKGROUND', (0, row), (-1, row), colors.HexColor('#F0ECE4')))
    table.setStyle(TableStyle(commands))
    return table

def metric_card(tag, first_value, first_title, retention, second_value, second_note):
    data = [[P(tag, 'tinyred')], [P(first_value, 'metric')], [P(first_title, 'metriclabel')],
            [P(retention, 'metricnote')], [P(second_value, 'metricsecondary')], [P(second_note, 'metricnote')]]
    table = Table(data, colWidths=[82 * mm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PALE), ('BOX', (0, 0), (-1, -1), .5, LINE),
        ('LINEBEFORE', (0, 0), (0, -1), 3, TEAL), ('LEFTPADDING', (0, 0), (-1, -1), 9),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8), ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return table

def page_art(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, width, height, stroke=0, fill=1)
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(.6)
    canvas.line(20 * mm, height - 17 * mm, width - 20 * mm, height - 17 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont('PlexSemi', 7.8)
    canvas.drawString(20 * mm, height - 13 * mm, 'LEBANESE UNIVERSITY  /  RESEARCH PROJECT')
    canvas.setFont('Plex', 7.8)
    canvas.drawRightString(width - 20 * mm, height - 13 * mm, 'CONDITIONAL HYBRID RETRIEVAL')
    canvas.line(20 * mm, 15 * mm, width - 20 * mm, 15 * mm)
    canvas.setFont('Plex', 7.5)
    canvas.drawString(20 * mm, 10 * mm, 'Four-section research showcase')
    canvas.drawRightString(width - 20 * mm, 10 * mm, f'{doc.page} / 4')
    canvas.restoreState()

story = []
# 1. Base paper
story += [Spacer(1, 4 * mm), P('RESEARCH SHOWCASE  /  FOUR-PAGE SUMMARY', 'eyebrow'),
          P('Conditional hybrid retrieval', 'title'),
          P('The selected paper, our adaptation, the measured changes, and what remains', 'subtitle'),
          P('Ali Omar  /  Abdallah Mestrah  /  Abbas Yazbek  |  Supervisor: Dr. Ahmad Abboud', 'byline'),
          P('1. Base paper', 'h1')]
story += [P("The selected paper, <i>Blended RAG</i> by Sawarkar, Mangal, and Solanki, combines keyword retrieval, dense vectors, and learned sparse retrieval in Elasticsearch. It uses ELSER for the learned sparse branch and evaluates both retrieval rankings and generated-answer quality [1]. The paper motivates our question: must every query run every retrieval branch?"),
          box('<b>Our question:</b> Can a BM25-first system skip neural retrieval on some queries while keeping recall close to a blended reference?'), Spacer(1, 4 * mm)]
base_rows = [
    [P('Dimension', 'tablehead'), P('Blended RAG paper', 'tablehead'), P('Our project', 'tablehead')],
    [P('Search stack', 'cellbold'), P('Elasticsearch; keyword, dense, and ELSER retrieval', 'cell'), P('Python BM25, MiniLM, SPLADE, reciprocal rank fusion', 'cell')],
    [P('Research focus', 'cellbold'), P('Hybrid retrieval and answer quality', 'cell'), P('Query cost under recall-retention targets', 'cell')],
    [P('Quality measures', 'cellbold'), P('Top-k, nDCG@10, generated-answer EM/F1', 'cell'), P('Recall@100 and retention, nDCG@10, MAP@10, latency', 'cell')],
    [P('Comparison', 'cellbold'), P('Published paper results', 'cell'), P('Within-project tests; no direct original-system timing', 'cell')],
]
story += [grid(base_rows, [34 * mm, 66 * mm, 70 * mm]), Spacer(1, 5 * mm),
          P("The project studies the execution cost of conditional retrieval in our implementation. The published values provide context, but they are not a matched baseline for our experiments.")]

# 2. Reproduction
story += [PageBreak(), Spacer(1, 4 * mm), P('RECREATION AND ADAPTATION', 'eyebrow'),
          P('2. A related recipe, implemented differently', 'h1'),
          P('We recreated the broad retrieval setup on overlapping datasets, then tested a BM25-first cascade. This is a methodological reproduction, not an exact reproduction of the original Elasticsearch and ELSER system.')]
repro_rows = [
    [P('What carries over', 'tablehead'), P('What changes in our study', 'tablehead')],
    [P('Combining lexical, dense, and learned-sparse retrieval signals.', 'cell'), P('Python backend, MiniLM, SPLADE, reciprocal rank fusion, and Best Fields BM25 replace the original stack.', 'cell')],
    [P('TREC-COVID and SQuAD provide task overlap with the paper.', 'cell'), P('SQuAD uses paragraph documents from official data; indexed follow-up tests a fixed sample of 256 questions.', 'cell')],
    [P('A blended retrieval result provides a within-project reference.', 'cell'), P('We report Recall@100, recall retention, ranking measures, and latency; generated-answer EM/F1 is not part of the completed evaluation.', 'cell')],
]
story += [grid(repro_rows, [84 * mm, 86 * mm]), Spacer(1, 5 * mm), P('The first cascade exposed a bottleneck', 'h2'),
          P('In the initial TREC-COVID run, BM25 took 1,410.14 ms out of 1,493.23 ms for full fusion. Nearly all measured time sat in the lexical first stage, so skipping later neural work could save little. This result motivated indexed BM25 before we assessed conditional routing.'),
          box("<b>Metric note:</b> Recall retention is the method's mean Recall@100 divided by the fusion reference's mean Recall@100. It does not measure result-list overlap, and it is not generated-answer accuracy."),
          Spacer(1, 4 * mm), P('The indexed SQuAD result covers a declared 256-question sample against the full corpus. It is not a full-query evaluation.', 'small')]

# 3. Contribution and implementation
story += [PageBreak(), Spacer(1, 4 * mm), P('CONTRIBUTION AND IMPLEMENTATION', 'eyebrow'),
          P('3. Index the lexical stage, then decide whether to escalate', 'h1'),
          P('Our engineering contribution is an indexed BM25 implementation applied to fusion and cascade, plus a frozen BM25-first gate. We do not claim a new retrieval algorithm.'),
          P('At inference, BM25 runs first. Eleven features from the query and BM25 ranking enter a standardized Ridge regression model trained on 2,590 NFCorpus training queries. It predicts the recall gain expected from neural retrieval. Low predicted gain returns BM25; otherwise MiniLM and SPLADE run, then RRF combines them with the BM25 ranking already available. The gate does not use relevance labels at inference.')]
pipeline = Image(str(ROOT / 'final_results' / 'assets' / 'pipeline.png'), width=170 * mm, height=51.7 * mm)
pipeline.hAlign = 'CENTER'
story += [pipeline, P('Conditional path in the indexed cascade. Corpus and index preparation happen before query timing.', 'caption'), Spacer(1, 3 * mm)]
trec_card = metric_card('TREC-COVID / ALL 50 QUERIES', '1.24x', 'Routing gain over indexed fusion',
                        '96.68% of indexed fusion mean Recall@100 retained.', '20.21x combined',
                        'vs earlier Python fusion; includes indexing and routing. Indexed fusion alone was 16.36x.')
squad_card = metric_card('SQUAD / FIXED SAMPLE OF 256 QUESTIONS', '1.59x', 'Routing gain over indexed fusion',
                         '99.22% retained on the declared sample.', '7.91x combined',
                         'vs earlier Python fusion; includes indexing and routing on the same sample.')
mt = Table([[trec_card, squad_card]], colWidths=[85 * mm, 85 * mm])
mt.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 0),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 0), ('TOPPADDING', (0, 0), (-1, -1), 3),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]))
story += [mt, Spacer(1, 2 * mm), P('The routing-only comparison uses equally indexed fusion. The combined ratios compare separate runs and must not be read as speedups against the original paper system.', 'small')]

# 4. Future work and references
story += [PageBreak(), Spacer(1, 4 * mm), P('FUTURE WORK / REFERENCES', 'eyebrow'),
          P('4. What we learned, and what remains', 'h1'),
          P('The measurements show an engineering gain in our Python setup, not that the cascade is the best retrieval choice in every setting. Standalone SPLADE is a strong speed-recall option on TREC-COVID; on sampled SQuAD, standalone BM25 and SPLADE also provide competitive choices.'),
          P('Limits of the current evidence', 'h2')]
for item in [
    '<b>System transfer:</b> we did not evaluate the original Elasticsearch and ELSER pipeline.',
    '<b>Evaluation scope:</b> indexed SQuAD uses 256 queries; full NQ, HotpotQA, CoQA, and generated-answer quality remain pending.',
    '<b>Selection and hardware:</b> indexed TREC quality had already been inspected when its primary setting was selected; timings use one GPU type and few sessions.',
    '<b>Practical costs:</b> production concurrency, energy, index memory, and update costs were not measured.',
]:
    story.append(P('&bull; ' + item))
story += [P('Next study', 'h2'),
          P('Implement the cascade in Elasticsearch with ELSER and matched queries, corpus, and relevance judgments. Predeclare the protocol, test held-out collections, report generated-answer EM/F1, compare strong standalone and adaptive baselines, and measure repeated timing, load, memory, and index-update costs.'),
          box('<b>Summary:</b> Indexed BM25 addressed the lexical bottleneck in our implementation, and routing added a measured gain over equally indexed fusion. Testing whether those gains transfer to the original system is the next step.'),
          Spacer(1, 4 * mm), P('Selected references', 'h2')]
references = [
    '[1] Sawarkar, Mangal, and Solanki (2024). <i>Blended RAG: Improving RAG Accuracy with Semantic Search and Hybrid Query-Based Retrievers.</i> arXiv:2404.07220.',
    '[2] Arabzadeh, Yan, and Clarke (2021). <i>Predicting Efficiency/Effectiveness Trade-offs for Dense vs. Sparse Retrieval Strategy Selection.</i> arXiv:2109.10739.',
    '[3] Lu (2024). <i>BM25S: Orders of Magnitude Faster Lexical Search via Eager Sparse Scoring.</i> arXiv:2407.03618.',
    '[4] Thakur et al. (2021). <i>BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.</i> NeurIPS Datasets and Benchmarks.',
    '[5] Rajpurkar et al. (2016). <i>SQuAD: 100,000+ Questions for Machine Comprehension of Text.</i> EMNLP.',
    '[6] Formal, Piwowarski, and Clinchant (2021). <i>SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking.</i> SIGIR.',
    '[7] Reimers and Gurevych (2019). <i>Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.</i> EMNLP-IJCNLP.',
]
for ref in references:
    story.append(P(ref, 'ref'))

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=23 * mm, bottomMargin=20 * mm,
                        title='Conditional Hybrid Retrieval - Four-Page Research Showcase',
                        author='Ali Omar, Abdallah Mestrah, Abbas Yazbek',
                        subject='Four-section research showcase: paper, reproduction, contribution, and future work')
doc.build(story, onFirstPage=page_art, onLaterPages=page_art)
print(f'Created {OUT}')
