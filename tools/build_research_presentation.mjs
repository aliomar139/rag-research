/** Build an editable, evidence-based deck with the bundled Artifact Tool. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const runtime=process.env.RUNTIME_NODE_MODULES ?? process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES
  ?? path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');
process.env.RUNTIME_NODE_MODULES=runtime;
const runtimePython=process.env.RUNTIME_PYTHON ?? path.resolve(runtime,'../../python/python.exe');
const skill=process.env.PRESENTATIONS_SKILL_DIR ?? path.join(process.env.USERPROFILE,'.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations');
const require=createRequire(path.join(runtime,'__runtime__.cjs'));
const artifactEntry=require.resolve('@oai/artifact-tool');
const artifactRequire=createRequire(artifactEntry);
const {FontLibrary}=artifactRequire('skia-canvas');
const fontDir=path.join(root,'docs/submission/assets/fonts');
FontLibrary.use('Newsreader',[path.join(fontDir,'Newsreader-400.ttf'),path.join(fontDir,'Newsreader-600.ttf')]);
FontLibrary.use('IBM Plex Sans',[path.join(fontDir,'IBMPlexSans-Regular.ttf'),path.join(fontDir,'IBMPlexSans-SemiBold.ttf')]);
try {
  const {GlobalFonts}=require('@napi-rs/canvas');
  for(const [name,family] of [['Newsreader-400.ttf','Newsreader'],['Newsreader-600.ttf','Newsreader'],
    ['IBMPlexSans-Regular.ttf','IBM Plex Sans'],['IBMPlexSans-SemiBold.ttf','IBM Plex Sans']])
    GlobalFonts.registerFromPath(path.join(fontDir,name),family);
} catch {}
const {Presentation,PresentationFile,FileBlob}=await import(pathToFileURL(artifactEntry).href);
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const data=JSON.parse(await fs.readFile(path.join(root,'docs/submission/assets/results.json'),'utf8'));
const css=await fs.readFile(path.join(root,'tokens.css'),'utf8');
const C=Object.fromEntries([...css.matchAll(/--color-([\w-]+):\s*(#[0-9a-fA-F]{6})/g)].map(x=>[x[1],x[2]]));
const out=path.join(root,'docs/submission');
const build=path.join(root,'.local/release_build/presentation');
await fs.mkdir(build,{recursive:true});
if(process.argv.includes('--render-existing')){
  const existing=await PresentationFile.importPptx(await FileBlob.load(path.join(out,'research_presentation.pptx')));
  for(let i=0;i<existing.slides.items.length;i++){
    const image=await existing.export({slide:existing.slides.items[i],format:'png',scale:1.25});
    await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await image.arrayBuffer()));
  }
  console.log('Rendered all',existing.slides.items.length,'slides from final PPTX.');
  process.exit(0);
}
const presentation=Presentation.create({slideSize:{width:1280,height:720}});
const notes=[]; const tableOwners=[]; const chartOwners=[];
const trec=data.datasets['trec-covid'].methods, squad=data.datasets.squad.methods;
const refs={paper:'https://arxiv.org/html/2404.07220v2',strategy:'https://arxiv.org/abs/2109.10739',
  bm25s:'https://arxiv.org/abs/2407.03618',beir:'https://arxiv.org/abs/2104.08663',
  squad:'https://arxiv.org/abs/1606.05250',splade:'https://arxiv.org/abs/2107.05720',sbert:'https://arxiv.org/abs/1908.10084'};
const measured='Project evidence: docs/submission/assets/results.json and docs/figures/all_measured_speedups.csv. Underlying original reports and SHA256 fingerprints are in results.json.';

function text(slide,value,x,y,w,h,size=28,{serif=false,bold=false,color=C.ink,align='left'}={}){
  const shape=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},
    fill:'none',line:{fill:'none',width:0}});
  shape.text=value;
  shape.text.style={fontSize:size,typeface:serif?'Newsreader':'IBM Plex Sans',bold,color,
    alignment:align,verticalAlignment:'top',autoFit:'none',wrap:'square',insets:0};
  return shape;
}
function slide(title,note,{cover=false}={}){
  const s=presentation.slides.add(); s.background.fill=C.paper;
  const number=presentation.slides.items.length;
  if(!cover) text(s,title,64,46,1150,90,48,{serif:true});
  text(s,String(number).padStart(2,'0'),1170,669,45,25,17,{color:C.muted,align:'right'});
  s.speakerNotes.text=note;
  notes.push({number,title,note}); return s;
}
function table(s,values,x=64,y=185,w=1152,h=360,widths=null,fontSize=25){
  const table=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,
    ...(widths?{columnWidths:widths}:{})});
  table.borders.assign({fill:C.rule,width:.5});
  table.styleOptions={headerRow:true,bandedRows:false};
  table.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length})
    .assign({margins:{left:10,right:10,top:6,bottom:6}});
  for(let r=0;r<values.length;r++){
    table.rows[r].height=h/values.length;
    for(let c=0;c<values[r].length;c++){
      const cell=table.getCell(r,c);
      cell.fill=r===0?C['paper-2']:C.paper;
      cell.text.style={typeface:'IBM Plex Sans',fontSize,bold:r===0,color:C.ink,
        alignment:c===0?'left':'right',verticalAlignment:'middle',insets:{top:8,right:12,bottom:8,left:12},
        autoFit:'none',wrap:'square'};
    }
  }
  tableOwners.push(presentation.slides.items.length); return table;
}
function chart(s,title,categories,values,{x=64,y=180,w=1152,h=350,max,unit='x',fill=C.good}={}){
  const c=s.charts.add('bar',{position:{left:x,top:y,width:w,height:h},categories,
    series:[{name:title,values:values.map(v=>Number(v.toFixed(6))),fill}],barOptions:{direction:'column',grouping:'clustered',gapWidth:100},
    hasLegend:false,chartFill:C.paper,plotAreaFill:C.paper,
    chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0},
    xAxis:{textStyle:{fontSize:23,fill:C.ink},line:{fill:C.rule,width:.5},majorGridlines:null},
    yAxis:{min:0,max,numberFormatCode:unit==='x'?'0.0"x"':'0',textStyle:{fontSize:20,fill:C.muted},
      majorGridlines:{fill:C.rule,width:.4}},
    dataLabels:{showValue:true,position:'outEnd',numberFormatCode:unit==='x'?'0.00"x"':'0.00',
      textStyle:{fontSize:23,fill:C.ink,bold:true}}});
  applyPresentationChartFont(c,{fontFamily:'IBM Plex Sans'});
  chartOwners.push(presentation.slides.items.length); return c;
}
function footer(s,value){text(s,value,64,582,1125,69,22,{color:C.muted});}
const pct=v=>(100*v).toFixed(2)+'%';

let s=slide('Conditional Hybrid Retrieval',
  'This presentation develops the completed research showcase. The central question is whether conditional retrieval saves actual query time while preserving a chosen fraction of fusion recall. We will distinguish the original Blended RAG paper from our adaptation, explain the metric differences, and separate the gain from indexing from the extra gain from routing. The work is a bounded engineering study. No new dataset or model run was performed for this presentation. Suggested talk length: 15–20 minutes for slides 1–18. Slides 19–22 are appendices.',{cover:true});
text(s,'Conditional\nhybrid retrieval',64,125,1120,240,84,{serif:true});
text(s,'Execution cost and recall budgets',68,408,1050,54,34);
text(s,'An indexed cascade study',68,480,1000,48,28,{color:C.muted});
text(s,'Research project report · 21 October 2026',68,609,1050,34,22,{color:C.muted});

s=slide('The original Blended RAG paper',
  'Sawarkar, Mangal and Solanki study keyword, dense-vector and learned-sparse retrieval with hybrid field queries. Their implementation uses Elasticsearch and ELSER. The published TREC nDCG and SQuAD answer values are reference results from that paper. Our system did not reproduce them. The paper motivates complementary retrieval signals, while this project investigates the runtime cost of always executing them. CoQA appears in the paper’s discussion of metadata-poor documents. Source: '+refs.paper);
text(s,'Keyword, dense and learned-sparse search\nwith hybrid field queries',64,180,1120,120,38);
text(s,'0.87',64,355,480,90,70,{serif:true,color:C.good});
text(s,'Published TREC-COVID nDCG@10',64,455,490,75,26);
text(s,'57.63 / 68.40',650,355,560,90,66,{serif:true,color:C.good});
text(s,'Published SQuAD answer EM / F1',650,455,560,75,26);
footer(s,'Published reference values. Our completed study measures retrieval quality and query latency.');

s=slide('The first-stage bottleneck',
  'The initial TREC implementation spent 1410.14 milliseconds in BM25 while fusion took 1493.23 milliseconds. The gate had already paid most of the runtime before deciding whether to skip neural branches. Neural-call reduction alone therefore could not establish a useful speedup. The backend optimization addresses this execution bottleneck and is applied to reference and proposed methods equally. '+measured);
chart(s,'Original mean query latency',['BM25','Three-branch fusion'],[1410.13554,1493.23017],{max:1750,unit:'ms',h:335});
footer(s,'Original TREC means in milliseconds. An expensive lexical stage leaves little query time to skip.');

s=slide('Reproduction and adaptation boundaries',
  'We reproduce the use of complementary retrieval signals and Best Fields lexical scoring, then make declared substitutions. Our accessible Python implementation uses MiniLM, SPLADE and RRF. The original paper studies a different search engine, sparse model and query formulations. We also construct an explicit SQuAD paragraph corpus. These differences prevent direct equivalence of quality or latency numbers. Source: '+refs.paper+'. Project method: docs/FINAL_MANUSCRIPT.md, sections 3–5.');
table(s,[['Dimension','Original paper','This project'],['Engine / sparse model','Elasticsearch / ELSER','Python / SPLADE'],
 ['Retrieval question','Blended-query quality','Latency under recall budgets'],['Answer generation','NQ / SQuAD EM and F1','Not evaluated'],
 ['Timing reference','Paper system','Our original / indexed fusion']],64,180,1152,350,[280,420,452],24);
footer(s,'Different models, corpus units and label rules are comparison limits. Their individual effects were not isolated.');

s=slide('Recall retention and absolute recall',
  'Retention is the ratio of mean Recall@100 values. It is not the overlap of two result lists and it does not imply the same quality for every query. In TREC, many documents can be relevant for a topic. Recovering at least one relevant item gives a hit, while recall asks how much of the relevant set was recovered. Our primary retains 96.68 percent of a reference recall near 0.12250, giving absolute recall near 0.11844. nDCG evaluates ranking order and relevance, while EM and F1 evaluate generated answers. '+measured);
text(s,'Mean method Recall@100\n÷ mean fusion Recall@100',64,174,1130,144,49,{serif:true,color:C.good});
text(s,'TREC primary: 0.11844 ÷ 0.12250 = 96.68%',64,357,1130,66,35);
text(s,'Hit accuracy: at least one relevant result\nRecall: fraction of relevant documents recovered\nnDCG: quality of the top-ranked ordering',64,459,1130,147,28);

s=slide('The live conditional decision',
  'Every cascade query first executes Best Fields BM25. Eleven features from the query and initial lexical ranking feed a standardized Ridge regressor. The regressor predicts clipped recall gain from escalation. A small predicted gain returns BM25 immediately. A larger prediction executes Dense and SPLADE and fuses those rankings with the already-computed BM25 result. Inference does not use evaluation relevance labels. Methods do not borrow precomputed neural query results from one another. Sources: experiments 12 and 25; frozen router artifact.');
text(s,'Best Fields BM25, then an 11-feature Ridge gate',64,174,1130,74,34);
table(s,[['Gate condition','Executed action'],['Predicted gain ≤ threshold','Return BM25 top 100'],
 ['Predicted gain > threshold','Run Dense + SPLADE, then RRF']],64,290,1152,210,[475,677],28);
footer(s,'Fallback reuses BM25. RRF uses k = 60. Every path returns up to 100 documents.');

s=slide('Four frozen recall budgets',
  'The calibration uses five-fold out-of-fold predictions on 2590 NFCorpus train queries, seed 42. A Ridge model with alpha 10 predicts nonnegative gain. Each policy limits the sum of observed clipped losses over the queries selected for exit. The final exported model fits all training queries. Training calibration is an empirical procedure and does not guarantee transfer to a new distribution. All four policies remain frozen in the reported follow-ups. Source: results/nfcorpus_frozen_retention_routers.json and experiment 12.');
table(s,[['Policy','Threshold','Train OOF retention'],['r99','0.0629951','99.11%'],
 ['r95','0.0756657','95.42%'],['r92','0.0810938','92.59%'],['r90','0.0847674','90.69%']],64,180,1152,335,[300,420,432],28);
footer(s,'2,590 NFCorpus train queries. Nominal policy targets are calibration settings, not transfer guarantees.');

s=slide('Completed evaluation scope',
  'NFCorpus dev is inspected development evidence. SCIDOCS is frozen confirmation. Original paper-dataset runs declared r90 primary. The indexed follow-up declared r99 primary against a 90 percent floor, using already-inspected TREC quality evidence. It therefore tests compatible execution rather than new independent TREC generalization. Timings include query encoding, retrieval and routing with GPU synchronization, one CPU thread and randomized order. Corpus encoding, index preparation, model loading, downloads and answer generation are outside the timing window. Source: frozen protocols and main manuscript section 5.');
table(s,[['Study','Queries','Documents','Timing passes'],['NFCorpus dev','324','3,633','3'],
 ['SCIDOCS','1,000','25,657','3'],['TREC original / indexed','50 / 50','171,332','3 / 3'],
 ['SQuAD original','10,570','20,958','1 complete'],['SQuAD indexed','256 fixed','20,958','3']],64,170,1152,350,[485,205,232,230],24);
footer(s,'Warmed T4 query timing. The indexed SQuAD run retains the full corpus but samples its questions.');

s=slide('TREC-COVID primary result',
  'The conservative r99 policy was preselected as primary for the indexed engineering follow-up. It takes 73.90 milliseconds against indexed fusion at 91.28 milliseconds, giving 1.2351 times speedup, rounded to 1.24. It retains 96.68 percent of fusion recall, with a query-bootstrap interval of 93.58 to 98.75 percent. Mean time saved is 19.04 percent, interval 9.47 to 30.11. This meets the user’s 90 percent floor for this measured setting and misses the policy’s nominal 99 percent target. The primary p95 is slightly higher than fusion, so the claim concerns mean time. '+measured);
text(s,'1.24×',64,175,510,120,94,{serif:true,color:C.good});
text(s,'speedup over indexed fusion',64,314,550,80,30);
text(s,'96.68%',665,175,535,120,85,{serif:true,color:C.good});
text(s,'Recall@100 retained',665,314,535,70,30);
text(s,'91.28 ms fusion / 73.90 ms cascade',64,438,1130,60,32);
footer(s,'Retention interval: 93.58–98.75%. All 50 queries, full corpus, three warmed passes.');

s=slide('TREC-COVID operating points',
  'All four frozen settings remain visible. r95 is the fastest secondary point above 90 percent observed retention. Its retention interval is 86.88 to 95.76 percent and crosses the floor. r92 and r90 both retain 84.53 percent, so their greater speed does not meet the requested quality budget. The 99/95/92/90 names refer to training calibration, and every nominal target still misses on TREC. The fair additional-routing comparison is against the same indexed fusion backend. '+measured);
table(s,[['Configuration','Retained recall','Mean ms','vs indexed'],
 ['Indexed fusion','100.00%','91.28','1.00x'],
 ...['ridge11_r99','ridge11_r95','ridge11_r92','ridge11_r90'].map(name=>[name.replace('ridge11_',''),pct(trec[name].recall_retention),trec[name].latency_ms.mean.toFixed(2),trec[name].mean_speedup_factor.toFixed(2)+'x'])],64,176,1152,340,[380,270,220,282],27);
footer(s,'r95: interval crosses 90%. r92 / r90: observed recall falls below 90%. Primary remains r99.');

s=slide('Indexing and routing combined',
  'This larger progression divides original fusion latency by the indexed method latency. It combines indexing and routing across different runs. Indexing alone gives 16.36 times, followed by r99 at 20.21, r95 at 26.83, r92 at 35.56 and r90 at 36.96. These are configurations, not scaling measurements as corpus size grows. The maximum is below the requested quality floor. The r95 secondary is above 90 percent at its point estimate but its interval crosses the floor. '+measured);
chart(s,'vs original fusion',['Original','Indexed','r99','r95','r92','r90'],[1,...['static_rrf','ridge11_r99','ridge11_r95','ridge11_r92','ridge11_r90'].map(n=>trec[n].speedup_vs_original_fusion)],{max:43,h:325});
footer(s,'Retained recall after indexing: 100%, 96.68%, 91.95%, 84.53%, 84.53%. Separate runs include session variation.');

s=slide('SQuAD sampled cascade results',
  'The indexed SQuAD result uses 256 fixed SHA-selected questions with the full 20958-document corpus. The fastest secondary setting r90 takes 18.59 milliseconds and retains 99.22 percent of fusion recall, producing 1.59 times speedup over indexed fusion. Its retention interval is 98.05 to 100 percent. The primary retains 100 percent sampled recall at 1.04 times. Original evidence covers 10570 questions with one complete pass plus separately preserved partial repetitions. Cross-run ratios use the same 256 questions, not the full-query original mean. '+measured);
chart(s,'vs indexed fusion',['Fusion','r99','r95','r92','r90'],[1,...['ridge11_r99','ridge11_r95','ridge11_r92','ridge11_r90'].map(n=>squad[n].mean_speedup_factor)],{max:1.85,h:325});
footer(s,'Recall retained: 100%, 100%, 100%, 100%, 99.22%. 256 questions / full corpus / three passes.');

s=slide('Standalone retrievers remain strong',
  'The static fusion reference is useful for measuring what the cascade preserves, but it is not automatically the best efficient action. On TREC, SPLADE has higher recall and nDCG than the primary at lower mean time. The primary has higher MAP, 0.01769 versus 0.01662. On sampled SQuAD, BM25 alone exceeds the 90 percent recall floor at 1.03 milliseconds, and SPLADE has the highest nDCG and MAP. Thus a fusion-relative cascade gain does not establish overall superiority under a recall/time objective. '+measured);
text(s,'TREC-COVID',64,166,530,42,30,{bold:true});
text(s,'SQuAD, same sample',665,166,550,42,30,{bold:true});
for(const [x,methods] of [[64,trec],[665,squad]])
  table(s,[['Method','Mean ms','Retained'],...['static_rrf','bm25','dense','splade'].map(n=>[
    {static_rrf:'Fusion',bm25:'BM25',dense:'Dense',splade:'SPLADE'}[n],methods[n].latency_ms.mean.toFixed(2),pct(methods[n].recall_retention)])],x,235,550,290,[200,150,200],24);
footer(s,'SPLADE favors TREC recall/time. BM25 already clears the sampled SQuAD 90% floor. Ranking metrics still differ.');

s=slide('Earlier transfer findings',
  'Earlier runs are part of the evidence, not discarded failures. The NFCorpus development primary r92 retains 94.74 percent and saves 23 percent mean time. Frozen SCIDOCS r92 retains 90.01 percent and saves only 5.53 percent, with a retention interval crossing 90. Every nominal target fails on SCIDOCS and TREC. Dense is a stronger recall/time action than the cascades on NFCorpus and SCIDOCS. These observations motivate the first-stage investigation and limit any universal transfer claim. Source: final_quality_speed.csv and the original benchmark reports.');
table(s,[['Original backend study','Primary','Retained','Time saved'],['NFCorpus dev','r92','94.74%','23.00%'],
 ['SCIDOCS','r92','90.01%','5.53%'],['TREC-COVID','r90','84.53%','3.56%'],['SQuAD, full queries','r90','99.38%','6.12%']],64,180,1152,330,[480,170,250,252],26);
footer(s,'Original primaries retain their labels. All nominal targets miss on SCIDOCS and TREC-COVID.');

s=slide('Saved evidence and compatibility',
  'We compare every saved first-pass result across original and indexed backends on the same query IDs. The 400 TREC records and 2048 sampled SQuAD records match in ranked IDs, scores, routes and recall. Independent scorers reproduce quality against exact raw relevance labels, and frozen source and weight hashes identify the executed code. These checks validate saved artifacts and quality calculations. They do not independently verify the GPU clock or establish production behavior. Source: experiments 22, 29, 30 and 31; recorded local verification and metric-check reports.');
text(s,'400',64,175,520,105,86,{serif:true,color:C.good});
text(s,'TREC first-pass rankings',64,298,540,64,30);
text(s,'2,048',665,175,530,105,86,{serif:true,color:C.good});
text(s,'Sampled SQuAD rankings',665,298,540,64,30);
text(s,'Exact IDs, scores, routes and recalls across backends',64,423,1130,90,34);
footer(s,'Independent raw-label scoring reproduces quality. Timing and wider hardware behavior remain experimental limits.');

s=slide('Implementation and reproducibility',
  'The code remains organized around retrieval modules, numbered experiments, frozen result protocols and compact saved reports. The indexed backend changes how lexical scores execute while preserving score definitions. Benchmark 25 executes the indexed cascade; import and scoring helpers check returned evidence. The publication builders only consume saved reports. They do not run neural encoding, training or query timing. The source-sharing ZIP excludes local datasets, environments and bulk query rankings. Their original exports are identified separately in the artifact catalog. Source: repository module indexes and reproduction guide.');
table(s,[['File or group','Responsibility'],['src/indexed_bm25.py','Compatible lexical postings'],
 ['Frozen routers + indexed protocol','Weights, thresholds, revisions and hashes'],['Experiment 25','Indexed query execution'],
 ['Experiments 22 / 29 / 30 / 31','Import, rescore and compare saved evidence'],['Experiment 32 + publication tools','Rebuild figures and report artifacts']],64,176,1152,350,[620,532],24);
footer(s,'Source paths remain recognizable. Large local inputs are ignored and restored separately when needed.');

s=slide('Limits and deferred evaluation',
  'The completed study does not evaluate full current NQ, HotpotQA or CoQA. The old NQ candidate pool is qrel-informed and the routing is simulated, so it cannot support the current method’s full-corpus savings. Generated-answer EM/F1, modern adaptive-retrieval baselines, concurrency, energy and index memory/update costs are unmeasured. One GPU type, small numbers of sessions and known TREC quality restrict generalization. A later study should preserve this release, declare its protocol and quality objective, and retain strong standalone actions.');
text(s,'Full NQ, HotpotQA and CoQA remain deferred',64,178,1130,96,40,{serif:true});
text(s,'Generated-answer EM/F1 and modern adaptive baselines remain unmeasured',64,324,1130,110,32);
text(s,'TREC quality was already inspected. Indexed SQuAD covers a fixed query sample.',64,482,1130,96,28,{color:C.muted});

s=slide('Conclusion',
  'The main result is that compatible first-stage indexing makes the frozen cascade deliver measurable extra mean-time savings against an equally optimized fusion reference. TREC primary gives 1.24 times at 96.68 percent retention. The fastest sampled SQuAD cascade gives 1.59 times at 99.22 percent. Every secondary point, uncertainty and standalone comparison remains visible. The contribution is execution optimization and empirical understanding of its quality/time limits. It does not establish overall superiority or a new routing principle. The current project can be reported at this scope while remaining datasets are tested later. '+measured);
text(s,'1.24× at 96.68%',64,180,1140,100,77,{serif:true,color:C.good});
text(s,'TREC-COVID primary over equally indexed fusion',64,300,1130,65,31);
text(s,'1.59× at 99.22%',64,422,1130,83,60,{serif:true});
text(s,'Fastest cascade on the fixed SQuAD sample',64,520,1130,65,28);
text(s,'A measured execution contribution with explicit quality and baseline limits',64,633,1110,30,22,{color:C.muted});

for(const [key,label,methods] of [['trec-covid','TREC-COVID',trec],['squad','SQuAD sample',squad]]){
  s=slide('Appendix: '+label+' quality metrics',
    'These are the complete indexed quality values and mean times for the same evaluated query set. Recall retained uses mean Recall@100 as the denominator. The SQuAD sample is not the full-query evaluation. Values are rounded for display; source reports retain full precision. '+measured);
  const order=['static_rrf','bm25','dense','splade','ridge11_r99','ridge11_r95','ridge11_r92','ridge11_r90'];
  table(s,[['Method','Recall@100','nDCG@10','MAP@10','Mean ms','Indexed x','Original x'],...order.map(n=>[
    {static_rrf:'Fusion',bm25:'BM25',dense:'Dense',splade:'SPLADE'}[n]??n.replace('ridge11_',''),
    methods[n].metrics['recall@100'].toFixed(4),methods[n].metrics['ndcg@10'].toFixed(4),
    methods[n].metrics['map@10'].toFixed(4),methods[n].latency_ms.mean.toFixed(2),
    methods[n].mean_speedup_factor.toFixed(2)+'x',methods[n].speedup_vs_original_fusion.toFixed(2)+'x'])],
    64,169,1152,380,[170,175,150,145,145,180,187],22);
  footer(s,key==='squad'?'256 fixed questions, 20,958 documents, three indexed passes.':'All 50 queries, 171,332 documents, three indexed passes.');
}

s=slide('Appendix: primary references',
  'Primary sources verified for this release: '+Object.values(refs).join('\n')+'. The main manuscript provides full bibliographic details. Sources for each measured claim appear in the corresponding speaker notes and the shared results JSON.');
text(s,'Sawarkar et al. (2024)\nBlended RAG\n\nArabzadeh et al. (2021)\nRetrieval strategy selection\n\nLu (2024)\nBM25S eager sparse scoring',64,174,530,390,27);
text(s,'Thakur et al. (2021)\nBEIR benchmark\n\nRajpurkar et al. (2016)\nSQuAD\n\nFormal et al. (2021) / Reimers & Gurevych (2019)\nSPLADE / Sentence-BERT',665,174,550,420,25);
footer(s,'Full URLs and source details are in speaker notes and the research paper.');

s=slide('Appendix: rebuilding the report',
  'To rebuild figures, run experiment 32 from the repository root using the documented analysis requirements. To rebuild the release figures, paper and HTML, use the publication tools and requirements. The editable presentation builder uses the bundled Codex Artifact Tool runtime. Headless browser rendering creates the showcase PDF. Exact commands and required environment variables appear in docs/submission/README.md. No new Kaggle run is necessary to read or regenerate current result plots. Full record-level rescoring needs original matching result archives separately.');
text(s,'1  Figures from saved reports\n    experiments/32_prepare_report_assets.py\n\n2  Release artifacts\n    tools/build_research_materials.py\n\n3  Editable deck and print showcase\n    tools/build_research_presentation.mjs\n    tools/render_showcase.mjs',64,176,1130,390,29);
footer(s,'See docs/submission/README.md for exact commands and runtime requirements.');

await fs.writeFile(path.join(out,'presentation_notes.md'), '# Presentation speaker notes\n\nSuggested talk: 15–20 minutes on slides 1–18. Slides 19–22 are appendices.\n\n'+notes.map(n=>`## ${n.number}. ${n.title}\n\n${n.note}\n`).join('\n'));
const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidate);
console.log('Draft exported:',candidate);
for(let i=0;i<presentation.slides.items.length;i++){
  const image=await presentation.export({slide:presentation.slides.items[i],format:'png',scale:1.25});
  await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await image.arrayBuffer()));
  console.log('Rendered slide',i+1);
}
const final=path.join(out,'research_presentation.pptx');
// Validate each revision to a new private output, then copy the checked bytes.
const finalizedDir=path.join(root,'.local/release_build/finalized');
await fs.mkdir(finalizedDir,{recursive:true});
const finalPath=process.env.FINAL_PPTX ?? path.join(finalizedDir,`research_presentation_${Date.now()}.pptx`);
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,
  pythonExecutable:runtimePython,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',
    ...[...new Set(tableOwners)].flatMap(number=>['--require-native-table-slide',String(number)])],
  explicitTotalSlideCount:22,requiredNativeTableOwnerSlides:[...new Set(tableOwners)],
  requiredNativeChartOwnerSlides:[...new Set(chartOwners)],materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'design',families:['Newsreader','IBM Plex Sans']},verifyArtifactToolImport:true,
  receiptPath:path.join(build,`validation-${path.basename(finalPath,'.pptx')}.json`)});
await fs.copyFile(finalPath,final);
console.log('Final presentation:',final);
process.exit(0);
