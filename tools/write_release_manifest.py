"""Fingerprint the completed report release without changing measured evidence."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=ROOT/'docs/submission/RELEASE_MANIFEST.json'
    names={p for p in (ROOT/'docs/submission').rglob('*') if p.is_file() and p!=output}
    names.update(ROOT/name for name in [
        'docs/FINAL_MANUSCRIPT.md','docs/review/PAPER_RELEASE_REVIEW.md','tokens.css',
        'tools/build_publication_figures.py','tools/build_research_materials.py',
        'tools/build_research_presentation.mjs','tools/render_showcase.mjs',
        'tools/export_presentation_pdf.py','tools/showcase.template.html',
    ])
    protocols=['nfcorpus_frozen_retention_routers.json','indexed_cascade_protocol.json','paper_datasets_protocol.json']
    report={
        'release_date':'2026-10-21',
        'scope':'Completed conditional retrieval engineering study. Additional full NQ, HotpotQA, CoQA and answer-generation evaluations are deferred.',
        'new_retrieval_training_or_timing_runs_for_materials':0,
        'main_evaluations':{'trec_covid':'all 50 queries, 171332 documents, 3 indexed passes',
                            'squad_indexed':'fixed 256 questions of 10570, 20958 documents, 3 passes',
                            'squad_original':'all 10570 questions, 1 complete timing pass'},
        'frozen_assets_sha256':{name:digest(ROOT/'cascaded-blended-rag/results'/name) for name in protocols},
        'files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(names)],
    }
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('Release manifest:',output,'files:',len(report['files']))


if __name__=='__main__':
    main()
