"""Build paper PDF and offline showcase from completed project reports."""
import base64
import csv
import hashlib
import html
import json
from pathlib import Path
import re
from string import Template

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/submission'
ASSETS = OUT / 'assets'
REPORTS = ROOT / 'cascaded-blended-rag/results'
PALETTE = dict(re.findall(r'--color-([\w-]+):\s*(#[0-9a-fA-F]{6})', (ROOT/'tokens.css').read_text()))


def load_data():
    with (ROOT/'docs/figures/all_measured_speedups.csv').open(newline='') as handle:
        ratios = {(r['dataset'],r['method']):r for r in csv.DictReader(handle)}
    bundle = {'release_date':'2026-10-21','datasets':{},'source_sha256':{}}
    for key,folder in [('trec-covid','indexed_trec-covid_q0'),('squad','indexed_squad_q256')]:
        path = REPORTS/folder/'indexed_cascade_benchmark.json'
        report = json.loads(path.read_text())
        methods = {}
        for name,method in report['methods'].items():
            ratio = ratios[(key,name)]
            methods[name] = {k:v for k,v in method.items() if k not in ['timings_per_query_ms','random_exit_reference']}
            methods[name]['speedup_vs_original_fusion'] = float(ratio['speedup_vs_original_fusion'])
            methods[name]['original_fusion_same_queries_ms'] = float(ratio['original_fusion_same_queries_ms'])
        bundle['datasets'][key]={'queries':report['queries'],'all_available_queries':report['all_available_queries'],
                                 'documents':report['corpus_documents'],'repeats':report['repeats'],
                                 'methods':methods, 'source':path.relative_to(ROOT).as_posix()}
        bundle['source_sha256'][path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return bundle


def image_uri(path):
    mime='image/svg+xml' if path.suffix=='.svg' else 'image/png'
    return f'data:{mime};base64,'+base64.b64encode(path.read_bytes()).decode()


def method_name(name):
    return {'static_rrf':'Indexed fusion','bm25':'BM25','dense':'Dense MiniLM','splade':'SPLADE',
            'ridge11_r99':'r99 · primary','ridge11_r95':'r95 · secondary',
            'ridge11_r92':'r92 · secondary','ridge11_r90':'r90 · secondary'}[name]


def web_table(headers, rows, classes=None):
    head=''.join(f'<th scope="col">{html.escape(h)}</th>' for h in headers)
    body=[]
    for index,row in enumerate(rows):
        cls = classes[index] if classes else ''
        cells=''.join(f'<{("th scope=\"row\"" if j==0 else "td")} data-label="{html.escape(headers[j])}">{html.escape(str(value))}</{("th" if j==0 else "td")}>' for j,value in enumerate(row))
        body.append(f'<tr class="{cls}">{cells}</tr>')
    return '<table><thead><tr>'+head+'</tr></thead><tbody>'+''.join(body)+'</tbody></table>'


def results_table(data,key):
    methods=data['datasets'][key]['methods']
    order=['static_rrf','ridge11_r99','ridge11_r95','ridge11_r92','ridge11_r90']
    rows=[]; classes=[]
    for name in order:
        method=methods[name]
        rows.append([method_name(name),f'{method["recall_retention"]*100:.2f}%',
                     f'{method["latency_ms"]["mean"]:.2f}',f'{method["mean_speedup_factor"]:.2f}x',
                     f'{method["speedup_vs_original_fusion"]:.2f}x'])
        classes.append('primary' if name=='ridge11_r99' else 'fails' if method['recall_retention']<.90 else '')
    return web_table(['Configuration','Recall retained','Mean ms','vs indexed','vs original'],rows,classes)


def baseline_table(data,key):
    methods=data['datasets'][key]['methods']
    rows=[]
    for name in ['static_rrf','bm25','dense','splade']:
        m=methods[name]
        rows.append([method_name(name),f'{m["recall_retention"]*100:.2f}%',f'{m["latency_ms"]["mean"]:.2f}',
                     f'{m["metrics"]["ndcg@10"]:.4f}',f'{m["metrics"]["map@10"]:.4f}',
                     f'{m["mean_speedup_factor"]:.2f}x',f'{m["speedup_vs_original_fusion"]:.2f}x'])
    return web_table(['Retriever','Recall retained','Mean ms','nDCG@10','MAP@10','vs indexed','vs original'],rows)


def showcase(data):
    css=(ROOT/'tokens.css').read_text()
    faces=[]
    for family,weight,name in [('Newsreader',400,'Newsreader-400.ttf'),('Newsreader',600,'Newsreader-600.ttf'),
                                ('IBM Plex Sans',400,'IBMPlexSans-Regular.ttf'),('IBM Plex Sans',600,'IBMPlexSans-SemiBold.ttf')]:
        encoded=base64.b64encode((ASSETS/'fonts'/name).read_bytes()).decode()
        faces.append(f"@font-face{{font-family:'{family}';font-style:normal;font-weight:{weight};font-display:swap;src:url(data:font/ttf;base64,{encoded}) format('truetype');}}")
    template=(ROOT/'tools/showcase.template.html').read_text(encoding='utf-8')
    values={'tokens':css,'fonts':'\n'.join(faces),'trec_table':results_table(data,'trec-covid'),
            'squad_table':results_table(data,'squad'),'trec_baselines':baseline_table(data,'trec-covid'),
            'squad_baselines':baseline_table(data,'squad'),
            'trec_chart':image_uri(ASSETS/'speedup_trec-covid.svg'),
            'squad_chart':image_uri(ASSETS/'speedup_squad.svg'),'pipeline':image_uri(ASSETS/'pipeline.svg')}
    output=OUT/'research_showcase.html'
    output.write_text(Template(template).substitute(values),encoding='utf-8')
    return output


def inline(text):
    text=html.escape(text,quote=False)
    def link(match):
        target=match.group(2)
        if not re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:',target) and not target.startswith('#'):
            target='../'+target
        return f'<a href="{html.escape(target,quote=True)}" color="{PALETTE["good"]}">{match.group(1)}</a>'
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)',r'<i>\1</i>',text)
    text=re.sub(r'`([^`]+)`',r'<font size="8.5">\1</font>',text)
    return text


def paper_pdf():
    fonts=ASSETS/'fonts'
    for family,file in [('Body','IBMPlexSans-Regular.ttf'),('BodyBold','IBMPlexSans-SemiBold.ttf'),
                         ('Display','Newsreader-400.ttf'),('DisplayBold','Newsreader-600.ttf')]:
        pdfmetrics.registerFont(TTFont(family,str(fonts/file)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
    pdfmetrics.registerFontFamily('Display',normal='Display',bold='DisplayBold',italic='Display',boldItalic='DisplayBold')
    ink=colors.HexColor(PALETTE['ink']); muted=colors.HexColor(PALETTE['muted'])
    body=ParagraphStyle('Body',fontName='Body',fontSize=9.8,leading=14.1,textColor=ink,spaceAfter=8,
                        splitLongWords=True,allowWidows=0,allowOrphans=0)
    styles={
        'body':body,
        'h1':ParagraphStyle('Title',parent=body,fontName='Display',fontSize=29,leading=31,spaceAfter=12,keepWithNext=True),
        'h2':ParagraphStyle('H2',parent=body,fontName='DisplayBold',fontSize=17,leading=20,spaceBefore=17,spaceAfter=8,keepWithNext=True),
        'h3':ParagraphStyle('H3',parent=body,fontName='BodyBold',fontSize=11.5,leading=15,spaceBefore=10,spaceAfter=5,keepWithNext=True),
        'caption':ParagraphStyle('Caption',parent=body,fontSize=8.4,leading=11,textColor=muted,spaceAfter=7,keepWithNext=True),
        'cell':ParagraphStyle('Cell',parent=body,fontSize=8,leading=10.4,spaceAfter=0),
        'code':ParagraphStyle('Code',parent=body,fontSize=8.3,leading=11.8,spaceAfter=9),
        'ref':ParagraphStyle('Reference',parent=body,fontSize=8.8,leading=12,spaceAfter=7),
    }
    lines=(ROOT/'docs/FINAL_MANUSCRIPT.md').read_text(encoding='utf-8').splitlines()
    story=[]; i=0; refs=False; figure_number=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1; continue
        if line.startswith('```'):
            language=line[3:]; block=[]; i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                block.append(lines[i]); i+=1
            if language=='mermaid':
                figure_number+=1
                story.append(Image(str(ASSETS/'pipeline.png'),width=491,height=150))
                story.append(Paragraph(f'Figure {figure_number}. The live conditional path reuses BM25 results during fallback fusion.',styles['caption']))
            else:
                story.append(Paragraph('<br/>'.join(html.escape(x) for x in block),styles['code']))
        elif line.startswith('!['):
            figure_number+=1
            relative=re.search(r'!\[[^\]]*\]\(([^)]+)\)',line).group(1)
            source=ROOT/'docs'/relative
            from PIL import Image as PILImage
            with PILImage.open(source) as im:
                width,height=im.size
            story.append(Image(str(source),width=491,height=491*height/width))
        elif line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[v.strip() for v in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',c) for c in cells):
                    rows.append(cells)
                i+=1
            count=len(rows[0]); widths={2:[170,321],3:[110,190,191],4:[200,97,97,97],5:[173,79.5,79.5,79.5,79.5],6:[156,67,67,67,67,67]}.get(count,[491/count]*count)
            cells=[[Paragraph(inline(c),styles['cell']) for c in row] for row in rows]
            table=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),
                ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor(PALETTE['paper-2'])),
                ('LINEABOVE',(0,0),(-1,0),.6,colors.HexColor(PALETTE['ink'])),
                ('LINEBELOW',(0,0),(-1,0),.4,colors.HexColor(PALETTE['rule'])),
                ('LINEBELOW',(0,-1),(-1,-1),.6,colors.HexColor(PALETTE['ink']))]))
            story.extend([table,Spacer(1,11)])
            continue
        elif line.startswith('# '):
            story.append(Paragraph(inline(line[2:]),styles['h1']))
        elif line.startswith('### '):
            story.append(Paragraph(inline(line[4:]),styles['h3']))
        elif line.startswith('## '):
            if line=='## References': refs=True
            elif line.startswith('## Appendix'): refs=False
            style=styles['h3'] if line=='## An Indexed Cascade Study' else styles['h2']
            story.append(Paragraph(inline(line[3:]),style))
        else:
            parts=[line]; i+=1
            while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```','![')):
                parts.append(lines[i].strip()); i+=1
            text=' '.join(parts)
            if text.startswith('**Figure '):
                text=re.sub(r'^\*\*Figure \d+\.',f'**Figure {figure_number}.',text)
            style=styles['caption'] if text.startswith('**Table ') else styles['ref'] if refs else styles['body']
            story.append(Paragraph(inline(text),style))
            continue
        i+=1
    def page_frame(canvas,doc):
        canvas.saveState()
        w,h=A4
        canvas.setStrokeColor(colors.HexColor(PALETTE['rule'])); canvas.setLineWidth(.4)
        canvas.line(52,42,w-52,42)
        canvas.setFont('Body',8); canvas.setFillColor(muted)
        canvas.drawString(52,29,'Conditional Hybrid Retrieval under Recall Budgets')
        canvas.drawRightString(w-52,29,str(doc.page))
        if doc.page>1:
            canvas.setFont('Body',7.4); canvas.drawString(52,h-30,'RESEARCH PROJECT REPORT  /  COMPLETED EXPERIMENTAL SCOPE')
        canvas.restoreState()
    output=OUT/'conditional_retrieval_paper.pdf'
    class AcademicDoc(SimpleDocTemplate):
        outline_parent=False
        def afterFlowable(self,flowable):
            if not isinstance(flowable,Paragraph):
                return
            name=flowable.style.name
            if name not in ('H2','H3') or (name=='H3' and not self.outline_parent):
                return
            label=flowable.getPlainText()
            key=hashlib.sha256(label.encode()).hexdigest()[:20]
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(label,key,level=0 if name=='H2' else 1,closed=False)
            if name=='H2':
                self.outline_parent=True
    doc=AcademicDoc(str(output),pagesize=A4,rightMargin=52,leftMargin=52,topMargin=48,bottomMargin=57,
                          title='Conditional Hybrid Retrieval under Recall Budgets',
                          author='Ali Omar; Abdallah Mestrah; Abbas Yazbek',
                          subject='Supervisor: Dr. Ahmad Abboud; Lebanese University')
    doc.build(story,onFirstPage=page_frame,onLaterPages=page_frame)
    return output


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=load_data()
    (ASSETS/'results.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print('Showcase:',showcase(data))
    print('Paper PDF:',paper_pdf())
    print('No retrieval, neural encoding, training or timing was performed.')


if __name__=='__main__':
    main()
