"""Create a portable presentation PDF from the inspected Artifact Tool renders."""
from pathlib import Path
from reportlab.pdfgen import canvas

ROOT=Path(__file__).resolve().parents[1]


def main():
    slides=sorted((ROOT/'.local/release_build/presentation').glob('slide-*.png'))
    if len(slides)!=22:
        raise RuntimeError('Render all 22 slides before exporting the presentation PDF.')
    output=ROOT/'docs/submission/research_presentation.pdf'
    pdf=canvas.Canvas(str(output),pagesize=(1280,720))
    pdf.setTitle('Conditional Hybrid Retrieval - Research Presentation')
    for slide in slides:
        pdf.drawImage(str(slide),0,0,width=1280,height=720)
        pdf.showPage()
    pdf.save()
    print('Presentation PDF:',output)


if __name__=='__main__':
    main()
