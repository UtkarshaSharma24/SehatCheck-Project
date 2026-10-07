"""Makes a simple PDF report. Needs: pip install fpdf2"""
from datetime import date
from fpdf import FPDF

DISCLAIMER = ("Educational project, not a medical diagnosis. The model uses practice data, "
              "not real patients. If you feel seriously unwell, see a doctor.")


def _latin(text):
    return str(text).encode("latin-1", "replace").decode("latin-1")


def make_pdf(kind, lines):
    """kind: report title. lines: list of text lines. Returns PDF as bytes."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(20, 86, 196)
    pdf.rect(0, 0, 210, 26, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(16, 6)
    pdf.cell(0, 8, "SehatCheck")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(16, 15)
    pdf.cell(0, 6, _latin(kind) + "  |  " + date.today().strftime("%d %b %Y"))
    pdf.set_text_color(20, 33, 61)
    pdf.set_xy(16, 36)
    pdf.set_font("Helvetica", "", 11)
    for line in lines:
        pdf.set_x(16)
        pdf.multi_cell(178, 6.5, _latin(line), new_x="LMARGIN", new_y="NEXT")
    pdf.set_y(-22)
    pdf.set_x(16)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(94, 107, 133)
    pdf.multi_cell(178, 4, DISCLAIMER)
    return bytes(pdf.output())
