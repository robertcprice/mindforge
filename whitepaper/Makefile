# Makefile for Conch DNA Whitepaper

PAPER = conch_dna
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean view

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	$(LATEX) $(PAPER)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

view: $(PAPER).pdf
	open $(PAPER).pdf

# Alternative: use latexmk for automatic rebuilding
watch:
	latexmk -pdf -pvc $(PAPER).tex
