PAPER = paper
PAPER_NIPS = nips
WRITE = writeup

all:
	latex $(PAPER).tex
	bibtex $(PAPER)
	latex $(PAPER).tex
	latex $(PAPER).tex
	dvipdf $(PAPER).dvi
	rm paper.aux paper.dvi paper.bbl paper.blg paper.log

writeup:
	latex $(WRITE).tex
	latex $(WRITE).tex
	dvipdf $(WRITE).dvi
	rm $(WRITE).aux $(WRITE).dvi $(WRITE).bbl $(WRITE).blg $(WRITE).log

one:
	latex $(PAPER).tex

icml:
	latex $(PAPER_ICML).tex
	bibtex $(PAPER_ICML)
	latex $(PAPER_ICML).tex
	latex $(PAPER_ICML).tex
	dvipdf $(PAPER_ICML).dvi

nips:
	latex $(PAPER_NIPS).tex
	bibtex $(PAPER_NIPS)
	latex $(PAPER_NIPS).tex
	latex $(PAPER_NIPS).tex
	dvipdf $(PAPER_NIPS).dvi
	rm $(PAPER_NIPS).aux $(PAPER_NIPS).dvi $(PAPER_NIPS).bbl $(PAPER_NIPS).blg $(PAPER_NIPS).log