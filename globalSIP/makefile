PAPER = paper
PAPER_SIP = globalSIP

all:
	latex $(PAPER).tex
	bibtex $(PAPER)
	latex $(PAPER).tex
	latex $(PAPER).tex
	dvipdf $(PAPER).dvi
	rm paper.aux paper.dvi paper.bbl paper.blg paper.log

sip:
	latex $(PAPER_SIP).tex
	bibtex $(PAPER_SIP)
	latex $(PAPER_SIP).tex
	latex $(PAPER_SIP).tex
	dvipdf $(PAPER_SIP).dvi
	rm $(PAPER_SIP).aux $(PAPER_SIP).dvi $(PAPER_SIP).bbl $(PAPER_SIP).blg $(PAPER_SIP).log