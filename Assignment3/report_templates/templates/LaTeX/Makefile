
all: template.pdf

%.pdf: %.tex
	pdflatex $<; bibtex $*; pdflatex $<; pdflatex $<

clean:
	rm -f *.log *.aux *~ *.toc *.bbl *.blg *.bak
