all: report clean

report:
	pdflatex -interaction batchmode report.tex
	pdflatex -interaction batchmode report.tex

verbose:
	pdflatex report.tex
	pdflatex report.tex

clean:
	rm -rf *.log *.aux *.toc *.out sections/*.aux

.PHONY: all verbose test clean
