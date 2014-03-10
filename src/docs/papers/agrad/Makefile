stan-autodiff.pdf: stan-autodiff.tex
	pdflatex stan-autodiff.tex
	bibtex stan-autodiff
	pdflatex stan-autodiff.tex
	pdflatex stan-autodiff.tex 

clean:
	rm -f stan-autodiff.aux stan-autodiff.bbl stan-autodiff.blg stan-autodiff.brf stan-autodiff.log stan-autodiff.out stan-autodiff.pdf

