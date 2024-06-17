SHELL := /bin/bash
.PHONY: help check autoformat notebook html clean
.DEFAULT: help

# Generates a useful overview/help message for various make features
help:
	@echo "make check"
	@echo "    Run code style and linting (black, flake, isort) *without* changing files!"
	@echo "make autoformat"
	@echo "    Run code styling (black, isort) and update in place - committing with pre-commit also does this."
	@echo "make notebook"
	@echo "    Use jupytext-light to build a notebook (.ipynb) from the s4/s4.py script."
	@echo "make html"
	@echo "    Use jupyter & jupytext to do the two-step conversion from the python script, to the HTML blog post."
	@echo "make clean"
	@echo "    Delete the generated, top-level s4.ipynb notebook."

check:
	isort --check s4/s4.py s4/data.py s4/train.py s4/sample.py
	black --check s4/s4.py s4/data.py s4/train.py s4/sample.py
	flake8 --show-source s4/s4.py s4/data.py s4/train.py s4/sample.py

autoformat:
	isort --atomic s4/s4.py s4/data.py s4/train.py s4/sample.py
	black s4/s4.py s4/data.py s4/train.py s4/sample.py
	flake8 --show-source s4/s4.py s4/data.py s4/train.py s4/sample.py

notebook: s4/s4.py s4/dss.py s4/s4d.py
	jupytext --to notebook s4/s4.py -o s4.ipynb
	jupytext --to notebook s4/dss.py -o dss.ipynb
	jupytext --to notebook s4/s4d.py -o s4d.ipynb

html: mamba.py
	jupytext --to notebook s4/s4.py -o s4.ipynb
	jupyter nbconvert --to html s4.ipynb
	jupytext --to notebook s4/dss.py -o dss.ipynb
	jupyter nbconvert --to html dss.ipynb
	jupytext --to notebook s4/s4d.py -o s4d.ipynb
	jupyter nbconvert --to html s4d.ipynb

md: blog.ipynb
	jupyter nbconvert --to markdown blog.ipynb --output=fullblog.md --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags remove_cell --TagRemovePreprocessor.remove_input_tags remove_in

blog:  blog.md
	pandoc docs/header-includes.yaml fullblog.md --from markdown+raw_html --mathjax --output=index.html --to=html5 --css=docs/github.min.css --css=docs/tufte.css --css=docs/extra.css --no-highlight  --standalone --metadata pagetitle="Differentiable Rasterization"
	
clean: s4.ipynb dss.ipynb s4d.ipynb
	rm -f s4.ipynb
	rm -f dss.ipynb
	rm -f s4d.ipynb
