SPHINXBUILD   = sphinx-build
SPHINXPROJ    = tdml
SOURCEDIR     = source
BUILDDIR      = build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)"

.PHONY: clean
clean:
	rm -rf source/tdml*.rst
	rm -rf source/modules.rst
	rm -rf build
