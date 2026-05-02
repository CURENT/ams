# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    =  "-W"  # This flag turns warnings into errors.
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = PackagingScientificPython
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help clean distclean Makefile

# Wipe the Sphinx build tree only (fast).
clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Wipe the build tree AND the generated source artifacts:
#   - source/_generated, source/modeling/_generated  (autosummary output)
#   - source/_examples                               (nbsphinx notebook copies)
# The next `make html` will be noticeably slower because notebooks re-execute.
distclean: clean
	@rm -rf "$(SOURCEDIR)/_generated" \
	        "$(SOURCEDIR)/modeling/_generated" \
	        "$(SOURCEDIR)/_examples"
	@echo "Removed autosummary + nbsphinx artifacts."

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
