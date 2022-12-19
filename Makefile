PKG_VERSION   = $(shell cat VERSION)
DOCS_FOLDER   = "docs"
SOURCE_FOLDER = "madminer"
TESTS_FOLDER  = "tests"


.PHONY: build
build:
	@echo "Building package"
	@python -m build --sdist --wheel --outdir dist .


.PHONY: check
check:
	@echo "Checking code format"
	@black --check $(SOURCE_FOLDER) $(TESTS_FOLDER) examples
	@isort --check $(SOURCE_FOLDER) $(TESTS_FOLDER)


.PHONY: docs
docs:
	@echo "Building documentation"
	@sphinx-build -W -b html $(DOCS_FOLDER) "$(DOCS_FOLDER)/_build"


.PHONY: tag
tag:
	@echo "Tagging current commit"
	@git tag --annotate "v$(PKG_VERSION)" --message "Tag v$(PKG_VERSION)"
	@git push --follow-tags


.PHONY: test
test:
	@echo "Testing code"
	@pytest -p no:cacheprovider
