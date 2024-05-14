DOCS_FOLDER   = "docs"
SOURCE_FOLDER = "madminer"
TESTS_FOLDER  = "tests"

.PHONY: build
build:
	@echo "Building package"
	@python -m build --installer uv .


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
	@git tag --annotate "v$(shell python -c 'from importlib.metadata import version; print(version("madminer"))')" \
	--message "Tag v$(shell python -c 'from importlib.metadata import version; print(version("madminer"))')"
	@git push --follow-tags


.PHONY: test
test:
	@echo "Testing code"
	@pytest -p no:cacheprovider
