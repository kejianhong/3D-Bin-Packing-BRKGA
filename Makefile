PYTHON_FILES = $(wildcard brkga/*py)

.PHONY: help
help:
	@echo "\e[1m\e[93m3D Bin Packing Module In Python\e[0m"
	@echo "make format-view"
	@echo "\e[90m- view the change done by black and format\e[0m"
	@echo "make format"
	@echo "\e[90m- format all python file used black and format\e[0m"
	@echo "make check"
	@echo "\e[90m- run mypy against python3 code\e[0m"

.PHONY: format-view
format-view:
	@python3 -m black $(PYTHON_FILES) -l 65535 --diff
	@python3 -m isort $(PYTHON_FILES) -l 65535 --diff

.PHONY: format
format:
	@python3 -m black $(PYTHON_FILES) -l 65535
	@python3 -m isort $(PYTHON_FILES) -l 65535

.PHONY: check
check:
	@python3 -m mypy --config-file mypy.ini

.PHONY: print
print:
	@echo $(PYTHON_FILES)

.PHONY: all-examples
all-examples:
	@for i in $$(seq 0 7); do \
                echo $$i; \
                python3 -m example.example$$i; \
        done

