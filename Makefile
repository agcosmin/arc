.PHONY: setup_lint lint format test

setup_lint:
	lintrunner init

lint:
	lintrunner

format:
	lintrunner format

test:
	pytest ./tests

