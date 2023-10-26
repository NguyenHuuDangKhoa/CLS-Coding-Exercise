setup-package:
	@echo "Installing src in development mode with ${MAGENTA}pip install -e .${RESET}"
	@echo "Now you can use the current state of code in src anywhere in this environment"
	pip install -e .