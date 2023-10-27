setup-environment:
	@echo "Install requirements"
	@conda config --set channel_priority strict
	@conda env create -f environment.yml
	@echo "${MAGENTA}Remember to activate your environment ^${RESET}"

setup-package:
	@echo "Installing src in development mode with ${MAGENTA}pip install -e .${RESET}"
	@echo "Now you can use the current state of code in src anywhere in this environment"
	pip install -e .

run-pipeline:
	@echo "Running main.py"
	@python main.py