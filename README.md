# Derma-CoT: Fine-tuning Vision LLMs for CoT reasoning on medical images of skin conditions

This project focuses on fine-tuning Vision Language Models to demonstrate
Chain-of-Thought reasoning abilities for dermatological image analysis.

## Setup Instructions

### Prerequisites

- Python 3.12+
- Git
- uv

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/drmaniak/derma-cot.git
   cd derma-cot
   ```

2. Set up a virtual environment:

   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   uv sync
   ```

4. Set up pre-commit hooks:

   ```bash
   pre-commit install
   pre-commit run --all-files
   ```

## Development Workflow

### Branching Strategy

- `main`: Production-ready code
- `dev`: Development branch for integration
- Feature branches: Create from `dev` using the format `feature/your-feature-name`
- Bug fix branches: Use the format `fix/issue-description`

### Workflow

1. Create a new branch for your feature or fix:
2. Make your changes and commit them:
3. Push your branch to GitHub:
4. Create a Pull Request to merge into the `dev` branch

### Code Quality

- All code must pass pre-commit checks
- Write tests for new features
- Follow the project's coding style (enforced by Ruff)

## Testing

Run tests with pytest:

```bash
pytest
```

## Project Structure

- `derma_cot/`: Main package code
- `tests/`: Test suite
- `notes/`: Project notes and documentation
