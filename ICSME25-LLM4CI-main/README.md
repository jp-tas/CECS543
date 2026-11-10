# LLM for CI Replication Package


This repository contains the replication package for the paper "_Can LLMs Write CI? A Study on Automatic Generation of GitHub Actions Configurations_," accepted at the 41st IEEE International Conference on Software Maintenance and Evolution 2025 (ICSME'25). The package provides all resources needed to reproduce the experiments and results presented in the paper.


## Package Structure

```
project-root/
├── data/                         # Datasets and LLM outputs used in the study
│   ├── GitHubActions_Docs.csv         # Documentation data for GitHub Actions
│   ├── models_contexts.csv            # Contexts provided to LLMs
│   ├── GitHubActions_gemma3-12b_output.csv   # Output from Gemma 3-12b
│   ├── GitHubActions_codegemma-7b_output.csv # Output from CodeGemma 7b
│   ├── GitHubActions_gpt-4o_output.csv       # Output from GPT-4o
│   ├── GitHubActions_codellama-7b_output.csv # Output from CodeLlama 7b
│   ├── GitHubActions_gpt-4.1_output.csv      # Output from GPT-4.1
│   └── GitHubActions_llama3.1-8b_output.csv  # Output from Llama 3.1-8b
├── scripts/                      # Scripts for data collection, LLM prompting, and analysis
│   ├── ci_docs_selenium_crawler.py         # Crawls and collects CI documentation using Selenium
│   ├── prompt_llms.py                     # Prompts LLMs and collects their outputs
│   └── similarity_calculation_and_analysis.py # Calculates similarity scores and performs analysis
├── results/                      # Experiment outputs and analysis results
│   ├── GitHubActions_Similarity_Scores_Six_LLMs.csv   # Similarity scores for six LLMs
│   └── GitHubActions_Similarity_All_LLMs_boxplot.pdf  # Boxplot visualization of similarity scores
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Installation


This package was developed and tested with **Python 3.13.2**.

Clone the repository and install the required dependencies:


```bash
pip install -r requirements.txt
```

### Additional Requirements

- **Chromedriver**: Required for running `ci_docs_selenium_crawler.py`. Please [download Chromedriver](https://chromedriver.chromium.org/downloads) and ensure it is in your system PATH or specify its location in the script.
- **Ollama Framework**: Required for running LLMs via the `ollama` Python package. See [Ollama documentation](https://ollama.com/) for installation and setup instructions.
- **OpenAI API Key**: Required for using OpenAI models in `prompt_llms.py`. Set your API key as an environment variable:
  ```bash
  export OPENAI_API_KEY=your_api_key_here
  ```

## Usage



1. **Collect CI Documentation:**
   This script accepts documentation URLs for various CI services, but was tested on GitHub Actions Documentation.
   ```bash
   python scripts/ci_docs_selenium_crawler.py
   ```


2. **Prompt LLMs and collect outputs:**
   This script supports running a variety of LLMs. In our experiments, we evaluated two OpenAI models and four Ollama-supported models, but you may use it with other compatible models as well.
   - For Ollama models, ensure the Ollama framework is running.
   - For OpenAI models, ensure your API key is set.
   ```bash
   python scripts/prompt_llms.py
   ```

3. **Calculate similarity and analyze results:**
   This script computes similarity scores between LLM-generated and reference GitHub Actions configurations using multiple metrics (e.g., embedding-based, CHRF, ROUGE-L, tree edit distance). It also performs correlation analysis and statistical significance testing to compare model performance, and generates summary tables and visualizations (such as boxplots) for further analysis.
   ```bash
   python scripts/similarity_calculation_and_analysis.py
   ```

All outputs and analysis results will be saved in the `results/` directory.

## Data

The `data/` folder contains all datasets and LLM outputs used in the study. See the paper and script comments for details on each file.

## Results

The `results/` folder contains:
- Similarity scores for LLM-generated GitHub Actions configurations.
- Visualizations and summary statistics as reported in the paper.


## Docker Setup

If you want to run the experiments on a standalone image using Docker, follow these steps:

1. **Install Docker:**
   Download and install Docker for your platform from the official website: https://www.docker.com/products/docker-desktop. Follow the instructions for your operating system.

2. **Build the Docker Image:**
   From the project root, build the Docker image using the `Dockerfile` provided, as follows:
   ```bash
   docker build -t llm4ci-project .
   ```

3. **Run the Docker Image:**
   Now you can run the Docker image and provide your OpenAI API key, as follows:
   ```bash
   docker run -it --rm -e OPENAI_API_KEY=your_api_key_here llm4ci-project
   ```

4. **Execute the steps in the _Usage_ section above:**


## License

Code in this repository is licensed under the MIT License. See the LICENSE file.

Data files in this repository are licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license unless otherwise noted. You are free to share and adapt the data with appropriate credit.

## How to Cite

If you use this package, please cite our paper:

> Taher A. Ghaleb and Dulina Rathnayake. "Can LLMs Write CI? A Study on Automatic Generation of GitHub Actions Configurations." In Proceedings of the 41st IEEE International Conference on Software Maintenance and Evolution (ICSME), 2025.

```bibtex
@inproceedings{ghaleb2025llm4ci,
  title={Can LLMs Write CI? A Study on Automatic Generation of GitHub Actions Configurations},
  author={Ghaleb, Taher A. and Rathnayake, Dulina},
  booktitle={Proceedings of the 41st IEEE International Conference on Software Maintenance and Evolution (ICSME)},
  year={2025}
  organization={IEEE}
}
```