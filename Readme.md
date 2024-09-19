# LLM Backbone (Pretraining model)

👋This is a simple project to code evaluation system of LLM using Python.🧙‍♂️

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Logging](#logging)
- [API Rate Limiting](#api-rate-limiting)
- [Contributing](#contributing)

## Prerequisites

Ensure you have the following installed on your machine:

- Python 3.12.4
- `pip` (Python package installer)
- `virtualenv` (optional but recommended)

## To run this project
you need to install virtual environment.
If you are not familiar with python venv setting, please consult any material. 
#### Here's brief introduction.
python3 -m venv env
source env/bin/activate

## Installation

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/yourusername/LLM.git
    cd LLM
    ```

2. **Create a Virtual Environment** (optional but recommended):
    ```sh
    python -m venv venv
    ```

3. **Activate the Virtual Environment**:

    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4. **Install Required Packages**:
    ```sh
    pip install -r requirements.txt
    ```

    or

    pip install pytorch, matplotlib, tiktoken, seaborn, numpy

## Usage

1. **Run the Script**:
    ```sh
    python showcase.py
    python principles.py
    ```

2. **Input Prompts**:
    - **State Name**: Enter the name of the state or press Enter for 'all'.
    - **City Name**: Enter the name of the city or press Enter for 'all'.
    - **Start Date**: Enter the start date in `YYYY-MM-DD` format or press Enter for '2014-01-01'.
    - **End Date**: Enter the end date in `YYYY-MM-DD` format or press Enter for '2023-12-31'.

    The script will fetch GitHub user data based on the provided inputs and save it into CSV files.

## Project Structure

- `main.py`: Main script that contains the logic to fetch and save GitHub user data.
- `states/`: Directory containing CSV files for each state with city information.
- `accounts/`: Directory where fetched account data will be saved.
- `users/`: Directory where fetched user data will be saved.
- `error.log`: Log file for capturing errors.

### -GPTModel
is the highest leveled class and it is consisted of TransformerBlock (backbone of LLM evaluation) and LayerNorm (normalization class).

### -TransformerBlocker
is consisted of LayerNorm, Attention (self-attention, mask mechanism), FeedForward (forward propagation procedure).

### -FeedForward
contains GELU (activation fucntion, the most ideal for non-linear solution at the moment).
You can see difference and preferable feature of GELU compared with RELU (another widely-used activation function) in "difference between RELU and GELU". It draws graph.

### *And also, running "test-activate function" will clarify you the mechanism of activate function via nice pictures.

### -"short story.txt"
is just a prepared training data. It will be used importantly in next chapter's code. Here, it is used to show you more brief result of evaluation.
principles.py used "short story.txt" as the source of its dictionary. Thus, its output will be consisted of words only in that ".txt" file. To achieve constructing dictionary in principles.py, we need to use SimpleTokenizerV2, which is implemented in chapter 2 for basic contetualization. Actually, it won't be used in real developing challenge, but here the most effective for fast, stable, normalized and vanishing gradients avoided understanding!!!!!


## Logging

Errors encountered during execution are logged in `error.log`. You can review this file to debug issues.

## API Rate Limiting

The script includes logic to handle GitHub API rate limiting by pausing execution and retrying requests when the rate limit is exceeded.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.
