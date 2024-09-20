# LLM Backbone (Pretraining model)

👋This is a simple project to code evaluation system of LLM using Python.🧙‍♂️

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Prerequisites

Ensure you have the following installed on your machine:

- Python 3.12.4
- `pip` (Python package installer)
- `virtualenv` (optional but recommended)

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
     ```sh
    pip install pytorch, matplotlib, tiktoken, seaborn, numpy
    ```

## Usage

1. **Run the Script**:
    ```sh
    python showcase.py
    python principles.py
    ```

2. **Change of input in code so as to evaluate various string**:
    - **In showcase.py**: line 35 : start_context = "Type your text"
    - **In principles.py**: line 47 : text = "Type your text"

    The script will print different output for the same input because each file of both is modeled with different initiative structure : for showcase.py GPT2, for principles.py manually coded one.

## Project Structure

- `showcase.py`: Main script that contains the logic to predict next string for various input using GPT2.
- `principles.py`: Another main script that contains the logic to predict next string for various input using manually coded model.
- `short story.txt`: Texts for pretraining manually coded model. You can change it to see how the model's dictionary changes.
- `difference between RELU and GELU.py`: Show the graphical description how RELU and GELU differ in their function.

    The other files are class files used in above files. Do not manipulate those. 

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.
