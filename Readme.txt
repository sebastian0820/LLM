#This is a simple project to code evaluation system of LLM using Python.

##To run this project, you need to install virtual environment. If you are not familiar with python venv setting, please consult any material. Here's brief introduction.
python3 -m venv env
source env/bin/activate

##Required packages are pytorch, matplotlib, tiktoken, seaborn, numpy. If you are not familiar with python package installation, please consult any material. 
Simply, you can install those by typing this command on your bash.
pip install "package name"

##and then run showcase.py or principles.py. The other files are consisting class structure of LLM.

##GPTModel is the highest leveled class and it is consisted of TransformerBlock (backbone of LLM evaluation) and LayerNorm (normalization class).

##TransformerBlocker is consisted of LayerNorm, Attention (self-attention, mask mechanism), FeedForward (forward propagation procedure).

##FeedForward contains GELU (activation fucntion, the most ideal for non-linear solution at the moment).

##You can see difference and preferable feature of GELU compared with RELU (another widely-used activation function) in "difference between RELU and GELU". It draws graph.

##And also, running "test-activate function" will clarify you the mechanism of activate function via nice pictures.

##"short story.txt" is just a prepared training data. It will be used importantly in next chapter's code. Here, it is used to show you more brief result of evaluation.
principles.py used "short story.txt" as the source of its dictionary. Thus, its output will be consisted of words only in that ".txt" file. To achieve constructing dictionary in principles.py, we need to use SimpleTokenizerV2, which is implemented in chapter 2 for basic contetualization. Actually, it won't be used in real developing challenge, but here the most effective for fast, stable, normalized and vanishing gradients avoided understanding!!!!!
