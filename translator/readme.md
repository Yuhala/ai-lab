## Setting up Fairseq language translation model 

- Install useful packages
```
pip install fairseq sacremoses tensorboardX subword-nmt

```
- Download pretrained model and extract; I use english to french translation model here as an example. See [pre-trained Fairseq models](https://github.com/facebookresearch/fairseq/blob/main/examples/translation/README.md#example-usage-torchhub) for more pre-trained models.

```bash
mkdir -p models/en-fr
wget https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2
tar -xvjf model.tar.bz2 -C models/en-fr

```

- Load pretrained model with Fairseq

```python
import torch
from fairseq.models.transformer import TransformerModel

# Load the pretrained model
model = TransformerModel.from_pretrained(
    'models/en-fr',
    checkpoint_file='en-fr-model.pt',
    tokenizer='moses',  # Tokenizer used during training
    bpe='subword_nmt'   # Byte Pair Encoding used during training
)
```

- Test translate some text
```python
# Example sentences
sentences = [
    "Hello, how are you?",
    "This is a simple example.",
    "The weather is nice today."
]

for sentence in sentences:
    translation = model.translate(sentence)
    print(f"English: {sentence}")
    print(f"French: {translation}")
```

## Using Microsoft phi model from hugging face
- Hugging face is an open 
- Install requirements
```
pip install transformers torch
```
- Import useful APIs
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
```
- Import model
```python
model_name = "microsoft/phi-4"  # Replace with the correct Phi model name
cache_dir = "./models/phi"  # Specify a local directory for caching
```