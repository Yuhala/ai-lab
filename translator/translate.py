import torch
from fairseq.models.transformer import TransformerModel

# Load the pretrained model
model = TransformerModel.from_pretrained(
    'models/en-fr/wmt14.en-fr.joined-dict.transformer',
    checkpoint_file='model.pt',
    tokenizer='moses',  # Tokenizer used during training
    bpe='subword_nmt',   # Byte Pair Encoding used during training
    bpe_codes='models/en-fr/wmt14.en-fr.joined-dict.transformer/bpecodes'
)

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