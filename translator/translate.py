# 
# Peterson Yuhala
# Simple program that uses a pretrained Fairseq model for English to French translation
#
#import torch

from fairseq.models.transformer import TransformerModel

model_path = '../../models/en-fr/wmt14.en-fr.joined-dict.transformer'
bpe_code_path = '../../models/en-fr/wmt14.en-fr.joined-dict.transformer/bpecodes'


# Load the pretrained model
model = TransformerModel.from_pretrained(
    model_path,
    checkpoint_file='model.pt',
    tokenizer='moses',  # Tokenizer used during training
    bpe='subword_nmt',   # Byte Pair Encoding used during training
    bpe_codes=bpe_code_path, 
    
)

# Example sentences to translate
sentences = [
    "Hello, how are you?",
    "This is a simple example.",
    "The weather is nice today."
]

"""for sentence in sentences:
    translation = model.translate(sentence)
    print(f"English: {sentence}")
    print(f"French: {translation}")
"""    
print("Type a phrase in English to translate to French (type 'exit' to quit):")

while True:
    user_input = input("Enter phrase: ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break
    translation = model.translate(user_input)
    print(f"French translation: {translation}")
    