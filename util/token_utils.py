
def encode(tokenizer, prompt, max_length, return_tensors=None):
    encodings_dict = tokenizer(prompt,
                               truncation=True,
                               padding="max_length",
                               max_length=max_length,
                               add_special_tokens=False,
                               return_tensors=return_tensors)

    return encodings_dict


def decode(tokenizer, input_ids):
    return tokenizer.decode(input_ids, skip_special_tokens=True, ).strip()
