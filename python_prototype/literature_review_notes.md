###AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE
* Split image into N patches of (P x P) -> Flatten patches -> Map to D dimensions with trainable linear projections ("patch embeddings") and add positional encoding
* Prepend learnable embedding whose output serves as y (true class)