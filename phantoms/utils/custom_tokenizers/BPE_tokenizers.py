import typing as T
from tokenizers import Tokenizer, models, trainers
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing
from phantoms.utils.constants import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN


class ByteBPETokenizerWithSpecialTokens:
    def __init__(
        self,
        max_len: T.Optional[int] = None,
        tokenizer_path: T.Optional[str] = None,
    ):
        """
        Initialize the Byte-Level BPE Tokenizer with special tokens, padding, and truncation.

        Args:
            max_len (int, optional): Maximum length for tokenized sequences. Defaults to None.
            tokenizer_path (str, optional): Path to a pre-trained tokenizer JSON file.
                                            If provided, loads the tokenizer from this path.
        """
        if tokenizer_path:
            # Load an existing tokenizer
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            print(f"Loaded tokenizer from {tokenizer_path}.")
        else:
            # Initialize a new Byte-Level BPE Tokenizer with <unk> as the unknown token
            bpe_model = models.BPE(unk_token=UNK_TOKEN)
            self.tokenizer = Tokenizer(bpe_model)

            # Set pre-tokenizer and de_novo_scripts
            self.tokenizer.pre_tokenizer = ByteLevel()
            self.tokenizer.decoder = ByteLevelDecoder()

            print("Initialized a new Byte-Level BPE Tokenizer.")

        self.max_length = max_len

        # Define and add special tokens
        self.special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.tokenizer.add_special_tokens(self.special_tokens)

        # Assign special token IDs
        self.pad_token_id = self.tokenizer.token_to_id(PAD_TOKEN)
        self.sos_token_id = self.tokenizer.token_to_id(SOS_TOKEN)
        self.eos_token_id = self.tokenizer.token_to_id(EOS_TOKEN)
        self.unk_token_id = self.tokenizer.token_to_id(UNK_TOKEN)

        # Enable padding and truncation if max_length is specified
        if self.max_length:
            self.tokenizer.enable_padding(
                direction="right",
                pad_token=PAD_TOKEN,
                pad_id=self.pad_token_id,
                length=self.max_length,
            )
            self.tokenizer.enable_truncation(max_length=self.max_length)

        # Set post-processing to add SOS and EOS tokens
        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{SOS_TOKEN} $A {EOS_TOKEN}",
            pair=f"{SOS_TOKEN} $A {EOS_TOKEN} {SOS_TOKEN} $B {EOS_TOKEN}",
            special_tokens=[
                (SOS_TOKEN, self.sos_token_id),
                (EOS_TOKEN, self.eos_token_id),
            ],
        )

    def train(
        self,
        texts: T.List[str],
        vocab_size: int = 1000,
        min_frequency: int = 2,
        save_path: T.Optional[str] = None,
        show_progress: bool = True,
    ):
        """
        Train the Byte-Level BPE Tokenizer on the provided texts.

        Args:
            texts (List[str]): List of strings (SMILES or SELFIES) to train the tokenizer on.
            vocab_size (int, optional): Size of the vocabulary. Defaults to 1000.
            min_frequency (int, optional): Minimum frequency a token must have to be included. Defaults to 2.
            save_path (str, optional): Path to save the trained tokenizer JSON file. If None, tokenizer is not saved.
            show_progress (bool, optional): Whether to display a progress bar during training. Defaults to True.
        """
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.special_tokens,
            show_progress=show_progress,
            initial_alphabet=ByteLevel.alphabet(),
        )

        print(f"Starting training on {len(texts)} texts...")
        self.tokenizer.train_from_iterator(
            texts,
            trainer=trainer,
            length=len(texts)
            # Removed 'batch_size' parameter to avoid TypeError
        )
        print("Training complete.")

        if save_path:
            self.tokenizer.save(save_path)
            print(f"Tokenizer saved to {save_path}.")

    def encode(self, text: str, add_special_tokens: bool = True) -> T.List[int]:
        """
        Encode a single string into token IDs.

        Args:
            text (str): The string (SMILES or SELFIES) to encode.
            add_special_tokens (bool, optional): Whether to add special tokens. Defaults to True.

        Returns:
            List[int]: List of token IDs.
        """
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

        return encoding.ids

    def decode(self, token_ids: T.List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs back into a string.

        Args:
            token_ids (List[int]): List of token IDs to decode.
            skip_special_tokens (bool, optional): Whether to skip special tokens. Defaults to True.

        Returns:
            str: The decoded string (SMILES or SELFIES).
        """
        decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        return decoded_text

    def encode_batch(
        self, texts: T.List[str], add_special_tokens: bool = True
    ) -> T.List[T.List[int]]:
        """
        Encode a batch of strings into lists of token IDs.

        Args:
            texts (List[str]): List of strings (SMILES or SELFIES) to encode.
            add_special_tokens (bool, optional): Whether to add special tokens. Defaults to True.

        Returns:
            List[List[int]]: List of token ID lists.
        """

        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        token_ids_batch = [enc.ids for enc in encodings]

        return token_ids_batch


    def decode_batch(
        self, token_ids_batch: T.List[T.List[int]], skip_special_tokens: bool = True
    ) -> T.List[str]:
        """
        Decode a batch of token ID lists back into strings.

        Args:
            token_ids_batch (List[List[int]]): Batch of token ID lists.
            skip_special_tokens (bool, optional): Whether to skip special tokens. Defaults to True.

        Returns:
            List[str]: List of decoded strings (SMILES or SELFIES).
        """
        decoded_texts = [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in token_ids_batch]

        return decoded_texts

    def get_vocab_size(self) -> int:
        """
        Get the size of the tokenizer's vocabulary.

        Returns:
            int: Vocabulary size.
        """
        vocab_size = self.tokenizer.get_vocab_size()
        return vocab_size

    def get_vocab(self) -> T.Dict[str, int]:
        """
        Get the tokenizer's vocabulary.

        Returns:
            Dict[str, int]: A dictionary mapping tokens to their corresponding IDs.
        """
        vocab = self.tokenizer.get_vocab()
        return vocab

    def token_to_id(self, token: str) -> int:
        """
        Convert a token to its corresponding ID.

        Args:
            token (str): The token to convert.

        Returns:
            int: The token ID.
        """
        token_id = self.tokenizer.token_to_id(token)
        return token_id

    def id_to_token(self, id_: int) -> str:
        """
        Convert a token ID back to its corresponding token.

        Args:
            id_ (int): The token ID to convert.

        Returns:
            str: The corresponding token.
        """
        token = self.tokenizer.id_to_token(id_)
        return token