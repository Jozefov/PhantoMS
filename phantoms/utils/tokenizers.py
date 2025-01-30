import pandas as pd
import typing as T
import selfies as sf
from tokenizers import ByteLevelBPETokenizer
from tokenizers.models import WordLevel
from tokenizers import Tokenizer, processors
from tokenizers.implementations import BaseTokenizer, ByteLevelBPETokenizer
import massspecgym.utils as utils  # Update the import path if necessary
from phantoms.utils.constants import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN

class SpecialTokensBaseTokenizer(BaseTokenizer):
    def __init__(
        self,
        tokenizer: Tokenizer,
        max_len: T.Optional[int] = None,
    ):
        """Initialize the base tokenizer with special tokens performing padding and truncation."""
        super().__init__(tokenizer)

        # Save the tokenizer as an instance attribute
        self.tokenizer = tokenizer

        # Save essential attributes
        self.pad_token = PAD_TOKEN
        self.sos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.unk_token = UNK_TOKEN
        self.max_length = max_len

        # Add special tokens
        self.tokenizer.add_special_tokens([self.pad_token, self.sos_token, self.eos_token, self.unk_token])

        # Get token IDs
        self.pad_token_id = self.tokenizer.token_to_id(self.pad_token)
        self.sos_token_id = self.tokenizer.token_to_id(self.sos_token)
        self.eos_token_id = self.tokenizer.token_to_id(self.eos_token)
        self.unk_token_id = self.tokenizer.token_to_id(self.unk_token)

        # Enable padding
        self.tokenizer.enable_padding(
            direction="right",
            pad_token=self.pad_token,
            pad_id=self.pad_token_id,
            length=max_len,
        )

        # Enable truncation
        self.tokenizer.enable_truncation(max_length=max_len)

        # Set post-processing to add SOS and EOS tokens
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.sos_token} $A {self.eos_token}",
            pair=f"{self.sos_token} $A {self.eos_token} {self.sos_token} $B {self.eos_token}",
            special_tokens=[
                (self.sos_token, self.sos_token_id),
                (self.eos_token, self.eos_token_id),
            ],
        )

    def train(
        self,
        texts: T.List[str],
        vocab_size: int = 1000,
        min_frequency: int = 2,
        model_type: str = "wordlevel",
    ):
        """
        Trains the tokenizer on the provided texts.

        Args:
            texts (List[str]): List of strings to train the tokenizer on.
            vocab_size (int, optional): Size of the vocabulary. Relevant for certain tokenizer models.
            min_frequency (int, optional): Minimum frequency for a token to be included.
            model_type (str, optional): Type of tokenizer model to use ('wordlevel', 'bytelevel_bpe', etc.).
                This parameter can be utilized by subclasses to specify the model.
        """
        raise NotImplementedError("The 'train' method should be implemented in subclasses.")


class SelfiesTokenizer(SpecialTokensBaseTokenizer):
    def __init__(self, **kwargs):
        """
        Initialize the SELFIES tokenizer.

        Args:
            **kwargs: Additional keyword arguments to pass to the base class.
        """
        # Initialize the WordLevel tokenizer for SELFIES
        tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOKEN))
        super().__init__(tokenizer, **kwargs)

    def train(
            self,
            selfies_list: T.List[str],
            min_frequency: int = 1,
    ):
        """
        Trains the SELFIES tokenizer.

        Args:
            selfies_list (List[str]): List of SELFIES strings to train the tokenizer on.
            min_frequency (int, optional): Minimum frequency for a token to be included.
        """
        print(f"Training SELFIES WordLevel Tokenizer with min_frequency={min_frequency}...")
        self.tokenizer.train_from_iterator(
            selfies_list,
            vocab_size=None,  # SELFIES uses a fixed set of symbols; vocab_size is not required
            min_frequency=min_frequency,
            special_tokens=[self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        )
        print("Training complete.")

    def encode_batch(
            self, texts: T.List[str], add_special_tokens: bool = True
    ) -> T.List[T.List[int]]:
        """Encodes a batch of SMILES strings into a list of SELFIES token IDs."""
        selfies_strings = [
            list(sf.split_selfies(sf.encoder(text, strict=False))) for text in texts
        ]
        encoded = self.tokenizer.encode_batch(
            selfies_strings, is_pretokenized=True, add_special_tokens=add_special_tokens
        )
        return [enc.ids for enc in encoded]

    def decode_batch(
            self, token_ids_batch: T.List[T.List[int]], skip_special_tokens: bool = True
    ) -> T.List[str]:
        """Decodes a batch of SELFIES token IDs back into SMILES strings."""
        decoded_selfies = self.tokenizer.decode_batch(
            token_ids_batch, skip_special_tokens=skip_special_tokens
        )
        decoded_smiles = [
            sf.decoder(self._decode_wordlevel_str_to_selfies(s))
            for s in decoded_selfies
        ]
        return decoded_smiles

    def _decode_wordlevel_str_to_selfies(self, text: str) -> str:
        """Converts a WordLevel string back to a SELFIES string."""
        return text.replace(" ", "")


class SmilesBPETokenizer(SpecialTokensBaseTokenizer):
    def __init__(self, **kwargs):
        """
        Initialize the BPE tokenizer for SMILES strings.

        Args:
            **kwargs: Additional keyword arguments to pass to the base class.
        """
        # Initialize the ByteLevelBPETokenizer
        tokenizer = ByteLevelBPETokenizer()
        super().__init__(tokenizer, **kwargs)

    def train(
        self,
        smiles_list: T.List[str],
        vocab_size: int = 1000,
        min_frequency: int = 2,
    ):
        """
        Trains the Byte-Level BPE tokenizer on SMILES strings.

        Args:
            smiles_list (List[str]): List of SMILES strings to train the tokenizer on.
            vocab_size (int, optional): Size of the vocabulary.
            min_frequency (int, optional): Minimum frequency for a token to be included.
        """
        print(f"Training SMILES Byte-Level BPE Tokenizer with vocab_size={vocab_size}, min_frequency={min_frequency}...")
        self.tokenizer.train_from_iterator(
            smiles_list,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        )
        print("Training complete.")