from tokenizers import Regex
from tokenizers.normalizers import (
    Lowercase,
    NFD,
    StripAccents,
    Replace,
    Strip,
    Sequence,
)
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
import torch
from typing import List, Union
import logging
import gc
from argparse import ArgumentParser
import os
from os.path import expanduser, join, exists
from os import listdir, cpu_count
import re




logger = logging.getLogger(__name__)


class SpokenNormalizer:
    """
    Normalizer (as in the `tokenizers` framework) which removes punctuation, force lowercase, etc
    """

    def __init__(self):
        self.normalizer = SpokenNormalizer.build_normalizer()

    def normalize_string(self, s):
        return self.normalizer.normalize_str(s)

    @staticmethod
    def build_normalizer():
        normalizer = Sequence(
            [
                NFD(),
                Lowercase(),
                StripAccents(),
                Replace(Regex(r'[\.\,\!\?\:\;\)\(\[\]"\-]'), ""),  # punctuation
                Replace(Regex(r"\s\s+"), " "),  # double spaces
                Strip(),
            ]
        )
        return normalizer




class tokenizer_AMI():
    @property
    def unk_token(self):
        return self.tokenizer.unk_token

    @property
    def unk_token_id(self):
        return self.tokenizer.unk_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id
    
    def __init__(self, tokenizer_type = 'gpt2', model_max_length = 1024):
        super().__init__()
        self.normalizer = SpokenNormalizer()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type, add_prefix_space=True)
        self.tokenizer.model_max_length = model_max_length
        speaker_tokens = ["<speaker1>", "<speaker2>", "<speaker3>", "<speaker4>"]
        self.tokenizer.add_special_tokens({'pad_token':'<pad>', "eos_token": "<ts>", 
                                           "additional_special_tokens": ["<speaker1>", "<speaker2>", "<speaker3>", "<speaker4>"]})
        self.map_from_speaker_id = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        # Speaker Tokens
        self.sp1_token = speaker_tokens[0]
        self.sp2_token = speaker_tokens[1]
        self.sp3_token = speaker_tokens[2]
        self.sp4_token = speaker_tokens[3]
        self.sp1_token_id = self.tokenizer.convert_tokens_to_ids(self.sp1_token)
        self.sp2_token_id = self.tokenizer.convert_tokens_to_ids(self.sp2_token)
        self.sp3_token_id = self.tokenizer.convert_tokens_to_ids(self.sp3_token)
        self.sp4_token_id = self.tokenizer.convert_tokens_to_ids(self.sp4_token)


    def __len__(self):
        return len(self.tokenizer)
    
    def adding_eos_list(self, words, speakers, word_ids, word_start_times, word_end_times):
        """
        Function:
          add eos_token to a list of words when turn-shifting happened.
        Args:
          words: list(str), a list of words 'before' normalized
          speakers: list(str), a list of speakers. Since variable 'words' is not normalized, len(speakers) == len(words)
          eos: str, eos token to be added to words
        Return:
          new_words: list(str), a list of words 'before' normalized including <eos>  
        """
        assert len(speakers) == len(words)
        eos = self.tokenizer.eos_token
        new_words = []
        new_speakers = []
        new_word_ids = []
        new_word_start_times = []
        new_word_end_times = []
        
        video_list = listdir('/ocean/projects/cis220078p/yasano/amicorpus')
        
        for i in range(len(words)-1):
            if word_ids[i][:7] in video_list:
                if words[i] not in ['.', ',', '!', '?', ':', ';',')', '(', '[',']','-',']']:
                    new_words.append(self.normalizer.normalize_string(words[i]))
                    new_speakers.append(self.map_from_speaker_id[speakers[i]])
                    new_word_ids.append(word_ids[i][:7])
                    new_word_start_times.append(word_start_times[i])
                    new_word_end_times.append(word_end_times[i])
                    
                if speakers[i] != speakers[i+1]:
                    new_words.append(eos)
                    new_speakers.append(self.map_from_speaker_id[speakers[i]])
                    new_word_ids.append(word_ids[i][:7])
                    new_word_start_times.append(word_end_times[i])
                    new_word_end_times.append(word_end_times[i])

        if word_ids[-1][:7] in video_list:
            if words[-1] not in ['.', ',', '!', '?', ':', ';',')', '(', '[',']','-',']']:
                new_words.append(self.normalizer.normalize_string(words[-1]))
                new_speakers.append(self.map_from_speaker_id[speakers[-1]])
                new_word_ids.append(word_ids[-1][:7])
                new_word_start_times.append(word_start_times[-1])
                new_word_end_times.append(word_end_times[-1])
      
        
        return new_words, new_speakers, new_word_ids, new_word_start_times, new_word_end_times
    
    def normalize(self, words):
        return self.normalizer.normalize_string(words)
                   

    def truncate(self, words, speakers, word_ids, word_start_times, word_end_times): 
        max_length = self.tokenizer.model_max_length
        new_words = []
        new_speakers = []
        new_word_ids = []
        new_word_start_times = []
        new_word_end_times = []

        while len(words) > max_length:
            new_words.append(words[0 :  max_length])
            new_speakers.append(speakers[0 : max_length])
            new_word_ids.append(word_ids[0 :  max_length])
            new_word_start_times.append(word_start_times[0 : max_length])
            new_word_end_times.append(word_end_times[0 :  max_length])
            
            words = words[max_length:]
            speakers = speakers[max_length:]
            word_ids = word_ids[max_length:]
            word_start_times = word_start_times[max_length:]
            word_end_times = word_end_times[max_length:]
          
        new_words.append(words)
        new_speakers.append(speakers)
        new_word_ids.append(word_ids)
        new_word_start_times.append(word_start_times)
        new_word_end_times.append(word_end_times)
        dataset = {'word': new_words, 'speaker_ids': new_speakers, 'word_ids': new_word_ids,
                   'word_start_times': new_word_start_times, 'word_end_times': new_word_end_times}

        return dataset

    def __call__(self, dialogue):
        """
        Function:
          create dictionary of dialogue: {token_ids: list of tokens, attention_mask: list, speaker_ids: tensor of speaker_id}
        Args:
          dialogue, retrieved by dataset['train'/'validation'/'test'][int] which contains:
              dialogue['words']: list(str), a list of words before normalized
              dialogue['word_speakers']: list(str), an order-synchronous list of speakers, one speaker corresponds to one word
              dialogue['word_ids']: list(str), a list of word_ids
              dialogue['word_start_times']: list(str), a list of word_ids
              dialogue['word_end_times']: list(str), a list of word_ids
        Return:
          data_list: {token_ids: list of tokens, attention_mask: list, speaker_ids: tensor of speaker_id}
        """
        words, speakers, word_ids, word_start_times, word_end_times = self.adding_eos_list(dialogue['words'], dialogue['word_speakers'], dialogue['word_ids'], dialogue['word_start_times'], dialogue['word_end_times'])
        
        embedding = self.tokenizer(words, is_split_into_words=True)
        word_ind_aligned = embedding.word_ids(0)
        
        token_aligned_speakers = []
        token_aligned_word_ids = []
        token_aligned_word_start_times = []
        token_aligned_word_end_times = []
        
        for i in word_ind_aligned:
            token_aligned_speakers.append(speakers[i])
            token_aligned_word_ids.append(word_ids[i])
            token_aligned_word_start_times.append(word_start_times[i])
            token_aligned_word_end_times.append(word_end_times[i])
            
        
        dataset = self.truncate(embedding['input_ids'], token_aligned_speakers, 
                                token_aligned_word_ids, token_aligned_word_start_times,
                                token_aligned_word_end_times)
        del words, speakers, word_ids, word_start_times, word_end_times
        del embedding, token_aligned_speakers, token_aligned_word_ids, token_aligned_word_start_times, token_aligned_word_end_times
        gc.collect()
        return dataset

    def idx_to_tokens(self, ids):
        def list_ids_to_string(ids):
            return [
                self.convert_tokens_to_string(t)
                for t in self.convert_ids_to_tokens(ids)
            ]

        # tokenize keep tokens
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        if isinstance(ids, list):
            if isinstance(ids[0], list):
                ret = [list_ids_to_string(ids_list) for ids_list in ids]
            else:
                ret = list_ids_to_string(ids)
        else:
            ret = self.convert_tokens_to_string(self.convert_ids_to_tokens(ids))
        return ret

    def pad(self, *args, **kwargs):
        return self.tokenizer.pad(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def convert_ids_to_tokens(self, *args, **kwargs):
        return self.tokenizer.convert_ids_to_tokens(*args, **kwargs)

    def convert_tokens_to_ids(self, *args, **kwargs):
        return self.tokenizer.convert_tokens_to_ids(*args, **kwargs)

    def convert_tokens_to_string(self, *args, **kwargs):
        return self.tokenizer.convert_tokens_to_string(*args, **kwargs).strip()
    
    
    def tokenize_old(
        self,
        text: Union[str, List[str], List[List[str]]],
        return_token_type_ids: bool = True,
        include_pre_space: bool = False,
        include_end_ts: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """
        SpokenDialogTokenizer tokenization, which is original tokenizing function in TurnGPT.
        Therefore, it would accept different data structure as follows, but only applicable when num_speakers = 2.

        `text` can be either a String, a List of Strings, or a List of Lists of Strings. The behaviour of
        this function depends on the `single_dialog` flag.

        `text` is String:           representation of entire dialog (including eos_token)
        `text` is List[str]:        representation of turns in a dialog (no eos_tokens)
        `text` is List[List[str]]:  multiple dialogs (lists of strings) (no eos_tokens)

        """

        # List of lists
        if isinstance(text, list) and isinstance(text[0], list):
            ret = {}
            for text_list in text:
                o = self.tokenize_old(
                    text_list,
                    include_pre_space=include_pre_space,
                    include_end_ts=include_end_ts,
                )

                for k, v in o.items():
                    if not k in ret:
                        ret[k] = []
                    ret[k].append(v)
            return ret

        # List of strings, a dialog: ['hello', 'hello to you']
        elif isinstance(text, List):
            dialog_string = ""
            if include_pre_space:
                dialog_string = " "
            dialog_string += self.normalizer.normalize_string(text[0])
            if len(text) > 1:
                dialog_string += self.eos_token
                for text_string in text[1:-1]:
                    dialog_string += " " + self.normalizer.normalize_string(text_string) + self.eos_token
                dialog_string += " " + self.normalizer.normalize_string(text[-1])
            if include_end_ts:
                dialog_string += self.eos_token
            text = dialog_string
        else:
            text = self.normalizer.normalize_string(text)

        encoding = self.tokenizer(
            text=text,
            **kwargs,
        )

        if return_token_type_ids:
            encoding["speaker_ids"] = self._extract_speaker_states_old(
                encoding["input_ids"]
            )
        return encoding

    def _extract_speaker_states_old(self, input_ids):
        
        #function called in tokenize_old
        
        # extract speaker states
        back_to_list = False
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids).unsqueeze(0)  # with batch dim
            back_to_list = True
        # initialize with speaker 1
        speaker_ids = torch.ones_like(input_ids) * self.sp1_token_id
        batch, eos_idx = torch.where(input_ids == self.eos_token_id)
        for b in batch.unique():
            tmp_eos = eos_idx[batch == b]
            if len(tmp_eos) == 1:
                speaker_ids[b, eos_idx + 1 :] = self.sp2_token_id
            else:
                start = tmp_eos[0]
                for i, eos in enumerate(tmp_eos[1:]):
                    if i % 2 == 0:
                        sp = self.sp2_token_id
                        speaker_ids[b, start + 1 : eos + 1] = sp
                    start = eos
                if i % 2 == 1:  # add sp2 tokens after last eos if i is odd
                    speaker_ids[b, start + 1 :] = self.sp2_token_id

        if back_to_list:
            speaker_ids = speaker_ids.squeeze().tolist()
            if isinstance(speaker_ids, int):
                speaker_ids = [speaker_ids]

        return speaker_ids
        



