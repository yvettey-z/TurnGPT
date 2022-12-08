from nltk.translate import bleu_score, meteor_score, nist_score
from typing import Dict, List

def zeroDivisionWrapper(func, references, hypothesis, **kwargs):
    try:
        score = func(references, hypothesis, **kwargs)
    except ZeroDivisionError:
        score = 0.
    return score

def calc_score(references, hypothesis) -> Dict[str, float]:
    '''
    Returns scores for language generation. Refer to NLTK documentation for the types 
    of `references` and `hypothesis`
    '''
    scores = {}
    scores['BLEU-2'] = zeroDivisionWrapper(bleu_score.sentence_bleu, references, hypothesis, weights=(0.5, 0.5))
    scores['BLEU-4'] = zeroDivisionWrapper(bleu_score.sentence_bleu, references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25))
    scores['METEOR'] = zeroDivisionWrapper(meteor_score.meteor_score, references, hypothesis)
    scores['NIST-2'] = zeroDivisionWrapper(nist_score.sentence_nist, references, hypothesis, n=2)
    scores['NIST-4'] = zeroDivisionWrapper(nist_score.sentence_nist, references, hypothesis, n=4)
    return scores

def process_for_nltk(sentence: str, tokenizer) -> List[str]:
    return sentence.replace(tokenizer.eos_token, ' ').replace(tokenizer.unk_token, ' {} '.format(tokenizer.unk_token)).strip().split()