from utils import *
import nlpaug
import nlpaug.augmenter.word as naw
import pandas as pd
import numpy as np
from googletrans import Translator, constants
from typing import Any

def back_translate(text: str, translator: Any, src_lang: str = 'en', host_lang: str = 'jw') -> str:
    """
    Perform back-translation on a given text string using a specified translation service.

    This function takes a text string and translates it from the source language to the host language,
    and then back from the host language to the source language. The purpose of back-translation is often
    to augment textual data for natural language processing tasks or to understand the nuances of translation.

    Parameters:
    - text (str): The text string to be back-translated.
    - translator (Any): An instance of a translation service (such as from the `googletrans` library).
    - src_lang (str, optional): The ISO 639-1 language code of the source language (default is 'en' for English).
    - host_lang (str, optional): The ISO 639-1 language code of the host language (default is 'jw' for Javanese).

    Returns:
    - str: The back-translated text, originally from the source language and back to it via the host language.

    Note:
    - The 'translator' should have a 'translate' method compatible with the signature used in this function.
    - ISO 639-1 language codes should be used for the 'src_lang' and 'host_lang' parameters.
    """
    
    forth = translator.translate(
        text=text,
        src=src_lang,
        dest=host_lang
    )
    
    back = translator.translate(
        text=forth.text,
        src=host_lang,
        dest=src_lang
    )
    
    return back.text

from typing import Optional, Tuple
import pandas as pd
from nlpaug.augmenter.word import ContextualWordEmbsAug, RandomWordAug
from googletrans import Translator

def up_sample_and_augment_true_class(
    filename: Optional[str] = None,
    insert: bool = True,
    substitute: bool = True,
    translate: bool = True,
    delete: bool = True,
    swap: bool = True,
    host_lang: str = 'jw'
) -> pd.DataFrame:
    """
    Augment the positive class samples in a dataset by applying various NLP augmentation techniques 
    and optionally save the augmented dataset to a CSV file.

    This function performs data augmentation on textual data specifically targeting the samples 
    classified as the positive class (bin_label=True). It applies a series of augmentation 
    techniques including contextual insertions, substitutions, deletions, swaps, and back translation 
    depending on the flags provided. The function assumes that the dataset contains a 'text' and 
    'bin_label' columns.

    Parameters:
    - filename (Optional[str], optional): The path to save the augmented dataset as a CSV file. 
      If None, the dataset is not saved. Defaults to None.
    - insert (bool, optional): Flag to enable contextual word insertions. Defaults to True.
    - substitute (bool, optional): Flag to enable contextual word substitutions. Defaults to True.
    - translate (bool, optional): Flag to enable back translation. Defaults to True.
    - delete (bool, optional): Flag to enable random word deletions. Defaults to True.
    - swap (bool, optional): Flag to enable random word swaps. Defaults to True.
    - host_lang (str, optional): The intermediate language used for back translation. 
      Defaults to 'jw' (Javanese).

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the augmented texts and their associated binary labels.

    Note:
    - This function requires the nlpaug and googletrans libraries for augmentation and translation respectively.
    - It is assumed that a function `load_train_and_val` exists and is responsible for loading the training 
      and validation datasets.
    """
    train_set, _, _ = load_train_and_val()  # This function should be defined elsewhere
    positives = list(train_set['text'][train_set['bin_label'] == True])

    # Initialize augmentation results
    inserted, subbed, deleted, swapped, back_translated = [], [], [], [], []

    # Contextual insertions
    if insert:
        aug_insert = ContextualWordEmbsAug(
            model_path='bert-base-uncased',
            action="insert",
            aug_max=30
        )
        inserted = aug_insert.augment(positives)

    # Contextual substitutions
    if substitute:
        aug_sub = ContextualWordEmbsAug(
            model_path='bert-base-uncased',
            action="substitute",
            aug_max=30
        )
        subbed = aug_sub.augment(positives)

    # Deletions
    if delete:
        deleter = RandomWordAug(action='delete')
        deleted = deleter.augment(positives)

    # Swaps
    if swap:
        swapper = RandomWordAug(action='swap')
        swapped = swapper.augment(positives)

    # Compiling all augmented texts into a DataFrame
    augmented_train_df = pd.DataFrame({
        'test': inserted + subbed + back_translated + deleted + swapped,
        'bin_label': [True] * (len(inserted) + len(subbed) + len(back_translated) + len(deleted) + len(swapped)),
        'augment_type': ['inserted'] * len(inserted) + ['subbed'] * len(subbed) + ['back_translated'] * len(back_translated) + ['deleted'] * len(deleted) + ['swapped'] * len(swapped)

    })

    # Save to CSV if filename is provided
    if filename:
        augmented_train_df.to_csv(filename, index=False)

    # Back translation
    if translate:
        translator = Translator()
        for string in inserted + subbed: # Back translate augmented data
        # for string in inserted: # Back translate augmented data
            try:
                back_translated.append(back_translate(string, translator, host_lang=host_lang))
            except:
                print('ERROR')
                print(string)

    # Compiling all augmented texts into a DataFrame
    augmented_train_df = pd.DataFrame({
        'test': inserted + subbed + back_translated + deleted + swapped,
        'bin_label': [True] * (len(inserted) + len(subbed) + len(back_translated) + len(deleted) + len(swapped)),
        'augment_type': ['inserted'] * len(inserted) + ['subbed'] * len(subbed) + ['back_translated'] * len(back_translated) + ['deleted'] * len(deleted) + ['swapped'] * len(swapped)

    })

    # Save to CSV if filename is provided
    if filename:
        augmented_train_df.to_csv(filename + '_translate', index=False)
    
    return augmented_train_df