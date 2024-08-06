import numpy as np
import pandas as pd
import codecs
from tqdm import tqdm
from typing import Tuple, Dict, List

from sklearn.model_selection import train_test_split

def load_train_and_val():
    """
    Load training and validation datasets for a text classification task.

    This function reads a TSV file containing text data and labels, assigns column names,
    and converts the numeric labels into binary labels. It then reads two separate CSV files
    containing IDs for training and validation sets, respectively, and segments the data into
    training and validation datasets based on these IDs.

    Returns:
        train_set (pandas.DataFrame): The training set containing paragraphs' IDs, keywords,
                                      country codes, text data, and binary labels.
        val_set (pandas.DataFrame): The validation set structured like the training set,
                                    containing only the entries corresponding to the validation IDs.
    """
    # Load the dataset, skipping the first four lines, and set column names
    train_data = pd.read_csv('data/dontpatronizeme_pcl.tsv', skiprows=[0,1,2,3], sep='\t', header=None)
    train_data.columns = ['par_id', 'art_id', 'keyword', 'countrycode', 'text', 'label']
    
    # Convert numeric labels into binary labels
    train_data['bin_label'] = train_data['label'] >= 2

    # Load the training and test (validation) IDs
    train_ids = pd.read_csv('data/train_semeval_parids-labels.csv')
    test_ids = pd.read_csv('data/dev_semeval_parids-labels.csv')

    # Segment the data into training and validation sets based on the IDs
    train_set = train_data[train_data['par_id'].isin(train_ids.par_id)]
    val_set = train_data[train_data['par_id'].isin(test_ids.par_id)]

    test_set = val_set.copy()
    train_set, val_set = train_test_split(train_set, test_size=0.3, random_state=42)

    return train_set, val_set, test_set


def load_glove(glove_path: str = 'glove.6B.300d.txt') -> Tuple[Dict[str, int], Dict[int, str], np.ndarray]:
    """
    Load the GloVe word embedding data from a file.

    This function reads the GloVe embedding data from the specified file. It constructs
    three data structures: a word-to-index dictionary, an index-to-word dictionary,
    and an array of word vectors. Note that the first line of the GloVe data file,
    which typically contains the vocabulary size and the dimensionality, is ignored.

    Parameters:
    - glove_path (str): Path to the GloVe data file. Default is 'glove.6B.300d.txt'.

    Returns:
    - Tuple containing:
      - w2i (Dict[str, int]): Dictionary mapping words to their indices in the vector space.
      - i2w (Dict[int, str]): Dictionary mapping indices in the vector space to their corresponding words.
      - wvecs (np.ndarray): Array of word vectors, where each row corresponds to a word's vector.

    Note:
    - The file at glove_path is expected to be a large file; loading may take a while.
    - Each word vector is assumed to be in the subsequent lines after the first, separated by spaces.
    - This function uses tqdm to show progress, which can be helpful for very large files.
    """
    # Initialize containers for word to index mapping, index to word mapping, and word vectors
    w2i = []  # word2index
    i2w = []  # index2word
    wvecs = []  # word vectors

    # Read file and construct mappings and vectors
    with codecs.open(glove_path, 'r', 'utf-8') as f:
        index = 0
        for line in tqdm(f.readlines()):
            # Skip the first line and process the rest
            if len(line.strip().split()) > 3:
                word, vec = line.strip().split()[0], list(map(float, line.strip().split()[1:]))
                wvecs.append(vec)
                w2i.append((word, index))
                i2w.append((index, word))
                index += 1

    # Convert lists to appropriate data structures
    w2i = dict(w2i)
    i2w = dict(i2w)
    wvecs = np.array(wvecs)

    return w2i, i2w, wvecs


def load_pcl_categories():
    """
    Returns the dataset which contains the PCL categories
    """
    df = pd.read_csv('data/dontpatronizeme_categories.tsv', skiprows=[0,1,2,3], sep='\t', header=None)
    df.columns = ['par_id', 'art_id', 'text', 'keyword', 'country_code', 'span_start', 'span_finish', 'span_text', 'pcl_category', 'num_annotators']
    df['text_len'] = df['text'].str.len()
    df['num_categories'] = df.groupby('par_id')['pcl_category'].transform('size')
    return df


def expand_train_data(df_train):
    """
    One category per patronising text
    """
    pass

def load_augmented(inserted: bool = True,
                   subbed: bool = True,
                   back_translated: bool = True,
                   deleted: bool = True,
                   swapped: bool = True) -> pd.DataFrame:
    """
    Loads and filters augmented data based on specified augmentation techniques.

    This function loads a dataset containing augmented training data with labels from a predefined CSV file. 
    It filters this dataset to return only the rows corresponding to the augmentation techniques that have been 
    enabled via the function's parameters.

    Parameters:
    - inserted (bool): If True, includes data augmented by insertion of words. Default is True.
    - subbed (bool): If True, includes data augmented by substituting words. Default is True.
    - back_translated (bool): If True, includes data augmented by back-translation. Default is True.
    - deleted (bool): If True, includes data augmented by deletion of words. Default is True.
    - swapped (bool): If True, includes data augmented by swapping words. Default is True.

    Returns:
    - pd.DataFrame: A DataFrame containing the filtered augmented data based on the specified techniques.

    Examples:
    - To load all augmented data:
        >>> load_augmented()
    - To load only back-translated and deleted data:
        >>> load_augmented(inserted=False, subbed=False, swapped=False)
    """

    augmented_full = pd.read_csv('data/augmented_train_positives_with_labels_translate.csv')

    aug_techniques_available = {
        "inserted": inserted,
        "subbed": subbed,
        "back_translated": back_translated,
        "deleted": deleted, 
        "swapped": swapped
        }

    aug_techniques_to_use = []
    
    for aug_tech in aug_techniques_available:
        print(aug_tech, aug_techniques_available[aug_tech])
        if aug_techniques_available[aug_tech]:
            aug_techniques_to_use.append(aug_tech)

    mask = augmented_full['augment_type'].isin(aug_techniques_to_use)

    return augmented_full[mask]