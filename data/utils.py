from __future__ import print_function

import fnmatch
import io
import os, sys
import json
from tqdm import tqdm
import subprocess

SPECIAL_SPACE_CHARACTERS = ['\n', '\t', '\r']

def generate_label_from_corpora(corpus_paths, output_path=None, lower_case=True):
    """Generating label data from a given corpus folder file path(s)

    This function will generate label data by performing character level tokenization 
    over all path in the specified `corpus_paths` and store the result as a json formatted 
    file in the specified `output_path`. If path is a folder, the label file will be generated 
    based on all files with `.txt` format inside the given path recursively.

    Args:
        corpus_paths (list[str]): list of file or folder path of the text corpus
        output_path (str): file path for the generated label file
        lower_case (bool): flag for performing lower case on the text data

    Returns:
        list(str): The label list generated from the given `corpus_paths`
    """
    
    label_set = set()
    for corpus_path in corpus_paths:
        label_set |= retrieve_label_from_corpus(corpus_path, lower_case)
    label_list = list(label_set)

    if output_path:
        with open(output_path, 'w') as outfile:
            json.dump(label_list, outfile, ensure_ascii=False)

    return label_list

def retrieve_label_from_corpus(corpus_path, lower_case=True):
    """Retrieve all unique character labels from a given corpus folder or file path

    This function will generate json label file by performing character level 
    tokenization over `corpus_path`. If `corpus_`path is a folder, the character labels
    will be retrieved from  all files with `.txt` format inside the given `corpus_path`.

    Args:
        corpus_path (str): list of file or folder path of the text corpus
        lower_case (bool): flag for performing lower case on the text data

    Returns:
        set(str): The return character labels
    """
    label_set = set()
    if os.path.isdir(corpus_path):
        # Recursive search over folder
        for f_path in os.listdir(corpus_path):
            f_path = '{}/{}'.format(corpus_path, f_path)
            if  os.path.isdir(f_path) or f_path[-4:] == '.txt':
                label_set |= retrieve_label_from_corpus(f_path)
            else:
                # Skip non-folder and non-txt file
                pass
    elif corpus_path[-4:] == '.txt':
        # Perform character level tokenization over corpus
        with open(corpus_path,'r') as corpus_file:
            data = corpus_file.read()

            # Turn special character to space
            for c in SPECIAL_SPACE_CHARACTERS:
                data = data.replace(c,' ')

            # Perform lower case if needed
            if lower_case:
                data = data.lower()

            # Add to result set
            label_set |= set(data)
    else:
        # Skip non-folder and non-txt file
        pass
    return label_set

def create_manifest(data_path, output_path, min_duration=None, max_duration=None):
    file_paths = [os.path.join(dirpath, f)
                  for dirpath, dirnames, files in os.walk(data_path)
                  for f in fnmatch.filter(files, '*.wav')]
    file_paths = order_and_prune_files(file_paths, min_duration, max_duration)
    with io.FileIO(output_path, "w") as file:
        for wav_path in tqdm(file_paths, total=len(file_paths)):
            transcript_path = wav_path.replace('/wav/', '/txt/').replace('.wav', '.txt')
            sample = os.path.abspath(wav_path) + ',' + os.path.abspath(transcript_path) + '\n'
            file.write(sample.encode('utf-8'))
    print('\n')


def order_and_prune_files(file_paths, min_duration, max_duration):
    print("Sorting manifests...")
    duration_file_paths = [(path, float(subprocess.check_output(
        ['soxi -D \"%s\"' % path.strip()], shell=True))) for path in file_paths]
    if min_duration and max_duration:
        print("Pruning manifests between %d and %d seconds" % (min_duration, max_duration))
        duration_file_paths = [(path, duration) for path, duration in duration_file_paths if
                               min_duration <= duration <= max_duration]

    def func(element):
        return element[1]

    duration_file_paths.sort(key=func)
    return [x[0] for x in duration_file_paths]  # Remove durations

if __name__ == '__main__':
    # Test for generating label file
    print('Test Gen Label File')
    print(generate_label_from_corpora(['./test.txt'], output_path=None, lower_case=True))

    print('Test Gen Label File No Lower Case')
    print(generate_label_from_corpora(['./test.txt'], output_path=None, lower_case=False))

    print('Test Gen Label File to Json file')
    print(generate_label_from_corpora(['./test.txt'], output_path='./label_file.json', lower_case=True))

    print('Test Gen Label Folder')
    print(generate_label_from_corpora(['./test.txt', './test_folder'], output_path=None, lower_case=True))

    print('Test Gen Label Folder No Lower Case')
    print(generate_label_from_corpora(['./test.txt', './test_folder'], output_path=None, lower_case=False))

    print('Test Gen Label Folder to Json file')
    print(generate_label_from_corpora(['./test.txt', './test_folder'], output_path='./label_folder.json', lower_case=True))