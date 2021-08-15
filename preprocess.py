import argparse
import collections
import copy
import os
import random
import re
import sys

import nltk
from tqdm import tqdm


class Dictionary:

    UNK = "[unk]"
    PAD = "[pad]"
    NUM = "[num]"

    def __init__(self):
        self._reset()

    def _reset(self):
        self.word2index = dict()
        self.index2word = list()
        self.word_freq = dict()
        self.add_a_word(self.PAD)
        self.add_a_word(self.UNK)
        self.add_a_word(self.NUM)

    def add_a_sentence(self, sentence):
        """Add a list of words into dictionary
        """
        for word in sentence:
            self.add_a_word(word)

    def add_a_word(self, word):
        """Add a single word into dictionary
        """
        if word not in self.word2index:
            self.word2index[word] = len(self.index2word)
            self.index2word.append(word)
            self.word_freq[word] = 1
        else:
            self.word_freq[word] += 1

    def drop_rare_words(self, threshold=2):
        """Drop words with occurance less than `threshold`
        """
        word_freq = copy.deepcopy(self.word_freq)
        print('Before dropping rare words, vocab size is {}'.format(
            len(self.word2index)))

        self._reset()
        for word, freq in word_freq.items():
            if freq >= threshold:
                self.add_a_word(word)
        print('After dropping rare words, vocab size is {}'.format(
            len(self.word2index)))

        # keep the orginal frequency table
        self.word_freq = word_freq

    def __contains__(self, word):
        """Check if a word is contained in the dictionary
        """
        return word in self.word2index

    def __getitem__(self, key):
        """Get index of a word, return `UNK` if not found
        """
        if key in self.word2index:
            return self.word2index[key]
        return self.word2index[self.UNK]

    def __len__(self):
        """Length of the vocabulary size
        """
        return len(self.index2word)


class Sentence:

    # regex for comma seperated numbers, e.g. 10,000
    number_regex = re.compile("^([0-9]{1,3})(,[0-9]{3})*$")

    punct_table = {
        "ptb": {",", ".", ":", "``", "''", "-LRB-", "-RRB-", "#", "LS", "LST", "SYM"},
        "ktb": {"PU", "-LRB-", "-RRB-", "LS", "LST", "META", "CODE", "QUOT"}
    }

    @classmethod
    def normalize_a_word(cls, word):
        ''' 1. lowercase the word
            2. replace number with `[num]`
        '''
        word = word.lower()

        if cls.number_regex.match(word):
            return Dictionary.NUM

        try:
            float(word)
            return Dictionary.NUM
        except ValueError:
            return word

    def replace_rare_words_in_tree(self, dictionary):
        def replace_rare_words(tree):
            nonlocal dictionary

            if tree.height() == 2:  # preterminal nodes
                if tree[0] not in dictionary:
                    tree[0] = dictionary.UNK
                return

            for child in tree:
                replace_rare_words(child)

        replace_rare_words(self.tree)

    def collapse_unary(self, collapse_root=False):
        tree = self.tree
        if not collapse_root and isinstance(tree, nltk.Tree) and len(tree) == 1:
            nodeList = [tree[0]]
        else:
            nodeList = [tree]

        # depth-first traversal of tree
        while nodeList != []:
            node = nodeList.pop()
            if isinstance(node, nltk.Tree):
                if len(node) == 1 and isinstance(node[0], nltk.Tree):
                    node[0:] = [child for child in node[0]]
                    nodeList.append(node)
                else:
                    for child in node:
                        nodeList.append(child)

    def is_ktb_null_element(self, sub):
        """Check if a subtree is a null element in KTB.
        """
        if self.data_type != "ktb":
            return False
        if len(sub.pos()) != 1:
            return False
        text = sub.pos()[0][0]
        # null elements without indexing
        if text.startswith("*") and text.endswith("*"):
            return True
        # null elements with indexing
        if re.match("^\*(.*)\*-[0-9]+$", text):
            return True
        return False

    def remove_nodes(self, tags_to_remove):
        '''Remove nodes with certain tags specified in `tags_to_remove`,
           Note: if the tag of a subtree is in `tags_to_remove`, 
               the whole subtree will be removed
        '''
        for sub in reversed(list(self.tree.subtrees())):
            if sub.label() in tags_to_remove or self.is_ktb_null_element(sub):
                position = sub.treeposition()
                if not position:
                    continue
                parent_position = position[:-1]
                parent = self.tree[parent_position]
                while parent and len(parent) == 1:
                    sub = parent
                    position = sub.treeposition()
                    parent_position = position[:-1]
                    if not parent_position:
                        break
                    parent = self.tree[parent_position]
                del self.tree[sub.treeposition()]

    def clean_tree(self, tree):
        """Clean terminals and nonterminals in the parse tree
        """
        label = tree.label()
        if label not in ["-RRB-", "-LRB-"]:
            label = re.split("[-_=]", label)[0]
        tree.set_label(label)

        if tree.height() == 2:
            tree[0] = self.normalize_a_word(tree[0])
        else:
            for child in tree:
                self.clean_tree(child)

    def __init__(self, raw, punct_type, remove_punct):
        if isinstance(raw, nltk.ParentedTree):
            self.tree = raw
        elif isinstance(raw, nltk.Tree):
            self.tree = nltk.ParentedTree.convert(raw)
        elif isinstance(raw, str):
            self.tree = nltk.ParentedTree.fromstring(
                raw, remove_empty_top_bracketing=True)
        else:
            raise ValueError(
                'Unexpected type of sentence: {}'.format(type(raw)))

        self.data_type = punct_type
        self.punct_set = self.punct_table[punct_type]

        # remove null elements and ID
        try:
            self.remove_nodes({"-NONE-", "ID", "META", "LS", "LST"})
            if remove_punct:
                self.remove_nodes(self.punct_set)
        except IndexError as e:
            # print("The whole tree is deleted:")
            # print(repr(raw))
            self.tree = None
            pass

        # clean terminals and nonterminals
        if self.tree is not None:
            self.clean_tree(self.tree)
            self.tree = nltk.Tree.fromstring(
                str(self.tree), remove_empty_top_bracketing=True)
            self.collapse_unary(punct_type == "ktb")
            self.tree = nltk.ParentedTree.fromstring(
                str(self.tree), remove_empty_top_bracketing=True)

        if self.tree is None or len(self) == 0:
            self.tree = None

    @property
    def pos(self):
        return self.tree.pos()

    @property
    def words(self):
        return [w for w, p in self.pos]

    def __repr__(self):
        return self.tree.pformat(margin=sys.maxsize)

    def __len__(self):
        return len(self.pos)

    @property
    def n_words(self):
        return sum([s not in self.punct_set for w, s in self.pos])


class Dataset:
    def __init__(self, train_sents, dev_sents, test_sents,
                 min_occurrence=2, max_len=None):
        """Dataset class
            train_sents: A list of Sentence objects, used for training
            dev_sents: dev set
            test_sents: test set
            min_occurrence: Drop words which occur less than this number
            max_len: Maximimum lenght of sentences to keep
        """
        self.dictionary = Dictionary()
        self.data = dict()

        self.data['train'] = self.build(train_sents, max_len, 'train')
        self.dictionary.drop_rare_words(min_occurrence)
        self.drop_rare_words_in_training_set()

        self.data['dev'] = self.build(dev_sents, max_len, 'dev')
        self.data['test'] = self.build(test_sents, None, 'test')

    def build(self, sentences, max_len, mode):
        new_sentences = list()
        for s in tqdm(sentences, desc="Building {}".format(mode)):
            if max_len is not None and len(s) > max_len:
                continue
            if mode == 'train':
                self.dictionary.add_a_sentence(s.words)
            else:
                s.replace_rare_words_in_tree(self.dictionary)
            new_sentences.append(s)
        return new_sentences

    def drop_rare_words_in_training_set(self):
        for sent in self.data['train']:
            sent.replace_rare_words_in_tree(self.dictionary)

    def print_statistics(self):
        print('Vocab size: {}'.format(len(self.dictionary)))
        for k, v in self.data.items():
            print('{} sentences for {}'.format(len(v), k))

    def write_to_folder(self, base_dir, sformat='bracket'):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        for key in self.data.keys():
            filepath = os.path.join(base_dir, key)
            with open(filepath, "w") as f:
                for sent in self.data[key]:
                    if sformat == 'bracket':
                        tree_string = str(sent)
                        print(tree_string, file=f)
                    elif sformat == 'txt':
                        print(' '.join(sent.words), file=f)
                    else:
                        raise ValueError


class PTBHelper:
    def __init__(self, path_to_raw_ptb):
        self.path_to_raw_ptb = path_to_raw_ptb

    def penn_treebank_iterator(self, mode, base_dir, remove_punct=False):
        """If the whole parse tree of a sentence is deleted, we will still return that sentence
        """
        if mode == 'train':
            sections = range(2, 22)
        elif mode == 'dev':
            sections = range(22, 23)
        elif mode == 'test':
            sections = range(23, 24)
        else:
            raise ValueError(mode)

        for section in tqdm(sections, desc="Iterate over sections"):
            section_dir = os.path.join(base_dir, "{:02d}".format(section))
            for filename in sorted(os.listdir(section_dir)):
                if not filename.endswith(".mrg"):
                    continue
                path = os.path.join(section_dir, filename)
                cur_str = ""
                n_left_brackets = n_right_brackets = 0
                with open(path, "r") as fp:
                    for line in fp:
                        if line == "\n":
                            continue
                        cur_str += line
                        n_left_brackets += line.count("(")
                        n_right_brackets += line.count(")")
                        if n_left_brackets == n_right_brackets:
                            try:
                                tree = nltk.Tree.fromstring(
                                    cur_str, remove_empty_top_bracketing=True)
                            except:
                                print(repr(cur_str))
                                return None
                            sent = Sentence(
                                tree, 'ptb', remove_punct=remove_punct)
                            yield sent
                            cur_str = ""
                            n_left_brackets = n_right_brackets = 0

    def drop_empty_sentences(self, sentences):
        return [sent for sent in sentences if sent.tree is not None]

    def load_splited_data(self, remove_punct):
        ptb_train = list(self.penn_treebank_iterator(
            'train', self.path_to_raw_ptb, remove_punct=remove_punct))
        ptb_train = self.drop_empty_sentences(ptb_train)
        ptb_dev = list(self.penn_treebank_iterator(
            'dev', self.path_to_raw_ptb, remove_punct=remove_punct))
        ptb_dev = self.drop_empty_sentences(ptb_dev)
        ptb_test = list(self.penn_treebank_iterator(
            'test', self.path_to_raw_ptb, remove_punct=remove_punct))
        ptb_test = self.drop_empty_sentences(ptb_test)
        return ptb_train, ptb_dev, ptb_test

    def process_dataset(self, path_to_ptb_output_dir, max_len=40, remove_punct=False):
        assert max_len in [10, 40]
        min_occurrence = 1 if max_len == 10 else 2
        dataset_name = "ptb_len{}_{}".format(
            max_len, "nopunct" if remove_punct else "punct")

        ptb_train, ptb_dev, ptb_test = self.load_splited_data(
            remove_punct=remove_punct)
        dataset = Dataset(ptb_train, ptb_dev, ptb_test,
                          max_len=max_len, min_occurrence=min_occurrence)
        dataset.print_statistics()
        dataset.write_to_folder(os.path.join(
            path_to_ptb_output_dir, dataset_name))


class KTBHelper:
    def __init__(self, path_to_raw_ktb):
        self.path_to_raw_ktb = path_to_raw_ktb
        self.train, self.dev, self.test = self.train_test_split()

    def keyaki_treebank_iterator(self, base_dir, remove_punct=False):
        for filename in tqdm(sorted(os.listdir(base_dir))):
            if not filename.endswith(".psd"):
                continue
            filepath = os.path.join(base_dir, filename)

            cur_str = ""
            with open(filepath, "r") as fp:
                for line in fp:
                    if line == "\n" and cur_str:
                        try:
                            tree = nltk.Tree.fromstring(
                                cur_str, remove_empty_top_bracketing=True)
                            sent = Sentence(tree, 'ktb', remove_punct)
                        except Exception as e:
                            print(e)
                            print(filepath)
                            print(repr(cur_str))
                            exit(0)
                        yield sent
                        cur_str = ""
                    else:
                        cur_str += line

    def train_test_split(self):
        sentences = list(self.keyaki_treebank_iterator(self.path_to_raw_ktb, remove_punct=True))
        indices = [i for i, sent in enumerate(sentences) if sent.tree is not None]
        n_dev = n_test = int(len(indices) * 0.1)
        random.seed(0)
        random.shuffle(indices)
        indices_dev = set(indices[:n_dev])
        indices_test = set(indices[n_dev:(n_dev+n_test)])
        indices_train = set(indices[(n_dev+n_test):])
        return indices_train, indices_test, indices_dev

    def load_splited_data(self, remove_punct):
        ktb_all = list(self.keyaki_treebank_iterator(
            self.path_to_raw_ktb, remove_punct=remove_punct))
        train = [sent for i, sent in enumerate(ktb_all) if i in self.train]
        dev = [sent for i, sent in enumerate(ktb_all) if i in self.dev]
        test = [sent for i, sent in enumerate(ktb_all) if i in self.test]
        return train, dev, test

    def process_dataset(self, path_to_ktb_output_dir, remove_punct=False, max_len=10):
        assert max_len in [10, 40]
        min_occurrence = 1 if max_len == 10 else 2
        dataset_name = "ktb_len{}_{}".format(
            max_len, "nopunct" if remove_punct else "punct")

        ktb_train, ktb_dev, ktb_test = self.load_splited_data(
            remove_punct=remove_punct)
        dataset = Dataset(ktb_train, ktb_dev, ktb_test,
                          max_len=max_len, min_occurrence=min_occurrence)
        dataset.print_statistics()
        dataset.write_to_folder(os.path.join(
            path_to_ktb_output_dir, dataset_name))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_raw_ptb", 
                        default="data/english/package/treebank_3/parsed/mrg/wsj")
    parser.add_argument("--path_to_raw_ktb",
                        default="data/japanese/KeyakiTreebank/treebank")
    parser.add_argument("--path_to_ptb_output_dir",
                        default="data/cleaned_datasets/english/ptb")
    parser.add_argument("--path_to_ktb_output_dir",
                        default="data/cleaned_datasets/japanese/ktb")
    args = parser.parse_args()
    return args


def preprocess_ptb(args):
    if args.path_to_raw_ptb is None or not os.path.exists(args.path_to_raw_ptb):
        print("No PTB data found, do nothing")
        return
    ptb_helper = PTBHelper(args.path_to_raw_ptb)
    print("Processing for the setting: ptb_len10_nopunct")
    ptb_helper.process_dataset(
        args.path_to_ptb_output_dir, remove_punct=True, max_len=10)
    print("Processing for the setting: ptb_len40_nopunct")
    ptb_helper.process_dataset(
        args.path_to_ptb_output_dir, remove_punct=True, max_len=40)
    print("Processing for the setting: ptb_len40_punct")
    ptb_helper.process_dataset(
        args.path_to_ptb_output_dir, remove_punct=False, max_len=40)


def preprocess_ktb(args):
    if args.path_to_raw_ktb is None or not os.path.exists(args.path_to_raw_ktb):
        print("No KTB data found, do nothing")
        return
    ktb_helper = KTBHelper(args.path_to_raw_ktb)
    print("Processing for the setting: ktb_len10_nopunct")
    ktb_helper.process_dataset(
        args.path_to_ktb_output_dir, remove_punct=True, max_len=10)
    print("Processing for the setting: ktb_len40_nopunct")
    ktb_helper.process_dataset(
        args.path_to_ktb_output_dir, remove_punct=True, max_len=40)
    print("Processing for the setting: ktb_len40_punct")
    ktb_helper.process_dataset(
        args.path_to_ktb_output_dir, remove_punct=False, max_len=40)


def main():
    args = parse_args()
    preprocess_ptb(args)
    preprocess_ktb(args)


if __name__ == "__main__":
    main()
