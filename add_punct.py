from typing import List, IO
import sys
import argparse


def tokens_to_str(tokens: List[str]) -> str:
    '''
    Concatenate the tokens of a parse tree to a string and add spaces properly.

    A space is added if and only if:
    - Before ( or non-paren
    - After ) or non-paren
    - Not after (

    @param tokens: a `list` of tokens
    @return: the concatenated `str`
    '''
    line = ''
    sp = False
    for token in tokens:
        if token == '(':
            if sp:
                line += ' '
                sp = False
            line += '('
        elif token == ')':
            sp = True
            line += ')'
        else:
            if sp:
                line += ' '
            else:
                sp = True
            line += token
    return line


def add_punct(raw: List[str], ref: List[str], tag: str = None) -> List[str]:
    '''
    Given the raw parse tree and the reference sentence, add punctuations
    to the raw parse tree according to the reference sentence.

    @param raw: the raw parse tree, a `list` of tokens
    @param ref: the reference sentence, a `list` of words and punctuations
    @param tag: the POS tag for the punctuations, use the punctuations
                themselves if `None`
    @return: a `list` of tokens with punctuations added
    '''
    i = j = 0
    k = 2
    out = []

    while i < len(raw):
        # left par together with the pos tag
        if raw[i] == '(':
            out.append('(')
            out.append(raw[i+1])
            i += 2

        # right par
        elif raw[i] == ')':
            out.append(')')
            i += 1

        # the ref word matches the raw word
        # in case of the case, compare lowercase and use the ref version
        elif raw[i].lower() == ref[j].lower():
            out.append(ref[j])
            k = len(out)
            i += 1
            j += 1

        # the ref word is a punct
        else:
            while k < len(out) and out[k] == ')':
                k += 1
            out[k:k] = ['(', ref[j] if tag is None else tag, ref[j], ')']
            k += 4
            j += 1

    # deal with punct at the end
    if j < len(ref):
        out.pop()
        while j < len(ref):
            out.extend(['(', ref[j] if tag is None else tag, ref[j], ')'])
            j += 1
        out.append(')')

    return out


def process(fileref: str, fileraw: str, fout: IO[str], tag: str = None, encoding: str = 'utf-8'):
    '''
    Given a file of raw parse trees and a file of reference sentences, read
    a line from two files and output a line of parse tree with punctuations
    added.

    @param fileref: the path of the reference sentences file
    @param fileraw: the path of the raw parse trees file
    @param fout: a file-like object, will write the output here
    @param tag: the POS tag for the punctuations, use the punctuations
                themselves if `None`
    @param encoding: encoding of the files
    '''
    with open(fileraw, encoding=encoding) as fraw, open(fileref, encoding=encoding) as fref:
        while True:
            raw = fraw.readline()
            ref = fref.readline()

            # stop reading if either of raw and ref ends
            if not raw or not ref:
                break

            # tokenize raw
            raw = raw.replace('(', ' ( ').replace(')', ' ) ').split()
            # tokenize ref and extract leaves
            ref = ref.replace('(', ' ( ').replace(')', ' ) ').split()
            ref = [ref[i] for i in range(len(ref)) if ref[i] not in (
                '(', ')') and ref[i-1] != '(']

            # add punct
            out = add_punct(raw, ref, tag=tag)

            # concatenate tokens to string, then output
            fout.write(tokens_to_str(out) + '\n')


if __name__ == '__main__':
    DESCRIPTION = '''
    A program that adds punctuations to a set of constituency-based parse
    trees, given a set of reference sentences with punctuations.
    '''
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--ref',
                        help='path of the reference file with punctuations (typically the gold test set)')
    parser.add_argument('--raw',
                        help='path of the raw file (the file that punctuations need to be added)')
    parser.add_argument('-o', '--output', default=None,
                        help='path of the output file (output to stdout by default)')
    parser.add_argument('-e', '--encoding', default='utf-8',
                        help="the encoding of the files ('%(default)s' by default)")
    parser.add_argument('-t', '--tag', default=None,
                        help='the POS tag of punctuations (use punctuations themselves by default)')

    args = parser.parse_args()
    if args.output is None:
        process(args.ref, args.raw, sys.stdout,
                tag=args.tag, encoding=args.encoding)
    else:
        with open(args.output, 'w', encoding=args.encoding) as fout:
            process(args.ref, args.raw, fout,
                    tag=args.tag, encoding=args.encoding)
