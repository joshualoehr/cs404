#!/bin/env python

import argparse
import glob
import os
from xml.etree import ElementTree as ET

def replace_text(nodes, replace):
    for node in nodes:
        node.text = replace(node.text)

def scrape_text(nxml):
    with open(nxml, 'r') as f:
        data = f.read().replace('\n', '')

    root = ET.fromstring(data)

    abstract = root.find('./front/article-meta/abstract')
    replace_text(abstract.findall('*/title'), lambda text: '{{H}}')
    abstract_text = ' '.join(list(abstract.itertext()))
    
    article  = root.find('./body')
    replace_text(article.findall('*/title'), lambda text: '{{H}}')
    article_text = ' '.join(list(article.itertext()))

    return article_text, abstract_text

def write_output(nxml, article, abstract):
    output_dir = os.path.dirname(nxml)
    with open('{}/article.txt'.format(output_dir), 'bw+') as f:
        f.write(article.encode('utf-8'))
    with open('{}/abstract.txt'.format(output_dir), 'bw+') as f:
        f.write(abstract.encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Scrape abstract and article text from MEDLINE .nxml files.')
    parser.add_argument('files', type=glob.iglob,
            help='Glob pattern of .nxml files scrape.')
    args = parser.parse_args()

    for nxml in args.files:
        article, abstract = scrape_text(nxml) 
        write_output(nxml, article, abstract)
