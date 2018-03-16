#!/bin/env python

import argparse
import glob
import os
from xml.etree import ElementTree as ET

def replace_text(nodes, replace):
    for node in nodes:
        node.text = replace(node)

def remove_references_section(article, parent_map):
    possible_titles = ['references', 'bibliography', 'works cited', 'footnotes', 'acknowledgements']
    for node in article.findall('.//HEADER'):
        if node.text.lower().strip() in possible_titles:
            div = parent_map[node]
            parent_map[div].remove(div)

def scrape_text(xml):
    try:
        with open(xml, 'r') as f:
            data = f.read().replace('\n', ' ')
        root = ET.fromstring(data)
    except Exception as e:
        print("Error parsing file: {}: {}".format(xml, e))
        return None, None

    parent_map = dict((c,p) for p in root.iter() for c in p)

    abstract = root.find('ABSTRACT')
    abstract_text = ' '.join(list(abstract.itertext()))
    
    article  = root.find('BODY')
    remove_references_section(article, parent_map)
    replace_text(article.findall('.//HEADER'), lambda node: '. {{H%s}}.' % parent_map[node].get('ID'))
    replace_text(article.findall('.//REF'), lambda node: '{{REF}}')
    replace_text(article.findall('.//CREF'), lambda node: '{{REF}}')
    replace_text(article.findall('.//EQN'), lambda node: '{{EQN}}')
    article_text = ' '.join(list(article.itertext()))
    article_text = ' '.join(article_text.split())

    return article_text, abstract_text

def write_output(xml, article, abstract):
    output_dir = xml.split('.')[0]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.rename(xml, "{}/{}".format(output_dir, os.path.basename(xml)))
    with open('{}/article.txt'.format(output_dir), 'bw+') as f:
        f.write(article.encode('utf-8'))
    with open('{}/abstract.txt'.format(output_dir), 'bw+') as f:
        f.write(abstract.encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Scrape abstract and article text from cmp-lg .xml files.')
    parser.add_argument('files', type=glob.glob,
            help='Glob pattern of .xml files to scrape.')
    args = parser.parse_args()

    for xml in args.files:
        article, abstract = scrape_text(xml) 
        if article and abstract: 
            write_output(xml, article, abstract)

