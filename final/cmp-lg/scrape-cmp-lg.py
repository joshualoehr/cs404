#!/bin/env python

import argparse
import glob
import os
from xml.etree import ElementTree as ET

def replace_text(nodes, replace):
    for node in nodes:
        node.text = replace(node.text)

def scrape_text(xml):
    with open(xml, 'r') as f:
        data = f.read().replace('\n', ' ')

    try:
        root = ET.fromstring(data)
    except Exception as e:
        print("Error parsing file: {}: {}".format(xml, e))
        return None, None

    abstract = root.find('ABSTRACT')
    abstract_text = ' '.join(list(abstract.itertext()))
    
    article  = root.find('BODY')
    replace_text(article.findall('*/HEADER'), lambda text: '{{%s}}' % text)
    article_text = ' '.join(list(article.itertext()))

    return article_text, abstract_text

def write_output(xml, article, abstract):
    output_dir = xml.split('.')[0]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.rename(xml, "{}/{}".format(output_dir, os.path.basename(xml)))
    with open('{}/article.txt'.format(output_dir), 'w+') as f:
        f.write(article.encode('utf-8'))
    with open('{}/abstract.txt'.format(output_dir), 'w+') as f:
        f.write(abstract.encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Scrape abstract and article text from cmp-lg .xml files.')
    parser.add_argument('files', type=glob.glob,
            help='Glob pattern of files to scrape.')
    args = parser.parse_args()

    for xml in args.files:
        article, abstract = scrape_text(xml) 
        if article and abstract: 
            write_output(xml, article, abstract)
