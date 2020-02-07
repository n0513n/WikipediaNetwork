#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Functions to interact with Wikimedia API.

Requirements and tested versions:

* Python 3.6.8+
* networkx==2.3
* wikipedia==1.4.0

Currently in WIP Stage (Work In Progress)!
'''

import networkx as nx
import pandas as pd
import wikipedia as wp

from time import time, sleep
from tqdm.auto import tqdm

def WikipediaNetwork(articles, output_file=None,
    lang='en', levels=1, n=None, G=None):
    '''
    Returns a network graph containing articles
    and their hyperlinks collected from Wikipedia.
    '''
    t0 = time()
    depth = 0
    G = nx.DiGraph() if not G else G
    wp.set_lang(lang)

    if isinstance(articles,str):
        articles = [articles]

    print('Starting...')

    while True:
        depth += 1
        links = set()

        t = tqdm(articles,
                 ascii=True,
                 total=len(articles),
                 desc='Level '+str(depth))

        for a in t:
            try: # get hyperlinks
                w = wp.page(a, lang)
                links = list(w.links)
                for i in links[:n]:
                    G.add_node(i)
                    G.add_edge(i,a)

            except Exception as e:
                print(str(e)) # raise
                sleep(3)

        articles = links

        print('Level %d finished with %d nodes and %d degrees.'%\
              (depth, G.order(), G.size()))

        if levels == depth:
            break

        if output_file:
            output = 'level'+str(depth)+output_file
            print('Writing to %s...' % output)
            nx.write_gexf(G, output)

    print('Finished in %0.2fs.' % (time() - t0))

    if output_file:
        print('Writing to %s...' % output_file)
        nx.write_gexf(G, output_file)

    return G