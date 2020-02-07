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
import wikipedia as wp

def wiki_summary(query, sentences=10, lang='en'):
    wp.set_lang(lang)
    return wp.summary(query, sentences=sentences)

def wiki_page(query, lang='en'):
    wp.set_lang(lang)
    return wp.page(query)

def wiki_search(query, lang='en'):
    wp.set_lang(lang)
    return wp.search(query)

def wiki_graph(query, lang='en', levels=1, n=None):
    depth = 0
    G = nx.DiGraph()
    wp.set_lang(lang)
    if isinstance(query,str):
        query = [query]
    while True:
        depth += 1
        for q in query:
            try:
                w = wp.page(q, lang)
                links = w.links[:n]
                for i in links:
                    G.add_node(i)
                    G.add_edge(i,q)
            except Exception as e:
                print(str(e)) # raise
            else: query = links
        if levels == depth:
            break
    return G