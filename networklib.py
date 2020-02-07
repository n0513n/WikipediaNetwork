#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import community
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import wikipedia as wp

from sklearn.preprocessing import MinMaxScaler

D = 2

NORM_DEGREE = False

def nx_centrality(G, it=100, deg=True, clu=True, clo=True,
    eig=True, bet=True, bro=True, bri=True, mod=True,
    normalized=False, is_directed=None):
    '''
    Returns a data frame containing network statistics for
    the nodes and their communities from a NetworkX.Graph():

    Accepts arguments:
    * G: networkx.graph()
    * it: iterations
    * deg: degree centrality
    * clu: clustering coefficient
    * clo: closeness centrality
    * eig: eigenvector centrality
    * bet: betweenness centrality
    * bro: brokering centrality
    * bri: bridgeness centrality
    * mod: modularity coefficient
    * normalized: min/max scale
    * is_directed: for DiGraphs

    Note: the number of iterations (it) may need to be raised
    to calculate the eigenvector centrality of the nodes.
    '''
    def normalize(D):
        max_d = max(D.values())
        for key in D:
            D[key] /= (max_d if max_d > 0 else 1)

    def add_m(m,c,norm=normalized):
        if norm:
            normalize(m)
        df[c] = pd.Series(m)
        print('Avg', c +':',
              round(sum(m.values())/len(m),3))

    df = pd.DataFrame()
    print('Nodes:', len(G.nodes()))
    print('Edges:', len(G.edges()))

    if is_directed == None:
        is_directed = G.is_directed()

    if deg == True:
        degree = dict(nx.degree(G))
        #degree = nx.degree_centrality(G)
        add_m(degree, 'degree', NORM_DEGREE)
        # indegree = dict(G.in_degree(G))
        # outdegree = dict(G.out_degree(G))
        indegree = defaultdict(int)
        outdegree = defaultdict(int)
        for u,v in G.edges():
            indegree[v] += 1
            outdegree[u] += 1
            if not is_directed:
                indegree[u] += 1
                outdegree[v] += 1
        add_m(indegree, 'in_degree', NORM_DEGREE)
        add_m(outdegree, 'out_degree', NORM_DEGREE)

    if clu == True:
        clustering = nx.clustering(G)
        add_m(clustering, 'clustering')

    if clo == True:
        closeness = nx.closeness_centrality(G)
        add_m(closeness, 'closeness')

    if eig == True:
        eigenvector = nx.eigenvector_centrality(G, it)
        add_m(eigenvector, 'eigenvector')

    if bet == True:
        betweenness = nx.betweenness_centrality(G, normalized=normalized)
        add_m(betweenness, 'betweenness')

    if bro == True:
        brokering = nx_brokering_centrality(G, normalized=normalized)
        add_m(brokering, 'brokering')

    if bri == True:
        bridgeness = nx_bridgeness_centrality(G, betweenness if bet else None, normalized=normalized)
        add_m(bridgeness, 'bridgeness')

    df.fillna(0, inplace=True)

    if mod == True:
        if G.is_directed():
            G = G.to_undirected()
        partition = community.best_partition(G)
        df['partition'] = pd.Series(partition)
        modularity = community.modularity(partition, G)
        modules = len(set(partition.values()))
        print('Modules:', modules,
              '\nModularity:', round(modularity,3))

    df.index.name = 'id'
    #df = pd.DataFrame.transpose(df)
    return df.round(D)

def nx_bridgeness_centrality(G, betweenness=None, normalized=True):
    '''
    Computes Bridgeness Centrality for a graph G.

    For each node the bridgeness coefficient is defined as:
        bridge_coeff[node] =  (1/degree[node]) / sum(1/degree[neighbors[node]])

    The bridgeness centrality of a node is defined as:
        bridgeness[node] = betweenness(node) * bridge_coeff[node]

    Since computing of betweennes can take a lot of time,
    it's possible to provide the betweenness as a parameter.

    Note that only nodes with degree >= 1 will be returned.
    '''
    if betweenness is None:
        betweenness = nx.betweenness_centrality(G)

    bridgeness = {}
    for node in G.nodes():
        degree_node = nx.degree(G,node)
        if degree_node > 0:
            neighbors_degree  = dict(nx.degree(G, nx.neighbors(G, node))).values()
            sum_neigh_inv_deg = sum((1.0/d) for d in neighbors_degree)
            if sum_neigh_inv_deg > 0:
                bridge_coeff = (1.0/degree_node) / sum_neigh_inv_deg
                bridgeness[node] = betweenness[node] * bridge_coeff
            else: bridgeness[node] = 0

    if normalized == True:
        max_d = max(bridgeness.values())
        for key in bridgeness:
            bridgeness[key] /= max_d

    return bridgeness

def nx_brokering_centrality(G, normalized=True):
    '''
    Computes Brokering Centrality for a graph G.

    For each node brokering centrality is defined as:
        brokering[node] =  (1 - clustering[node]) * degree[node]
    '''
    degree     = nx.degree_centrality(G)
    clustering = nx.clustering(G)

    brokering  = {}
    for node in G.nodes():
        brokering[node] =  (1 - clustering[node]) * degree[node]

    if normalized == True:
        max_d = max(brokering.values())
        for key in brokering:
            brokering[key] /= max_d

    return brokering

def nx_modules(G, df=[], measures=[], column='partition', normalized=False, it=100,
    deg=True, clu=True, clo=True, eig=True, bet=True, bro=True, bri=True, mod=True):
    '''
    Returns a data frame of modules (aka. communities or partitions) as
    identified by the Louvain Method, containing their network proprierties
    and summed centrality measures. Accepts centrality data frames (df).
    '''
    if len(df) == 0:
        df = nx_centrality(G, it, deg, clu, clo, eig, bet, bro, bri, mod, normalized)

    if not measures:
        measures = df.select_dtypes(np.number).columns.to_list()
        measures.remove(column) if column in measures else None

    df_modules = pd.DataFrame()
    df_modules.index.name = 'id'

    for i in df[column].unique():
        part  = df[df[column] == i]
        SG    = nx.subgraph(G, part.index)
        order = SG.order() # nodes
        size  = SG.size()  # edges
        df_modules.loc[i, 'nodes'] = order
        df_modules.loc[i, 'edges'] = size
        df_modules.loc[i, 'density'] = nx.density(SG)
        #df_modules.loc[i, 'diameter'] = nx.diameter(SG)
        for m in measures:
            v = part[m].sum()
            if normalized == True:
                v = v/part[m].max()
            df_modules.loc[i, m] = v

    return df_modules.round(D)

def df_describe(df, columns=[], output_file=None, low_memory=False):
    '''
    Return most common statistics for a data frame.
    Accepts a list of columns to consider (optional).
    '''
    if isinstance(df, str):
        df = df_load(df, low_memory=low_memory)

    d = df.describe()

    if not columns:
        columns = d.columns

    for c in columns:
        mean = df[c].mean()
        median = df[c].median()
        mad = df[c].mad() # mean absolute deviation
        std = df[c].std() # sample standard deviation (S)
        var = df[c].var() # unbiased variance (S²)
        cova = var/mean if mean > 0 else 0 # coefficient of variation
        #d.loc['mean'], c] = mean
        d.loc['median', c] = median
        #d.loc['std', c] = std
        d.loc['mad', c] = mad
        d.loc['var', c] = var
        d.loc['cova', c] = cova
        d.loc['10%', c] = df[c].quantile(0.1) # 10% percentile
        #d.loc['25%', c] = df[c].quantile(0.25) # first quartile
        #d.loc['50%', c] = df[c].quantile(0.5) # same as mean
        ##d.loc['75%', c] = df[c].quantile(0.75) # last quartile
        d.loc['90%', c] = df[c].quantile(0.9) # 90% percentile

    d = d.reindex(['count', 'mean', 'median', 'std', 'min', 'max',
        '10%', '25%', '50%', '75%', '90%', 'mad', 'var', 'cova'])

    if output_file:
        df_write(d, output_file)

    return d.round(D)

def degree_histogram(G, title="Histograma de grau"):
    '''
    Returns node degree histogram.
    '''
    deg = sorted([d for n, d in G.degree()], reverse=True)
    degree_count = collections.Counter(deg)
    deg, cnt = zip(*degree_count.items())
    fig, ax = plt.subplots(figsize=(10,6))
    plt.bar(deg, cnt, width=0.80, color='b')
    plt.title(title)
    plt.ylabel("Nós")
    plt.xlabel("Grau")
    ax.set_xticks([d for d in deg])
    ax.set_xticklabels(d for d in deg)
    plt.plot()