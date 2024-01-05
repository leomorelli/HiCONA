import graph_tool.all as gt
import pandas as pd
import numpy as np
import hicona
import cooler
import pybedtools
import time

def filtered_edges(cool,chromosomes):
    '''generation of filtered edge list from a cool file, according to the optimal alpha'''
    ## 1. Load File
    h=hicona.hicona_cooler.HiconaCooler(cool)
    
    ## 2. Extraction of filtered edges
    tables=h.tables(chrom_selection=chromosomes,dist_thr=100000000,quant_thr=0.01)

    spar_tabs=[]
    for i in tables:
        i.alpha_column=f'alpha_min'
        i.compute_opt_alpha(decimals=2, num_pts=11)
        spar_tabs.append(i.filter_alpha(alpha='optimal'))
    df=pd.concat(spar_tabs,ignore_index=True)

    bins=h.bins()[:]
    anno=cooler.annotate(df,bins)
    anno_f=anno[anno['chrom1']==chromosomes[0]]
    elist=anno_f[['bin1_id','bin2_id','chrom1','start1','end1','chrom2','start2','end2']]
    
    return elist,bins

def prep_elist(elist,bins):
    '''prepare the edge list for graph generation: 
            A. assign to each node the label of the cool bin;
            B. label the interaction as HiC or genomic interactions '''
    ## 1. Encode node information for graphtool
    node=pd.DataFrame()
    node['n']=sorted(list(set(elist['bin1_id'])|set(elist['bin2_id'])))
    nodes=bins.T[node['n'].tolist()].T
    nodes['n']=nodes.index
    nodes.index=[x for x in range(nodes.shape[0])]

    n_to_code = {n: i for i, n in enumerate(nodes['n'])}
    nodes['codes']=n_to_code.values()

    elist['start_code'] = elist['bin1_id'].map(n_to_code)
    elist['end_code'] = elist['bin2_id'].map(n_to_code)
    elist_f=elist[['start_code','end_code']]

    ## 2. Classification of edges: 1 = Hi-C edge ; 2 = Genomic Edges
    elist_f['link']=1

    genomic=[]
    for i in range(elist_f.max().max()):
        if i==elist_f.max().max():
            continue
        else:
            genomic.append([i,i+1,0])

    elist_f=pd.concat([elist_f,pd.DataFrame(genomic,columns=['start_code','end_code','link'])])
    return elist_f,nodes

def cluster_analysis(graph,c):
    ''' two step cluster analysis: nested unweighted and multilayered
     --> see https://graph-tool.skewed.de/static/doc/demos/inference/inference.html'''
    g=graph
    #### a. Minimization Step
    states=[]
    a=time.time()
    for i in range(100):
        state=gt.minimize_nested_blockmodel_dl(g, state_args=dict(base_type=gt.LayeredBlockState,state_args=dict(ec=g.ep.link, layers=True,deg_corr=True)))
        states.append(state)
    print('minimization step:',time.time()-a,'seconds')

    #### b. selection of the best model across 100 initialization
    min_entropy=10**9

    for i,s in enumerate(states):
        if s.entropy()<min_entropy:
            min_entropy=s.entropy()
            state_index=i
        else:
            continue

    state=states[state_index]

    #### c. Equilibration Step
    a=time.time()
    gt.mcmc_equilibrate(state, wait=1000, mcmc_args=dict(niter=10,c=c))
    print('equilibration step:',time.time()-a,'seconds')
    return state

def clusters_df(state):
    bs=state.get_bs()
    groups = np.zeros((g.num_vertices(), len(bs)), dtype=int)

    for x in range(len(bs)):
        # for each level, project labels to the vertex level
        # so that every cell has a name. Note that at this level
        # the labels are not necessarily consecutive
        groups[:, x] = state.project_partition(x, 0).get_array()

    groups = pd.DataFrame(groups).astype('category')

    # rename categories from 0 to n

    for c in groups.columns:
        ncat = len(groups[c].cat.categories)
        new_cat = [u'%s' % x for x in range(ncat)]
        groups[c] = groups[c].cat.rename_categories(new_cat)

    #selection of columns in which we have more than one clusters
    for i in groups.columns:
        if len(set(groups[i]))==1:
            columns=i
            break
        else:
            continue
    groups=groups.iloc[:,:columns]
    return groups


def cluster_hic(cell,chrom):
    c = 0.1
    ## 1. Filtered edges
    elist,bins=filtered_edges(cell,[chrom])
    print('1. edges filtering done')
    
    ## 2. Filtered edges info for graphtool
    elist_f,nodes=prep_elist(elist,bins)
    print('2. edges information prepared')

    ## 3. Graph Generation
    g=gt.Graph(directed=False)
    g.add_edge_list(elist_f.values)

    link = g.new_edge_property("short",vals=elist_f['link'].tolist()) 
    g.ep['link']=link
    print('3. graph generated')

    ## 4. Cluster Analysis
    state=cluster_analysis(g,c)
    print('4. clusters computed')

    ## 5. Assign to each node its cluster at different levels
    groups=clusters_df(state)
    
    nodes=pd.concat([nodes.T,groups.T]).T
    return nodes
