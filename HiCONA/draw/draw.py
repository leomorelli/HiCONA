import graph_tool.all as gt
import warnings
from typing import Optional, Tuple, Sequence, Type, Union, Literal








def normalize_rgb(rgb): #list    return the right rgb for graph tool
    rgb_n=[]
    for c in rgb:
        rgb_n.append(c/255)
    return rgb_n


def colors(n_colors):  #int        rgbs automatically available  (10 colors)
    if n_colors>7:
        raise ValueError('n_colors must not exceed 7 colors')    
    colors=[[16,119,243,249],[198,160,0,249],[51,185,131,249],[232,51,38,249],[191,240,252,249],[0,0,0,249],[255,255,255,249],[204,102,0,249],[88,47,5,249],[255,0,127,249]]   #blue,yellow,green,red,purple,black,white, orange,brown,pink
    return colors[:n_colors]



def defined_colormap(color_dictionary):
    if len(color_dictionary)>0:
        categories=list(color_dictionary.keys())
        colormap=list(color_dictionary.values())
        if (len(colormap[0])<3)|(len(colormap[0])>4):
            raise ValueError('colors must be provided in RGB format: i.e. [50,234,256]')
        if len(colormap[0])==3:
            for c in range(len(colormap)):
                colormap[c].append(249)
    else:
        categories=[]
        colormap=[]
    return categories,colormap






def colormap_auto(g,annotation,type_anno, categories, colormap):
    if type_anno=='edge':
        anno=[x for x in g.ep[annotation]]
    elif type_anno=='vertex':
        anno=[x for x in g.vp[annotation]]
    if len(categories)>0:
        categ=categories
        cmap=colormap
    else:
        categ=list(set(anno))
        cmap=colors(len(categ))
    color_list=[0 for x in range(len(anno))]
    for c in range(len(categ)):
        for v in range(len(anno)):
            if anno[v] == categ[c]:
                color_list[v]=normalize_rgb(cmap[c])
    if type_anno=='edge':
        color = g.new_ep("vector<double>", vals=color_list)
    elif type_anno=='vertex':
        color = g.new_vp("vector<double>", vals=color_list)
    return color








def draw(
    g,
    color_dictionary_vertex:Optional[dict]={},
    vertex_color:Optional[Union[list,str]]='',
    color_dictionary_edge:Optional[dict]={},
    edge_color:Optional[Union[list,str]]='',
    vertex_size:Optional[Union[int,str]]='',
    edge_width:Optional[Union[int,str]]='',
    vertex_text:Optional[str]='',
    edge_text:Optional[str]='',
    output_size:Optional[tuple]=(600,600),
    output:Optional[Union[str,bool]]=None      #if you want to store the figure
):
    #1. selecting colors for edge and vertices
    categories_vertex,colormap_vertex=defined_colormap(color_dictionary_vertex)
    categories_edge,colormap_edge=defined_colormap(color_dictionary_edge)
    ##### 1.1 if vertex color is a list, values are taken as an RGB color
    if type(vertex_color)==list:
        color_vertex=vertex_color
    else:
    ##### 1.2 else the variable vertex_color will be considered as a property map (same for edges)
        try:
            g.vp[vertex_color]
            color_vertex=colormap_auto(g,vertex_color,'vertex', categories_vertex, colormap_vertex)
        except KeyError:
            warnings.warn(f'no vertex property map is named as {vertex_color}: vertex colors will follow the default setting')
            color_vertex=[0.640625, 0, 0, 0.9]
    if type(edge_color)==list:
        color_edge=edge_color
    else:
        try:
            g.ep[edge_color]
            color_edge=colormap_auto(g,edge_color,'edge', categories_edge, colormap_edge)
        except KeyError:
            warnings.warn(f'no edge property map is named as {edge_color}: edge colors will follow the default setting')
            color_edge=[0,0,0,0.7]
    #2. Vertex size and Edge width
    if type(vertex_size)==int:
        size_vertex=vertex_size
    else:
        try:
            g.vp[vertex_size]
            size_vertex=gt.prop_to_size(g.vp[vertex_size])
        except KeyError:
            warnings.warn(f'no vertex property map is named as {vertex_size}: vertex size will follow the default setting')
            size_vertex=10

    if type(edge_width)==int:
        width_edge=edge_width
    else:
        try:
            g.ep[edge_width]
            width_edge=gt.prop_to_size(g.vp[edge_width])
        except KeyError:
            warnings.warn(f'no edge property map is named as {edge_width}: edge width will follow the default setting')
            width_edge=1.0
    #3. vertex text and edge text
    try:
        g.vp[vertex_text]
        text_vertex=g.vp[vertex_text]
    except KeyError:
        warnings.warn(f'no vertex property map is named as {vertex_text}: vertex text will follow the default setting')
        text_vertex=''
    try:
        g.ep[edge_text]
        text_edge=g.vp[edge_text]
    except KeyError:
        warnings.warn(f'no edge property map is named as {edge_text}: edge text will follow the default setting')
        text_edge=''
    #4. plot
    pl = gt.graph_draw(g,vertex_fill_color=color_vertex,edge_color=color_edge,edge_pen_width=width_edge,vertex_size=size_vertex,vertex_text=text_vertex,edge_text=text_edge,output_size=output_size,output=output)
    return pl 