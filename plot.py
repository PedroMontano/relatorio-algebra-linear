import matplotlib.pyplot as plt
import numpy as np


def bar_plot( height_list, labels, y_label = '', title = '', colors = 'random' ):
    """Bar Plot

    :param y_label: y label
    :param title: plot title
    :param colors: list of colors ('random' if random)
    :param height_list: height of bars
    :param labels: label of each bar
    :return: matplotlib object
    """
    fig , ax = plt.subplots( figsize = ( 10 , 7 ) )
    for i, h in enumerate( height_list ):
        if colors == 'random' :
            r, g, b = np.random.random( ) , np.random.random( ) , np.random.random( )
            rgb = [ r , g , b ]
        else :
            rgb = colors[ i ]
        ax.bar( labels[ i ], height_list[ i ], color = rgb, width = 0.3 )
    ax.set_xticklabels( labels, rotation = 15, ha = "right" )
    ax.set_title( title , fontsize = 16 )
    ax.set_ylabel( y_label, fontsize = 12 )
    ax.set_ylim( [ .95 * min( height_list ), 1.05 * max( height_list ) ] )
    ax.grid( )
    return  fig