import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def topics_fig(df_top, top_doc_topics_gensim, tsne_x, tsne_y):


    fig = plt.figure(figsize=(10,10))

    gs= fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[1,0])
    ax2 = fig.add_subplot(gs[1,1])
    ax_tab = fig.add_subplot(gs[0,:], frame_on=False)
    ax_tab.xaxis.set_visible(False)
    ax_tab.yaxis.set_visible(False)
    cmap = plt.get_cmap('tab20')

    #Topic count bar plot
    n_topics = len(df_top.index)

    #some topics may have zero texts with that top topic, so value counts will give incomplete index
    topic_counts = pd.Series(index=range(n_topics))
    topic_counts_temp = top_doc_topics_gensim.value_counts()
    topic_counts.iloc[topic_counts_temp.index] = topic_counts_temp.values
    topic_counts = topic_counts.fillna(0).sort_values(ascending=False)

    #Table
    topic_strs = df_top.apply(" ".join, axis=1)
    topic_strs.name = 'Top Topic Words'
    topic_strs = topic_strs.iloc[topic_counts.index]

    colors_key = np.linspace(0, 1, n_topics)
    colors_key = [colors_key[i] for i in topic_counts.index]
    colors = cmap(colors_key)
    tab = pd.plotting.table(ax_tab, topic_strs, loc='center', rowColours=colors, cellLoc='left')

    for key,cell in tab.get_celld().items():
        if key[1]==0:                   
            cell.PAD = 0.01

    tab.auto_set_font_size(False)
    tab.set_fontsize(8)

    #TSNE plot
    ax1.scatter(tsne_x,tsne_y,c = top_doc_topics_gensim.values, cmap = cmap, s = 3)
    ax1.set_xlabel('TSNE x')
    ax1.set_ylabel('TSNE y')



    topic_counts.index = topic_counts.index + 1
    topic_counts.plot.bar(ax=ax2)
    ax2.set_xlabel('Topic Number')
    ax2.set_ylabel('Count')

    


    return fig