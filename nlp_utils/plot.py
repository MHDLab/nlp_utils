import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import gensim_utils
from . import sklearn_utils

def topics_fig(df_topickeywords, tsne_x, tsne_y ,doc_topic_probs, titlestr = None):

    df_doc_topic_probs = pd.DataFrame(doc_topic_probs)
    top_topic_doc = df_doc_topic_probs.idxmax(axis=1)
    top_topic_doc.name = 'top_topic_doc'

    fig = plt.figure(figsize=(10,10))

    gs= fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[1,0])
    ax2 = fig.add_subplot(gs[1,1])
    ax_tab = fig.add_subplot(gs[0,:], frame_on=False)
    ax_tab.xaxis.set_visible(False)
    ax_tab.yaxis.set_visible(False)
    cmap = plt.get_cmap('tab20')

    #Topic count bar plot
    n_topics = len(df_topickeywords.index)

    #some topics may have zero texts with that top topic, so value counts will give incomplete index
    topic_counts = pd.Series(index=range(n_topics))
    topic_counts_temp = top_topic_doc.value_counts()
    topic_counts.iloc[topic_counts_temp.index] = topic_counts_temp.values
    topic_counts = topic_counts.fillna(0).sort_values(ascending=False)

    #Table
    topic_strs = df_topickeywords.apply(" ".join, axis=1)
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
    ax1.scatter(tsne_x,tsne_y,c = top_topic_doc.values, cmap = cmap, s = 3)
    ax1.set_xlabel('TSNE x')
    ax1.set_ylabel('TSNE y')



    topic_counts.index = topic_counts.index + 1
    topic_counts.plot.bar(ax=ax2)
    ax2.set_xlabel('Topic Number')
    ax2.set_ylabel('Count')

    #TODO: have subplots adjust adapts to titlestr length
    if titlestr == None:
        fig.tight_layout()
    else:
        fig.suptitle(titlestr)
        fig.tight_layout()
        fig.subplots_adjust(top=0.93)
    
    return fig

def topics_fig_bigramlda(texts, bigram_kwargs, lda_kwargs):

    texts_bigram, id2word, data_words, lda_model = gensim_utils.gensim_lda_bigram(texts, bigram_kwargs, lda_kwargs)

    df_topickeywords, doc_topic_probs = gensim_utils.gensim_topic_info(lda_model, data_words, id2word)

    num_bigrams, total_words = gensim_utils.bigram_stats(data_words, id2word)
    titlestr = str(lda_kwargs) + str(bigram_kwargs) + "\n Num Bigrams: " + str(num_bigrams) + ", Total Words: " + str(total_words) + ", Bigram Fraction: " + str(round(num_bigrams/total_words, 3))

    tsne_x, tsne_y = sklearn_utils.calc_tsne(texts_bigram)

    fig = topics_fig(df_topickeywords, tsne_x, tsne_y, doc_topic_probs, titlestr)

    return fig

def top_word_plot(tw, titlestr=''):
    """histogram plot of word counts"""
    counts = [w[1] for w in tw]
    bins = np.logspace(np.log10(0.9),np.log10(tw[0][1]), 100)
    plt.hist(counts, bins=bins)
    plt.yscale('log')
    plt.xscale('log')

    titlestr = titlestr + ',  Num Words: ' + str(len(tw))
    print(titlestr)
    plt.suptitle(titlestr)


import math
from textwrap import wrap
from .common import fit_topic_year


def top_slopes_plot(df_topicsyear, topic_strs, year_range_fit, n_plots = 5, ascending=False, highlight_topics=[]):
    col_wrap = 5
    
    df_fit_params  = fit_topic_year(df_topicsyear, year_range_fit)
    top_slopes = df_fit_params['slope'].sort_values(ascending=ascending)[0:n_plots]
    
    n_rows = math.trunc(len(top_slopes)/col_wrap)

    fig, axes = plt.subplots(n_rows, min(col_wrap,len(top_slopes.index)), figsize=(20,4*n_rows), sharey=True, squeeze=False)
    
    for i, topic_id in enumerate(top_slopes.index):
        
        col = int(i % col_wrap)
        row = int(math.trunc(i/col_wrap))

        #Time Series plot
        s_time = df_topicsyear[topic_id]
        s_time.plot(ax=axes[row][col])

        #Fit Plot
        z_topic = (df_fit_params.loc[topic_id]['slope'], df_fit_params.loc[topic_id]['offset'])
        p = np.poly1d(z_topic)
        ys = p(s_time.loc[year_range_fit].index)
        axes[row][col].plot(s_time.loc[year_range_fit].index, ys, color='red')

        #Annotation
        t_w = topic_strs.loc[topic_id]
        t_w = "\n".join(wrap(t_w, width = 30))
        title_obj = axes[row][col].set_title(topic_id + ": " + t_w)

        if topic_id in highlight_topics:
            plt.setp(title_obj, color='r')


    current_ylim = axes[0][0].get_ylim()
    for ax in axes.flatten():
        ax.set_ylim(0, current_ylim[1]*1.2)

    axes[0][0].set_ylabel('Normalized (over years) Topic Probability')

    fig.tight_layout()