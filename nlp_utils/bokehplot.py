#%%


import random


# required libraries for plot
# from bokeh_functions.plot_text import cite, notes, dataset_description, header, description, description2
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, Slider, TapTool, TextInput, Div, Paragraph
from bokeh.palettes import Category20

from bokeh.transform import linear_cmap, transform, factor_mark
from bokeh.plotting import figure, show
from bokeh.layouts import column, widgetbox, row, layout



def lit_cluster_plot(tsne_x, tsne_y, y_labels, topics, keywords, display_text, search_text, hover_df):

        num_clusters = len(set(y_labels))

        #data sources


        data_dict = dict(
                x= tsne_x,
                y= tsne_y,
                x_backup = tsne_x,
                y_backup = tsne_y,
                desc= y_labels,
                keywords = keywords,
                labels = ["C-" + str(x) for x in y_labels],
                display_text = display_text,
                search_text = search_text,
                )

        tooltips = []
        for col in hover_df.columns:
                data_dict[col] = hover_df[col]
                tooltips.append((col, "@" + col + "{safe}"))

        source = ColumnDataSource(data=data_dict)

        hover = HoverTool(tooltips=tooltips,
        point_policy="follow_mouse")

        #map colors
        initial_palette = Category20[20]
        random.Random(42).shuffle(list(initial_palette))

        mapper = linear_cmap(field_name='desc',
                        palette=Category20[20],
                        low=min(y_labels), high=max(y_labels))

        x_buff = (tsne_x.max() - tsne_x.min())/10
        y_buff = (tsne_y.max() - tsne_y.min())/10

        x_range = (tsne_x.min() - x_buff,tsne_x.max() + x_buff)
        y_range = (tsne_y.min() - y_buff,tsne_y.max() + y_buff)

        #prepare the figure
        plot = figure( aspect_ratio = 16/9, 
                tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'save', 'tap', 'crosshair'],
                title="Clustering of Literature with t-SNE and K-Means",
                toolbar_location="above",
                x_range = x_range,
                y_range = y_range)

        #plot settings
        plot.scatter('x', 'y', size=11,
                source=source,
                fill_color=mapper,
                line_alpha=1,
                line_width=1.1,
                line_color="blue",
                legend = 'labels')

        plot.legend.background_fill_alpha = 0.6

        #Keywords
        out_text = Paragraph(text= 'Keywords: Slide to specific cluster to see the keywords.', height=25)
        input_callback_1 = input_callback(plot, source, out_text, topics, num_clusters)

        # currently selected article
        div_curr = Div(text="""Click on a point to view the metadata of the research paper.""",height=150)
        callback_selected = CustomJS(args=dict(source=source, current_selection=div_curr), code=selected_code())
        taptool = plot.select(type=TapTool)
        taptool.callback = callback_selected

        #WIDGETS
        slider = Slider(start=0, end=num_clusters, value=num_clusters, step=1, title="Cluster #", callback = input_callback_1)
        # slider.js_on_change('value', callback)bo
        keyword = TextInput(title="Search:", callback = input_callback_1)
        # keyword.js_on_change('value', callback)

        #pass call back arguments
        input_callback_1.args["text"] = keyword
        input_callback_1.args["slider"] = slider
        # column(,,widgetbox(keyword),,widgetbox(slider),, notes, cite, cite2, cite3), plot

        # slider.sizing_mode = "scale_both"
        slider.width = 590
        slider.margin=15

        # keyword.sizing_mode = "scale_both"
        keyword.width = 615
        keyword.margin=15

        plot.sizing_mode = "scale_both"
        plot.align = 'center'
        plot.margin = (50,100,50,100)

        div_curr.style={'color': '#BF0A30', 'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 'font-size': '1.1em'}
        div_curr.sizing_mode = "scale_both"
        div_curr.margin = 20

        out_text.style={'color': '#0269A4', 'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 'font-size': '1.1em'}
        out_text.sizing_mode = "scale_both"
        out_text.margin = 20


        r = row(div_curr,out_text)
        r.sizing_mode = "stretch_width"

        description_search = Div(text="""<h3>Filter by Text:</h3><p1>Search keyword to filter out the plot. It will search pre-processed texts,
        titles, and authors. Press enter when ready.
        Clear and press enter to reset the plot.</p1>""")

        description_slider = Div(text="""<h3>Filter by the Clusters:</h3><p1>The slider below can be used to filter the target cluster.
        Adjust the slider to the desired cluster number to display the plots that belong to that cluster.
        Slide back to the right to show all.</p1>""")

        description_slider.style ={'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 'font-size': '1.1em'}
        description_slider.sizing_mode = "stretch_width"

        description_search.style ={'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 'font-size': '1.1em'}
        description_search.sizing_mode = "stretch_width"
        description_search.margin = 5


        layout_plot = layout([
                [description_slider, description_search],
                [slider, keyword],
                [out_text],
                [plot],
                [div_curr],
        ])

        return layout_plot


from bokeh.models.callbacks import CustomJS
from bokeh.models import ColumnDataSource, CustomJS, Slider

# handle the currently selected article
def selected_code():
    code = """
            var texts = [];
            cb_data.source.selected.indices.forEach(index => texts.push(source.data['display_text'][index].slice(0,2000)));

            text = "<h4>" + texts[0].toString() + "</h4>";

            current_selection.text =  text
            current_selection.change.emit();
        """

    return code

# handle the keywords and search
def input_callback(plot, source, out_text, topics, num_clusters):

    # slider call back for cluster selection

    callback = CustomJS(args=dict(p=plot, source=source, out_text=out_text, topics=topics, num_clusters=num_clusters), code="""
                    var key = text.value;
                    key = key.toLowerCase();
                    var cluster = slider.value;
                    var data = source.data;


                    x = data['x'];
                    y = data['y'];
                    x_backup = data['x_backup'];
                    y_backup = data['y_backup'];
                    labels = data['desc'];
                    text = data['search_text'];


                    if (cluster == num_clusters) {
                        out_text.text = 'Keywords: Slide to specific cluster to see the keywords.';
                        for (i = 0; i < x.length; i++) {
                            						if(text[i].includes(key)) {
                            							x[i] = x_backup[i];
                            							y[i] = y_backup[i];
                            						} else {
                            							x[i] = undefined;
                            							y[i] = undefined;
                            						}
                        }
                    }
                    else {
                        out_text.text = 'Keywords: ' + topics[Number(cluster)];
                        for (i = 0; i < x.length; i++) {
                            if(labels[i] == cluster) {
                                							if(text[i].includes(key))  {
                                								x[i] = x_backup[i];
                                								y[i] = y_backup[i];
                                							} else {
                                								x[i] = undefined;
                                								y[i] = undefined;
                                							}
                            } else {
                                x[i] = undefined;
                                y[i] = undefined;
                            }
                        }
                    }
                source.change.emit();
        """)
    return callback
