import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import chart_studio.plotly as py
from dash import Dash, dcc, html, Input, Output

import dash_daq as daq
external_stylesheets =['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP, 'style.css']

import numpy as np# read csv data


# ======================== Setting the margins
layout = go.Layout(
    margin=go.layout.Margin(
        l=40,  # left margin
        r=40,  # right margin
        b=10,  # bottom margin
        t=35  # top margin
    )
)

def lookuplat(x):
    lat = 0
    if(x== 'Lille'):
     lat = 50.62925
    if(x== 'Paris'):
     lat = 48.856614
    if (x== 'Rennes'):
     lat = 48.117266
    if (x== 'Strasbourg'):
     lat = 48.5734053
    if (x== 'Bordeaux'):
     lat = 44.837789
    if (x == 'Marseille'):
     lat = 43.296482
    if (x == 'Lyon'):
     lat = 45.764043
    if (x == 'Tours'):
     lat = 47.394144

    return lat

def lookuplong(x):
    longitude = 0
    if (x == 'Lille'):
        longitude = 3.057256
    if (x == 'Paris'):
        longitude = 2.3522219
    if (x == 'Rennes'):
        longitude = -1.6777926
    if (x == 'Strasbourg'):
        longitude = 7.7521113
    if (x == 'Bordeaux'):
        longitude = -0.57918
    if (x == 'Marseille'):
        longitude = 5.36978
    if (x == 'Lyon'):
        longitude = 4.835659
    if (x == 'Tours'):
        longitude = 0.68484

    return longitude


df = pd.read_csv('data/LeadsFranceDate.csv', parse_dates=['CreatedDate'])
#df.drop('Unnamed: 0', axis=1, inplace=True)
df = df.sort_values(by=['CreatedDate', 'Lead Source'])
df['year'] = pd.DatetimeIndex(df['CreatedDate']).year
df['latitude'] = df.apply(lambda row: lookuplat(row.City), axis=1)
df['longitude'] = df.apply(lambda row: lookuplong(row.City), axis=1)
index = df.index
number_of_rows = len(index)
print(number_of_rows)
navbar = dbc.Nav()

app = dash.Dash(__name__)
server = app.server

region_names = df.City.unique()
region_names.sort()
total_volume = df.groupby(['year'])['Prospect ID'].size()
print(total_volume )
total_volume = total_volume.reset_index()
print(total_volume )

# pie chart - rating

col_label = "Converted"
col_values = "Count"

v = df[col_label].value_counts()
new2 = pd.DataFrame({
    col_label: v.index,
    col_values: v.values
})


# ======================== Plotly Graphs
def get_pie_chart():
    pieChart = dcc.Graph(
        figure=go.Figure(layout=layout).add_trace(go.Pie(
            labels=new2['Converted'],
            values=new2['Count'],
            marker=dict(line=dict(color='#ffffff', width=2)))).update_layout(title='Lead Conversion Rate',
                                                                             plot_bgcolor='rgba(0,0,0,0)',
                                                                             showlegend=False),
        style={'width': '50%', 'display': 'inline-block'})
    return pieChart



# ======================== Plotly Graphs
def get_scatter_plot():
    scatterPlot = dcc.Graph(figure=go.Figure(layout=layout).add_trace(go.Scatter(x=df['City'],
                                                                                 y=df['Converted'],
                                                                                 marker=dict(
                                                                                     color='#351e15'),
                                                                                 mode='markers')).update_layout(
        title='Conversion by City', plot_bgcolor='rgba(0,0,0,0)'),
        style={'width': '50%', 'height': '40vh', 'display': 'inline-block'})
    return scatterPlot



# ======================== Plotly Graphs
def get_bar_chart():
    barChart = dcc.Graph(figure=go.Figure(layout=layout).add_trace(go.Bar(x=df['Converted'],
                                                                          y=df.groupby(['City','Converted'])['Prospect ID'].agg('count'),
                                                                          marker=dict(color='#00008B'))).update_layout(
        title='Conversion by City'),
        style={'width': '50%', 'display': 'inline-block'})
    return barChart


# ========================


def get_bar_chart_stacked():
    barChart = dcc.Graph(figure=go.Figure(layout=layout).add_trace(go.Bar(x=df['Converted'],
                                                                          y=df.groupby(['City','Converted'])['Prospect ID'].agg('count'),
                                                                          marker=dict(color='#00008B'))).update_layout(
        title='Conversion by City'),
        style={'width': '50%', 'display': 'inline-block'})
    return barChart

bars = []
for city in df['City'].unique():
    bar = go.Bar(name=city, x=df[df['City'] == city]
                 ['Converted'], y=df[df['City'] == city].agg('count'))
    bars.append(bar)
figbarChartStacked = go.Figure(data=bars)
figbarChartStacked.update_layout(barmode='stack', title='Conversion By City')


# -------------------------

def get_bar_chart_ls():
    barChartls = dcc.Graph(figure=go.Figure(layout=layout).add_trace(go.Bar(x=df['Lead Source'],
                                                                          y=df.groupby(['Lead Source'])['Prospect ID'].agg('count'),
                                                                          marker=dict(color='#8080ff'))).update_layout(
        title='Lead Source'),
        style={'width': '100%', 'display': 'inline-block'})
    return barChartls


# -------------------------
def get_bar_chart_lo():
    barChartlo = dcc.Graph(figure=go.Figure(layout=layout).add_trace(go.Bar(x=df['Lead Origin'],
                                                                          y=df.groupby(['Lead Origin'])['Prospect ID'].agg('count'),
                                                                          marker=dict(color='#000080'))).update_layout(
        title='Lead Origin'),
        style={'width': '100%', 'display': 'inline-block'})
    return barChartlo

# -------------------------

volume_fig = px.bar(
    total_volume,
    x='year',
    y='Prospect ID',
    color='year',
    # barmode='group',
    labels={'year':'Year'},
    title = 'Lead volume per year', barmode = 'stack'
)
volume_fig.update_layout(xaxis_type='category', plot_bgcolor="white")
# Map
fig = go.Figure(data=go.Scattergeo(
    lon=df['longitude'],
    lat=df['latitude'],
    text=df['City']  ,
    mode='markers',
    marker_color=df.groupby(['City'])['Prospect ID'].value_counts()
))

fig.update_layout(
    geo_scope='europe'
)

print(df)
df['year'] = pd.DatetimeIndex(df['CreatedDate']).year
total_volumeLeadOrigin = df.groupby("Lead Origin")['Prospect ID'].agg('count')
total_volumeLeadOrigin = total_volumeLeadOrigin.reset_index()
print(total_volumeLeadOrigin)

fig1 = px.bar(total_volumeLeadOrigin,
    x="Lead Origin", #x
    y='Prospect ID',
    labels={"x": "Lead Origin", "y": "Count"}, #define lable
    color="Lead Origin",
    color_continuous_scale=px.colors.sequential.RdBu,#color
    text='Lead Origin',#text
    title="Recorded leads by Lead Origin", #title
    orientation="h"  #horizonal bar chart
)


# Box plots
fig4 = px.box(df, x="Converted", y="TotalVisits",points="all",color="Converted")

fig5 = px.box(df, x="Converted", y="Total Time Spent on Website",points="all",color="Converted")



tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #151B54',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #151B54',
    'borderBottom': '1px solid #151B54',
    'backgroundColor': '#151B54',
    'color': 'white',
    'padding': '6px'
}

bar1 = dcc.Graph(
                        id='volume-bar-graph',
                        figure=volume_fig
                    )
pie3 = dcc.Graph(
                        id='volume-pie',
                        figure=volume_fig
                    )



# 2 Bars plot
s1 = df[df['Converted']== 0]
print(s1.groupby("City")["Converted"].value_counts())
print(s1)
s2 = df[df['Converted']== 1]
print(s2.groupby("City")["Converted"].value_counts())
print(s2)
trace1 = go.Bar(    #setup the chart for Resolved records
    x=s1["City"].unique(), #x for Resolved records
    y=s1.groupby("City")["Converted"].value_counts(),#y for Resolved records

    marker_color=px.colors.qualitative.Dark24[0],  #color
    text=s1.groupby("City")["Converted"].value_counts(), #label/text
    textposition="outside", #text position
    name="1", #legend name
)
trace2 = go.Bar(   #setup the chart for Unresolved records
    x=s2["City"].unique(),
    y=s2.groupby("City")["Converted"].value_counts(),
    text=s2.groupby("City")["Converted"].value_counts(),
    marker_color=px.colors.qualitative.Dark24[1],
    textposition="outside",
    name="0",
)

data = [trace1, trace2] #combine two charts/columns
layout = go.Layout(barmode="group", title="Converted vs Not Converted") #define how to display the columns
fig3 = go.Figure(data=data, layout=layout)
fig3.update_layout(
    title=dict(x=0.5), #center the title
    xaxis_title="City",#setup the x-axis title
    yaxis_title="Converted", #setup the x-axis title
    margin=dict(l=20, r=20, t=60, b=20),#setup the margin
    paper_bgcolor="aliceblue", #setup the background color
)
fig3.update_traces(texttemplate="%{text:.2s}") #text formart

# Box plot



app.layout = html.Div([
    dcc.Tabs(id="tabs-inline", value='tab-1', children=[
        dcc.Tab(label='Discovery', value='tab-1', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Exploration', value='tab-2', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Machine Learning', value='tab-3', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Simulation', value='tab-4', style=tab_style, selected_style=tab_selected_style),
    ], style=tabs_styles),
    html.Div(id='tabs-content-inline-3')
])

@app.callback(Output('tabs-content-inline-3', 'children'),
              Input('tabs-inline', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            #html.H1('Leads Data Set Discovery : Number of Records ' + str(number_of_rows) , className="app-header"),
            html.H1(''),
            daq.LEDDisplay(
                label='Leads Data Set Discovery - Records Count',
                labelPosition='bottom',
                value=str(number_of_rows),
                backgroundColor="#00008B"
            ),
            html.Div(
                children=[
                    html.Div(
                        children=dcc.Graph(
                            id='volume-bar-graph',
                            figure=volume_fig,
                            #  config={"displayModeBar": False},
                        ),
                        style={'width': '50%', 'height': '50vh', 'display': 'inline-block'},


                    ),
                    get_pie_chart(),
                    html.Div(dcc.Graph(figure=figbarChartStacked),style={'width': '50%', 'height': '50vh', 'display': 'inline-block'}),

                ],
                className='double-graph',
            ),
            html.Div(
                children=[
                    html.Div(get_bar_chart_ls(),style={'width': '60%', 'height': '20vh', 'display': 'inline-block'}),
                    html.Div(get_bar_chart_lo(),style={'width': '60%', 'height': '20vh', 'display': 'inline-block'})
                ],
                className='double-graph',
            ),



        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Leads Data Set Exploration'),
            html.Div(children=[
                         html.Div(children='Year', style={'fontSize': "24px"}, className='menu-title'),
                         dcc.Dropdown(
                             id='year-filter',
                             options=[
                                 {'label': Year, 'value': Year}
                                 for Year in df.year.unique()
                             ],  # 'Year' is the filter
                             value='2022',
                             clearable=False,
                             searchable=False,
                             className='dropdown', style={'fontSize': "24px", 'textAlign': 'center'},
                         ),
                     ],
                     className = 'menu',
        ), # the dropdown function
         html.Div(
                        children=dcc.Graph(
                            id='bibar',
                            figure=fig3,
                            # config={"displayModeBar": False},
                        ),
                        style={'width': '100%', 'display': 'inline-block'},
                    ),
         dcc.Graph(figure=fig4),
         dcc.Graph(figure=fig5),

        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Leads Data Set Modelisation')
        ])
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Leads Simulation'),
        ])

@app.callback(
    Output("bibar", "figure"),
    [Input("year-filter", "value")],
)


def update_charts(Year):
    filtered_s1 = s1[s1["year"] == Year]
    filtered_s2 = s2[s2["year"] == Year]
    trace1 = go.Bar(
        x=filtered_s1["City"].unique(),
        y=filtered_s1.groupby("City")["Converted"].value_counts(),
        text=filtered_s1.groupby("City")["Converted"].value_counts(),
        textposition="outside",
        marker_color=px.colors.qualitative.Dark24[0],
        name="0",
    )
    trace2 = go.Bar(
        x=filtered_s2["City"].unique(),
        y=filtered_s2.groupby("City")["Converted"].value_counts(),
        text=filtered_s2.groupby("City")["Converted"].value_counts(),
        textposition="outside",
        marker_color=px.colors.qualitative.Dark24[1],
        name="1",
    )
    data = [trace1, trace2]
    layout = go.Layout(barmode="group", title="Converted vs Not Converted By City and year : " + str(Year))
    bibar = go.Figure(data=data, layout=layout)
    bibar.update_layout(
        title=dict(x=0.5),
        xaxis_title="City",
        yaxis_title="Converted",
        paper_bgcolor="aliceblue",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    bibar.update_traces(texttemplate="%{text:.2s}")
    return bibar


if __name__ == '__main__':
    app.run_server(debug=True)
