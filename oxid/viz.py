import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np
import pandas as pd

from data import IronOxide, DATA_TYPES, Data

pio.kaleido.scope.mathjax = None


def process_fig(fig:go.Figure, output:Path|None=None, show:bool=False) -> go.Figure:
    if show:
        fig.show()
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing to {output}")
        if output.suffix == ".html":
            fig.write_html(output)
        else:
            fig.write_image(output)
    return fig


def format_fig(fig):
    """Formats a plotly figure in a nicer way."""
    fig.update_layout(
        width=1200,
        height=550,
        plot_bgcolor="white",
        title_font_color="black",
        font=dict(
            family="Linux Libertine Display O",
            size=18,
            color="black",
        ),
    )
    gridcolor = "#dddddd"
    fig.update_xaxes(gridcolor=gridcolor, zerolinecolor="#eeeeee")
    fig.update_yaxes(gridcolor=gridcolor, zerolinecolor="#111111")

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside')

    return fig


def plot_moment(data:Data, fig:go.Figure|None=None, row=1, col=1, title:str="", show_x_axis:bool=True, show:bool=False, output:Path|None=None) -> go.Figure:
    if fig is None:
        fig = make_subplots(rows=1, cols=1)

    extracted_data = data.extract()
    for key, (x, y) in extracted_data.items():
        color = data.color(key)
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='lines+markers', showlegend=row==1, line_color=color, name=key),
            row=row,
            col=col,
        )
    if show_x_axis:
        fig.update_xaxes(title_text=data.x_axis, row=row, col=col)

    if col == 1:
        fig.update_yaxes(title_text='Moment (A⋅m2/kg)', row=row, col=col)

    fig.update_layout(title=title)
    format_fig(fig)

    return process_fig(fig, output, show)


def plot_standards(width:int=1100, height:int=1100, show:bool=False, output:Path|None=None) -> go.Figure:
    fig = go.Figure()
    fig = make_subplots(
        rows=len(IronOxide), cols=len(DATA_TYPES),
        subplot_titles=[data_type.title() for data_type in DATA_TYPES.values()],
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
    )

    for i, oxide in enumerate(IronOxide):
        show_x_axis = i == len(IronOxide) - 1
        for data_type_index, data_type in enumerate(DATA_TYPES):
            plot_moment(oxide.standard_data(data_type), fig=fig, row=i+1, col=data_type_index+1, show_x_axis=show_x_axis)

        fig.update_yaxes(title_text=oxide.title(), row=i+1, col=1)
        
    format_fig(fig)
    fig.update_layout(width=width, height=height)

    return process_fig(fig, output, show)


def plot_inputs(
    observations:list[np.ndarray], 
    basis_functions_list:list[list[np.ndarray]], 
    regimes:list[str],
    iron_oxides:list[IronOxide], 
    rescale:bool=False, 
    show:bool=False, 
    output:Path|None=None,
    mode:str='lines+markers',
) -> go.Figure:
    fig = make_subplots(rows=len(observations), cols=1, shared_xaxes=False, vertical_spacing=0.05, subplot_titles=regimes)

    row = 0
    for observed, basis_functions, regime in zip(observations, basis_functions_list, regimes):
        row += 1
        if rescale:
            observed = observed / np.max(observed)
            basis_functions = [basis_function / np.max(basis_function) for basis_function in basis_functions]

        fig.add_trace(
            go.Scatter(y=observed, mode=mode, name='Observed', marker_color='black', marker_line_color='black', marker_line_width=3, marker_size=5, showlegend=row==1),
            row=row,
            col=1,
        )

        for basis_function, iron_oxide in zip(basis_functions, iron_oxides):
            fig.add_trace(
                go.Scatter(
                    y=basis_function, 
                    mode=mode, 
                    name=f'{iron_oxide.title()} Basis Function',
                    marker=dict(color=iron_oxide.color),
                    showlegend=row==1,
                ),
                row=row,
                col=1,
            )

    title = 'Observed vs Basis Functions'
    if rescale:
        title += ' (Rescaled)'
        fig.update_yaxes(title_text='Rescaled Moment (A⋅m2/kg)', tickformat=".1%")
    else:
        fig.update_yaxes(title_text='Moment (A⋅m2/kg)')

    fig.update_xaxes(title_text='')
    fig.update_layout(**{f"xaxis{row}_title_text": 'Index'})

    fig.update_layout(title=title)
    format_fig(fig)
    fig.update_layout(height=200+300*len(observations))
    return process_fig(fig, output, show)


def plot_components(
    transformed_data:np.ndarray,
    df:pd.DataFrame,
    title:str="UMAP Projection",
    output:Path=None,
    show:bool=True,
    color_column: str = "Group",
    color_map:dict|None = None,
):
    """
    Plot the components of the transformed data using Plotly.
    """
    names = df["Name"].values
    cluster = df[color_column].values if color_column and color_column in df else None

    x = transformed_data[:,0]
    y = transformed_data[:,1] if transformed_data.shape[1] > 1 else np.zeros_like(x)
    
    fig = px.scatter(
        x=x, 
        y=y, 
        color=cluster, 
        hover_data={"Name": names},
        color_discrete_map=color_map,
    )
    fig.update_traces(marker_size=14)
    format_fig(fig)
    fig.update_layout(
        width=900,
        height=800,
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        title=title,
        legend_title="Category",
        xaxis=dict(
            zerolinecolor='#dddddd',
            zerolinewidth=1,
        ),
        yaxis=dict(
            zerolinecolor='#dddddd',
            zerolinewidth=1,
        ),
    )
    return process_fig(fig, output, show)


def plot_strip(
    transformed_data: np.ndarray,
    df: pd.DataFrame,
    title: str = "Component vs. Category",
    output: Path = None,
    show: bool = True,
    color_column: str = "Group",
    color_map:dict|None = None,
) -> go.Figure:
    """
    Plot scatter points like a rug chart:
    - x-axis: first component of transformed data
    - y-axis: category (Cluster)
    """
    df = df.copy()
    df["Component1"] = transformed_data[:, 0]

    fig = px.strip(
        df,
        x="Component1",
        y=color_column,
        color=color_column,
        stripmode="overlay",
        orientation="h",
        hover_data=["Name"],
        color_discrete_map=color_map,
    )

    fig.update_traces(jitter=0.3, marker=dict(opacity=0.8, size=8))
    format_fig(fig)
    fig.update_layout(
        xaxis_title="Component 1",
        yaxis_title="Category",
        title=title,
        legend_title="Category",
        xaxis=dict(zerolinecolor='#dddddd', zerolinewidth=1),
        yaxis=dict(zerolinecolor='#dddddd', zerolinewidth=1),
    )

    return process_fig(fig, output, show)

