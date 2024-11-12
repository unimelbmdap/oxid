import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np

from .data import IronOxide, DATA_TYPES, Data

pio.kaleido.scope.mathjax = None


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
    fig.update_xaxes(gridcolor=gridcolor)
    fig.update_yaxes(gridcolor=gridcolor)

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

    if show:
        fig.show()
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(output)

    return fig


def plot_standards() -> go.Figure:
    fig = go.Figure()
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=["Hysteresis", "RT-SIRM", "ZFC-FC"],
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
    )

    for i, oxide in enumerate(IronOxide):
        show_x_axis = i == len(IronOxide) - 1
        for data_type_index, data_type in enumerate(DATA_TYPES):
            plot_moment(oxide.standard_data(data_type), fig=fig, row=i+1, col=data_type_index+1, show_x_axis=show_x_axis)

    fig.update_layout(width=1100, height=800)

    return fig


def plot_inputs(observed:np.ndarray, basis_functions:list[np.ndarray]) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(y=observed, mode='markers', name='Observed', marker=dict(color='black')),
    )

    for i, basis_function in enumerate(basis_functions):
        fig.add_trace(
            go.Scatter(y=basis_function, mode='markers', name=f'Basis Function {i}'),
        )

    fig.update_layout(title='Observed vs Basis Functions')
    format_fig(fig)
    return fig