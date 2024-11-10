import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .data import IronOxide, Measurement

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


def plot_moment(df, fig:go.Figure|None=None, row=1, col=1, x_axis='Temperature (K)', title:str="", show_x_axis:bool=True) -> go.Figure:
    if fig is None:
        fig = make_subplots(rows=1, cols=1)

    decreasing = df[x_axis].diff().fillna(0) > 0

    decreasing_df = df[decreasing]
    increasing_df = df[~decreasing]

    fig.add_trace(
        go.Scatter(x=decreasing_df[x_axis], y=decreasing_df['Moment_Am2_per_kg'], mode='lines+markers', name='Decreasing', showlegend=row==1 and col==1, line_color='red'),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(x=increasing_df[x_axis], y=increasing_df['Moment_Am2_per_kg'], mode='lines+markers', name='Increasing', showlegend=row==1 and col==1, line_color='blue'),
        row=row,
        col=col,
    )
    if show_x_axis:
        fig.update_xaxes(title_text=x_axis, row=row, col=col)

    if col == 1:
        fig.update_yaxes(title_text='Moment_Am2_per_kg', row=row, col=col)
        
    fig.update_layout(title=title)
    format_fig(fig)
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
        plot_moment(oxide.read_data(Measurement.HYSTERESIS), fig=fig, row=i+1, col=1, x_axis='Magnetic Field (Oe)', show_x_axis=show_x_axis)
        plot_moment(oxide.read_data(Measurement.RTSIRM), fig=fig, row=i+1, col=2, x_axis='Temperature (K)', show_x_axis=show_x_axis)
        plot_moment(oxide.read_data(Measurement.ZFCFC), fig=fig, row=i+1, col=3, x_axis='Temperature (K)', show_x_axis=show_x_axis)

    fig.update_layout(width=1100, height=800)

    return fig