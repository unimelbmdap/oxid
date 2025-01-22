import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np

from .data import IronOxide, DATA_TYPES, Data

pio.kaleido.scope.mathjax = None


def process_fig(fig:go.Figure, output:Path|None=None, show:bool=False) -> None:
    if show:
        fig.show()
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(output)


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

    process_fig(fig, output, show)
    return fig


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

    process_fig(fig, output, show)
    return fig


def plot_inputs(
    observed:np.ndarray, 
    basis_functions:list[np.ndarray], 
    iron_oxides:list[IronOxide], 
    rescale:bool=False, 
    show:bool=False, 
    output:Path|None=None,
    mode:str='markers',
) -> go.Figure:
    fig = go.Figure()

    if rescale:
        observed = observed / np.max(observed)
        basis_functions = [basis_function / np.max(basis_function) for basis_function in basis_functions]

    fig.add_trace(
        go.Scatter(y=observed, mode=mode, name='Observed', marker=dict(color='black')),
    )

    for basis_function, iron_oxide in zip(basis_functions, iron_oxides):
        fig.add_trace(
            go.Scatter(
                y=basis_function, 
                mode=mode, 
                name=f'{iron_oxide.title()} Basis Function',
            ),
        )

    title = 'Observed vs Basis Functions'
    if rescale:
        title += ' (Rescaled)'
        fig.update_yaxes(title_text='Moment (A⋅m2/kg) rescaled by maximum value', tickformat=".1%")
    else:
        fig.update_yaxes(title_text='Moment (A⋅m2/kg)')

    fig.update_xaxes(title_text='Interpolated Data Point')

    fig.update_layout(title=title)
    format_fig(fig)
    process_fig(fig, output, show)
    return fig


def get_bin_size(data):
    num_bins = int(np.ceil(np.log2(len(data)) + 1))
    bin_size = (data.max() - data.min()) / num_bins
    return bin_size if bin_size > 0 else 0.05


def plot_posterior_histograms(
    inference_data,
    show: bool = False,
    output: Path | None = None,
) -> go.Figure:
    fig = go.Figure()

    for iron_oxide in IronOxide:
        key = f"{iron_oxide}_proportion"
        if key in inference_data.posterior:
            data = inference_data.posterior[key].values.flatten()
            mean_value = np.mean(data)

            # Compute histogram bins manually
            bin_heights, bin_edges = np.histogram(data, bins="auto", density=True)
            bin_width = np.diff(bin_edges)[0]  # Get uniform bin width
            max_y = max(bin_heights) if len(bin_heights) > 0 else 0  # Find max bin height

            # Set bin centers by shifting the edges by half the bin width
            bin_centers = bin_edges[:-1] + bin_width / 2

            # Use go.Bar with width equal to bin width to remove gaps
            fig.add_trace(
                go.Bar(
                    x=bin_centers,
                    y=bin_heights * bin_width,  # Normalize to match Plotly's probability scaling
                    width=bin_width,  # Set bar width to be the bin width
                    name=iron_oxide.title(),
                    marker=dict(color=iron_oxide.color),
                )
            )

            # Add annotation at the top of the highest bin
            fig.add_annotation(
                x=mean_value,
                y=max_y * bin_width * 1.05,  # Slightly above the highest bar
                xref="x",
                yref="y",
                text=f"{mean_value:.2%}",  # Convert mean to percentage
                showarrow=False,
                arrowhead=0,
                arrowcolor=iron_oxide.color,
                font=dict(color=iron_oxide.color),
            )

    # Update layout to remove gaps between bars
    fig.update_layout(
        barmode="overlay",  # Overlay histograms
        bargap=0.0,  # No space between bars
        bargroupgap=0.0,  # No space between histogram groups
        xaxis_title_text="Proportion",
        xaxis_tickformat=".1%",
    )

    format_fig(fig)
    process_fig(fig, output, show)
    
    return fig


def plot_posterior_predictive_check(
    inference_data,
    show: bool = False,
    output: Path | None = None,
) -> go.Figure:

    keys = []
    for key in inference_data.keys():
        if key.startswith("posterior_predictive_"):
            keys.append(key)

    fig = make_subplots(rows=len(keys), cols=1, shared_xaxes=False, subplot_titles=[key.replace("posterior_predictive_", "") for key in keys], vertical_spacing=0.1)

    for index, key in enumerate(keys):
        # ppc = inference_data[key]["likelihood"].stack(draws=("chain", "draw")).values
        # for i in range(ppc.shape[1]):
        #     fig.add_trace(
        #         go.Scatter(
        #             y=ppc[:, i],
        #             line=dict(color="gray", width=0.5),
        #             showlegend=False,
        #         ),
        #         row=index + 1,
        #         col=1,
        #     )

        linear_combinations = inference_data[key.replace("posterior_predictive", "linear_combination")]["linear_combination"].stack(draws=("chain", "draw")).values
        for i in range(linear_combinations.shape[1]):
            fig.add_trace(
                go.Scatter(
                    y=linear_combinations[:, i],
                    line=dict(color="gray", width=0.5),
                    showlegend=False,
                    opacity=0.05,
                ),
                row=index + 1,
                col=1,
            )

        observed = inference_data[key.replace("posterior_predictive", "observed")]["likelihood"].values
        fig.add_trace(
            go.Scatter(
                y=observed,
                mode="markers+lines",
                name=f"Observed {key}",
                showlegend=False,
            ),
            row=index + 1,
            col=1,
        )

    format_fig(fig)
    fig.update_layout(height=200+300*len(keys))

    process_fig(fig, output, show)

    
    return fig
