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
    # Get datatypes from the posterior keys
    datatypes = []
    for key in inference_data.posterior.keys():
        if key.endswith("_factor"):
            datatype = key.split("_")[1]
            if datatype not in datatypes:
                datatypes.append(datatype)

    fig = make_subplots(rows=len(datatypes), cols=1, shared_xaxes=True, subplot_titles=datatypes, vertical_spacing=0.03)

    for datatype in datatypes:
        row = datatypes.index(datatype) + 1
        for iron_oxide in IronOxide:
            key = f"{iron_oxide}_{datatype}_factor"
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
                        marker=dict(color=iron_oxide.color, line_width=0),
                        showlegend=row==1,
                    ),
                    row=row,
                    col=1,
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
                    row=row,
                    col=1,
                )

    # Update layout to remove gaps between bars
    fig.update_layout(
        barmode="overlay",  # Overlay histograms
        bargap=0.0,  # No space between bars
        bargroupgap=0.0,  # No space between histogram groups
        # xaxis_title_text="Factor",
        xaxis_tickformat=".1%",
    )

    format_fig(fig)
    fig.update_layout(
        height=len(datatypes) * 200 + 200
    )
    process_fig(fig, output, show)
    
    return fig


def plot_posterior_predictive_check(
    inference_data,
    show: bool = False,
    output: Path | None = None,
) -> go.Figure:

    keys = []
    for key in inference_data['posterior'].keys():
        if key.startswith("predicted_"):
            keys.append(key)

    fig = make_subplots(rows=len(keys), cols=1, shared_xaxes=False, subplot_titles=[key.replace("predicted_", "") for key in keys], vertical_spacing=0.1)

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
        predictions = inference_data['posterior'][key].stack(draws=("chain", "draw")).values
        for i in range(predictions.shape[1]):
            fig.add_trace(
                go.Scatter(
                    y=predictions[:, i],
                    line=dict(color="gray", width=0.5),
                    showlegend=False,
                    opacity=0.05,
                ),
                row=index + 1,
                col=1,
            )

        linear_combinations = inference_data['posterior'][key.replace('predicted', 'linear_combination')].stack(draws=("chain", "draw")).values
        for i in range(linear_combinations.shape[1]):
            fig.add_trace(
                go.Scatter(
                    y=linear_combinations[:, i],
                    line=dict(color="purple", width=0.5),
                    showlegend=False,
                    opacity=0.05,
                ),
                row=index + 1,
                col=1,
            )

        observed = inference_data['observed_data'][key.replace("predicted", "likelihood")].values
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
