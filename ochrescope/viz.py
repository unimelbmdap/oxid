import plotly.graph_objects as go
import plotly.io as pio

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

def plot_moment(df) -> go.Figure:
    fig = go.Figure()
    x_axis = 'Temperature (K)'
    decreasing = df[x_axis].diff().fillna(0) > 0

    decreasing_df = df[decreasing]
    increasing_df = df[~decreasing]

    fig.add_trace(go.Scatter(x=decreasing_df['Temperature (K)'], y=decreasing_df['Moment_Am2_per_kg'], mode='lines+markers', name='Decreasing'))
    fig.add_trace(go.Scatter(x=increasing_df['Temperature (K)'], y=increasing_df['Moment_Am2_per_kg'], mode='lines+markers', name='Increasing'))
    fig.update_layout(title='Moment vs Temperature', xaxis_title='Temperature (K)', yaxis_title='Moment_Am2_per_kg')
    format_fig(fig)
    return fig