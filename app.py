import os, base64, io, threading
import polars as pl
import networkx as nx
import pandas as pd
import igraph as ig
import leidenalg
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx
import dash

app = Dash(__name__, suppress_callback_exceptions=True)

# ── Global state ──────────────────(All Data from Pipeline)───────────────────────────────────────────
pipeline_data = {}  

# ── Styles ───────────────────────────────────────────────────────────────────
CARD = {
    "background": "white",
    "borderRadius": "10px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
    "padding": "16px"
}

# ── Upload page layout ────────────────────────────────────────────────────────
upload_layout = html.Div([
    # Background grid pattern
    html.Div(style={
        "position": "fixed", "inset": "0", "zIndex": "0",
        "backgroundImage": "linear-gradient(rgba(59,130,246,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(59,130,246,0.04) 1px, transparent 1px)",
        "backgroundSize": "40px 40px",
        "pointerEvents": "none"
    }),

    html.Div([

        # ── Header ──────────────────────────────────────────────
        html.Div([
            html.Div("⬡", style={
                "fontSize": "32px", "color": "#3b82f6",
                "lineHeight": "1", "marginBottom": "12px"
            }),
            html.H1("Temporal Network Explorer", style={
                "fontFamily": "'Georgia', serif",
                "fontWeight": "700", "fontSize": "36px",
                "color": "#0f172a", "margin": "0 0 8px",
                "letterSpacing": "-0.5px"
            }),
            html.P("Detect, visualize and animate community evolution in temporal graphs.", style={
                "color": "#64748b", "fontSize": "15px",
                "fontFamily": "Georgia, serif", "margin": "0",
                "fontStyle": "italic"
            }),
        ], style={"textAlign": "center", "marginBottom": "36px"}),

        # ── Upload Zone ─────────────────────────────────────────
        dcc.Upload(
            id="upload-data",
            children=html.Div([
                html.Div("📂", style={"fontSize": "40px", "marginBottom": "10px"}),
                html.Div("Drop your file here or click to browse", style={
                    "fontWeight": "600", "fontSize": "15px",
                    "color": "#1e293b", "fontFamily": "Georgia, serif",
                    "marginBottom": "6px"
                }),
                html.Div(".edges · .csv · .txt  —  columns: u, v, weight, timestamp", style={
                    "fontSize": "12px", "color": "#94a3b8",
                    "fontFamily": "monospace", "letterSpacing": "0.3px"
                }),
            ], style={"textAlign": "center", "padding": "8px"}),
            style={
                "border": "2px dashed #cbd5e1",
                "borderRadius": "14px",
                "padding": "44px 32px",
                "cursor": "pointer",
                "background": "linear-gradient(135deg, #f8fafc 0%, #f0f7ff 100%)",
                "transition": "all 0.2s",
                "marginBottom": "28px",
            },
            multiple=False
        ),

        html.Div(id="upload-status", style={"marginBottom": "20px", "fontFamily": "Georgia, serif"}),

        # ── Config Grid ─────────────────────────────────────────
        html.Div([

            # Max Activity
            html.Div([
                html.Label("Max Activity / Window", style={
                    "display": "block", "fontSize": "11px", "fontWeight": "700",
                    "color": "#64748b", "letterSpacing": "0.8px",
                    "textTransform": "uppercase", "marginBottom": "8px",
                    "fontFamily": "Georgia, serif"
                }),
                dcc.Input(id="max-activity", type="number", value=15000, min=1000, step=1000,
                          style={
                              "width": "100%","height": "50px", "padding": "10px 14px",
                              "borderRadius": "8px", "border": "1.5px solid #e2e8f0",
                              "fontSize": "14px", "fontFamily": "monospace",
                              "background": "white", "color": "#0f172a",
                              "boxSizing": "border-box", "outline": "none"
                          }),
            ], style={"flex":"1","minWidth": "0"}),

            # Top Communities
            html.Div([
                html.Label("Top Communities", style={
                    "display": "block", "fontSize": "11px", "fontWeight": "700",
                    "color": "#64748b", "letterSpacing": "0.8px",
                    "textTransform": "uppercase", "marginBottom": "8px",
                    "fontFamily": "Georgia, serif"
                }),
                dcc.Input(id="top-comms", type="number", value=30, min=5, max=100, step=5,
                          style={
                              "width": "100%","height": "50px", "padding": "10px 14px",
                              "borderRadius": "8px", "border": "1.5px solid #e2e8f0",
                              "fontSize": "14px", "fontFamily": "monospace",
                              "background": "white", "color": "#0f172a",
                              "boxSizing": "border-box", "outline": "none"
                          }),
            ], style={"flex": "1", "minWidth": "0"}),

            # Resolution
            html.Div([
                html.Label("Leiden Resolution", style={
                    "display": "block", "fontSize": "11px", "fontWeight": "700",
                    "color": "#64748b", "letterSpacing": "0.8px",
                    "textTransform": "uppercase", "marginBottom": "8px",
                    "fontFamily": "Georgia, serif"
                }),
                dcc.Input(id="resolution", type="number", value=1.0, min=0.01, max=5.0, step=0.1,
                          style={
                              "width": "100%","height": "50px", "padding": "10px 14px",
                              "borderRadius": "8px", "border": "1.5px solid #e2e8f0",
                              "fontSize": "14px", "fontFamily": "monospace",
                              "background": "white", "color": "#0f172a",
                              "boxSizing": "border-box", "outline": "none"
                          }),
            ], style={"flex": "1", "minWidth": "0"}),

            # Separator
            html.Div([
                html.Label("Separator", style={
                    "display": "block", "fontSize": "11px", "fontWeight": "700",
                    "color": "#64748b", "letterSpacing": "0.8px",
                    "textTransform": "uppercase", "marginBottom": "8px",
                    "fontFamily": "Georgia, serif"
                }),
                dcc.Dropdown(
                    id="separator",
                    options=[
                        {"label": "Comma (,)", "value": ","},
                        {"label": "Space ( )", "value": " "},
                        {"label": "Tab (\\t)", "value": "\t"},
                    ],
                    value=",",
                    clearable=False,
                    style={"fontFamily": "monospace", "fontSize": "14px"}
                ),
            ], style={"flex": "1", "minWidth": "0"}),

        ], style={
            "display": "flex", "gap": "16px", "marginBottom": "28px","height":"50px",
            "flexWrap": "wrap"
        }),

        # ── Run Button ──────────────────────────────────────────
        html.Button("🚀  Run Pipeline", id="run-btn", n_clicks=0, style={
            "width": "100%", "padding": "15px",
            "fontSize": "15px", "fontWeight": "700",
            "fontFamily": "Georgia, serif", "letterSpacing": "0.3px",
            "background": "linear-gradient(135deg, #2563eb, #3b82f6)",
            "color": "white", "border": "none",
            "borderRadius": "10px", "cursor": "pointer",
            "boxShadow": "0 4px 14px rgba(59,130,246,0.35)",
            "transition": "all 0.2s"
        }),

        dcc.Store(id="file-store"),
        dcc.Store(id="pipeline-done", data=False),
        dcc.Interval(id="progress-interval", interval=1000, disabled=True),

        html.Div(id="progress-display", style={
            "marginTop": "18px", "textAlign": "center",
            "fontFamily": "Georgia, serif", "fontSize": "14px", "color": "#64748b"
        }),

    ], style={
        "maxWidth": "760px", "margin": "60px auto",
        "background": "white",
        "borderRadius": "18px",
        "boxShadow": "0 4px 32px rgba(15,23,42,0.10), 0 1px 4px rgba(15,23,42,0.06)",
        "padding": "44px 48px",
        "position": "relative", "zIndex": "1"
    })

], style={
    "backgroundColor": "#f1f5f9",
    "minHeight": "100vh",
    "padding": "20px"
})


# ── Dashboard layout (built dynamically after pipeline) ───────────────────────
def build_dashboard_layout(file_name):
    d = pipeline_data
    time_keys = d["time_keys"]
    trend_fig_nodes   = d["trend_fig_nodes"]
    trend_fig_edges   = d["trend_fig_edges"]
    trend_fig_density = d["trend_fig_density"]

    return html.Div([
        html.Div([
            html.H2(file_name,
                    style={"margin": 0, "fontFamily": "sans-serif",
                           "fontWeight": "700", "color": "#1e293b"}),
            html.Button("⬅ Upload New File", id="back-btn", n_clicks=0,
                        style={"fontSize": "13px", "padding": "6px 14px",
                               "cursor": "pointer", "borderRadius": "6px",
                               "border": "1px solid #cbd5e1", "background": "white"})
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "marginBottom": "16px"}),

        # ── top row ──────────────────────────────────────────────────────
        html.Div([
            # LEFT (Metrics and Sliders)
            html.Div([
                html.H4("Control Panel", style={"marginTop": 0, "fontFamily": "sans-serif"}),
                html.Label("Time Window", style={"fontWeight": "600", "fontFamily": "sans-serif"}),
                html.Div(id="slider-container", style={"marginBottom": "20px"}),
                html.Button("▶ Play", id="play-pause-btn", n_clicks=0,
                            style={"fontSize": "14px", "padding": "6px 20px",
                                   "cursor": "pointer", "marginBottom": "20px",
                                   "borderRadius": "6px", "border": "1px solid #ccc",
                                   "fontFamily": "sans-serif"}),
                dcc.Interval(id="interval", interval=800, disabled=True),
                dcc.Store(id="is-playing", data=False),
                html.Hr(),
                html.H4("Metrics & Statistics",
                        style={"marginTop": "10px", "fontFamily": "sans-serif"}),
                html.Div(id="metrics-panel")
            ], style={**CARD, "width": "25%"}),

            # CENTRE ( Forece Directed Graph )
            html.Div([
                html.H4("Network Visualization",
                        style={"marginTop": 0, "fontFamily": "sans-serif"}),
                dcc.Graph(id="community-graph", style={"height": "400px"})
            ], style={**CARD, "width": "72%", "overflow": "hidden"}),

        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

        # ── bottom row ──────────────────────────(trends chart)──────────────────────────
        html.Div([
            html.H4("Temporal Trends & Charts",
                    style={"marginTop": 0, "marginBottom": "12px", "fontFamily": "sans-serif"}),
            html.Div([
                dcc.Graph(figure=trend_fig_nodes,   style={"flex": "1", "minWidth": 0}),
                dcc.Graph(figure=trend_fig_edges,   style={"flex": "1", "minWidth": 0}),
                dcc.Graph(figure=trend_fig_density, style={"flex": "1", "minWidth": 0}),
            ], style={"display": "flex", "gap": "12px"})
        ], style={**CARD}),

    ], style={"backgroundColor": "#f1f5f9", "padding": "24px", "minHeight": "100vh"})


# ── Root layout ───────────────────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Store(id="page", data="upload"),
    dcc.Store(id="filename-store", data=""),
    html.Div(id="page-content")
])


# ── Page routing ──────────────────────────────────────────────────────────────
@app.callback(
    Output("page-content", "children"),
    Input("page", "data"),
    State("filename-store", "data"),
)
def render_page(page, filename):
    if page == "dashboard":
        return build_dashboard_layout(filename)
    return upload_layout


# ── File upload handler ───────────────────────────────────────────────────────
@app.callback(
    Output("upload-status", "children"),
    Output("file-store", "data"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True
)
def handle_upload(contents, filename):
    if contents is None:
        return "", None
    
    # decode and save to disk immediately — don't keep in browser memory
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    
    save_path = os.path.join("/tmp", filename)
    with open(save_path, "wb") as f:
        f.write(decoded)
    
    return (
        html.Div([
            html.Span("✅ ", style={"fontSize": "16px"}),
            html.Span(f"Loaded: {filename}",
                      style={"fontWeight": "600", "color": "#16a34a"})
        ]),
        {"filepath": save_path, "filename": filename}  # store path not contents
    )

# ── Pipeline runner ───────────────────────────────────────────────────────────
def run_pipeline(filepath, filename, max_activity, top_comms_n, resolution, separator):
    global pipeline_data
    pipeline_data["status"] = "running"
    pipeline_data["progress"] = "Loading file..."

    try:
        df = pl.read_csv(
            filepath,  # read directly from disk
            separator=separator,
            has_header=False,
            comment_prefix="%",
            new_columns=["u", "v", "weight", "timestamp"]
        )
        pipeline_data["progress"] = "Computing time windows..."
        activity = (
            df.group_by("timestamp")
            .agg(
                pl.concat_list(["u", "v"]).list.explode().unique().len().alias("active_nodes")
            )
            .sort("active_nodes")
        )
        activity = activity.with_columns(pl.col("active_nodes").cum_sum().alias("cum_activity"))
        activity = activity.with_columns(
            (pl.col("cum_activity") / max_activity).floor().cast(pl.Int32).alias("time_cluster")
        )
        df = df.join(activity.select(["timestamp", "time_cluster"]), on="timestamp", how="left")

        pipeline_data["progress"] = "Running Leiden community detection"
        results = []
        time_clusters = df["time_cluster"].unique().sort().to_list()
        for i, t in enumerate(time_clusters):
            pipeline_data["progress"] = f"Community detection: window {i+1}/{len(time_clusters)}"
            df_t = df.filter(pl.col("time_cluster") == t)
            edges = df_t.select(["u", "v", "weight"]).to_numpy()
            g = ig.Graph.TupleList(edges, weights=True, directed=False)
            partition = leidenalg.find_partition(
                g, leidenalg.RBConfigurationVertexPartition,
                weights="weight", resolution_parameter=resolution
            )
            for node_id, comm in zip(g.vs["name"], partition.membership):
                results.append({"node": int(node_id), "time_cluster": t, "community": int(comm)})

        pipeline_data["progress"] = "Building community data..."
        communities = pl.DataFrame(results)
        comm_raw = communities

        communities = communities.with_columns(
            pl.len().over(["time_cluster", "community"]).alias("size")
        )
        top_comms = (
            communities.group_by(["time_cluster", "community"])
            .agg(pl.count().alias("size"))
            .sort(["time_cluster", "size"], descending=True)
            .group_by("time_cluster").head(top_comms_n)
        )
        communities = communities.join(
            top_comms.select(["time_cluster", "community"]),
            on=["time_cluster", "community"], how="inner"
        )

        df_c = df.join(communities.rename({"node": "u", "community": "cu"}),
                       on=["u", "time_cluster"], how="left")
        df_c = df_c.join(communities.rename({"node": "v", "community": "cv"}),
                         on=["v", "time_cluster"], how="left")
        df_c_raw = df.join(comm_raw.rename({"node": "u", "community": "cu"}),
                           on=["u", "time_cluster"], how="left")
        df_c_raw = df_c_raw.join(comm_raw.rename({"node": "v", "community": "cv"}),
                                 on=["v", "time_cluster"], how="left")
        df_c = df_c.filter(pl.col("cu").is_not_null() & pl.col("cv").is_not_null())

        community_sizes = df_c.group_by(["time_cluster", "cu"]).agg(pl.len().alias("size"))
        community_sizes_raw = df_c_raw.group_by(["time_cluster", "cu"]).agg(pl.len().alias("size"))

        community_edges = df_c.group_by(["time_cluster", "cu", "cv"]).agg(pl.len().alias("weight"))
        community_edges_inter = (
            df_c.filter(pl.col("cu") != pl.col("cv"))
            .group_by(["time_cluster", "cu", "cv"]).agg(pl.len().alias("weight"))
        )
        community_edges_inter_raw = (
            df_c_raw.filter(pl.col("cu") != pl.col("cv"))
            .group_by(["time_cluster", "cu", "cv"]).agg(pl.len().alias("weight"))
        )
        community_edges_intra = (
            df_c.filter(pl.col("cu") == pl.col("cv"))
            .group_by(["time_cluster", "cu"]).agg(pl.len().alias("internal_weight"))
        )

        pipeline_data["progress"] = "Computing graph layout..."
        Gc = nx.Graph()
        for cu, cv, w in community_edges.select(["cu", "cv", "weight"]).iter_rows():
            Gc.add_edge(int(cu), int(cv), weight=float(w))
        pos = nx.kamada_kawai_layout(Gc)
        pos_df = (
            pd.DataFrame.from_dict(pos, orient="index", columns=["x", "y"])
            .reset_index().rename(columns={"index": "community"})
        )

        community_edges = (
            community_edges
            .join(pl.from_pandas(pos_df).rename({"community": "cu"}), on="cu")
            .rename({"x": "x_u", "y": "y_u"})
        )

        community_edges = (
            community_edges
            .join(pl.from_pandas(pos_df).rename({"community": "cv"}), on="cv")
            .rename({"x": "x_v", "y": "y_v"})
        )

        # ── deduplicate: one row per (time_cluster, cu, cv) ──
        community_edges = (
            community_edges
            .group_by(["time_cluster", "cu", "cv", "x_u", "y_u", "x_v", "y_v"])
            .agg(pl.sum("weight").alias("weight"))
        )
        community_sizes = df_c.group_by(["time_cluster", "cu"]).agg(pl.len().alias("size"))
        community_edges = community_edges.join(community_sizes, on=["time_cluster", "cu"], how="left")

        pipeline_data["progress"] = "Building snapshot figures..."

        def community_snapshot(t, centrality_dict=None):
            g = community_edges.filter(pl.col("time_cluster") == t)
            g_intra = community_edges_intra.filter(pl.col("time_cluster") == t)

            if len(g) == 0 and len(g_intra) == 0:
                return go.Figure()

           # ---------- EDGES ----------
            x_edges, y_edges = [], []
            seen_edges = set()
            for row in g.filter(pl.col("weight") >= 1).iter_rows(named=True):
                edge_key = tuple(sorted([row["cu"], row["cv"]]))  # canonical key
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                x_edges += [row["x_u"], row["x_v"], None]
                y_edges += [row["y_u"], row["y_v"], None]

            # ---------- NODES ----------
            intra_lookup = dict(zip(g_intra["cu"].to_list(), g_intra["internal_weight"].to_list()))
            connected = set(g["cu"].to_list()) | set(g["cv"].to_list())
            top_intra = (
                g_intra.sort("internal_weight", descending=True)
                .head(10).select("cu").to_series().to_list()
            )
            nodes_to_show = connected | set(top_intra)

            nodes_u = g.select(["cu", "x_u", "y_u"]).rename({"cu": "community", "x_u": "x", "y_u": "y"})
            nodes_v = g.select(["cv", "x_v", "y_v"]).rename({"cv": "community", "x_v": "x", "y_v": "y"})
            nodes_intra = (
                g_intra.join(pl.from_pandas(pos_df), left_on="cu", right_on="community", how="left")
                .select(["cu", "x", "y"]).rename({"cu": "community"})
            )

            nodes = (
                nodes_u.vstack(nodes_v).vstack(nodes_intra)
                .unique(subset=["community"], keep="first")
                .filter(pl.col("community").is_in(list(nodes_to_show)))
                .filter(pl.col("x").is_not_null() & pl.col("y").is_not_null())
                .to_pandas()
            )

            nodes["internal_weight"] = nodes["community"].map(lambda c: intra_lookup.get(c, 1))
            nodes["size"] = nodes["community"].map(
                lambda c: g.filter(pl.col("cu") == c)["size"].to_list()[0]
                if len(g.filter(pl.col("cu") == c)) > 0 else 1
            )
            nodes["centrality"] = nodes["community"].map(
                lambda c: round(centrality_dict.get(c, 0), 3) if centrality_dict else 0
            )

            fig = go.Figure()

            # ---- edges ----
            fig.add_trace(go.Scatter(
                x=x_edges, y=y_edges,
                mode="lines",
                line=dict(width=2, color="rgba(120,170,255,0.4)"),
                hoverinfo="none"
            ))

            # ---- nodes ----
            fig.add_trace(go.Scatter(
                x=nodes["x"], y=nodes["y"],
                mode="markers+text",
                marker=dict(
                    size=nodes["internal_weight"].apply(lambda s: (s**0.08) * 15),
                    color=nodes["size"],
                    colorscale="Turbo",
                    showscale=True,
                    colorbar=dict(title="Community Size"),
                    line=dict(width=1, color="black")
                ),
                text=nodes["community"],
                textposition="top center",
                textfont=dict(size=8),
                customdata=list(zip(nodes["internal_weight"], nodes["centrality"])),
                hovertemplate=(
                    "Community %{text}<br>"
                    "Size: %{marker.color}<br>"
                    "Internal Activity: %{customdata[0]}<br>"
                    "Degree Centrality: %{customdata[1]}<extra></extra>"
                )
            ))

            fig.update_layout(
                title=f"Temporal Community Graph — Window {t}",
                height=400,
                showlegend=False,
                plot_bgcolor="white",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )

            return fig

        community_figures = {}
        for t in community_edges["time_cluster"].unique().sort().to_list():
            pipeline_data["progress"] = f"Building figures: window {t}"
            G_t = nx.Graph()
            g_inter_t = community_edges_inter.filter(pl.col("time_cluster") == t)
            for cu, cv, w in g_inter_t.select(["cu", "cv", "weight"]).iter_rows():
                G_t.add_edge(int(cu), int(cv))
            centrality_dict = nx.degree_centrality(G_t)
            community_figures[int(t)] = community_snapshot(int(t), centrality_dict=centrality_dict)
        def window_stats(t):
            n_nodes = comm_raw.filter(pl.col("time_cluster") == t).select(pl.n_unique("community")).item()
            g_inter = community_edges_inter_raw.filter(pl.col("time_cluster") == t)
            n_edges = len(g_inter)
            avg_size = round(
                comm_raw.filter(pl.col("time_cluster") == t)
                .group_by("community").len().select(pl.mean("len")).item(), 2
            )
            top = (
                community_sizes_raw.filter(pl.col("time_cluster") == t)
                .sort("size", descending=True).head(3).select(["cu", "size"]).to_pandas()
            )
            top_nodes = [(str(row["cu"]), int(row["size"])) for _, row in top.iterrows()]
            G_t = nx.Graph()
            for cu, cv, w in g_inter.select(["cu", "cv", "weight"]).iter_rows():
                G_t.add_edge(int(cu), int(cv))
            centrality = nx.degree_centrality(G_t)
            top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
            return n_nodes, n_edges, avg_size, top_nodes, centrality

        pipeline_data["progress"] = "Computing trend charts..."
        time_keys = sorted(community_figures.keys())
        trend_nodes, trend_edges, trend_avg_size = [], [], []
        for t in time_keys:
            n, e, a, _, _ = window_stats(t)
            trend_nodes.append(n)
            trend_edges.append(e)
            trend_avg_size.append(a)

        def make_trend_fig(y_vals, title, color):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(y_vals))), y=y_vals,
                                     mode="lines", line=dict(color=color, width=2)))
            fig.update_layout(
                title=dict(text=title, font=dict(size=12)),
                height=180, margin=dict(l=30, r=10, t=30, b=30),
                plot_bgcolor="white",
                xaxis=dict(showgrid=False, title="Window"),
                yaxis=dict(showgrid=True, gridcolor="#eee")
            )
            return fig

        pipeline_data.update({
            "status": "done",
            "progress": "Done!",
            "community_figures": community_figures,
            "community_edges_inter": community_edges_inter,
            "community_edges_intra": community_edges_intra,
            "community_sizes_raw": community_sizes_raw,
            "comm_raw": comm_raw,
            "community_edges_inter_raw": community_edges_inter_raw,
            "window_stats": window_stats,
            "time_keys": time_keys,
            "trend_fig_nodes":   make_trend_fig(trend_nodes,    "Communities vs Time",       "#3b82f6"),
            "trend_fig_edges":   make_trend_fig(trend_edges,    "Edges vs Time",             "#10b981"),
            "trend_fig_density": make_trend_fig(trend_avg_size, "Avg Community Size vs Time","#f59e0b"),
        })

    except Exception as ex:
        pipeline_data["status"] = "error"
        pipeline_data["error"] = str(ex)


# ── Run pipeline callback ─────────────────────────────────────────────────────
@app.callback(
    Output("progress-interval", "disabled"),
    Output("progress-display", "children"),
    Output("pipeline-done", "data"),
    Input("run-btn", "n_clicks"),
    State("file-store", "data"),
    State("max-activity", "value"),
    State("top-comms", "value"),
    State("resolution", "value"),
    State("separator", "value"),
    prevent_initial_call=True
)
def start_pipeline(n_clicks, file_data, max_activity, top_comms_n, resolution, separator):
    if not file_data:
        return True, html.Span("⚠️ Please upload a file first.", style={"color": "#ef4444"}), False
    pipeline_data.clear()
    pipeline_data["status"] = "running"
    pipeline_data["progress"] = "Starting..."
    t = threading.Thread(
        target=run_pipeline,
        args=(file_data["filepath"], file_data["filename"],  # pass filepath
              max_activity or 15000, top_comms_n or 30,
              resolution or 1.0, separator or ",")
    )
    t.daemon = True
    t.start()
    return False, "🔄 Pipeline starting...", False
#__________P
@app.callback(
    Output("progress-display", "children", allow_duplicate=True),
    Output("progress-interval", "disabled", allow_duplicate=True),
    Output("pipeline-done", "data", allow_duplicate=True),
    Input("progress-interval", "n_intervals"),
    prevent_initial_call=True
)
def poll_progress(n):
    status = pipeline_data.get("status", "idle")
    progress = pipeline_data.get("progress", "")
    if status == "done":
        return html.Span("✅ Pipeline complete! Loading dashboard...", style={"color": "#16a34a"}), True, True
    if status == "error":
        return html.Span(f"❌ Error: {pipeline_data.get('error')}", style={"color": "#ef4444"}), True, False
    return html.Span(f"🔄 {progress}"), False, False


@app.callback(
    Output("page", "data"),
    Output("filename-store", "data"),
    Input("pipeline-done", "data"),
    State("file-store", "data"),
    prevent_initial_call=True
)
def go_to_dashboard(done, file_data):
    if done and file_data:
        fname = os.path.splitext(file_data["filename"])[0].replace("-", " ").replace("_", " ").title()
        return "dashboard", fname
    return dash.no_update, dash.no_update


# ── Back button ───────────────────────────────────────────────────────────────
@app.callback(
    Output("page", "data", allow_duplicate=True),
    Input("back-btn", "n_clicks"),
    prevent_initial_call=True
)
def go_back(n):
    if n:
        pipeline_data.clear()
        return "upload"
    return dash.no_update


# ── Dashboard callbacks ───────────────────────────────────────────────────────
@app.callback(
    Output("slider-container", "children"),
    Input("page", "data"),
)
def build_slider(page):
    if page != "dashboard" or not pipeline_data.get("time_keys"):
        return []
    time_keys = pipeline_data["time_keys"]
    return dcc.Slider(
        id="time-slider",
        min=min(time_keys),
        max=max(time_keys),
        step=1,
        value=min(time_keys),
        marks={k: str(k) for k in time_keys[::max(1, len(time_keys)//6)]},
        tooltip={"placement": "bottom"},
    )
@app.callback(
    Output("is-playing", "data"),
    Output("play-pause-btn", "children"),
    Output("interval", "disabled"),
    Input("play-pause-btn", "n_clicks"),
    State("is-playing", "data"),
    prevent_initial_call=True
)
def toggle_play(n_clicks, is_playing):
    now_playing = not is_playing
    return now_playing, "⏸ Pause" if now_playing else "▶ Play", not now_playing


@app.callback(
    Output("time-slider", "value"),
    Input("interval", "n_intervals"),
    State("time-slider", "value"),
    prevent_initial_call=True
)
def advance_frame(n, current_t):
    keys = pipeline_data.get("time_keys", [])
    if not keys:
        return dash.no_update
    idx = keys.index(current_t)
    return keys[(idx + 1) % len(keys)]


@app.callback(
    Output("community-graph", "figure"),
    Output("metrics-panel", "children"),
    Input("time-slider", "value"),
    prevent_initial_call=True
)
def update_graph(t):
    d = pipeline_data
    if not d.get("community_figures"):
        return go.Figure(), ""

    community_figures = d["community_figures"]
    community_edges_inter = d["community_edges_inter"]
    window_stats = d["window_stats"]

    G_t = nx.Graph()
    g_inter = community_edges_inter.filter(pl.col("time_cluster") == t)
    for cu, cv, w in g_inter.select(["cu", "cv", "weight"]).iter_rows():
        G_t.add_edge(int(cu), int(cv))
    centrality = nx.degree_centrality(G_t)
    top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]

    # rebuild figure with centrality
    from dash import html as dhtml
    fig = community_figures.get(t, go.Figure())

    n_nodes, n_edges, avg_size, top_nodes, _ = window_stats(t)

    central_rows = [
        html.Div([
            html.Span(f"Community {cu}", style={"fontWeight": "700", "marginRight": "8px"}),
            html.Span(f"{round(score, 3)}")
        ], style={"marginBottom": "4px", "fontFamily": "sans-serif", "fontSize": "13px"})
        for cu, score in top_central
    ]

    metrics = html.Div([
        html.Div([
            html.P("Number of communities", style={"margin": "0", "color": "#666", "fontSize": "13px", "fontFamily": "sans-serif"}),
            html.H3(str(n_nodes), style={"margin": "4px 0 12px", "fontFamily": "sans-serif"}),
        ]),
        html.Div([
            html.P("Inter-community edges", style={"margin": "0", "color": "#666", "fontSize": "13px", "fontFamily": "sans-serif"}),
            html.H3(str(n_edges), style={"margin": "4px 0 12px", "fontFamily": "sans-serif"}),
        ]),
        html.Div([
            html.P("Avg community size", style={"margin": "0", "color": "#666", "fontSize": "13px", "fontFamily": "sans-serif"}),
            html.H3(str(avg_size), style={"margin": "4px 0 12px", "fontFamily": "sans-serif"}),
        ]),
        html.Div([
            html.P("Top Central Communities", style={"margin": "0", "color": "#666", "fontSize": "13px", "fontFamily": "sans-serif"}),
            html.Div(central_rows, style={"marginTop": "4px"})
        ]),
    ])

    return fig, metrics


if __name__ == "__main__":
    app.run(debug=True, port=8053)
