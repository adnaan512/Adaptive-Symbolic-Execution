"""
Modular visualization components for the Streamlit dashboard (Phase 8).
"""

import json
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_coverage_over_time(raw_results: List[Dict[str, Any]]):
    """
    Renders a line chart showing Branch Coverage over Time.
    Compares the average coverage curve of each heuristic across all programs.
    """
    st.subheader("📈 Branch Coverage Over Time")
    
    if not raw_results:
        st.info("No raw data available for coverage chart.")
        return
        
    # Flatten snapshots into a dataframe
    records = []
    for run in raw_results:
        prog = run.get("program_name")
        heuristic = run.get("heuristic")
        seed = run.get("seed")
        
        for snapshot in run.get("coverage_over_time", []):
            records.append({
                "program": prog,
                "heuristic": heuristic,
                "seed": seed,
                "time_s": snapshot.get("elapsed_seconds"),
                "branch_coverage": snapshot.get("branch_coverage")
            })
            
    df = pd.DataFrame(records)
    if df.empty:
        st.info("No coverage snapshots found.")
        return
        
    # Average across seeds and programs for a global view (or could add a selectbox for programs)
    # We round time_s to nearest integer to group them roughly
    df['time_bin'] = df['time_s'].round()
    agg_df = df.groupby(['heuristic', 'time_bin'])['branch_coverage'].mean().reset_index()
    
    fig = px.line(
        agg_df, 
        x="time_bin", 
        y="branch_coverage", 
        color="heuristic",
        title="Average Branch Coverage vs Time",
        labels={"time_bin": "Elapsed Time (s)", "branch_coverage": "Branch Coverage"},
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)


def render_heuristic_comparison(metrics_df: pd.DataFrame):
    """
    Renders a grouped bar chart comparing AI vs Baselines for key metrics.
    """
    st.subheader("📊 Heuristic Comparison")
    
    if metrics_df.empty:
        st.info("No metrics data available.")
        return
        
    # Let user select the metric
    metric_options = {
        "Branch Coverage": "branch_coverage_mean",
        "Unique Bugs Found": "unique_bugs_mean",
        "Execution Time (s)": "execution_time_seconds_mean",
        "Unique Paths": "unique_paths_mean"
    }
    
    selected_metric_label = st.selectbox("Select Metric to Compare:", list(metric_options.keys()))
    metric_col = metric_options[selected_metric_label]
    
    # We expect metrics_df to have program_name, heuristic, and the metric_col
    if metric_col not in metrics_df.columns:
        st.warning(f"Column {metric_col} not found in metrics data.")
        return
        
    fig = px.bar(
        metrics_df,
        x="program_name",
        y=metric_col,
        color="heuristic",
        barmode="group",
        title=f"{selected_metric_label} by Program and Heuristic",
        labels={"program_name": "Program", metric_col: selected_metric_label}
    )
    st.plotly_chart(fig, use_container_width=True)


def render_live_execution_tree_mock():
    """
    Renders a mocked active execution tree using Plotly.
    Represents the live state space being explored.
    """
    st.subheader("🌳 Live Execution Tree (Mock)")
    
    # A simple tree visualization using Plotly Graph Objects
    # Node positions
    X = [0, -1, 1, -1.5, -0.5, 0.5, 1.5]
    Y = [0, -1, -1, -2, -2, -2, -2]
    
    # Edges
    edges_x = [0, -1, None, 0, 1, None, -1, -1.5, None, -1, -0.5, None, 1, 0.5, None, 1, 1.5, None]
    edges_y = [0, -1, None, 0, -1, None, -1, -2, None, -1, -2, None, -1, -2, None, -1, -2, None]
    
    # Colors (Simulating coverage/exploration)
    node_colors = ['green', 'green', 'orange', 'green', 'red', 'lightgrey', 'lightgrey']
    node_texts = [
        "Root (Covered)", 
        "Branch A (Covered)", "Branch B (Active)", 
        "Path 1 (Covered)", "Path 2 (Bug!)", "Path 3 (Pending)", "Path 4 (Pending)"
    ]
    
    fig = go.Figure()
    
    # Add Edges
    fig.add_trace(go.Scatter(
        x=edges_x, y=edges_y,
        mode='lines',
        line=dict(color='gray', width=2),
        hoverinfo='none'
    ))
    
    # Add Nodes
    fig.add_trace(go.Scatter(
        x=X, y=Y,
        mode='markers+text',
        marker=dict(
            symbol='circle',
            size=30,
            color=node_colors,
            line=dict(color='black', width=1)
        ),
        text=node_texts,
        textposition="bottom center",
        hoverinfo='text'
    ))
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Symbolic Execution Path Exploration",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
