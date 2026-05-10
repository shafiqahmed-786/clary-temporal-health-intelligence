"""
app/ui/graph.py — Event Graph Visualisation

Renders the causal event graph as an interactive network diagram.
Uses pyvis to generate an HTML network, embedded via st.components.v1.html.

Node types rendered:
  User (🧑)  Session (📋)  Symptom (🤒)  Trigger (⚡)  Pattern (🔍)  Mechanism (🔬)

Edge types:
  HAS_SESSION · REPORTED_SYMPTOM · CONTAINS_TRIGGER
  PRECEDES · CAUSED_BY · EVIDENCE_FOR · EXPLAINED_BY · DOWNSTREAM_OF

If Neo4j is unavailable, falls back to a synthetic graph built from
st.session_state patterns + timeline events (always renderable).
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

from app.state import ClaryState, run_async

# Try pyvis import
try:
    from pyvis.network import Network
    _PYVIS_AVAILABLE = True
except ImportError:
    _PYVIS_AVAILABLE = False

# Try networkx import
try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False


# ── Node/Edge styling ──────────────────────────────────────────────────────────

_NODE_STYLES = {
    "User":      {"color": "#3b82f6", "size": 28, "icon": "🧑", "shape": "dot"},
    "Session":   {"color": "#6b7280", "size": 16, "icon": "📋", "shape": "square"},
    "Symptom":   {"color": "#ef4444", "size": 20, "icon": "🤒", "shape": "dot"},
    "Trigger":   {"color": "#f59e0b", "size": 20, "icon": "⚡", "shape": "diamond"},
    "Pattern":   {"color": "#8b5cf6", "size": 24, "icon": "🔍", "shape": "star"},
    "Mechanism": {"color": "#10b981", "size": 18, "icon": "🔬", "shape": "dot"},
}

_EDGE_STYLES = {
    "HAS_SESSION":       {"color": "#374151", "width": 1, "dashes": False},
    "REPORTED_SYMPTOM":  {"color": "#ef4444", "width": 2, "dashes": False},
    "CONTAINS_TRIGGER":  {"color": "#f59e0b", "width": 2, "dashes": False},
    "PRECEDES":          {"color": "#3b82f6", "width": 1, "dashes": True},
    "CAUSED_BY":         {"color": "#8b5cf6", "width": 2.5, "dashes": False},
    "EVIDENCE_FOR":      {"color": "#6b7280", "width": 1, "dashes": True},
    "EXPLAINED_BY":      {"color": "#10b981", "width": 2, "dashes": False},
    "DOWNSTREAM_OF":     {"color": "#f97316", "width": 2, "dashes": False},
    "SIMILAR_TO":        {"color": "#1f2937", "width": 1, "dashes": True},
}


# ── Synthetic graph (fallback from session state) ──────────────────────────────

def _build_synthetic_graph(user_id: str, patterns: list[dict], events: list[dict]) -> dict:
    """
    Build a graph dict from local state when Neo4j is unavailable.
    Returns {"nodes": [...], "edges": [...]} in the same format as EventGraph.
    """
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_nodes: set[str] = set()

    def add_node(nid: str, label: str, node_type: str, title: str = "") -> None:
        if nid not in seen_nodes:
            nodes.append({"id": nid, "label": label, "type": node_type, "title": title})
            seen_nodes.add(nid)

    # User node
    add_node(user_id, user_id, "User", f"User: {user_id}")

    # Session nodes from events
    session_map: dict[str, dict] = {}
    for raw in events:
        meta = raw.get("metadata", {})
        sid = meta.get("session_id", "")
        if not sid or sid in session_map:
            continue
        epoch = meta.get("timestamp_epoch", 0)
        try:
            date_str = datetime.fromtimestamp(float(epoch)).strftime("%b %d")
        except Exception:
            date_str = "?"
        session_map[sid] = meta
        add_node(sid, f"Session\n{date_str}", "Session", f"Session: {sid}\n{date_str}")
        edges.append({"from": user_id, "to": sid, "type": "HAS_SESSION", "label": ""})

        # Symptom nodes
        for sym in (meta.get("symptoms", "") or "").split(","):
            sym = sym.strip()
            if not sym:
                continue
            add_node(sym, sym, "Symptom", f"Symptom: {sym}")
            edges.append({"from": sid, "to": sym, "type": "REPORTED_SYMPTOM", "label": ""})

        # Trigger nodes
        for trig in (meta.get("triggers", "") or "").split(","):
            trig = trig.strip()
            if not trig:
                continue
            add_node(trig, trig, "Trigger", f"Trigger: {trig}")
            edges.append({"from": sid, "to": trig, "type": "CONTAINS_TRIGGER", "label": ""})

    # Pattern nodes
    for pat in patterns:
        pid = str(pat.get("pattern_id", ""))
        if not pid:
            continue
        sym = pat.get("symptom", "")
        trig = pat.get("trigger", "")
        conf = pat.get("confidence", "low")
        status = pat.get("status", "watching")
        label = f"Pattern\n{trig[:10]}→{sym[:10]}"
        add_node(pid, label, "Pattern", f"Pattern [{status}]\n{trig} → {sym}\nConf: {conf}")

        # CAUSED_BY edge
        if sym and trig:
            add_node(sym, sym, "Symptom")
            add_node(trig, trig, "Trigger")
            edges.append({"from": sym, "to": trig, "type": "CAUSED_BY",
                          "label": f"{pat.get('lag_days_min',0)}–{pat.get('lag_days_max',0)}d"})

        # EVIDENCE_FOR edges
        for ev in pat.get("evidence", [])[:4]:
            sid = ev.get("session_id", "")
            if sid in seen_nodes:
                edges.append({"from": sid, "to": pid, "type": "EVIDENCE_FOR", "label": ""})

        # Mechanism node
        mech = pat.get("lag_registry_match") or {}
        mname = mech.get("mechanism_name", "")
        if mname:
            add_node(mname, mname[:20], "Mechanism", mech.get("description", "")[:120])
            edges.append({"from": pid, "to": mname, "type": "EXPLAINED_BY", "label": ""})

    return {"nodes": nodes, "edges": edges}


# ── pyvis graph builder ────────────────────────────────────────────────────────

def _build_pyvis_network(graph_data: dict, visible_types: set[str]) -> str:
    """Build pyvis HTML string from graph dict."""
    if not _PYVIS_AVAILABLE:
        return "<p style='color:#ef4444;'>pyvis not installed. Run: pip install pyvis</p>"

    net = Network(
        height="580px",
        width="100%",
        bgcolor="#111827",
        font_color="#94a3b8",
        notebook=False,
    )
    net.set_options(json.dumps({
        "nodes": {
            "font": {"size": 12, "face": "Inter"},
            "borderWidth": 1,
            "shadow": {"enabled": True, "color": "rgba(0,0,0,0.5)", "size": 8},
        },
        "edges": {
            "font": {"size": 9, "face": "JetBrains Mono", "color": "#4b5563"},
            "smooth": {"type": "continuous"},
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
        },
        "physics": {
            "enabled": True,
            "stabilization": {"iterations": 100},
            "barnesHut": {"gravitationalConstant": -8000, "centralGravity": 0.3},
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 150,
            "zoomView": True,
        },
        "layout": {"randomSeed": 42},
    }))

    # Add nodes
    for node in graph_data.get("nodes", []):
        node_type = node.get("type", "Session")
        if node_type not in visible_types:
            continue
        style = _NODE_STYLES.get(node_type, _NODE_STYLES["Session"])
        label = node.get("label", node.get("id", "?"))[:30]
        title = node.get("title", label)
        net.add_node(
            node["id"],
            label=label,
            title=title,
            color={"background": style["color"], "border": "#0a0c12", "highlight": {"background": "#60a5fa"}},
            size=style["size"],
            shape=style["shape"],
            font={"color": "#e5e7eb"},
        )

    # Add edges
    node_ids = {n["id"] for n in graph_data.get("nodes", []) if n.get("type") in visible_types}
    for edge in graph_data.get("edges", []):
        src = edge.get("from", "")
        dst = edge.get("to", "")
        if src not in node_ids or dst not in node_ids:
            continue
        etype = edge.get("type", "")
        style = _EDGE_STYLES.get(etype, {"color": "#374151", "width": 1, "dashes": False})
        net.add_edge(
            src, dst,
            title=etype,
            label=edge.get("label", ""),
            color=style["color"],
            width=style["width"],
            dashes=style["dashes"],
        )

    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w")
    tmp.close()
    net.save_graph(tmp.name)
    with open(tmp.name, "r") as f:
        html = f.read()
    os.unlink(tmp.name)

    # Inject dark background
    html = html.replace(
        "<body>",
        '<body style="background:#111827;margin:0;padding:0;">',
    )
    return html


# ── Legend ─────────────────────────────────────────────────────────────────────

def _render_legend() -> None:
    st.markdown('<div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Node Legend</div>', unsafe_allow_html=True)
    for node_type, style in _NODE_STYLES.items():
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
            f'<div style="width:10px;height:10px;border-radius:50%;background:{style["color"]};flex-shrink:0;"></div>'
            f'<span style="font-size:12px;color:#94a3b8;">{style["icon"]} {node_type}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin:12px 0 8px;">Edge Legend</div>', unsafe_allow_html=True)
    important_edges = ["CAUSED_BY", "PRECEDES", "EXPLAINED_BY", "DOWNSTREAM_OF"]
    for etype in important_edges:
        style = _EDGE_STYLES[etype]
        dash = "- -" if style["dashes"] else "—"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
            f'<span style="color:{style["color"]};font-weight:700;font-family:monospace;">{dash}</span>'
            f'<span style="font-size:11px;color:#6b7280;">{etype}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Graph stats ────────────────────────────────────────────────────────────────

def _render_graph_stats(graph_data: dict) -> None:
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    by_type: dict[str, int] = {}
    for n in nodes:
        by_type[n.get("type", "Unknown")] = by_type.get(n.get("type", "Unknown"), 0) + 1

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#3b82f6;">{len(nodes)}</div><div class="metric-label">Nodes</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#8b5cf6;">{len(edges)}</div><div class="metric-label">Edges</div></div>', unsafe_allow_html=True)
    with c3:
        causal_edges = sum(1 for e in edges if e.get("type") == "CAUSED_BY")
        st.markdown(f'<div class="metric-box"><div class="metric-val" style="color:#ef4444;">{causal_edges}</div><div class="metric-label">Causal Links</div></div>', unsafe_allow_html=True)


# ── Main render ────────────────────────────────────────────────────────────────

def render() -> None:
    ClaryState.init()
    user_id = ClaryState.get_user_id()

    st.markdown('<h2 style="color:#f1f5f9;font-weight:600;margin:0;">🕸️ Event Graph</h2>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:12px;color:#4b5563;margin-top:2px;margin-bottom:16px;font-family:monospace;">{user_id} · Causal Network</div>', unsafe_allow_html=True)

    col_main, col_sidebar = st.columns([4, 1])

    with col_sidebar:
        st.markdown("#### Filters")
        all_types = list(_NODE_STYLES.keys())
        visible_types = set(st.multiselect(
            "Node types",
            options=all_types,
            default=["Symptom", "Trigger", "Pattern", "Mechanism"],
            key="graph_node_filter",
        ))

        st.markdown("---")
        _render_legend()

        st.markdown("---")
        if st.button("🔄 Reload Graph", use_container_width=True):
            ClaryState.set_graph_data(None)
            st.rerun()

    with col_main:
        # Load graph data
        graph_data = ClaryState.get_graph_data()

        if graph_data is None:
            with st.spinner("Building event graph…"):
                # Try Neo4j first
                try:
                    from graph.event_graph import EventGraph
                    eg = EventGraph()
                    graph_data = run_async(eg.get_subgraph_for_user(user_id))
                    if not graph_data.get("nodes"):
                        raise ValueError("Empty graph from Neo4j")
                except Exception as neo4j_exc:
                    st.caption(f"ℹ️ Using local graph (Neo4j: {str(neo4j_exc)[:60]}…)")
                    patterns = ClaryState.get_patterns()
                    events = ClaryState.get_timeline_events()
                    graph_data = _build_synthetic_graph(user_id, patterns, events)

                ClaryState.set_graph_data(graph_data)

        # Stats row
        _render_graph_stats(graph_data)
        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

        # Render graph
        if not graph_data.get("nodes"):
            st.markdown("""
            <div style="text-align:center;padding:80px 20px;color:#4b5563;">
                <div style="font-size:32px;margin-bottom:12px;">🕸️</div>
                <div style="font-size:14px;">No graph data yet.</div>
                <div style="font-size:12px;margin-top:6px;">Start chatting to build the event graph.</div>
            </div>
            """, unsafe_allow_html=True)
        elif not _PYVIS_AVAILABLE:
            st.warning("Install pyvis to view the interactive graph: `pip install pyvis`")

            # Fallback: text adjacency list
            st.markdown("#### Text Adjacency (fallback)")
            edges = graph_data.get("edges", [])
            for e in edges[:30]:
                etype = e.get("type", "")
                style = _EDGE_STYLES.get(etype, {})
                color = style.get("color", "#6b7280")
                st.markdown(
                    f'<div style="font-family:monospace;font-size:11px;padding:3px 0;color:{color};">'
                    f'{e.get("from","?")[:20]} ──[{etype}]──▶ {e.get("to","?")[:20]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            graph_html = _build_pyvis_network(graph_data, visible_types or set(all_types))
            components.html(graph_html, height=600, scrolling=False)

        # Node table (expandable)
        nodes = graph_data.get("nodes", [])
        if nodes:
            with st.expander(f"📋 Node List ({len(nodes)} nodes)", expanded=False):
                node_rows = [
                    {"Type": n.get("type", "?"), "ID": n.get("id", "?")[:40], "Label": n.get("label", "")[:30]}
                    for n in nodes
                ]
                import pandas as pd
                st.dataframe(pd.DataFrame(node_rows), use_container_width=True, hide_index=True)


__all__ = ["render"]