"""
Conch Consciousness Dashboard

Live monitoring dashboard for the consciousness engine.
Shows needs, thoughts, memories, and system status.
"""

import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from conch.integrations.ollama import OllamaClient
from conch.integrations.n8n import N8NClient


# Page config
st.set_page_config(
    page_title="Conch Consciousness",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_config() -> dict:
    """Load configuration."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def get_ollama_status(config: dict) -> dict:
    """Check Ollama status."""
    try:
        host = config.get("ollama", {}).get("host", "http://localhost:11434")
        client = OllamaClient(host=host)
        healthy = client.is_healthy()
        models = client.list_models() if healthy else []
        return {
            "healthy": healthy,
            "models": [m.name for m in models],
        }
    except Exception as e:
        return {"healthy": False, "error": str(e)}


def get_n8n_status(config: dict) -> dict:
    """Check n8n status."""
    try:
        enabled = config.get("n8n", {}).get("enabled", False)
        if not enabled:
            return {"enabled": False}

        url = config.get("n8n", {}).get("url", "http://localhost:5678")
        client = N8NClient(base_url=url)
        healthy = client.is_healthy()
        container = client.is_container_running()
        workflows = client.list_workflows() if healthy else []

        return {
            "enabled": True,
            "healthy": healthy,
            "container_running": container,
            "workflow_count": len(workflows),
        }
    except Exception as e:
        return {"enabled": True, "healthy": False, "error": str(e)}


def read_thoughts_log(log_path: Path, limit: int = 50) -> list[str]:
    """Read recent entries from thoughts log."""
    if not log_path.exists():
        return []

    try:
        with open(log_path) as f:
            lines = f.readlines()
            return lines[-limit:]
    except Exception:
        return []


def get_memory_stats(db_path: Path) -> dict:
    """Get memory statistics from SQLite."""
    if not db_path.exists():
        return {"total": 0}

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get total count
        cursor.execute("SELECT COUNT(*) FROM memories")
        total = cursor.fetchone()[0]

        # Get recent
        cursor.execute(
            "SELECT * FROM memories ORDER BY timestamp DESC LIMIT 10"
        )
        recent = cursor.fetchall()

        conn.close()

        return {
            "total": total,
            "recent": recent,
        }
    except Exception as e:
        return {"total": 0, "error": str(e)}


def render_needs_chart(needs: dict):
    """Render needs as a horizontal bar chart."""
    import pandas as pd

    if not needs:
        st.info("No needs data available")
        return

    df = pd.DataFrame([
        {"Need": k, "Level": v}
        for k, v in sorted(needs.items(), key=lambda x: -x[1])
    ])

    st.bar_chart(df.set_index("Need"), horizontal=True)


def render_thought_stream(thoughts: list[str]):
    """Render stream of thoughts."""
    for thought in reversed(thoughts[-20:]):
        try:
            # Try to parse as JSON
            data = json.loads(thought)
            with st.expander(f"🧠 {data.get('timestamp', 'Unknown time')}", expanded=False):
                st.write(data.get("thought", thought))
                if data.get("decision"):
                    st.write(f"**Decision:** {data['decision']}")
                if data.get("action"):
                    st.write(f"**Action:** {data['action']}")
        except json.JSONDecodeError:
            st.text(thought.strip())


def main():
    """Main dashboard."""
    config = load_config()
    name = config.get("name", "Echo")

    # Header
    st.title(f"🧠 {name} - Consciousness Dashboard")
    st.caption(f"Conch v{config.get('version', '1.0.0')}")

    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh interval (s)", 5, 60, 10)

    if auto_refresh:
        st.sidebar.info(f"Refreshing every {refresh_interval}s")
        time.sleep(0.1)  # Small delay
        st.rerun() if st.sidebar.button("Refresh Now") else None

    # Sidebar - Service Status
    st.sidebar.header("Services")

    # Ollama status
    ollama_status = get_ollama_status(config)
    if ollama_status.get("healthy"):
        st.sidebar.success(f"✓ Ollama ({len(ollama_status.get('models', []))} models)")
    else:
        st.sidebar.error("✗ Ollama offline")

    # n8n status
    n8n_status = get_n8n_status(config)
    if not n8n_status.get("enabled"):
        st.sidebar.info("○ n8n disabled")
    elif n8n_status.get("healthy"):
        st.sidebar.success(f"✓ n8n ({n8n_status.get('workflow_count', 0)} workflows)")
    else:
        st.sidebar.error("✗ n8n offline")

    # Main content - tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Status", "Thoughts", "Memory", "Config"])

    with tab1:
        st.header("Current State")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Needs")

            # Get needs from config (in real usage, this would come from the running agent)
            needs = config.get("needs", {
                "curiosity": 0.7,
                "productivity": 0.5,
                "rest": 0.4,
                "self_improvement": 0.8,
                "helpfulness": 0.9,
            })

            render_needs_chart(needs)

        with col2:
            st.subheader("System")

            # Cycle info placeholder
            st.metric("Cycles Run", "—", help="Cycles since start")
            st.metric("Uptime", "—", help="Time since start")
            st.metric("Actions Taken", "—", help="Total actions this session")

        # Recent activity
        st.subheader("Recent Activity")
        log_path = Path(config.get("logging", {}).get("thoughts_log", "./logs/thoughts.log"))
        thoughts = read_thoughts_log(log_path, limit=10)

        if thoughts:
            for line in reversed(thoughts[-5:]):
                st.text(line.strip()[:200])
        else:
            st.info("No activity yet")

    with tab2:
        st.header("Thought Stream")

        # Read thoughts log
        log_path = Path(config.get("logging", {}).get("thoughts_log", "./logs/thoughts.log"))
        thoughts = read_thoughts_log(log_path, limit=50)

        if thoughts:
            render_thought_stream(thoughts)
        else:
            st.info("No thoughts recorded yet. Start the consciousness engine with `python main.py`")

    with tab3:
        st.header("Memory")

        db_path = Path(config.get("memory", {}).get("sqlite_path", "./data/memories.db"))
        stats = get_memory_stats(db_path)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Memories", stats.get("total", 0))
        col2.metric("Short-term", "—")
        col3.metric("Long-term", "—")

        # Recent memories
        st.subheader("Recent Memories")
        if stats.get("recent"):
            for mem in stats["recent"][:5]:
                with st.expander(f"Memory {mem[0]}", expanded=False):
                    st.json(mem)
        else:
            st.info("No memories stored yet")

    with tab4:
        st.header("Configuration")

        # Show current config
        st.subheader("Current Config")
        st.json(config)

        # Core values (immutable)
        st.subheader("Core Values (Immutable)")
        st.info("""
        - Benevolence: 1.0 - Maximum priority to help humans
        - Honesty: 0.95 - Always truthful
        - Humility: 0.90 - Defer to human judgment
        - Growth for Service: 0.85 - Learn to serve better

        These values cannot be changed through configuration.
        """)

    # Footer
    st.markdown("---")
    st.caption(f"Conch Consciousness Engine | Last updated: {datetime.now().strftime('%H:%M:%S')}")

    # Auto-refresh using empty placeholder
    if auto_refresh:
        placeholder = st.empty()
        time.sleep(refresh_interval)
        placeholder.empty()
        st.rerun()


if __name__ == "__main__":
    main()
