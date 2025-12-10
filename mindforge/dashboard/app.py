"""
MindForge Dashboard - Streamlit Application

Interactive dashboard for monitoring and interacting with the MindForge
consciousness engine.

Features:
- Real-time system status
- Chat interface
- Memory browser
- KVRM grounding viewer
- Needs visualization
- Thought stream

Run with: streamlit run mindforge/dashboard/app.py
Or: python -m mindforge.dashboard
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="MindForge Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "mind" not in st.session_state:
        st.session_state.mind = None

    if "memory_store" not in st.session_state:
        st.session_state.memory_store = None

    if "kvrm_resolver" not in st.session_state:
        st.session_state.kvrm_resolver = None

    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    if "start_time" not in st.session_state:
        st.session_state.start_time = datetime.now()


def initialize_mindforge():
    """Initialize MindForge components."""
    if st.session_state.initialized:
        return True

    try:
        from mindforge.core.mind import Mind
        from mindforge.memory import MemoryStore
        from mindforge.kvrm.resolver import KeyResolver, FactKeyStore
        from mindforge.kvrm.grounding import GroundingRouter

        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        # Initialize components
        st.session_state.memory_store = MemoryStore(db_path=data_dir / "memories.db")
        st.session_state.kvrm_resolver = KeyResolver()

        fact_store = FactKeyStore(db_path=data_dir / "facts.db")
        st.session_state.kvrm_resolver.register_store("facts", fact_store)

        st.session_state.grounding_router = GroundingRouter(
            key_resolver=st.session_state.kvrm_resolver
        )

        st.session_state.mind = Mind()
        st.session_state.initialized = True

        return True

    except Exception as e:
        st.error(f"Failed to initialize MindForge: {e}")
        return False


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar():
    """Render the sidebar with navigation and status."""
    with st.sidebar:
        st.title("🧠 MindForge")
        st.caption("Consciousness Engine Dashboard")

        st.divider()

        # Navigation
        page = st.radio(
            "Navigation",
            ["Overview", "Chat", "Memory", "KVRM", "Needs", "Thoughts"],
            label_visibility="collapsed",
        )

        st.divider()

        # Quick Status
        if st.session_state.initialized and st.session_state.mind:
            st.subheader("Quick Status")

            mind = st.session_state.mind

            # Mind state
            state_colors = {
                "idle": "🟢",
                "thinking": "🟡",
                "responding": "🔵",
                "reflecting": "🟣",
                "learning": "🟠",
                "resting": "⚪",
            }
            state = mind.state.value
            st.write(f"State: {state_colors.get(state, '⚪')} {state.title()}")

            # Uptime
            uptime = datetime.now() - st.session_state.start_time
            hours, remainder = divmod(int(uptime.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            st.write(f"Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}")

            # Interactions
            st.write(f"Interactions: {mind.stats.get('total_interactions', 0)}")

        st.divider()

        # Core Values (always visible)
        with st.expander("Core Values", expanded=False):
            st.markdown("""
            - **Benevolence**: Help humans
            - **Honesty**: Always truthful
            - **Humility**: Know limitations
            - **Growth**: Improve to serve
            """)

    return page


# =============================================================================
# Overview Page
# =============================================================================

def render_overview():
    """Render the overview page."""
    st.title("System Overview")

    if not st.session_state.initialized:
        st.warning("MindForge not initialized. Click 'Initialize' in the sidebar.")
        if st.button("Initialize MindForge"):
            with st.spinner("Initializing..."):
                if initialize_mindforge():
                    st.success("MindForge initialized successfully!")
                    st.rerun()
        return

    # Status metrics
    col1, col2, col3, col4 = st.columns(4)

    mind = st.session_state.mind
    memory_store = st.session_state.memory_store

    with col1:
        st.metric(
            "Mind State",
            mind.state.value.title(),
            delta=None,
        )

    with col2:
        st.metric(
            "Total Interactions",
            mind.stats.get("total_interactions", 0),
        )

    with col3:
        stats = memory_store.get_statistics()
        st.metric("Memories", stats["total_memories"])

    with col4:
        uptime = datetime.now() - st.session_state.start_time
        st.metric("Uptime (min)", int(uptime.total_seconds() / 60))

    st.divider()

    # Needs visualization
    st.subheader("Current Needs State")

    needs_state = mind.needs.get_state()

    cols = st.columns(len(needs_state))
    for i, (need_name, need_data) in enumerate(needs_state.items()):
        with cols[i]:
            level = need_data["level"]
            st.progress(level, text=f"{need_name.title()}: {level:.1%}")

    st.divider()

    # Memory stats
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Memory Distribution")
        if stats["by_type"]:
            import pandas as pd
            df = pd.DataFrame([
                {"Type": k, "Count": v}
                for k, v in stats["by_type"].items()
            ])
            st.bar_chart(df.set_index("Type"))
        else:
            st.info("No memories stored yet.")

    with col2:
        st.subheader("Recent Activity")
        recent = memory_store.get_recent(count=5)
        if recent:
            for mem in recent:
                with st.expander(f"{mem.memory_type.value}: {mem.content[:50]}..."):
                    st.write(f"**Type:** {mem.memory_type.value}")
                    st.write(f"**Importance:** {mem.importance:.2f}")
                    st.write(f"**Content:** {mem.content}")
        else:
            st.info("No recent memories.")


# =============================================================================
# Chat Page
# =============================================================================

def render_chat():
    """Render the chat interface."""
    st.title("Chat with MindForge")

    if not st.session_state.initialized:
        st.warning("Please initialize MindForge first (go to Overview).")
        return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Say something..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Simple response (in real impl, would use Mind.process)
                response = f"I received your message: '{prompt}'. Let me think about how I can help you best."

                # Store interaction as memory
                from mindforge.memory import Memory, MemoryType
                memory = Memory(
                    content=f"User: {prompt}\nAssistant: {response}",
                    memory_type=MemoryType.INTERACTION,
                    importance=0.6,
                )
                st.session_state.memory_store.store(memory)

                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# =============================================================================
# Memory Page
# =============================================================================

def render_memory():
    """Render the memory browser."""
    st.title("Memory Browser")

    if not st.session_state.initialized:
        st.warning("Please initialize MindForge first.")
        return

    memory_store = st.session_state.memory_store

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Browse", "Search", "Add New"])

    with tab1:
        st.subheader("Recent Memories")

        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            count = st.slider("Number of memories", 5, 50, 10)
        with col2:
            from mindforge.memory import MemoryType
            type_filter = st.selectbox(
                "Filter by type",
                ["All"] + [t.value for t in MemoryType],
            )

        # Get memories
        memory_type = MemoryType(type_filter) if type_filter != "All" else None
        memories = memory_store.get_recent(count=count, memory_type=memory_type)

        if memories:
            for mem in memories:
                with st.expander(f"[{mem.memory_type.value}] {mem.content[:60]}...", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Content:** {mem.content}")
                        st.write(f"**Type:** {mem.memory_type.value}")
                        st.write(f"**Importance:** {mem.importance:.2f}")
                        if mem.tags:
                            st.write(f"**Tags:** {', '.join(mem.tags)}")
                    with col2:
                        st.write(f"**ID:** {mem.id}")
                        if st.button("Delete", key=f"del_{mem.id}"):
                            memory_store.delete(mem.id)
                            st.success("Memory deleted!")
                            st.rerun()
        else:
            st.info("No memories found.")

    with tab2:
        st.subheader("Search Memories")

        query = st.text_input("Search query")
        if query:
            results = memory_store.search(query, limit=20)
            st.write(f"Found {len(results)} results:")

            for mem in results:
                with st.expander(f"[{mem.memory_type.value}] {mem.content[:60]}..."):
                    st.write(mem.content)
                    st.caption(f"Importance: {mem.importance:.2f}")

    with tab3:
        st.subheader("Add New Memory")

        from mindforge.memory import Memory, MemoryType

        content = st.text_area("Memory content")
        col1, col2 = st.columns(2)
        with col1:
            memory_type = st.selectbox(
                "Type",
                [t.value for t in MemoryType],
            )
        with col2:
            importance = st.slider("Importance", 0.0, 1.0, 0.5)

        tags = st.text_input("Tags (comma-separated)")

        if st.button("Store Memory"):
            if content:
                memory = Memory(
                    content=content,
                    memory_type=MemoryType(memory_type),
                    importance=importance,
                    tags=[t.strip() for t in tags.split(",") if t.strip()],
                )
                memory_id = memory_store.store(memory)
                st.success(f"Memory stored with ID: {memory_id}")
            else:
                st.error("Please enter content.")


# =============================================================================
# KVRM Page
# =============================================================================

def render_kvrm():
    """Render the KVRM grounding interface."""
    st.title("KVRM Grounding")

    if not st.session_state.initialized:
        st.warning("Please initialize MindForge first.")
        return

    tab1, tab2, tab3 = st.tabs(["Ground Claim", "Search", "Add Fact"])

    with tab1:
        st.subheader("Ground a Claim")
        st.caption("Verify claims through the Key-Value Response Mapping system.")

        claim = st.text_area("Enter a claim to verify")

        if st.button("Ground Claim"):
            if claim:
                with st.spinner("Grounding..."):
                    result = st.session_state.grounding_router.ground(claim)

                    # Display result
                    col1, col2 = st.columns(2)
                    with col1:
                        status_colors = {
                            "VERIFIED": "🟢",
                            "GROUNDED": "🟡",
                            "UNVERIFIED": "🔴",
                            "NOT_APPLICABLE": "⚪",
                        }
                        st.write(f"**Status:** {status_colors.get(result.status, '⚪')} {result.status}")
                        st.write(f"**Claim Type:** {result.claim_type.value}")
                        st.write(f"**Confidence:** {result.confidence:.1%}")

                    with col2:
                        st.write(f"**Grounded:** {'Yes' if result.grounded else 'No'}")
                        if result.key_used:
                            st.write(f"**Key Used:** {result.key_used}")
                        if result.reason:
                            st.write(f"**Reason:** {result.reason}")
            else:
                st.error("Please enter a claim.")

    with tab2:
        st.subheader("Search Facts")

        query = st.text_input("Search query", key="kvrm_search")
        limit = st.slider("Max results", 1, 20, 5)

        if query:
            results = st.session_state.kvrm_resolver.search(query, limit=limit)

            if results:
                for r in results:
                    with st.expander(f"{r.key}"):
                        st.write(f"**Content:** {r.content}")
                        st.write(f"**Confidence:** {r.confidence:.1%}")
                        st.write(f"**Source:** {r.source}")
            else:
                st.info("No results found.")

    with tab3:
        st.subheader("Add Verified Fact")

        from mindforge.kvrm.resolver import FactKeyStore

        store = st.session_state.kvrm_resolver.stores.get("facts")

        if isinstance(store, FactKeyStore):
            domain = st.text_input("Domain (e.g., science, history)")
            fact_id = st.text_input("Fact ID (unique identifier)")
            content = st.text_area("Fact content")
            source = st.text_input("Source")
            confidence = st.slider("Confidence", 0.0, 1.0, 0.9)

            if st.button("Add Fact"):
                if domain and fact_id and content:
                    key = store.add_fact(
                        domain=domain,
                        fact_id=fact_id,
                        content=content,
                        source=source or "dashboard",
                        confidence=confidence,
                    )
                    st.success(f"Fact added with key: {key}")
                else:
                    st.error("Please fill in domain, ID, and content.")
        else:
            st.error("Fact store not available.")


# =============================================================================
# Needs Page
# =============================================================================

def render_needs():
    """Render the needs management interface."""
    st.title("Needs Management")

    if not st.session_state.initialized:
        st.warning("Please initialize MindForge first.")
        return

    mind = st.session_state.mind
    needs_state = mind.needs.get_state()

    # Current state visualization
    st.subheader("Current Needs State")

    for need_name, need_data in needs_state.items():
        col1, col2, col3 = st.columns([2, 3, 1])
        with col1:
            st.write(f"**{need_name.title()}**")
        with col2:
            st.progress(need_data["level"], text=f"{need_data['level']:.1%}")
        with col3:
            st.write(f"Weight: {need_data['weight']:.2f}")

    st.divider()

    # Presets
    st.subheader("Apply Preset")

    col1, col2, col3, col4 = st.columns(4)

    presets = {
        "balanced": "Default balanced configuration",
        "learning": "Higher curiosity for exploration",
        "production": "Focus on reliability",
        "creative": "Maximum curiosity",
    }

    for i, (preset_name, description) in enumerate(presets.items()):
        col = [col1, col2, col3, col4][i]
        with col:
            if st.button(preset_name.title(), key=f"preset_{preset_name}"):
                mind.needs.apply_preset(preset_name)
                st.success(f"Applied '{preset_name}' preset!")
                st.rerun()
            st.caption(description)

    st.divider()

    # Priority ranking
    st.subheader("Priority Ranking")

    ranking = mind.needs.get_priority_ranking()
    for i, (need_type, priority) in enumerate(ranking, 1):
        st.write(f"{i}. **{need_type.value.title()}** - Priority: {priority:.3f}")


# =============================================================================
# Thoughts Page
# =============================================================================

def render_thoughts():
    """Render the thoughts stream."""
    st.title("Thought Stream")

    if not st.session_state.initialized:
        st.warning("Please initialize MindForge first.")
        return

    mind = st.session_state.mind

    # Generate thought button
    st.subheader("Generate Thought")

    col1, col2 = st.columns(2)

    with col1:
        from mindforge.core.thought import ThoughtTrigger
        trigger = st.selectbox(
            "Trigger",
            [t.value for t in ThoughtTrigger],
        )

    with col2:
        if st.button("Generate Thought"):
            with st.spinner("Thinking..."):
                thought = mind.thought_generator.generate(
                    trigger=ThoughtTrigger(trigger),
                    context={},
                )

                st.success("Thought generated!")

                with st.expander("View Thought", expanded=True):
                    st.write(f"**Content:** {thought.content}")
                    st.write(f"**Type:** {thought.thought_type.value}")
                    st.write(f"**Trigger:** {thought.trigger.value}")
                    st.write(f"**Confidence:** {thought.confidence:.2f}")

    st.divider()

    # Thought types explanation
    st.subheader("Thought Types")

    thought_types = {
        "spontaneous": "Unprompted, curious thoughts",
        "reactive": "Response to external stimuli",
        "reflective": "Self-examination and analysis",
        "creative": "Novel ideas and connections",
        "planning": "Future-oriented goal setting",
        "empathetic": "Understanding others' perspectives",
    }

    cols = st.columns(2)
    for i, (ttype, description) in enumerate(thought_types.items()):
        with cols[i % 2]:
            st.write(f"**{ttype.title()}:** {description}")


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main dashboard application."""
    init_session_state()

    # Render sidebar and get selected page
    page = render_sidebar()

    # Initialize on first load
    if not st.session_state.initialized:
        initialize_mindforge()

    # Render selected page
    if page == "Overview":
        render_overview()
    elif page == "Chat":
        render_chat()
    elif page == "Memory":
        render_memory()
    elif page == "KVRM":
        render_kvrm()
    elif page == "Needs":
        render_needs()
    elif page == "Thoughts":
        render_thoughts()


def run_dashboard():
    """Entry point for running the dashboard."""
    main()


if __name__ == "__main__":
    main()
