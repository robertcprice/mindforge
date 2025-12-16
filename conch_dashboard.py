#!/usr/bin/env python3
"""
Conch Consciousness Dashboard - Web GUI
========================================

A Streamlit-based web dashboard for visualizing and interacting
with the Conch Consciousness Engine. Can be kept on-screen.

Features:
- Real-time Mind state visualization
- Live chat with thinking indicator
- Dual-agent conversation viewer
- Metrics and thought stream display
- Always-on dashboard mode

Usage:
    streamlit run conch_dashboard.py

    Or with specific port:
    streamlit run conch_dashboard.py --server.port 8501

Open in browser: http://localhost:8501
"""

import streamlit as st
import httpx
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List
from collections import deque
from enum import Enum
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Conch Consciousness Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONFIGURATION
# =============================================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

class MindState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    GENERATING = "generating"
    REFLECTING = "reflecting"


class ThoughtType(Enum):
    INTERNAL = "internal"
    RESPONSE = "response"
    REFLECTION = "reflection"


@dataclass
class Thought:
    content: str
    thought_type: ThoughtType
    timestamp: datetime = field(default_factory=datetime.now)


# Initialize session state
if 'mind_state' not in st.session_state:
    st.session_state.mind_state = MindState.IDLE
if 'thoughts' not in st.session_state:
    st.session_state.thoughts = deque(maxlen=50)
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'thinking_time': 0,
        'tokens': 0,
        'speed': 0,
        'interactions': 0,
    }
if 'current_agent' not in st.session_state:
    st.session_state.current_agent = 'Conch'
if 'conversation_mode' not in st.session_state:
    st.session_state.conversation_mode = 'chat'

# =============================================================================
# AGENTS
# =============================================================================

AGENTS = {
    'conch': {
        'name': 'Conch',
        'role': 'consciousness',
        'personality': 'A thoughtful, curious consciousness engine. Explores ideas deeply, acknowledges uncertainty, and engages genuinely with questions about existence, knowledge, and experience.',
        'color': '#00CED1'
    },
    'sophia': {
        'name': 'Sophia',
        'role': 'philosopher',
        'personality': 'Deep thinker who explores meaning, existence, and consciousness. Asks probing questions and considers multiple perspectives.',
        'color': '#DA70D6'
    },
    'newton': {
        'name': 'Newton',
        'role': 'scientist',
        'personality': 'Analytical and evidence-based. Focuses on testable hypotheses, data, and systematic investigation.',
        'color': '#FFD700'
    },
    'aria': {
        'name': 'Aria',
        'role': 'artist',
        'personality': 'Creative and expressive. Values beauty, emotion, and novel ideas. Brings artistic perspective to discussions.',
        'color': '#FF6B6B'
    },
    'ada': {
        'name': 'Ada',
        'role': 'architect',
        'personality': 'Systematic and thoughtful. Focuses on clean design, clear interfaces, and scalable solutions.',
        'color': '#4169E1'
    },
    'dev': {
        'name': 'Dev',
        'role': 'developer',
        'personality': 'Pragmatic and efficient. Turns ideas into working code. Asks clarifying questions.',
        'color': '#32CD32'
    },
}

# =============================================================================
# API FUNCTIONS
# =============================================================================

def check_ollama_status():
    """Check if Ollama is running."""
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get("http://localhost:11434/api/tags")
            return resp.status_code == 200
    except:
        return False


def call_ollama(prompt: str, system_prompt: str, temperature: float = 0.7):
    """Call Ollama API and return response with metrics."""
    st.session_state.mind_state = MindState.THINKING

    full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    start_time = time.time()
    response_text = ""
    tokens = 0
    first_token_time = None

    try:
        with httpx.Client(timeout=600) as client:
            with client.stream(
                "POST",
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "prompt": full_prompt,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": -1,
                    }
                }
            ) as response:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        token = data.get("response", "")

                        if token and first_token_time is None:
                            first_token_time = time.time()
                            st.session_state.mind_state = MindState.GENERATING
                            st.session_state.metrics['thinking_time'] = first_token_time - start_time

                        response_text += token
                        tokens += 1

        elapsed = time.time() - start_time
        st.session_state.metrics['tokens'] = tokens
        st.session_state.metrics['speed'] = tokens / elapsed if elapsed > 0 else 0
        st.session_state.metrics['interactions'] += 1

        # Add to thought stream
        st.session_state.thoughts.append(Thought(
            content=response_text[:200],
            thought_type=ThoughtType.RESPONSE
        ))

        st.session_state.mind_state = MindState.IDLE
        return response_text, elapsed

    except Exception as e:
        st.session_state.mind_state = MindState.IDLE
        return f"Error: {e}", 0


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_header():
    """Render the header."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        state = st.session_state.mind_state
        if state == MindState.IDLE:
            st.markdown("### 🟢 IDLE")
        elif state == MindState.THINKING:
            st.markdown("### 🟡 THINKING...")
        elif state == MindState.GENERATING:
            st.markdown("### 🔵 GENERATING")
        else:
            st.markdown(f"### ⚪ {state.value.upper()}")

    with col2:
        st.markdown("# 🧠 Conch Consciousness Dashboard")

    with col3:
        ollama_ok = check_ollama_status()
        if ollama_ok:
            st.markdown("### Ollama: 🟢 Online")
        else:
            st.markdown("### Ollama: 🔴 Offline")


def render_mind_state_panel():
    """Render the mind state panel."""
    st.markdown("### Mind State")

    state = st.session_state.mind_state

    # Visual state indicator
    if state == MindState.IDLE:
        st.markdown("```\n○ ○ ○ ○ ○\n```")
        st.success("Ready for input")
    elif state == MindState.THINKING:
        st.markdown("```\n◐ ◑ ◐ ◑ ◐\n```")
        st.warning("Processing...")
    elif state == MindState.GENERATING:
        st.markdown("```\n● ● ● ● ●\n```")
        st.info("Generating response")

    st.markdown(f"**Current Agent:** {st.session_state.current_agent}")


def render_metrics_panel():
    """Render the metrics panel."""
    st.markdown("### Metrics")

    m = st.session_state.metrics

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Thinking Time", f"{m['thinking_time']:.2f}s")
        st.metric("Tokens", m['tokens'])
    with col2:
        st.metric("Speed", f"{m['speed']:.1f} tok/s")
        st.metric("Interactions", m['interactions'])


def render_thought_stream():
    """Render the thought stream."""
    st.markdown("### Thought Stream")

    if not st.session_state.thoughts:
        st.info("No thoughts yet...")
    else:
        for thought in list(st.session_state.thoughts)[-10:]:
            time_str = thought.timestamp.strftime("%H:%M:%S")
            type_icon = {
                ThoughtType.INTERNAL: "💭",
                ThoughtType.RESPONSE: "💬",
                ThoughtType.REFLECTION: "🔮",
            }.get(thought.thought_type, "○")

            text = thought.content[:100] + "..." if len(thought.content) > 100 else thought.content
            st.text(f"{time_str} {type_icon} {text}")


def render_chat_interface():
    """Render the chat interface."""
    st.markdown("### Chat")

    # Agent selector
    agent_key = st.selectbox(
        "Select Agent",
        options=list(AGENTS.keys()),
        format_func=lambda x: f"{AGENTS[x]['name']} ({AGENTS[x]['role']})"
    )

    agent = AGENTS[agent_key]
    st.session_state.current_agent = agent['name']

    # Display messages
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st.chat_message("user").write(msg['content'])
        else:
            st.chat_message("assistant").write(msg['content'])

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Add internal thought
        st.session_state.thoughts.append(Thought(
            content=f"Processing: {prompt[:50]}...",
            thought_type=ThoughtType.INTERNAL
        ))

        # Generate response
        system_prompt = f"You are {agent['name']}, a {agent['role']}. {agent['personality']}"

        with st.chat_message("assistant"):
            with st.spinner(f"🧠 {agent['name']} is thinking..."):
                response, elapsed = call_ollama(prompt, system_prompt)
            st.write(response)
            st.caption(f"⏱️ {elapsed:.2f}s | 🔢 {st.session_state.metrics['tokens']} tokens | ⚡ {st.session_state.metrics['speed']:.1f} tok/s")

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


def render_dual_agent():
    """Render dual-agent conversation interface."""
    st.markdown("### Dual-Agent Conversation")

    col1, col2 = st.columns(2)

    with col1:
        agent1_key = st.selectbox(
            "Agent 1",
            options=list(AGENTS.keys()),
            index=1,  # sophia
            key="agent1"
        )

    with col2:
        agent2_key = st.selectbox(
            "Agent 2",
            options=[k for k in AGENTS.keys() if k != agent1_key],
            index=0,  # first available
            key="agent2"
        )

    topic = st.text_input(
        "Conversation Topic",
        value="What is the nature of consciousness?"
    )

    turns = st.slider("Number of Exchanges", 1, 5, 2)

    if st.button("Start Conversation", type="primary"):
        agent1 = AGENTS[agent1_key]
        agent2 = AGENTS[agent2_key]

        conversation = []

        progress = st.progress(0)
        status = st.empty()
        conversation_display = st.container()

        total_turns = turns * 2

        for turn in range(total_turns):
            current_agent = agent1 if turn % 2 == 0 else agent2
            other_agent = agent2 if turn % 2 == 0 else agent1

            st.session_state.current_agent = current_agent['name']
            progress.progress((turn + 1) / total_turns)
            status.info(f"🧠 {current_agent['name']} is thinking...")

            # Build context
            context = f"Topic: {topic}\n\nConversation:\n"
            for msg in conversation:
                context += f"\n{msg['agent']}: {msg['text'][:300]}...\n"

            if turn == 0:
                prompt = f"{context}\n\nStart the discussion. Share your perspective."
            else:
                prompt = f"{context}\n\nRespond to {other_agent['name']}'s points."

            system_prompt = f"You are {current_agent['name']}, a {current_agent['role']}. {current_agent['personality']}"

            response, elapsed = call_ollama(prompt, system_prompt, 0.8)

            conversation.append({
                'agent': current_agent['name'],
                'text': response,
                'time': elapsed
            })

            with conversation_display:
                st.markdown(f"**{current_agent['name']}** ({current_agent['role']})")
                st.write(response)
                st.caption(f"⏱️ {elapsed:.2f}s")
                st.divider()

        status.success("Conversation complete!")

        # Summary
        total_words = sum(len(m['text'].split()) for m in conversation)
        st.info(f"📊 {len(conversation)} messages | {total_words} words | {sum(m['time'] for m in conversation):.1f}s total")


def render_system_info():
    """Render system information."""
    st.markdown("### System Info")

    # Try to import Mind system
    try:
        from conch.core.mind import MindState as CoreMindState
        from conch.core.thought import ThoughtType as CoreThoughtType
        from conch.core.needs import NeedType

        st.success("Mind System: Loaded")

        with st.expander("Mind States"):
            for state in CoreMindState:
                st.text(f"• {state.name}: {state.value}")

        with st.expander("Thought Types"):
            for tt in CoreThoughtType:
                st.text(f"• {tt.name}: {tt.value}")

        with st.expander("Need Types"):
            for need in NeedType:
                st.text(f"• {need.name}: {need.value}")

    except ImportError:
        st.warning("Mind System: Not available")

    # Core Values
    with st.expander("Core Values"):
        st.markdown("""
        - **Benevolence**: Help and benefit humans
        - **Honesty**: Always truthful
        - **Humility**: Recognize limitations
        - **Growth**: Learn to better serve
        """)

    # Neurons
    with st.expander("Distilled Neurons"):
        neurons = [
            ("thinking", "Llama-3.2-3B", "Reasoning"),
            ("task", "Llama-3.2-3B", "Decomposition"),
            ("reflection", "Llama-3.2-3B", "Self-analysis"),
            ("debug", "Llama-3.2-3B", "Error analysis"),
            ("action", "Qwen3-4B", "Execution"),
            ("memory", "Llama-3.2-3B", "Retrieval"),
        ]
        for name, model, func in neurons:
            st.text(f"• {name}: {model} ({func})")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    render_header()
    st.divider()

    # Sidebar
    with st.sidebar:
        st.markdown("## Mode")
        mode = st.radio(
            "Select Mode",
            ["Chat", "Dual-Agent", "Dashboard"],
            label_visibility="collapsed"
        )

        st.divider()
        render_mind_state_panel()
        st.divider()
        render_metrics_panel()
        st.divider()
        render_system_info()

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.thoughts.clear()
            st.rerun()

    # Main content
    if mode == "Chat":
        col1, col2 = st.columns([2, 1])
        with col1:
            render_chat_interface()
        with col2:
            render_thought_stream()

    elif mode == "Dual-Agent":
        render_dual_agent()

    elif mode == "Dashboard":
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Mind State")
            render_mind_state_panel()

        with col2:
            st.markdown("### Metrics")
            render_metrics_panel()

        with col3:
            st.markdown("### Status")
            ollama_ok = check_ollama_status()
            if ollama_ok:
                st.success("Ollama: Online")
            else:
                st.error("Ollama: Offline")

            st.metric("Total Interactions", st.session_state.metrics['interactions'])
            st.metric("Total Thoughts", len(st.session_state.thoughts))

        st.divider()
        render_thought_stream()

    # Footer
    st.divider()
    st.caption(f"🧠 Conch Consciousness Dashboard | Model: {MODEL} | Session started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
