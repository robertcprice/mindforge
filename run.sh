#!/bin/bash
#
# Conch Consciousness Engine - Run Script
# Starts both the terminal console and web dashboard
#

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Conch Consciousness Engine                     ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Activate virtual environment
source venv/bin/activate

# Start the Streamlit dashboard in background
echo -e "${BLUE}Starting web dashboard...${NC}"
streamlit run conch/dashboard/app.py --server.headless true &
DASHBOARD_PID=$!
echo -e "${GREEN}✓ Dashboard started (PID: $DASHBOARD_PID)${NC}"
echo -e "${YELLOW}  Open in browser: http://localhost:8501${NC}"
echo ""

# Wait for dashboard to initialize
sleep 3

# Open browser automatically (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:8501 2>/dev/null &
fi

# Trap to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    kill $DASHBOARD_PID 2>/dev/null
    echo -e "${GREEN}✓ Dashboard stopped${NC}"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Run the consciousness engine in foreground
echo -e "${BLUE}Starting consciousness engine...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop both${NC}"
echo ""
echo "════════════════════════════════════════════════════════════════"

# Parse arguments
CYCLES=""
if [[ "$1" == "--cycles" ]] && [[ -n "$2" ]]; then
    CYCLES="--cycles $2"
fi

python main.py $CYCLES

# Cleanup when main.py exits
cleanup
