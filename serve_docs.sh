#!/bin/bash

# Serve Sphinx Documentation on Local Network
# Your colleagues can access the docs via http://YOUR_IP:8000

set -e

cd "$(dirname "$0")/sphinx_docs/_build/html"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔═════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  LeRobot Testing Branch Documentation Server       ║${NC}"
echo -e "${BLUE}╚═════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if documentation is built
if [ ! -f "index.html" ]; then
    echo -e "${RED}✗ Documentation not built yet!${NC}"
    echo ""
    echo "Please build the documentation first:"
    echo "  ./build_sphinx_docs.sh"
    exit 1
fi

# Get local IP address
echo -e "${BLUE}📡 Detecting network interfaces...${NC}"
echo ""

# Try to get the main IP address
IP_ADDR=$(hostname -I | awk '{print $1}')

if [ -z "$IP_ADDR" ]; then
    IP_ADDR="localhost"
    echo -e "${YELLOW}⚠ Could not detect IP address${NC}"
else
    echo -e "${GREEN}✓ Your IP address: ${IP_ADDR}${NC}"
fi

echo ""
echo -e "${BLUE}🌐 Starting HTTP server...${NC}"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  📚 Documentation is now available at:"
echo ""
echo "     Local:   ${GREEN}http://localhost:8000${NC}"
echo "     Network: ${GREEN}http://${IP_ADDR}:8000${NC}"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  👥 Share with colleagues:"
echo "     Tell them to open: ${YELLOW}http://${IP_ADDR}:8000${NC}"
echo ""
echo "  🛑 To stop the server: Press ${RED}Ctrl+C${NC}"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

# Start Python HTTP server
python3 -m http.server 8000 --bind 0.0.0.0
