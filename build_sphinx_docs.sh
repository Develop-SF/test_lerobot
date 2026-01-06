#!/bin/bash

# Sphinx Documentation Build Script for Testing Branch

set -e

cd "$(dirname "$0")"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  LeRobot Testing Branch Documentation  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Menu
echo "1. Build documentation (HTML)"
echo "2. Build and open in browser"
echo "3. Clean build directory"
echo "4. Rebuild from scratch"
echo "5. Check Sphinx installation"
echo "6. Exit"
echo ""
read -p "Select option (1-6): " choice

case $choice in
    1)
        echo -e "\n${BLUE}Building Sphinx documentation...${NC}"
        cd sphinx_docs
        sphinx-build -b html . _build/html
        echo -e "\n${GREEN}✓ Documentation built successfully!${NC}"
        echo -e "Location: $(pwd)/_build/html/index.html"
        ;;
    2)
        echo -e "\n${BLUE}Building Sphinx documentation...${NC}"
        cd sphinx_docs
        sphinx-build -b html . _build/html
        echo -e "\n${GREEN}✓ Documentation built successfully!${NC}"
        echo -e "\n${BLUE}Opening in browser...${NC}"
        xdg-open _build/html/index.html || firefox _build/html/index.html || google-chrome _build/html/index.html
        ;;
    3)
        echo -e "\n${YELLOW}Cleaning build directory...${NC}"
        rm -rf sphinx_docs/_build
        echo -e "${GREEN}✓ Build directory cleaned${NC}"
        ;;
    4)
        echo -e "\n${YELLOW}Cleaning and rebuilding...${NC}"
        rm -rf sphinx_docs/_build
        cd sphinx_docs
        sphinx-build -b html . _build/html
        echo -e "\n${GREEN}✓ Documentation rebuilt successfully!${NC}"
        echo -e "\n${BLUE}Opening in browser...${NC}"
        xdg-open _build/html/index.html || firefox _build/html/index.html || google-chrome _build/html/index.html
        ;;
    5)
        echo -e "\n${BLUE}Checking Sphinx installation...${NC}"
        if command -v sphinx-build &> /dev/null; then
            VERSION=$(sphinx-build --version)
            echo -e "${GREEN}✓ Sphinx is installed: $VERSION${NC}"
            echo ""
            echo "Installed extensions:"
            python3 -c "import myst_parser; print('  ✓ myst-parser:', myst_parser.__version__)" 2>/dev/null || echo "  ✗ myst-parser not found"
            python3 -c "import sphinx_rtd_theme; print('  ✓ sphinx-rtd-theme:', sphinx_rtd_theme.__version__)" 2>/dev/null || echo "  ✗ sphinx-rtd-theme not found"
        else
            echo -e "${YELLOW}✗ Sphinx not found${NC}"
            echo ""
            echo "Install with:"
            echo "  pip install --user --index-url https://pypi.org/simple sphinx sphinx-rtd-theme myst-parser"
        fi
        ;;
    6)
        echo -e "\n${BLUE}Goodbye!${NC}"
        exit 0
        ;;
    *)
        echo -e "\n${YELLOW}Invalid option${NC}"
        exit 1
        ;;
esac

echo ""
