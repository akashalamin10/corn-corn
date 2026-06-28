#!/bin/bash
mkdir -p ~/.streamlit/
cat > ~/.streamlit/credentials.toml << 'EOF'
[general]
email = ""
EOF
