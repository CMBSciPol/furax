#!/bin/sh
# Build the documentation site into site/.
# Run from the repository root: uv run --group docs sh docs/build.sh
set -e

# The notebooks expect several devices, like the test suite (tests/conftest.py)
if [ -z "${XLA_FLAGS}" ]; then
    XLA_FLAGS='--xla_force_host_platform_device_count=8'
    export XLA_FLAGS
fi

# Execute the notebooks and convert them to the Markdown pages the nav references
if ls docs/user-guide/*.ipynb > /dev/null 2>&1; then
    jupyter nbconvert --to markdown --execute docs/user-guide/*.ipynb
fi

zensical build

# The theme pins the navigation sidebar only from 76.25em (1220px) up and uses a
# hamburger drawer below; lower that breakpoint to 60em (960px) so the desktop
# layout, with the navigation always visible on the left, applies on laptop-sized
# windows too. The values are hardcoded in the minified assets, shared by the CSS
# media queries and the JS matchMedia observers.
sed -i 's/76\.25em/60em/g; s/76\.234375em/59\.984375em/g' \
    site/assets/stylesheets/*/main.*.min.css \
    site/assets/javascripts/bundle.*.min.js
