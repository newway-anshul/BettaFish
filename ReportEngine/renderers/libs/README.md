# Third-Party JavaScript Libraries

This directory contains third-party JavaScript libraries required for HTML report rendering. These libraries are inlined into the generated HTML files for use in offline environments.

## Included Libraries

1. **chart.js** (204KB) - For chart rendering
   - Version: 4.5.1
   - Source: https://cdn.jsdelivr.net/npm/chart.js

2. **chartjs-chart-sankey.js** (10KB) - Sankey chart plugin
   - Version: 0.12.0
   - Source: https://unpkg.com/chartjs-chart-sankey@0.12.0/dist/chartjs-chart-sankey.min.js

3. **html2canvas.min.js** (194KB) - HTML to Canvas utility
   - Version: 1.4.1
   - Source: https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js

4. **jspdf.umd.min.js** (356KB) - PDF export library
   - Version: 2.5.1
   - Source: https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js

5. **mathjax.js** (1.1MB) - Math formula rendering engine
   - Version: 3.2.2
   - Source: https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js

## Functionality

The HTML renderer (`html_renderer.py`) automatically loads these library files from this directory and inlines them into the generated HTML. Benefits:

- ✅ Offline-ready — works without a network connection
- ✅ Fast loading — no dependency on external CDNs
- ✅ High stability — unaffected by CDN outages
- ✅ Pinned versions — ensures consistent behavior

## Fallback Mechanism

If a library file fails to load (e.g., missing or unreadable), the renderer automatically falls back to CDN links to ensure the report always renders correctly.

## Updating Libraries

To update a library:

1. Download the latest version from the corresponding CDN
2. Replace the file in this directory
3. Update the version information in this README

## Notes

- Total size is approximately 1.86 MB, which increases the size of the generated HTML file
- These libraries are included even for simple reports that do not use charts or math formulas
- Consider lighter-weight alternatives if a smaller file size is needed
