#!/usr/bin/env python3
"""Convert markdown documentation to PDF with proper formatting."""

import markdown2
from weasyprint import HTML, CSS
from pathlib import Path

def markdown_to_pdf(md_file, pdf_file):
    """Convert markdown file to PDF with styling."""
    
    # Read markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown2.markdown(
        md_content,
        extras=[
            'fenced-code-blocks',
            'tables',
            'header-ids',
            'toc',
            'code-friendly',
            'break-on-newline'
        ]
    )
    
    # Create styled HTML document
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Federated Learning for Fraud Detection - Project Documentation</title>
        <style>
            @page {{
                size: A4;
                margin: 2cm 1.5cm;
                @bottom-center {{
                    content: "Page " counter(page) " of " counter(pages);
                    font-size: 9pt;
                    color: #666;
                }}
            }}
            
            body {{
                font-family: 'DejaVu Sans', Arial, sans-serif;
                font-size: 10pt;
                line-height: 1.5;
                color: #333;
                max-width: 100%;
            }}
            
            h1 {{
                color: #1a1a1a;
                font-size: 20pt;
                font-weight: bold;
                margin-top: 0.5cm;
                margin-bottom: 0.3cm;
                page-break-after: avoid;
                border-bottom: 2px solid #2c5aa0;
                padding-bottom: 0.2cm;
            }}
            
            h2 {{
                color: #2c5aa0;
                font-size: 14pt;
                font-weight: bold;
                margin-top: 0.6cm;
                margin-bottom: 0.3cm;
                page-break-after: avoid;
            }}
            
            h3 {{
                color: #4a4a4a;
                font-size: 12pt;
                font-weight: bold;
                margin-top: 0.4cm;
                margin-bottom: 0.2cm;
                page-break-after: avoid;
            }}
            
            h4 {{
                color: #666;
                font-size: 11pt;
                font-weight: bold;
                margin-top: 0.3cm;
                margin-bottom: 0.2cm;
            }}
            
            p {{
                margin: 0.2cm 0;
                text-align: justify;
            }}
            
            ul, ol {{
                margin: 0.2cm 0;
                padding-left: 0.8cm;
            }}
            
            li {{
                margin: 0.1cm 0;
            }}
            
            code {{
                background-color: #f4f4f4;
                padding: 0.05cm 0.15cm;
                border-radius: 2px;
                font-family: 'DejaVu Sans Mono', 'Courier New', monospace;
                font-size: 9pt;
                color: #c7254e;
            }}
            
            pre {{
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 3px;
                padding: 0.3cm;
                overflow-x: auto;
                font-family: 'DejaVu Sans Mono', 'Courier New', monospace;
                font-size: 8pt;
                line-height: 1.3;
                margin: 0.3cm 0;
                page-break-inside: avoid;
            }}
            
            pre code {{
                background-color: transparent;
                padding: 0;
                color: #333;
            }}
            
            strong {{
                font-weight: bold;
                color: #1a1a1a;
            }}
            
            em {{
                font-style: italic;
            }}
            
            hr {{
                border: none;
                border-top: 1px solid #ddd;
                margin: 0.4cm 0;
            }}
            
            blockquote {{
                border-left: 3px solid #2c5aa0;
                padding-left: 0.4cm;
                margin: 0.3cm 0;
                color: #666;
                font-style: italic;
            }}
            
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 0.3cm 0;
                font-size: 9pt;
            }}
            
            th, td {{
                border: 1px solid #ddd;
                padding: 0.15cm;
                text-align: left;
            }}
            
            th {{
                background-color: #2c5aa0;
                color: white;
                font-weight: bold;
            }}
            
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            
            .page-break {{
                page-break-before: always;
            }}
            
            /* Prevent orphans and widows */
            p, li {{
                orphans: 3;
                widows: 3;
            }}
            
            /* Keep headings with following content */
            h1, h2, h3, h4, h5, h6 {{
                page-break-after: avoid;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Convert HTML to PDF
    HTML(string=styled_html).write_pdf(pdf_file)
    print(f"✓ PDF created successfully: {pdf_file}")
    print(f"  File size: {Path(pdf_file).stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    markdown_file = "project_documentation.md"
    pdf_file = "Project_Documentation.pdf"
    
    print(f"Converting {markdown_file} to PDF...")
    markdown_to_pdf(markdown_file, pdf_file)
    print("\nDone! Your project documentation is ready.")
