"""ClauseFinder Gradio theme: NVIDIA-green accent on a neutral slate base."""
from __future__ import annotations

import gradio as gr

# NVIDIA green ramp centered on #76B900
_nvidia = gr.themes.Color(
    c50="#f3fae4", c100="#e4f4c3", c200="#cdea92", c300="#b1dd5c",
    c400="#94cc2b", c500="#76b900", c600="#5e9400", c700="#487100",
    c800="#365500", c900="#2a4200", c950="#1a2900",
)

theme = gr.themes.Soft(
    primary_hue=_nvidia,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_400",
    button_primary_text_color="#0b0f0a",
    block_title_text_weight="600",
)

CSS = """
.badge {display:inline-block;padding:2px 10px;margin:2px 4px 2px 0;border-radius:999px;
        font-size:0.78rem;font-weight:600;line-height:1.4;white-space:nowrap;}
.badge-green {background:#e4f4c3;color:#2a4200;border:1px solid #94cc2b;}
.badge-amber {background:#fdf0d5;color:#7a4b00;border:1px solid #e8b04a;}
.badge-red   {background:#fde3e3;color:#7a1010;border:1px solid #e87070;}
.dark .badge-green {background:#213500;color:#cdea92;border-color:#487100;}
.dark .badge-amber {background:#3a2c07;color:#f3d8a0;border-color:#8a6414;}
.dark .badge-red   {background:#3a0f0f;color:#f0b0b0;border-color:#8a2626;}
.source-card {border-left:3px solid #76b900;padding:4px 10px;margin:6px 0;
              font-size:0.85rem;}
.source-card .meta {opacity:0.7;}
.cf-footer {text-align:center;font-size:0.85rem;opacity:0.85;padding:10px 0 4px;
            border-top:1px solid rgba(128,128,128,0.25);margin-top:8px;}
.cf-footer strong {color:#5e9400;}
.dark .cf-footer strong {color:#94cc2b;}
"""
