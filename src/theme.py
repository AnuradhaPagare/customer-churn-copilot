"""Dark theme: palette, CSS injection and small HTML components.

All HTML is emitted on a single line without blank lines or indentation, because
Streamlit's markdown renderer would otherwise treat indented HTML as a code block.
Every dynamic string is escaped with `esc()`.
"""
from __future__ import annotations

import html
import re

import pandas as pd
import streamlit as st

from . import config

BG, SIDEBAR, CARD, PANEL = "#0B1120", "#111827", "#151F32", "#1E293B"
INDIGO, CYAN, EMERALD, AMBER, ROSE = "#6366F1", "#06B6D4", "#10B981", "#F59E0B", "#F43F5E"
TEXT, MUTED, BORDER = "#F8FAFC", "#94A3B8", "#293548"
FONT = "Inter, 'Segoe UI', system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif"
RISK_COLORS = {"Low": EMERALD, "Medium": AMBER, "High": ROSE}

ICONS = {  # 24x24 stroke icons
    "users": '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "trend": '<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/>',
    "alert": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "dollar": '<line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "columns": '<path d="M12 3h7a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-7m0-18H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h7m0-18v18"/>',
    "inbox": '<polyline points="22 12 16 12 14 15 10 15 8 12 2 12"/><path d="M5.45 5.11L2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
}
TONES = {"pos": EMERALD, "warn": AMBER, "neg": ROSE, "info": CYAN, "brand": INDIGO, "muted": MUTED}


def esc(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def svg(name: str, size: int = 18, color: str = "currentColor") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            f'{ICONS.get(name, ICONS["info"])}</svg>')


def set_active_nav(url_path: str) -> None:
    """Highlight the current page's sidebar link (Streamlit exposes no stable active-state attribute)."""
    sel = '[href=""], [href="/"]' if url_path == "overview" else f'[href$="{url_path}"]'
    rules = ", ".join(f'[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"]{x.strip()}' for x in sel.split(","))
    st.markdown(f"<style>{rules} {{ background: rgba(99,102,241,.18) !important; border-color: rgba(99,102,241,.6) !important; "
                f"box-shadow: inset 3px 0 0 {INDIGO}; color: #fff !important; }}</style>", unsafe_allow_html=True)


def md(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def inject_css() -> None:
    css = (config.ASSETS_DIR / "style.css").read_text(encoding="utf-8")
    css = (css.replace("{{BG}}", BG).replace("{{SIDEBAR}}", SIDEBAR).replace("{{CARD}}", CARD)
              .replace("{{PANEL}}", PANEL).replace("{{INDIGO}}", INDIGO).replace("{{CYAN}}", CYAN)
              .replace("{{EMERALD}}", EMERALD).replace("{{AMBER}}", AMBER).replace("{{ROSE}}", ROSE)
              .replace("{{TEXT}}", TEXT).replace("{{MUTED}}", MUTED).replace("{{BORDER}}", BORDER)
              .replace("{{FONT}}", FONT))
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- components
def pill(text: str, tone: str = "muted", icon: str | None = None) -> str:
    c = TONES.get(tone, MUTED)
    ic = svg(icon, 14) if icon else ""
    return f'<span class="pill" style="--c:{c}">{ic}{esc(text)}</span>'


def risk_pill(level) -> str:
    lv = "—" if level is None or (isinstance(level, float) and pd.isna(level)) else str(level)
    c = RISK_COLORS.get(lv, MUTED)
    return f'<span class="risk-pill" style="--c:{c}"><i></i>{esc(lv)}</span>'


def app_header(title: str, subtitle: str, pills: list[str]) -> None:
    md(f'<div class="app-header"><div><div class="app-title">{esc(title)}</div>'
       f'<div class="app-sub">{esc(subtitle)}</div></div><div class="status-row">{"".join(pills)}</div></div>')


def page_heading(title: str, blurb: str = "") -> None:
    md(f'<div class="page-heading"><h2>{esc(title)}</h2>' + (f'<p>{esc(blurb)}</p>' if blurb else "") + "</div>")


def kpi_card(label: str, value: str, note: str | None = None, tone: str = "muted", icon: str = "activity",
             accent: str = INDIGO) -> str:
    n = f'<div class="kpi-note" style="--c:{TONES.get(tone, MUTED)}">{esc(note)}</div>' if note else ""
    return (f'<div class="kpi"><div class="kpi-top"><span class="kpi-label">{esc(label)}</span>'
            f'<span class="kpi-icon" style="--c:{accent}">{svg(icon, 18)}</span></div>'
            f'<div class="kpi-value">{esc(value)}</div>{n}</div>')


def kpi_row(cards: list[str]) -> None:
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        col.markdown(card, unsafe_allow_html=True)


def callout(text: str, kind: str = "info", title: str | None = None) -> None:
    c = {"info": CYAN, "warn": AMBER, "danger": ROSE, "success": EMERALD}.get(kind, CYAN)
    t = f'<b>{esc(title)}</b> ' if title else ""
    md(f'<div class="callout" style="--c:{c}">{svg("info" if kind != "warn" else "alert", 18)}<div>{t}{esc(text)}</div></div>')


def empty_state(title: str, message: str, icon: str = "inbox") -> None:
    md(f'<div class="empty"><div class="empty-icon">{svg(icon, 28)}</div><div class="empty-title">{esc(title)}</div>'
       f'<div class="empty-msg">{esc(message)}</div></div>')


def card_header(title: str, subtitle: str = "") -> str:
    sub = f'<div class="card-sub">{esc(subtitle)}</div>' if subtitle else ""
    return f'<div class="card-head"><div class="card-title">{esc(title)}</div>{sub}</div>'


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]


PLOT_CONFIG = {"displayModeBar": "hover", "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"], "responsive": True}


def chart_card(title: str, fig, subtitle: str = "", key: str | None = None) -> None:
    """Dark card containing a Plotly figure. Uses a keyed container so CSS targets a stable class."""
    k = key or slug(title)
    with st.container(key=f"card-{k}"):
        md(card_header(title, subtitle))
        st.plotly_chart(fig, theme=None, width="stretch", config=PLOT_CONFIG, key=f"plot-{k}")


def panel(key: str):
    """A dark card container for arbitrary content (tables, text, widgets)."""
    return st.container(key=f"card-{slug(key)}")


def html_table(df: pd.DataFrame, risk_cols: tuple = (), prob_cols: tuple = ()) -> None:
    """Compact dark table with alternating rows, hover state and risk pills."""
    if df.empty:
        empty_state("Nothing to show", "No rows match the current filters.")
        return
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if c in risk_cols:
                cells.append(f"<td>{risk_pill(v)}</td>")
            elif c in prob_cols and pd.notna(v):
                pct = max(0.0, min(1.0, float(v)))
                col = ROSE if pct >= 0.5 else AMBER if pct >= 0.25 else EMERALD
                cells.append(f'<td><div class="probcell"><span>{pct:.0%}</span><div class="bar"><div style="width:{pct * 100:.0f}%;'
                             f'background:{col}"></div></div></div></td>')
            else:
                cells.append(f"<td>{esc('' if pd.isna(v) else v)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    md(f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def style_dataframe(df: pd.DataFrame, prob_col: str | None = None):
    """Styler with alternating row shading and risk colouring for st.dataframe. Skipped for huge frames."""
    if df.size > 150_000:
        return df

    def zebra(d: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame("", index=d.index, columns=d.columns)
        out.iloc[1::2, :] = f"background-color: {PANEL}"
        out.iloc[0::2, :] = f"background-color: {CARD}"
        return out

    sty = df.style.apply(zebra, axis=None)
    fmt = {}
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]) and c != prob_col and c != "churn_probability":
            vals = df[c].dropna()
            fmt[c] = "{:,.0f}" if len(vals) and (vals % 1 == 0).all() else "{:,.2f}"
    if fmt:
        sty = sty.format(fmt, na_rep="—")
    if "risk_category" in df.columns:
        sty = sty.map(lambda v: f"color: {RISK_COLORS.get(str(v), TEXT)}; font-weight: 600", subset=["risk_category"])
    return sty
