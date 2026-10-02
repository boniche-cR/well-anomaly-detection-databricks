# Databricks notebook source
# MAGIC %md
# MAGIC # _viz_helper — diagramas SVG para el accelerator
# MAGIC Define `draw_graph(...)` que devuelve un SVG (se muestra con `displayHTML`). Renderiza en
# MAGIC cualquier workspace (no depende de Mermaid). Se invoca con `%run ./_viz_helper`.

# COMMAND ----------

def draw_graph(nodes, edges, highlight_ids=None, caption=None):
    """
    nodes: dict id -> {"col": int, "row": int, "label": str}  (label admite '\\n')
    edges: list de (src, dst) | (src, dst, etiqueta) | (src, dst, etiqueta, dashed_bool)
    highlight_ids: lista de ids a resaltar (lava)
    Devuelve un string SVG para displayHTML().
    """
    highlight = set(highlight_ids or [])
    BW, BH = 170, 58          # tamaño de caja
    CX, CY = 250, 100         # paso de grilla
    OX, OY = 24, 24           # margen
    LAVA, NAVY, OAT, GREY = "#FF3621", "#1B3139", "#F9F7F4", "#5A6B70"

    pos, maxc, maxr = {}, 0, 0
    for nid, n in nodes.items():
        pos[nid] = (OX + n["col"] * CX, OY + n["row"] * CY)
        maxc, maxr = max(maxc, n["col"]), max(maxr, n["row"])
    W = OX * 2 + maxc * CX + BW
    H = OY * 2 + maxr * CY + BH + (26 if caption else 0)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" style="font-family:DM Sans,Segoe UI,Arial,sans-serif;background:{OAT}">']
    s.append(f'<defs><marker id="ah" markerWidth="10" markerHeight="10" refX="8" refY="3" '
             f'orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="{NAVY}"/></marker></defs>')

    # aristas
    for e in edges:
        src, dst = e[0], e[1]
        lbl = e[2] if len(e) > 2 else None
        dashed = e[3] if len(e) > 3 else False
        sx, sy = pos[src]; dx, dy = pos[dst]
        if dx >= sx:
            x1, y1, x2, y2 = sx + BW, sy + BH / 2, dx, dy + BH / 2
        else:
            x1, y1, x2, y2 = sx, sy + BH / 2, dx + BW, dy + BH / 2
        dash = ' stroke-dasharray="5,4"' if dashed else ''
        s.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{NAVY}" '
                 f'stroke-width="1.6" marker-end="url(#ah)"{dash}/>')
        if lbl:
            s.append(f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 - 5}" font-size="11" '
                     f'fill="{GREY}" text-anchor="middle">{lbl}</text>')

    # nodos
    for nid, n in nodes.items():
        x, y = pos[nid]
        hl = nid in highlight
        fill = LAVA if hl else "#FFFFFF"
        stroke = LAVA if hl else NAVY
        tcol = "#FFFFFF" if hl else NAVY
        s.append(f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" rx="9" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="1.8"/>')
        lines = n["label"].split("\n")
        ty = y + BH / 2 - (len(lines) - 1) * 8 + 4
        for i, ln in enumerate(lines):
            weight = "600" if i == 0 else "400"
            s.append(f'<text x="{x + BW / 2}" y="{ty + i * 15}" font-size="12.5" '
                     f'font-weight="{weight}" fill="{tcol}" text-anchor="middle">{ln}</text>')

    if caption:
        s.append(f'<text x="{OX}" y="{H - 8}" font-size="11" fill="{GREY}">{caption}</text>')
    s.append('</svg>')
    return "".join(s)

print("draw_graph() listo.")
