# -*- coding: utf-8 -*-
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
fig, ax = plt.subplots(figsize=(15.5, 7.6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

COLORS = {
    'border': '#334155', 'muted': '#64748b', 'stage_fill': '#f8fafc', 'stage_edge': '#cbd5e1',
    'text': '#0f172a', 'text_fill': '#e8f0ff', 'audio_fill': '#e8fbf8', 'visual_fill': '#f3ebff',
    'proj_fill': '#f8fafc', 'ev_fill': '#fff7ed', 'stable_fill': '#ecfdf3', 'conf_fill': '#fff1e6',
    'ctx_fill': '#eef2ff', 'head_fill': '#fff1f5', 'loss_fill': '#fff7e8', 'mag_fill': '#f4efff',
    'green': '#4aa564', 'orange': '#e67e22', 'blue': '#4f83cc', 'purple': '#8b6ccf'
}


def rounded_box(x, y, w, h, text, fc, ec, fontsize=10, weight='regular', radius=0.02,
                align='center', textcolor=None, lw=1.6, linestyle='solid', z=1):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0.01,rounding_size={radius}",
                           linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=linestyle, zorder=z)
    ax.add_patch(patch)
    ax.text(x + (w / 2 if align == 'center' else 0.02), y + h / 2, text,
            ha=align, va='center', fontsize=fontsize, weight=weight,
            color=textcolor or COLORS['text'], zorder=z + 1, linespacing=1.18)
    return patch


def stage_frame(x, y, w, h, title, subtitle):
    rounded_box(x, y, w, h, '', COLORS['stage_fill'], COLORS['stage_edge'], lw=1.5,
                linestyle=(0, (4, 3)), radius=0.025, z=0)
    ax.text(x + 0.016, y + h - 0.035, title, ha='left', va='center', fontsize=14.5,
            weight='bold', color=COLORS['text'])
    ax.text(x + 0.016, y + h - 0.070, subtitle, ha='left', va='center', fontsize=9.7,
            color=COLORS['muted'])


def draw_tokens(x, y, w, h, title, footer, fill, edge, token_labels=None):
    rounded_box(x, y, w, h, '', fill, edge, lw=1.5)
    ax.text(x + w / 2, y + h - 0.03, title, ha='center', va='center', fontsize=11.5, weight='bold')
    inner_y = y + 0.05
    inner_h = h - 0.11
    inner_x = x + 0.02
    inner_w = w - 0.04
    if token_labels:
        tw = inner_w / len(token_labels)
        shades = ['#dbeafe', '#eff6ff', '#e5efff', '#d0e2ff']
        for i, label in enumerate(token_labels):
            rect = Rectangle((inner_x + i * tw, inner_y), tw - 0.004, inner_h,
                             linewidth=1, edgecolor=edge, facecolor=shades[i % len(shades)])
            ax.add_patch(rect)
            ax.text(inner_x + i * tw + (tw - 0.004) / 2, inner_y + inner_h / 2, label,
                    ha='center', va='center', fontsize=8.5, color=COLORS['border'])
    else:
        bars = 7
        bw = inner_w / bars
        heights = [0.55, 0.85, 0.35, 0.72, 0.48, 0.92, 0.60]
        for i, scale in enumerate(heights):
            rect = Rectangle((inner_x + i * bw + 0.003, inner_y), bw - 0.008, inner_h * scale,
                             linewidth=0.8, edgecolor='none', facecolor=edge, alpha=0.75)
            ax.add_patch(rect)
    ax.text(x + w / 2, y + 0.02, footer, ha='center', va='center', fontsize=8.9, color=COLORS['muted'])


def arrow(p1, p2, color=None, lw=1.8, style='-|>', ls='solid', rad=0.0, z=2):
    patch = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14, linewidth=lw,
                            color=color or COLORS['border'], linestyle=ls,
                            connectionstyle=f"arc3,rad={rad}", zorder=z)
    ax.add_patch(patch)
    return patch


stage_frame(0.03, 0.58, 0.28, 0.34, 'Stage 1  Utterance encoders',
            'Lexical, acoustic, and facial evidence are projected to a shared space')
stage_frame(0.33, 0.20, 0.35, 0.72, 'Stage 2  EV-Gate fusion',
            'Expectation-violation scores separate stable evidence from conflicting cues')
stage_frame(0.70, 0.20, 0.27, 0.72, 'Stage 3  Context + optimization',
            'Dialogue reasoning, classification, and momentum-aligned masked updates')

card_w, card_h = 0.076, 0.16
xs = [0.046, 0.138, 0.230]
draw_tokens(xs[0], 0.69, card_w, card_h, 'Text', 'RoBERTa-Large', COLORS['text_fill'], '#7aa6f7', ['[CLS]', 'I', 'am', 'fine'])
draw_tokens(xs[1], 0.69, card_w, card_h, 'Audio', 'openSMILE', COLORS['audio_fill'], '#5bc0b8')
draw_tokens(xs[2], 0.69, card_w, card_h, 'Visual', 'DenseNet / FER+', COLORS['visual_fill'], '#9b87f5')

proj_w, proj_h = 0.083, 0.082
proj_xs = [0.042, 0.134, 0.226]
proj_texts = [
    'Text projection\nLayerNorm + Linear + GELU\n1024 -> 320',
    'Audio projection\nLayerNorm + Linear + GELU\n1582/300 -> 320',
    'Visual projection\nLayerNorm + Linear + GELU\n342 -> 320',
]
for x, txt in zip(proj_xs, proj_texts):
    rounded_box(x, 0.595, proj_w, proj_h, txt, COLORS['proj_fill'], '#94a3b8', fontsize=9.3)
    arrow((x + proj_w / 2, 0.69), (x + proj_w / 2, 0.595 + proj_h))

rounded_box(0.37, 0.71, 0.145, 0.11,
            'Discrepancy scores\n$\\delta^{ta}_i=\\|h^t_i-h^a_i\\|_2$\n$\\delta^{tv}_i=\\|h^t_i-h^v_i\\|_2$',
            COLORS['ev_fill'], '#f59e0b', fontsize=10.2)
rounded_box(0.545, 0.71, 0.11, 0.11,
            'Violation router\n$g_i=\\sigma(f_c([\\delta^{ta}_i;\\delta^{tv}_i]))$',
            COLORS['ev_fill'], '#f59e0b', fontsize=10.2)
rounded_box(0.37, 0.49, 0.13, 0.125,
            'Stable path\n$s_i=f_s([h^t_i;h^a_i;h^v_i])$\n2-layer MLP + GELU',
            COLORS['stable_fill'], COLORS['green'], fontsize=10.3)
rounded_box(0.525, 0.49, 0.13, 0.125,
            'Conflict path\n$c_i=f_c([h^t_i;h^a_i;h^v_i;$\n$|h^t_i-h^a_i|;|h^t_i-h^v_i|])$\n2-layer MLP + GELU',
            COLORS['conf_fill'], COLORS['orange'], fontsize=9.8)
rounded_box(0.43, 0.33, 0.17, 0.12,
            'Utterance fusion\n$z_i=s_i+g_i\\odot c_i$\nSpeaker residual: $z_i+e_{spk}$',
            '#fffaf5', '#fb923c', fontsize=10.6, weight='bold')
ax.text(0.515, 0.285, 'fused utterance state', fontsize=9.4, color=COLORS['muted'], ha='center')

rounded_box(0.74, 0.62, 0.19, 0.12,
            'BiGRU context encoder\nDialogue sequence: $u_1,u_2,\\ldots,u_T$\nBidirectional temporal reasoning',
            COLORS['ctx_fill'], '#6b7fe3', fontsize=10.1)
rounded_box(0.74, 0.45, 0.19, 0.095,
            'Emotion head\nDropout + MLP + Softmax', COLORS['head_fill'], '#ef7298', fontsize=10.2)
rounded_box(0.74, 0.30, 0.19, 0.095,
            'Weighted cross-entropy\nValidation-selected weighted-F1', COLORS['loss_fill'], '#d7a029', fontsize=10.1)
rounded_box(0.74, 0.13, 0.19, 0.11,
            'MoMask optimizer\n$m_t=\\beta m_{t-1}+(1-\\beta)g_t$\nmask$(g_t \\odot m_t<0)$ -> AdamW step',
            COLORS['mag_fill'], COLORS['purple'], fontsize=10.0)

centers = [x + proj_w / 2 for x in proj_xs]
for c in centers:
    arrow((c, 0.61), (0.44, 0.765), color='#94a3b8', lw=1.5, rad=0.05)
    arrow((c, 0.63), (0.43, 0.555), color=COLORS['green'], lw=1.8, rad=0.10)
    arrow((c, 0.63), (0.59, 0.555), color=COLORS['orange'], lw=1.8, rad=-0.12)
arrow((0.515, 0.765), (0.545, 0.765), color='#f59e0b')
arrow((0.435, 0.49), (0.49, 0.45), color=COLORS['green'])
arrow((0.59, 0.49), (0.54, 0.45), color=COLORS['orange'])
arrow((0.60, 0.71), (0.565, 0.45), color='#f59e0b', rad=-0.05)
arrow((0.60, 0.38), (0.74, 0.68), color=COLORS['blue'], lw=2.1)
arrow((0.835, 0.62), (0.835, 0.545), color=COLORS['blue'])
arrow((0.835, 0.45), (0.835, 0.395), color='#ef7298')
arrow((0.835, 0.30), (0.835, 0.24), color='#d7a029')
arrow((0.74, 0.175), (0.59, 0.335), color=COLORS['purple'], lw=1.9, ls=(0, (3, 3)), rad=0.12)
ax.text(0.67, 0.19, 'optimizer-side feedback', fontsize=9.2, color=COLORS['purple'], ha='center')

legend_x, legend_y, legend_w, legend_h = 0.035, 0.035, 0.60, 0.10
rounded_box(legend_x, legend_y, legend_w, legend_h, '', '#ffffff', '#cbd5e1', lw=1.1, radius=0.018)
legend_items = [
    ('Stable branch', COLORS['green'], 'agreement-preserving path for aligned modalities'),
    ('Conflict branch', COLORS['orange'], 'nonlinear route for expectancy-violation cues'),
    ('Dashed feedback', COLORS['purple'], 'MoMask suppresses momentum-conflicting coordinates'),
]
start_x = legend_x + 0.02
for idx, (name, col, desc) in enumerate(legend_items):
    y = legend_y + legend_h - 0.026 - idx * 0.028
    ax.add_line(Line2D([start_x, start_x + 0.03], [y, y], color=col, linewidth=2.2,
                       linestyle='--' if idx == 2 else '-'))
    ax.text(start_x + 0.036, y, f'{name}: {desc}', va='center', ha='left', fontsize=9.15, color=COLORS['border'])

rounded_box(0.66, 0.03, 0.31, 0.11,
            'Design rationale\nEV-Gate isolates modality contradiction before temporal modeling;\nMoMask stabilizes late-stage multimodal optimization with minimal overhead.',
            '#ffffff', '#cbd5e1', fontsize=9.8, lw=1.1, radius=0.018)

out_pdf = FIG_DIR / 'evgate_momask_architecture.pdf'
fig.savefig(out_pdf, bbox_inches='tight')
print(f'Wrote {out_pdf}')
