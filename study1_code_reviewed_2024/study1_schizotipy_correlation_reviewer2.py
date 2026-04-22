import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'browser'

# Load and preprocess
data = pd.read_csv('data_study1.csv')
data = data[['sub_idx', 'PDI', 'caps', 'asi']]
data = data.groupby('sub_idx').mean()

# Compute correlation (Spearman)
corr, pvals = stats.spearmanr(data[['PDI', 'caps', 'asi']])

# Convert to DataFrame
labels = ['PDI', 'CAPS', 'ASI']
corr_df = pd.DataFrame(corr, index=labels, columns=labels)
pval_df = pd.DataFrame(pvals, index=labels, columns=labels)

# Create annotation matrix with r and significance (skip diagonal text)
annot_text = []
for i in range(len(labels)):
    row = []
    for j in range(len(labels)):
        r = corr_df.iloc[i, j]
        p = pval_df.iloc[i, j]
        if i == j:  
            row.append("")   # keep color but no text
            continue
        sig = ""
        if p < 0.05:
            sig = "*"
        if p < 0.01:
            sig = "**"
        if p < 0.001:
            sig = "***"
        row.append(f"{r:.2f}{sig}")
    annot_text.append(row)

# Plot heatmap
fig = px.imshow(
    corr_df,
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    text_auto=False,
)
fig.add_annotation(
      y=1, x=1.26,
      text="<b>ρ<b>",
      showarrow=False,
      font=dict(size=22, color="blue", family="Courier New, monospace"),
      xref='paper'
      )    

# Add custom annotations
for i, row in enumerate(annot_text):
    for j, text in enumerate(row):
        if text != "":  # show only off-diagonal text
            fig.add_annotation(
                x=j, y=i,
                text=text,
                showarrow=False,
                font=dict(color="white", size=14)
            )

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=20,
    width=600, height=500,
    xaxis=dict(side="top"),
    title_x=0.5
)

fig.show()
fig.write_image("study1_fig_psych_correlation_reviewer2.png",scale=3)
