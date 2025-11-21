import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

def plot_correlation(x, y, xlabel, ylabel, title, output_path):
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.6)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20

    plt.figure(figsize=(8, 8))
    ax = sns.regplot(
        x=x, y=y,
        scatter_kws={'alpha': 0.6, 's': 70, 'color': '#1f77b4'},
        line_kws={'color': '#d62728', 'lw': 2}
    )

    rho_s, p_s = spearmanr(y, x)
    rho_p, p_p = pearsonr(y, x)

    def format_p(p):
        return "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}"

    stats_text = (
        f"Pearson r = {rho_p:.4f} ({format_p(p_p)})\n"
        f"Spearman ρ = {rho_s:.4f} ({format_p(p_s)})\n"
        f"N = {len(x)}"
    )

    plt.text(
        0.03, 0.97, stats_text, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='left',
        fontsize=20, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

    plt.title(title, pad=18, fontsize=22)
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    ax.tick_params(axis='both', labelsize=18, width=1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"已保存图像: {output_path}")

# 地市级对比
your_data_path = r"E:\Data\sdata_Score_CompleteYear_1016\Validation\prefecture\OurScore_2005_2016_Filtered.xlsx"
ref_data_path  = r"E:\Data\sdata_Score_CompleteYear_1016\Validation\地市级Score对比样本.xlsx"
output_dir     = r"E:\Data\sdata_Score_CompleteYear_1016\Validation\prefecture\Test5"
os.makedirs(output_dir, exist_ok=True)

df_y = pd.read_excel(your_data_path)[['City', 'Mean_2005_2016']].dropna()
df_r = pd.read_excel(ref_data_path)[['City', 'Average']].dropna()
merged = pd.merge(df_y, df_r, on='City', how='inner')

plot_correlation(
    x=merged['Average'],
    y=merged['Mean_2005_2016'],
    xlabel='Reference Study: Mean Score (2005–2016)',
    ylabel='Our Calculated Mean Score(2005–2016)',
    title='Pearson & Spearman Correlation (Prefecture Level, 2005–2016)',
    output_path=os.path.join(output_dir, "Correlation_Prefecture_TNR.png")
)

# 省级对比 
your_data_path = r"E:\Data\sdata_Score_CompleteYear_1016\Validation\province\OurScore_Province_Average_2000_2005_2010_2015.xlsx"
ref_data_path  = r"E:\Data\sdata_Score_CompleteYear_1016\Validation\Province_Comparison_Average.xlsx"
output_dir     = r"E:\Data\sdata_Score_CompleteYear_1016\Validation\province\Test5"
os.makedirs(output_dir, exist_ok=True)

df_y = pd.read_excel(your_data_path).rename(columns={'Province': 'Provinces'})
df_r = pd.read_excel(ref_data_path)
merged = pd.merge(df_y, df_r, on='Provinces', how='inner')

plot_correlation(
    x=merged['Ref_Mean_17_Score'],
    y=merged['Mean_17_Score'],
    xlabel='Reference Study: Mean Score (2000–2015)',
    ylabel='Our Calculated Mean Score (2000–2015)',
    title='Pearson & Spearman Correlation (Province Level, 2000–2015)',
    output_path=os.path.join(output_dir, "Correlation_Province_TNR.png")
)
