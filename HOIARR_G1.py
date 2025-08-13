# ==============================================================================
# IMPORT LIBRARIES
# ==============================================================================
import streamlit as st
import pandas as pd
from pathlib import Path
import base64
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import numpy as np
import re
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
try:
    import google.generativeai as genai
except ImportError:
    genai = None
try:
    from weasyprint import HTML
except ImportError:
    HTML = None

# ==============================================================================
# --- 1. การตั้งค่าและตัวแปรหลัก ---
# ==============================================================================
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
PERSISTED_DATA_PATH = DATA_DIR / "processed_incident_data.parquet"
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin1234")

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
LOGO_URL = "https://raw.githubusercontent.com/HOIARRTool/hoiarr/refs/heads/main/logo1.png"
st.set_page_config(page_title="HOIA-RR", page_icon=LOGO_URL, layout="wide")
st.markdown("""
<style>
/* CSS to style the text area inside the chat input */
[data-testid="stChatInput"] textarea { min-height: 80px; height: 100px; resize: vertical; background-color: transparent; border: none; }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
@media print {
    div[data-testid="stHorizontalBlock"] { display: grid !important; grid-template-columns: repeat(5, 1fr) !important; gap: 1.2rem !important; }
    .stDataFrame, .stTable { break-inside: avoid; page-break-inside: avoid; }
    thead, tr, th, td { break-inside: avoid !important; page-break-inside: avoid !important; }
    h1, h2, h3, h4, h5 { page-break-after: avoid; }
}
.custom-header { font-size: 20px; font-weight: bold; margin-top: 0px !important; padding-top: 0px !important; }
div[data-testid="stHorizontalBlock"] > div div[data-testid="stMetric"] { border: 1px solid #ddd; padding: 0.75rem; border-radius: 0.5rem; height: 100%; display: flex; flex-direction: column; justify-content: center; }
div[data-testid="stHorizontalBlock"] > div:nth-child(1) div[data-testid="stMetric"] { background-color: #e6fffa; border-color: #b2f5ea; }
div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stMetric"] { background-color: #fff3e0; border-color: #ffe0b2; }
div[data-testid="stHorizontalBlock"] > div:nth-child(3) div[data-testid="stMetric"] { background-color: #fce4ec; border-color: #f8bbd0; }
div[data-testid="stHorizontalBlock"] > div:nth-child(4) div[data-testid="stMetric"] { background-color: #e3f2fd; border-color: #bbdefb; }
div[data-testid="stHorizontalBlock"] > div div[data-testid="stMetric"] [data-testid="stMetricLabel"] > div,
div[data-testid="stHorizontalBlock"] > div div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stHorizontalBlock"] > div div[data-testid="stMetric"] [data-testid="stMetricDelta"] { color: #262730 !important; }
div[data-testid="stMetric"] [data-testid="stMetricLabel"] > div { font-size: 0.8rem !important; line-height: 1.2 !important; white-space: normal !important; overflow-wrap: break-word !important; word-break: break-word; display: block !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }
div[data-testid="stHorizontalBlock"] > div .stExpander { border: none !important; box-shadow: none !important; padding: 0 !important; margin-top: 0.5rem; }
div[data-testid="stHorizontalBlock"] > div .stExpander header { padding: 0.25rem 0.5rem !important; font-size: 0.75rem !important; border-radius: 0.25rem; }
div[data-testid="stHorizontalBlock"] > div .stExpander div[data-testid="stExpanderDetails"] { max-height: 200px; overflow-y: auto; }
.stDataFrame table td, .stDataFrame table th { color: black !important; font-size: 0.9rem !important; }
.stDataFrame table th { font-weight: bold !important; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================
def load_data(uploaded_file):
    try:
        return pd.read_excel(uploaded_file, engine='openpyxl', keep_default_na=False)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ Excel: {e}")
        return pd.DataFrame()


@st.cache_data
def calculate_persistence_risk_score(_df: pd.DataFrame, total_months: int):
    risk_level_map_to_score = {"51": 21, "52": 22, "53": 23, "54": 24, "55": 25, "41": 16, "42": 17, "43": 18, "44": 19,
                               "45": 20, "31": 11, "32": 12, "33": 13, "34": 14, "35": 15, "21": 6, "22": 7, "23": 8,
                               "24": 9, "25": 10, "11": 1, "12": 2, "13": 3, "14": 4, "15": 5}
    if _df.empty or 'รหัส' not in _df.columns or 'Risk Level' not in _df.columns: return pd.DataFrame()
    analysis_df = _df[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Risk Level']].copy()
    analysis_df['Ordinal_Risk_Score'] = analysis_df['Risk Level'].astype(str).map(risk_level_map_to_score)
    analysis_df.dropna(subset=['Ordinal_Risk_Score'], inplace=True)
    if analysis_df.empty: return pd.DataFrame()
    persistence_metrics = analysis_df.groupby('รหัส').agg(
        Average_Ordinal_Risk_Score=('Ordinal_Risk_Score', 'mean'),
        Total_Occurrences=('รหัส', 'size')
    ).reset_index()
    total_months = max(1, total_months)
    persistence_metrics['Incident_Rate_Per_Month'] = persistence_metrics['Total_Occurrences'] / total_months
    max_rate = max(1, persistence_metrics['Incident_Rate_Per_Month'].max())
    persistence_metrics['Frequency_Score'] = persistence_metrics['Incident_Rate_Per_Month'] / max_rate
    persistence_metrics['Avg_Severity_Score'] = persistence_metrics['Average_Ordinal_Risk_Score'] / 25.0
    persistence_metrics['Persistence_Risk_Score'] = persistence_metrics['Frequency_Score'] + persistence_metrics[
        'Avg_Severity_Score']
    incident_names = _df[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง']].drop_duplicates()
    final_df = pd.merge(persistence_metrics, incident_names, on='รหัส', how='left')
    return final_df.sort_values(by='Persistence_Risk_Score', ascending=False)


@st.cache_data
def calculate_frequency_trend_poisson(_df: pd.DataFrame):
    if _df.empty or 'รหัส' not in _df.columns or 'Occurrence Date' not in _df.columns: return pd.DataFrame()
    analysis_df = _df[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Occurrence Date']].copy()
    analysis_df.dropna(subset=['Occurrence Date'], inplace=True)
    if analysis_df.empty: return pd.DataFrame()
    analysis_df['YearMonth'] = pd.to_datetime(analysis_df['Occurrence Date']).dt.to_period('M')
    full_date_range = pd.period_range(start=analysis_df['YearMonth'].min(), end=analysis_df['YearMonth'].max(),
                                      freq='M')
    results = []
    for code in analysis_df['รหัส'].unique():
        incident_subset = analysis_df[analysis_df['รหัส'] == code]
        if len(incident_subset) < 3 or len(incident_subset.groupby('YearMonth')) < 2: continue
        monthly_counts = incident_subset.groupby('YearMonth').size().reindex(full_date_range, fill_value=0)
        y = monthly_counts.values
        X = sm.add_constant(np.arange(len(monthly_counts)))
        try:
            model = sm.Poisson(y, X).fit(disp=0)
            results.append({
                'รหัส': code, 'Poisson_Trend_Slope': model.params[1],
                'Total_Occurrences': y.sum(), 'Months_Observed': len(y)
            })
        except Exception:
            continue
    if not results: return pd.DataFrame()
    final_df = pd.DataFrame(results)
    incident_names = _df[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง']].drop_duplicates()
    final_df = pd.merge(final_df, incident_names, on='รหัส', how='left')
    return final_df.sort_values(by='Poisson_Trend_Slope', ascending=False)


def create_poisson_trend_plot(df, selected_code_for_plot, display_df=None, show_ci=True):
    # เตรียมช่วงเดือนเต็มของทั้งชุดข้อมูล (แกน x)
    full_date_range_for_plot = pd.period_range(
        start=pd.to_datetime(df['Occurrence Date']).dt.to_period('M').min(),
        end=pd.to_datetime(df['Occurrence Date']).dt.to_period('M').max(),
        freq='M'
    )

    # นับจำนวนครั้งต่อเดือนของรหัสที่เลือก
    subset = df[df['รหัส'] == selected_code_for_plot].copy()
    subset['YearMonth'] = pd.to_datetime(subset['Occurrence Date']).dt.to_period('M')
    counts = subset.groupby('YearMonth').size().reindex(full_date_range_for_plot, fill_value=0)

    # เตรียมข้อมูลสำหรับโมเดล Poisson: y = counts, X = [const, time]
    y = counts.values.astype(float)
    t = np.arange(len(counts), dtype=float)
    X = sm.add_constant(t)

    # fit Poisson (log link): mu_t = exp(beta0 + beta1 * t)
    beta0 = beta1 = None
    mu_hat = None
    mu_lo = mu_hi = None

    if len(y) >= 2 and y.sum() > 0:
        try:
            model = sm.Poisson(y, X).fit(disp=0)
            beta0, beta1 = model.params

            # คาดการณ์ค่าเฉลี่ยที่คาดหวังต่อเดือนจากโมเดล
            eta = beta0 + beta1 * t                     # linear predictor
            mu_hat = np.exp(eta)                        # expected counts

            if show_ci:
                # 95% CI บนสเกลลอก -> แปลงกลับเป็นสเกลนับ
                cov = model.cov_params()
                design = np.column_stack([np.ones_like(t), t])
                se_eta = np.sqrt(np.einsum('ij,jk,ik->i', design, cov, design))
                eta_lo = eta - 1.96 * se_eta
                eta_hi = eta + 1.96 * se_eta
                mu_lo = np.exp(eta_lo)
                mu_hi = np.exp(eta_hi)
        except Exception as e:
            st.warning(f"คำนวณเส้นแนวโน้ม Poisson ไม่สำเร็จ: {e}")

    # วาดกราฟ
    fig_plot = go.Figure()

    # แท่งจำนวนครั้งจริงต่อเดือน
    fig_plot.add_trace(go.Bar(
        x=counts.index.strftime('%Y-%m'),
        y=y,
        name='จำนวนครั้งที่เกิดจริง',
        marker=dict(color='#AED6F1', cornerradius=8)
    ))

    # เส้นแนวโน้มจาก Poisson + ช่วงความเชื่อมั่น (ถ้ามี)
    if mu_hat is not None:
        fig_plot.add_trace(go.Scatter(
            x=counts.index.strftime('%Y-%m'),
            y=mu_hat,
            mode='lines',
            name='แนวโน้มคาดหมาย (Poisson)',
            line=dict(width=2)
        ))

        if show_ci and (mu_lo is not None) and (mu_hi is not None):
            # วาด band 95% CI (บนก่อนล่าง แล้ว fill='tonexty')
            fig_plot.add_trace(go.Scatter(
                x=counts.index.strftime('%Y-%m'),
                y=mu_hi,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig_plot.add_trace(go.Scatter(
                x=counts.index.strftime('%Y-%m'),
                y=mu_lo,
                mode='lines',
                fill='tonexty',
                name='95% CI',
                line=dict(width=0),
                fillcolor='rgba(0,0,0,0.08)'
            ))

    fig_plot.update_layout(
        title=f'การกระจายตัวของอุบัติการณ์: {selected_code_for_plot}',
        xaxis_title='เดือน-ปี',
        yaxis_title='จำนวนครั้งที่เกิด',
        barmode='overlay',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)')
    )

    # ข้อความประกอบ: ใช้พารามิเตอร์จาก Poisson (ถ้าคำนวณได้) ไม่ต้องพึ่ง display_df
    if beta1 is not None:
        factor = float(np.exp(beta1))
        annot_text = (f"<b>Poisson slope: {beta1:.4f}</b><br>"
                      f"อัตราเปลี่ยนแปลง: x{factor:.2f} ต่อเดือน")
    else:
        annot_text = "<b>Poisson slope: N/A</b><br>อัตราเปลี่ยนแปลง: N/A"

    fig_plot.add_annotation(
        x=0.5, y=0.98,
        xref="paper", yref="paper",
        text=annot_text,
        showarrow=False,
        font=dict(size=12, color="black"),
        align="center",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        bgcolor="rgba(255, 255, 224, 0.7)"
    )
    return fig_plot



def create_goal_summary_table(data_df_goal, goal_category_name_param,
                              e_up_non_numeric_levels_param, e_up_numeric_levels_param=None,
                              is_org_safety_table=False):
    goal_category_name_param = str(goal_category_name_param).strip()
    if 'หมวด' not in data_df_goal.columns:
        return pd.DataFrame()
    df_filtered_by_goal_cat = data_df_goal[
        data_df_goal['หมวด'].astype(str).str.strip() == goal_category_name_param].copy()
    if df_filtered_by_goal_cat.empty: return pd.DataFrame()
    if 'Incident Type' not in df_filtered_by_goal_cat.columns or 'Impact' not in df_filtered_by_goal_cat.columns: return pd.DataFrame()
    try:
        pvt_table_goal = pd.crosstab(df_filtered_by_goal_cat['Incident Type'],
                                     df_filtered_by_goal_cat['Impact'].astype(str).str.strip(), margins=True,
                                     margins_name='รวมทั้งหมด')
    except Exception:
        return pd.DataFrame()
    if 'รวมทั้งหมด' in pvt_table_goal.index: pvt_table_goal = pvt_table_goal.drop(index='รวมทั้งหมด')
    if pvt_table_goal.empty: return pd.DataFrame()
    if 'รวมทั้งหมด' not in pvt_table_goal.columns: pvt_table_goal['รวมทั้งหมด'] = pvt_table_goal.sum(axis=1)
    all_impact_columns_goal = [str(col).strip() for col in pvt_table_goal.columns if col != 'รวมทั้งหมด']
    e_up_non_numeric_levels_param_stripped = [str(level).strip() for level in e_up_non_numeric_levels_param]
    e_up_numeric_levels_param_stripped = [str(level).strip() for level in
                                          e_up_numeric_levels_param] if e_up_numeric_levels_param else []
    e_up_columns_goal = [col for col in all_impact_columns_goal if
                         col not in e_up_non_numeric_levels_param_stripped and (
                                 not e_up_numeric_levels_param_stripped or col not in e_up_numeric_levels_param_stripped)]
    report_data_goal = []
    for incident_type_goal, row_data_goal in pvt_table_goal.iterrows():
        total_e_up_count_goal = sum(row_data_goal[col] for col in e_up_columns_goal if
                                    col in row_data_goal.index and pd.notna(row_data_goal[col]))
        total_all_impacts_goal = row_data_goal['รวมทั้งหมด'] if 'รวมทั้งหมด' in row_data_goal and pd.notna(
            row_data_goal['รวมทั้งหมด']) else 0
        percent_e_up_goal = (total_e_up_count_goal / total_all_impacts_goal * 100) if total_all_impacts_goal > 0 else 0
        report_data_goal.append(
            {'Incident Type': incident_type_goal, 'รวม E-up': total_e_up_count_goal, 'ร้อยละ E-up': percent_e_up_goal})
    report_df_goal = pd.DataFrame(report_data_goal)
    if report_df_goal.empty:
        merged_report_table_goal = pvt_table_goal.reset_index()
        merged_report_table_goal['รวม E-up'] = 0
        merged_report_table_goal['ร้อยละ E-up'] = 0.0
    else:
        merged_report_table_goal = pd.merge(pvt_table_goal.reset_index(), report_df_goal, on='Incident Type',
                                            how='outer')
    if 'รวม E-up' not in merged_report_table_goal.columns:
        merged_report_table_goal['รวม E-up'] = 0
    else:
        merged_report_table_goal['รวม E-up'].fillna(0, inplace=True)
    if 'ร้อยละ E-up' not in merged_report_table_goal.columns:
        merged_report_table_goal['ร้อยละ E-up'] = 0.0
    else:
        merged_report_table_goal['ร้อยละ E-up'].fillna(0.0, inplace=True)
    cols_to_drop_from_display_goal = [col for col in e_up_non_numeric_levels_param_stripped if
                                      col in merged_report_table_goal.columns]
    if e_up_numeric_levels_param_stripped: cols_to_drop_from_display_goal.extend(
        [col for col in e_up_numeric_levels_param_stripped if col in merged_report_table_goal.columns])
    merged_report_table_goal = merged_report_table_goal.drop(columns=cols_to_drop_from_display_goal, errors='ignore')
    total_col_original_name, e_up_col_name, percent_e_up_col_name = 'รวมทั้งหมด', 'รวม E-up', 'ร้อยละ E-up'
    if is_org_safety_table:
        total_col_display_name, e_up_col_display_name, percent_e_up_display_name = 'รวม 1-5', 'รวม 3-5', 'ร้อยละ 3-5'
        merged_report_table_goal.rename(
            columns={total_col_original_name: total_col_display_name, e_up_col_name: e_up_col_display_name,
                     percent_e_up_col_name: percent_e_up_display_name}, inplace=True, errors='ignore')
    else:
        total_col_display_name, e_up_col_display_name, percent_e_up_display_name = 'รวม A-I', e_up_col_name, percent_e_up_col_name
        merged_report_table_goal.rename(columns={total_col_original_name: total_col_display_name}, inplace=True,
                                        errors='ignore')
    merged_report_table_goal['Incident Type Name'] = merged_report_table_goal['Incident Type'].map(type_name).fillna(
        merged_report_table_goal['Incident Type'])
    final_columns_goal_order = ['Incident Type Name'] + [col for col in e_up_columns_goal if
                                                         col in merged_report_table_goal.columns] + [
                                   e_up_col_display_name, total_col_display_name, percent_e_up_display_name]
    final_columns_present_goal = [col for col in final_columns_goal_order if col in merged_report_table_goal.columns]
    merged_report_table_goal = merged_report_table_goal[final_columns_present_goal]
    if percent_e_up_display_name in merged_report_table_goal.columns and pd.api.types.is_numeric_dtype(
            merged_report_table_goal[percent_e_up_display_name]):
        try:
            merged_report_table_goal[percent_e_up_display_name] = merged_report_table_goal[
                percent_e_up_display_name].astype(float).map('{:.2f}%'.format)
        except ValueError:
            pass
    return merged_report_table_goal.set_index('Incident Type Name')


def create_severity_table(input_df, row_column_name, table_title, specific_row_order=None):
    if not isinstance(input_df,
                      pd.DataFrame) or input_df.empty or row_column_name not in input_df.columns or 'Impact Level' not in input_df.columns: return None
    temp_df = input_df.copy()
    temp_df['Impact Level'] = temp_df['Impact Level'].astype(str).str.strip().replace('N/A', 'ไม่ระบุ')
    if temp_df[row_column_name].dropna().empty: return None
    try:
        severity_crosstab = pd.crosstab(temp_df[row_column_name].astype(str).str.strip(), temp_df['Impact Level'])
    except Exception:
        return None
    impact_level_map_cols = {'1': 'A-B (1)', '2': 'C-D (2)', '3': 'E-F (3)', '4': 'G-H (4)', '5': 'I (5)',
                             'ไม่ระบุ': 'ไม่ระบุ LV'}
    desired_cols_ordered_keys = ['1', '2', '3', '4', '5', 'ไม่ระบุ']
    for col_key in desired_cols_ordered_keys:
        if col_key not in severity_crosstab.columns: severity_crosstab[col_key] = 0
    present_ordered_keys = [key for key in desired_cols_ordered_keys if key in severity_crosstab.columns]
    if not present_ordered_keys: return None
    severity_crosstab = severity_crosstab[present_ordered_keys].rename(columns=impact_level_map_cols)
    final_display_cols_renamed = [impact_level_map_cols[key] for key in present_ordered_keys if
                                  key in impact_level_map_cols]
    if not final_display_cols_renamed: return None
    severity_crosstab['รวมทุกระดับ'] = severity_crosstab[
        [col for col in final_display_cols_renamed if col in severity_crosstab.columns]].sum(axis=1)
    if specific_row_order:
        severity_crosstab = severity_crosstab.reindex([str(i) for i in specific_row_order]).fillna(0).astype(int)
    else:
        severity_crosstab = severity_crosstab[severity_crosstab['รวมทุกระดับ'] > 0]
    if severity_crosstab.empty: return None
    st.markdown(f"##### {table_title}")
    display_column_order_from_map = [impact_level_map_cols.get(key) for key in desired_cols_ordered_keys]
    display_column_order_present = [col for col in display_column_order_from_map if
                                    col in severity_crosstab.columns] + (
                                       ['รวมทุกระดับ'] if 'รวมทุกระดับ' in severity_crosstab.columns else [])
    st.dataframe(
        severity_crosstab[[col for col in display_column_order_present if col in severity_crosstab.columns]].astype(
            int), use_container_width=True)
    return severity_crosstab


def create_psg9_summary_table(input_df):
    if not isinstance(input_df,
                      pd.DataFrame) or 'หมวดหมู่มาตรฐานสำคัญ' not in input_df.columns or 'Impact' not in input_df.columns: return None
    psg9_placeholders = ["ไม่จัดอยู่ใน PSG9 Catalog", "ไม่สามารถระบุ (Merge PSG9 ล้มเหลว)",
                         "ไม่สามารถระบุ (เช็คคอลัมน์ใน PSG9code.xlsx)",
                         "ไม่สามารถระบุ (PSG9code.xlsx ไม่ได้โหลด/ว่างเปล่า)",
                         "ไม่สามารถระบุ (Merge PSG9 ล้มเหลว - rename)", "ไม่สามารถระบุ (Merge PSG9 ล้มเหลว - no col)",
                         "ไม่สามารถระบุ (PSG9code.xlsx ไม่ได้โหลด/ข้อมูลไม่ครบถ้วน)"]
    df_filtered = input_df[
        ~input_df['หมวดหมู่มาตรฐานสำคัญ'].isin(psg9_placeholders) & input_df['หมวดหมู่มาตรฐานสำคัญ'].notna()].copy()
    if df_filtered.empty: return pd.DataFrame()
    try:
        summary_table = pd.crosstab(df_filtered['หมวดหมู่มาตรฐานสำคัญ'], df_filtered['Impact'], margins=True,
                                    margins_name='รวม A-I')
    except Exception:
        return pd.DataFrame()
    if 'รวม A-I' in summary_table.index: summary_table = summary_table.drop(index='รวม A-I')
    if summary_table.empty: return pd.DataFrame()
    all_impacts, e_up_impacts = list('ABCDEFGHI'), list('EFGHI')
    for impact_col in all_impacts:
        if impact_col not in summary_table.columns: summary_table[impact_col] = 0
    if 'รวม A-I' not in summary_table.columns: summary_table['รวม A-I'] = summary_table[
        [col for col in all_impacts if col in summary_table.columns]].sum(axis=1)
    summary_table['รวม E-up'] = summary_table[[col for col in e_up_impacts if col in summary_table.columns]].sum(axis=1)
    summary_table['ร้อยละ E-up'] = (summary_table['รวม E-up'] / summary_table['รวม A-I'] * 100).fillna(0)
    psg_order = [PSG9_label_dict[i] for i in sorted(PSG9_label_dict.keys())]
    summary_table = summary_table.reindex(psg_order).fillna(0)
    display_cols_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'รวม E-up', 'รวม A-I', 'ร้อยละ E-up']
    final_table = summary_table[[col for col in display_cols_order if col in summary_table.columns]].copy()
    for col in final_table.columns:
        if col != 'ร้อยละ E-up': final_table[col] = final_table[col].astype(int)
    final_table['ร้อยละ E-up'] = final_table['ร้อยละ E-up'].map('{:.2f}%'.format)
    return final_table


def get_text_color_for_bg(hex_color):
    try:
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6: return '#000000'
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
        return '#FFFFFF' if luminance < 0.5 else '#000000'
    except ValueError:
        return '#000000'

def prioritize_incidents_nb_logit_v2(_df: pd.DataFrame,
                                     horizon: int = 3,
                                     min_months: int = 4,
                                     min_total: int = 5,
                                     w_expected_severe: float = 0.7,
                                     w_freq_growth: float = 0.2,
                                     w_sev_growth: float = 0.1,
                                     alpha_floor: float = 1e-8):
    """
    เวอร์ชันแก้บั๊ก: คำนวณ Priority แบบ NB+Logit และทำสกอร์หลังรวม DataFrame (ไม่เรียก fillna บนสเกลาร์)
    """
    req = ['รหัส', 'Occurrence Date', 'Impact Level']
    if any(c not in _df.columns for c in req):
        return pd.DataFrame()

    d = _df.copy()
    d = d[pd.to_datetime(d['Occurrence Date'], errors='coerce').notna()]
    if d.empty:
        return pd.DataFrame()
    d['YearMonth'] = pd.to_datetime(d['Occurrence Date']).dt.to_period('M')

    full_range = pd.period_range(d['YearMonth'].min(), d['YearMonth'].max(), freq='M')

    rows = []
    for code, sub in d.groupby('รหัส'):
        if len(sub) < min_total:
            continue

        # ===== ความถี่รายเดือน (NB) =====
        counts = sub.groupby('YearMonth').size().reindex(full_range, fill_value=0).astype(float)
        y = counts.values
        n_months = len(counts)
        t = np.arange(n_months, dtype=float)
        X = sm.add_constant(t)

        nb_beta0 = np.nan; nb_beta1 = np.nan; nb_p = np.nan; nb_factor = np.nan; alpha_hat = np.nan
        mu_future = np.zeros(horizon, dtype=float)

        if n_months >= min_months and y.sum() > 0:
            try:
                pois = sm.GLM(y, X, family=sm.families.Poisson()).fit()
                mu = pois.fittedvalues
                num = float(((y - mu)**2 - y).sum())
                den = float(max((mu**2).sum(), 1e-12))
                alpha_hat = max(num/den, alpha_floor)

                nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha_hat)).fit()
                nb_beta0, nb_beta1 = float(nb.params[0]), float(nb.params[1])
                nb_p = float(nb.pvalues[1])
                nb_factor = float(np.exp(nb_beta1))

                t_future = np.arange(n_months, n_months + horizon, dtype=float)
                eta_future = nb_beta0 + nb_beta1 * t_future
                mu_future = np.exp(eta_future)
            except Exception:
                pass

        # ===== สัดส่วนเหตุรุนแรง LV3-5 (Logit) =====
        sub['__sev__'] = sub['Impact Level'].astype(str).isin(['3','4','5']).astype(int)
        sev_counts = sub.groupby('YearMonth')['__sev__'].sum().reindex(full_range, fill_value=0).astype(float)
        n_counts   = sub.groupby('YearMonth').size().reindex(full_range, fill_value=0).astype(float)
        mask = n_counts > 0

        lg_beta0 = np.nan; lg_beta1 = np.nan; lg_p = np.nan; sev_or = np.nan
        p_future = np.full(horizon, np.nan, dtype=float)

        if mask.sum() >= min_months and sev_counts[mask].sum() > 0 and (sev_counts[mask] < n_counts[mask]).any():
            try:
                endog = (sev_counts[mask] / n_counts[mask]).values
                Xt = sm.add_constant(np.arange(n_months)[mask].astype(float))
                logit = sm.GLM(endog, Xt, family=sm.families.Binomial(), freq_weights=n_counts[mask].values).fit()
                lg_beta0, lg_beta1 = float(logit.params[0]), float(logit.params[1])
                lg_p = float(logit.pvalues[1])
                sev_or = float(np.exp(lg_beta1))

                t_future_all = np.arange(n_months, n_months + horizon, dtype=float)
                lin = lg_beta0 + lg_beta1 * t_future_all
                p_future = 1/(1 + np.exp(-lin))
                p_future = np.clip(p_future, 1e-6, 1-1e-6)
            except Exception:
                pass
        else:
            base_p = (sev_counts.sum()/n_counts.sum()) if n_counts.sum() > 0 else 0.0
            p_future = np.full(horizon, base_p, dtype=float)

        expected_all_nextH = float(np.nansum(mu_future))
        expected_sev_nextH = float(np.nansum(mu_future * p_future))

        freq_rising = (nb_beta1 > 0) and (pd.notna(nb_p) and nb_p < 0.05)
        sev_rising  = (lg_beta1 > 0) and (pd.notna(lg_p) and lg_p < 0.05)

        rows.append({
            'รหัส': code,
            'ชื่ออุบัติการณ์ความเสี่ยง': sub['ชื่ออุบัติการณ์ความเสี่ยง'].iloc[0] if 'ชื่ออุบัติการณ์ความเสี่ยง' in sub else '',
            'Months_Observed': int(n_months),
            'Total_Occurrences': int(y.sum()),
            'NB_alpha_hat': alpha_hat,
            'Freq_NB_Slope': nb_beta1,
            'Freq_p_value': nb_p,
            'Freq_Factor_per_month': nb_factor,        # เก็บค่า “ดิบ” มาก่อน
            'Severity_Logit_Slope': lg_beta1,
            'Severity_p_value': lg_p,
            'Severe_OR_per_month': sev_or,             # เก็บค่า “ดิบ” มาก่อน
            'Expected_All_nextH': expected_all_nextH,
            'Expected_Severe_nextH': expected_sev_nextH,
            'Freq_Rising': freq_rising,
            'Sev_Rising': sev_rising
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    # ---------- ช่วยฟังก์ชัน: รับอาเรย์/Series เท่านั้น (ไม่รองรับสเกลาร์) ----------
    def _safe_log_pos_arr(x):
        arr = np.asarray(x, dtype=float)
        arr[~np.isfinite(arr)] = 1.0
        arr = np.clip(arr, 1e-12, None)
        return np.log(arr)

    def _norm01_pos_arr(x):
        arr = np.asarray(x, dtype=float)
        arr[~np.isfinite(arr)] = 0.0
        arr = np.clip(arr, 0, None)
        rng = arr.max() - arr.min()
        return (arr - arr.min())/rng if rng > 0 else np.zeros_like(arr)

    # ---------- เตรียมคอลัมน์ก่อนทำสกอร์ ----------
    out['Freq_Factor_per_month'] = pd.to_numeric(out['Freq_Factor_per_month'], errors='coerce').fillna(1.0)
    out['Severe_OR_per_month']   = pd.to_numeric(out['Severe_OR_per_month'], errors='coerce').fillna(1.0)
    out['Expected_Severe_nextH'] = pd.to_numeric(out['Expected_Severe_nextH'], errors='coerce').fillna(0.0)

    # ---------- คำนวณสกอร์แบบเวคเตอร์ (ไม่ยุ่งกับสเกลาร์) ----------
    out['Score_expected_severe'] = _norm01_pos_arr(out['Expected_Severe_nextH'].values)
    out['Score_freq_growth']     = _norm01_pos_arr(_safe_log_pos_arr(out['Freq_Factor_per_month'].values))
    out['Score_sev_growth']      = _norm01_pos_arr(_safe_log_pos_arr(out['Severe_OR_per_month'].values))

    # bonus ถ้า 2 แนวโน้มมีนัย
    bonus = np.where((out['Freq_Rising']) & (out['Sev_Rising']), 0.05, 0.0)

    out['Priority_Score'] = (
        w_expected_severe * out['Score_expected_severe'] +
        w_freq_growth     * out['Score_freq_growth'] +
        w_sev_growth      * out['Score_sev_growth'] +
        bonus
    )

    cols = [
        'รหัส','ชื่ออุบัติการณ์ความเสี่ยง','Months_Observed','Total_Occurrences',
        'Expected_All_nextH','Expected_Severe_nextH',
        'Freq_Factor_per_month','Freq_p_value',
        'Severe_OR_per_month','Severity_p_value',
        'NB_alpha_hat','Priority_Score',
        'Freq_Rising','Sev_Rising'
    ]
    out = out[cols].sort_values('Priority_Score', ascending=False).reset_index(drop=True)
    return out

def generate_executive_summary_pdf(df_filtered, metrics_data, total_month, df_freq, min_date_str, max_date_str):
    """
    สร้างไฟล์ PDF จากข้อมูลสรุปสำหรับผู้บริหาร (เวอร์ชันสมบูรณ์ 8 หัวข้อ)
    """
    if HTML is None:
        st.error("กรุณาติดตั้ง WeasyPrint เพื่อใช้งานฟังก์ชันนี้: pip install weasyprint")
        return None

    # --- เตรียมข้อมูลสำหรับแต่ละส่วน ---

    # 2. Risk Matrix & Top 10
    impact_level_keys = ['5', '4', '3', '2', '1']
    freq_level_keys = ['1', '2', '3', '4', '5']
    matrix_df = df_filtered[
        df_filtered['Impact Level'].isin(impact_level_keys) & df_filtered['Frequency Level'].isin(freq_level_keys)]
    matrix_data_html = "ไม่มีข้อมูล"
    if not matrix_df.empty:
        matrix_data = pd.crosstab(matrix_df['Impact Level'], matrix_df['Frequency Level'])
        matrix_data = matrix_data.reindex(index=impact_level_keys, columns=freq_level_keys, fill_value=0)
        impact_labels = {'5': "5 (Extreme)", '4': "4 (Major)", '3': "3 (Moderate)", '2': "2 (Minor)",
                         '1': "1 (Insignificant)"}
        freq_labels = {'1': "F1", '2': "F2", '3': "F3", '4': "F4", '5': "F5"}
        matrix_data_html = matrix_data.rename(index=impact_labels, columns=freq_labels).to_html(
            classes="styled-table",
            table_id="risk-matrix-table"
        )

        frequency_legend_html = """
        <div class="legend">
            <h4>หมายเหตุ คำอธิบายความถี่ (Frequency)</h4>
            <p>
                <b>F1</b> = Remote (น้อยกว่า 2 ครั้ง/เดือน)<br>
                <b>F2</b> = Uncommon (2-3 ครั้ง/เดือน)<br>
                <b>F3</b> = Occasional (4-6 ครั้ง/เดือน)<br>
                <b>F4</b> = Probable (7-29 ครั้ง/เดือน)<br>
                <b>F5</b> = Frequent (มากกว่าหรือเท่ากับ 30 ครั้ง/เดือน)
            </p>
        </div>
        """

    top10_df = df_freq.nlargest(10, 'count').copy()
    incident_names = df_filtered[['Incident', 'ชื่ออุบัติการณ์ความเสี่ยง']].drop_duplicates()
    top10_df = pd.merge(top10_df, incident_names, on='Incident', how='left')
    top10_html = top10_df[['Incident', 'count']].to_html(
        classes="styled-table",
        index=False,
        table_id="top10-table"
    )
    # 3. Sentinel Events
    sentinel_html = "<p>ไม่พบ Sentinel Events ในช่วงเวลานี้</p>"
    if 'Sentinel code for check' in df_filtered.columns:
        sentinel_df = df_filtered[df_filtered['Sentinel code for check'].isin(sentinel_composite_keys)]
        if not sentinel_df.empty:
            sentinel_html = sentinel_df[['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด']].to_html(
                classes="styled-table",
                index=False,
                table_id="sentinel-table"
            )

    psg9_summary_table = create_psg9_summary_table(df_filtered)
    psg9_html = "<p>ไม่พบข้อมูล PSG9 ในช่วงเวลานี้</p>"
    if psg9_summary_table is not None and not psg9_summary_table.empty:
        # to_html จะสร้าง index เป็นคอลัมน์แรกโดยอัตโนมัติ
        psg9_html = psg9_summary_table.to_html(
            classes="styled-table",
            table_id="psg9-table"
        )
        #เตรียมข้อมูลสำหรับตาราง "สรุปอุบัติการณ์ตามเป้าหมาย"
        goal_definitions = {
            "Patient Safety/ Common Clinical Risk": "P:Patient Safety Goals หรือ Common Clinical Risk Incident",
            "Specific Clinical Risk": "S:Specific Clinical Risk Incident",
            "Personnel Safety": "P:Personnel Safety Goals",
            "Organization Safety": "O:Organization Safety Goals"
        }
        safety_goals_html_parts = []
        for display_name, cat_name in goal_definitions.items():
            is_org_safety = (display_name == "Organization Safety")
            summary_table = create_goal_summary_table(
                df_filtered, cat_name,
                e_up_non_numeric_levels_param=[] if is_org_safety else ['A', 'B', 'C', 'D'],
                e_up_numeric_levels_param=['1', '2'] if is_org_safety else None,
                is_org_safety_table=is_org_safety
            )
            if summary_table is not None and not summary_table.empty:
                safety_goals_html_parts.append(f"<h3>{display_name}</h3>")
                # ใช้ class 'auto-width-table' สำหรับตารางที่มีคอลัมน์ไม่แน่นอน
                safety_goals_html_parts.append(summary_table.to_html(classes="styled-table auto-width-table"))

        safety_goals_html = "".join(
            safety_goals_html_parts) if safety_goals_html_parts else "<p>ไม่มีข้อมูลสำหรับสรุปตามเป้าหมาย</p>"

    #เตรียมข้อมูลสำหรับตาราง "อุบัติการณ์ที่ยังไม่แก้ไข"
    unresolved_severe_df = df_filtered[
        df_filtered['Impact Level'].isin(['3', '4', '5']) &
        df_filtered['Resulting Actions'].astype(str).isin(['None', '', 'nan'])
        ]
    unresolved_severe_html = "<p>ไม่พบอุบัติการณ์รุนแรงที่ยังไม่ถูกแก้ไขในช่วงเวลานี้</p>"
    if not unresolved_severe_df.empty:
        # เลือกคอลัมน์และจัดรูปแบบวันที่
        df_for_pdf = unresolved_severe_df[['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด']].copy()
        df_for_pdf['Occurrence Date'] = df_for_pdf['Occurrence Date'].dt.strftime('%d/%m/%Y')

        unresolved_severe_html = df_for_pdf.to_html(
            classes="styled-table",
            index=False,
            table_id="unresolved-table"
        )

    #เตรียมข้อมูลสำหรับ "ความเสี่ยงเรื้อรัง"
    persistence_html = "<p>ไม่มีข้อมูลเพียงพอสำหรับวิเคราะห์ความเสี่ยงเรื้อรัง</p>"
    persistence_df = calculate_persistence_risk_score(df_filtered, total_month)
    if not persistence_df.empty:
        top_persistence = persistence_df.head(5)
        p_list_items = ["<ol>"]
        for index, row in top_persistence.iterrows():
            item_text = (
                f"<li><b>{row['รหัส']}: {row['ชื่ออุบัติการณ์ความเสี่ยง']}</b><br>"
                f"<small><i>ดัชนีความเรื้อรัง: {row['Persistence_Risk_Score']:.2f} "
                f"(เกิดเฉลี่ย: {row['Incident_Rate_Per_Month']:.2f} ครั้ง/เดือน)</i></small></li>"
            )
            p_list_items.append(item_text)
        p_list_items.append("</ol>")
        persistence_html = "".join(p_list_items)

    #เตรียมข้อมูลสำหรับ "Early Warning"
    early_warning_html = "<p>ไม่มีข้อมูลเพียงพอสำหรับวิเคราะห์ Early Warning</p>"
    if 'prioritize_incidents_nb_logit_v2' in globals():
        ew_df = prioritize_incidents_nb_logit_v2(df_filtered, horizon=3, min_months=4, min_total=5)
        if not ew_df.empty:
            top_ew = ew_df.head(5)
            # สร้าง HTML List จากข้อมูล Top 5
            ew_list_items = ["<ol>"] # <ol> คือ Ordered List (รายการเรียงลำดับ)
            for index, row in top_ew.iterrows():
                item_text = (
                    f"<li><b>{row['รหัส']}: {row['ชื่ออุบัติการณ์ความเสี่ยง']}</b><br>"
                    f"<small><i>คะแนนความสำคัญ: {row['Priority_Score']:.3f}, "
                    f"คาดการณ์เหตุรุนแรงใน 3 เดือน: {row['Expected_Severe_nextH']:.2f} ครั้ง</i></small></li>"
                )
                ew_list_items.append(item_text)
            ew_list_items.append("</ol>")
            early_warning_html = "".join(ew_list_items)
    # --- สร้าง HTML Content ---
    html_string = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>            
            @page {{
                size: A4;
                margin: 2cm 1.5cm; /* เพิ่ม margin บนล่างเล็กน้อยเพื่อให้มีที่สำหรับ footer */

                /* สร้าง footer ที่มุมขวาล่าง */
                @bottom-center {{
                    content: "หน้า " counter(page) " / " counter(pages);
                    font-family: "TH SarabunPSK", sans-serif;
                    font-size: 9pt;
                    color: #888; /* สีเทา */
                }}
            }}
            /* --- จบส่วนที่แก้ไข --- */
            body {{ font-family: "TH SarabunPSK", sans-serif; font-size: 12pt; }}
            h1, h2, h3 {{ font-family: "TH SarabunPSK", sans-serif; color: #001f3f; border-bottom: 2px solid #001f3f; padding-bottom: 5px;}}
            h2 {{ page-break-before: always; }} /* ขึ้นหน้าใหม่ทุกครั้งที่เจอ h2 */
            h1 + h2 {{ page-break-before: auto; }} /* ยกเว้น h2 แรกสุด */
            .styled-table {{ width: 100%; border-collapse: collapse; margin-top: 1em; table-layout: fixed; }}
            .styled-table th, .styled-table td {{ border: 1px solid #ddd; padding: 6px; text-align: left; word-wrap: break-word; }}
            .styled-table th {{ background-color: #f2f2f2;  }}
            .metric-container {{ display: flex; justify-content: space-around; padding: 10px; background-color: #f9f9f9; border-radius: 5px; }}
            .metric {{ text-align: top; }}
            .metric-label {{ font-size: 11pt; color: #555; }}
            .metric-value {{ font-size: 16pt; font-weight: bold; }}
            
            #sentinel-table th:nth-child(1), #sentinel-table td:nth-child(1) {{ width: 24%; }}
            #sentinel-table th:nth-child(2), #sentinel-table td:nth-child(2) {{ width: 24%; }}
            #sentinel-table th:nth-child(3), #sentinel-table td:nth-child(3) {{ width: 10%; }} 
            #sentinel-table th:nth-child(4), #sentinel-table td:nth-child(4) {{ width: 38%; }}                 
            
            #top10-table th:nth-child(1), #top10-table td:nth-child(1) {{ width: 80%; }} 
            #top10-table th:nth-child(2), #top10-table td:nth-child(2) {{ width: 20%; }} 
            
            #risk-matrix-table th:nth-child(1), #risk-matrix-table td:nth-child(1) {{ width: 40%; }} 
            #risk-matrix-table th:nth-child(n+2), #risk-matrix-table td:nth-child(n+2) {{ width: 10%; }} 
           
            #psg9-table th:nth-child(1), #psg9-table td:nth-child(1) {{ width: 28%; text-align: left; }}
            #psg9-table th:nth-child(n+2):nth-child(-n+10),
            #psg9-table td:nth-child(n+2):nth-child(-n+10) {{ width: 3.4%; text-align: center; }}
            #psg9-table th:nth-child(n+11):nth-child(-n+12),
            #psg9-table td:nth-child(n+11):nth-child(-n+12) {{ width: 6.5%; text-align: center; }}
            #psg9-table th:nth-child(13), #psg9-table td:nth-child(13) {{ width: 10%; text-align: center; }}
            
            #unresolved-table th:nth-child(1), #unresolved-table td:nth-child(1) {{ width: 16%; }}
            #unresolved-table th:nth-child(2), #unresolved-table td:nth-child(2) {{ width: 22%; }}
            #unresolved-table th:nth-child(3), #unresolved-table td:nth-child(3) {{ width: 10%; }}
            #unresolved-table th:nth-child(4), #unresolved-table td:nth-child(4) {{ width: 48%; }}
            .auto-width-table {{
                table-layout: auto;
            }}
            
            ol {{ padding-left: 30px; }}
            li {{ margin-bottom: 10px; }}
            small {{ color: #555; }}
        </style>
    </head>
    <body>
        <h1>บทสรุปสำหรับผู้บริหาร</h1>
        <p><b>ช่วงข้อมูลที่วิเคราะห์:</b> {min_date_str} ถึง {max_date_str} (รวม {total_month} เดือน)</p>
        <p><b>จำนวนอุบัติการณ์ที่พบทั้งหมด:</b> {metrics_data.get('total_processed_incidents', 0):,} รายการ</p>

        <h1>1. แดชบอร์ดสรุปภาพรวม</h1>
        <div class="metric-container">
            <div class="metric"><div class="metric-label">อุบัติการณ์ทั้งหมด</div><div class="metric-value">{metrics_data.get('total_processed_incidents', 0):,}</div></div>
            <div class="metric"><div class="metric-label">Sentinel Events</div><div class="metric-value">{metrics_data.get('total_sentinel_incidents_for_metric1', 0):,}</div></div>
            <div class="metric"><div class="metric-label">PSG9</div><div class="metric-value">{metrics_data.get('total_psg9_incidents_for_metric1', 0):,}</div></div>
            <div class="metric"><div class="metric-label">ความรุนแรงสูง</div><div class="metric-value">{metrics_data.get('total_severe_incidents', 0):,}</div></div>
            <div class="metric"><div class="metric-label">รุนแรง & ยังไม่แก้ไข</div><div class="metric-value">{metrics_data.get('total_severe_unresolved_incidents_val', 'N/A')}</div></div>
        </div>

        <h1>2. Risk Matrix และ Top 10 อุบัติการณ์</h1>
        <h3>Risk Matrix</h3>
        {matrix_data_html}
        {frequency_legend_html}
        <h2>Top 10 อุบัติการณ์ (ตามความถี่)</h2>
        {top10_html}
        <h2>3. รายการ Sentinel Events</h2>             
        {sentinel_html}        
        <h2>4. วิเคราะห์ตามหมวดหมู่ มาตรฐานสำคัญจำเป็นต่อความปลอดภัย 9 ข้อ</h2>
        {psg9_html}        
        <h2>5. สรุปอุบัติการณ์ตามเป้าหมาย (Safety Goals)</h2>
        {safety_goals_html}
        <h2>6. สรุปอุบัติการณ์ที่เป็นปัญหาเรื้อรัง (Persistence Risk - Top 5)</h2>
        {persistence_html}        
        <h3>7. Early Warning: อุบัติการณ์ที่มีแนวโน้มสูงขึ้น ใน 3 เดือนข้างหน้า (Top 5)</h3>
        {early_warning_html}        
        <h2>8. รายการอุบัติการณ์รุนแรง (E-I & 3-5) ที่ยังไม่ถูกแก้ไข</h2>
        {unresolved_severe_html}
        
        </body>
    </html>
    """

    # --- แปลง HTML เป็น PDF ---
    return HTML(string=html_string).write_pdf()

# ==============================================================================
# STATIC DATA DEFINITIONS
# ==============================================================================
PSG9_FILE_PATH = "PSG9code.xlsx"
SENTINEL_FILE_PATH = "Sentinel2024.xlsx"
ALLCODE_FILE_PATH = "Code2024.xlsx"
psg9_r_codes_for_counting = set()
sentinel_composite_keys = set()
df2 = pd.DataFrame()
try:
    if Path(PSG9_FILE_PATH).is_file():
        PSG9code_df_master = pd.read_excel(PSG9_FILE_PATH)
        if 'รหัส' in PSG9code_df_master.columns:
            psg9_r_codes_for_counting = set(PSG9code_df_master['รหัส'].astype(str).str.strip().unique())
    if Path(SENTINEL_FILE_PATH).is_file():
        Sentinel2024_df = pd.read_excel(SENTINEL_FILE_PATH)
        if 'รหัส' in Sentinel2024_df.columns and 'Impact' in Sentinel2024_df.columns:
            Sentinel2024_df['รหัส'] = Sentinel2024_df['รหัส'].astype(str).str.strip()
            Sentinel2024_df['Impact'] = Sentinel2024_df['Impact'].astype(str).str.strip()
            Sentinel2024_df.dropna(subset=['รหัส', 'Impact'], inplace=True)
            sentinel_composite_keys = set((Sentinel2024_df['รหัส'] + '-' + Sentinel2024_df['Impact']).unique())
    if Path(ALLCODE_FILE_PATH).is_file():
        allcode2024_df = pd.read_excel(ALLCODE_FILE_PATH)
        if 'รหัส' in allcode2024_df.columns and all(c in allcode2024_df.columns for c in ["ชื่ออุบัติการณ์ความเสี่ยง", "กลุ่ม", "หมวด"]):
            df2 = allcode2024_df[["รหัส", "ชื่ออุบัติการณ์ความเสี่ยง", "กลุ่ม", "หมวด"]].drop_duplicates().copy()
            df2['รหัส'] = df2['รหัส'].astype(str).str.strip()
except Exception as e:
    st.error(f"เกิดปัญหาในการโหลดไฟล์นิยาม: {e}")

risk_color_data = {
    'Category Color': ["Critical", "Critical", "Critical", "Critical", "Critical", "High", "High", "Critical", "Critical", "Critical", "Medium", "Medium", "High", "Critical", "Critical", "Low", "Medium", "Medium", "High", "High", "Low", "Low", "Low", "Medium", "Medium"],
    'Risk Level': ["51", "52", "53", "54", "55", "41", "42", "43", "44", "45", "31", "32", "33", "34", "35", "21", "22", "23", "24", "25", "11", "12", "13", "14", "15"]}
risk_color_df = pd.DataFrame(risk_color_data)
display_cols_common = ['Occurrence Date', 'รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Impact', 'Impact Level', 'รายละเอียดการเกิด', 'Resulting Actions']
month_label = {1: '01 มกราคม', 2: '02 กุมภาพันธ์', 3: '03 มีนาคม', 4: '04 เมษายน', 5: '05 พฤษภาคม', 6: '06 มิถุนายน', 7: '07 กรกฎาคม', 8: '08 สิงหาคม', 9: '09 กันยายน', 10: '10 ตุลาคม', 11: '11 พฤศจิกายน', 12: '12 ธันวาคม'}

PSG9_label_dict = {
    1: '01 ผ่าตัดผิดคน ผิดข้าง ผิดตำแหน่ง ผิดหัตถการ', 2: '02 บุคลากรติดเชื้อจากการปฏิบัติหน้าที่',
    3: '03 การติดเชื้อสำคัญ (SSI, VAP,CAUTI, CLABSI)', 4: '04 การเกิด Medication Error และ Adverse Drug Event',
    5: '05 การให้เลือดผิดคน ผิดหมู่ ผิดชนิด', 6: '06 การระบุตัวผู้ป่วยผิดพลาด',
    7: '07 ความคลาดเคลื่อนในการวินิจฉัยโรค', 8: '08 การรายงานผลการตรวจทางห้องปฏิบัติการ/พยาธิวิทยา คลาดเคลื่อน',
    9: '09 การคัดกรองที่ห้องฉุกเฉินคลาดเคลื่อน'
}

# ✅ *** ส่วนที่เพิ่มกลับเข้ามา ***
type_name = {'CPS': 'Safe Surgery', 'CPI': 'Infection Prevention and Control', 'CPM': 'Medication & Blood Safety',
             'CPP': 'Patient Care Process', 'CPL': 'Line, Tube & Catheter and Laboratory', 'CPE': 'Emergency Response',
             'CSG': 'Gynecology & Obstetrics diseases and procedure', 'CSS': 'Surgical diseases and procedure',
             'CSM': 'Medical diseases and procedure', 'CSP': 'Pediatric diseases and procedure',
             'CSO': 'Orthopedic diseases and procedure', 'CSD': 'Dental diseases and procedure',
             'GPS': 'Social Media and Communication', 'GPI': 'Infection and Exposure',
             'GPM': 'Mental Health and Mediation', 'GPP': 'Process of work', 'GPL': 'Lane (Traffic) and Legal Issues',
             'GPE': 'Environment and Working Conditions', 'GOS': 'Strategy, Structure, Security',
             'GOI': 'Information Technology & Communication, Internal control & Inventory',
             'GOM': 'Manpower, Management', 'GOP': 'Policy, Process of work & Operation',
             'GOL': 'Licensed & Professional certificate', 'GOE': 'Economy'}

colors2 = np.array([["#e1f5fe", "#f6c8b6", "#dd191d", "#dd191d", "#dd191d", "#dd191d", "#dd191d"],
                    ["#e1f5fe", "#f6c8b6", "#ff8f00", "#ff8f00", "#dd191d", "#dd191d", "#dd191d"],
                    ["#e1f5fe", "#f6c8b6", "#ffee58", "#ffee58", "#ff8f00", "#dd191d", "#dd191d"],
                    ["#e1f5fe", "#f6c8b6", "#42db41", "#ffee58", "#ffee58", "#ff8f00", "#ff8f00"],
                    ["#e1f5fe", "#f6c8b6", "#42db41", "#42db41", "#42db41", "#ffee58", "#ffee58"],
                    ["#e1f5fe", "#f6c8b6", "#f6c8b6", "#f6c8b6", "#f6c8b6", "#f6c8b6", "#f6c8b6"],
                    ["#e1f5fe", "#e1f5fe", "#e1f5fe", "#e1f5fe", "#e1f5fe", "#e1f5fe", "#e1f5fe"]])

# ==============================================================================
# MAIN APP STRUCTURE
# ==============================================================================
def display_admin_page():
    st.title("🔑 Admin: Data Upload")
    st.warning("ส่วนนี้สำหรับผู้ดูแลระบบเท่านั้น")
    password = st.text_input("กรุณาใส่รหัสผ่าน:", type="password")
    if password == ADMIN_PASSWORD:
        st.success("เข้าสู่ระบบสำเร็จ!")
        st.header("อัปโหลดไฟล์รายงานอุบัติการณ์ (.xlsx)")
        uploaded_file = st.file_uploader("เลือกไฟล์ Excel ของคุณที่นี่:", type=".xlsx")
        if uploaded_file:
            with st.spinner("กำลังประมวลผลไฟล์ กรุณารอสักครู่..."):
                df = load_data(uploaded_file)
                if not df.empty:
                    df.replace('', 'None', inplace=True)
                    df = df.fillna('None')
                    df.rename(columns={'วดป.ที่เกิด': 'Occurrence Date', 'ความรุนแรง': 'Impact'}, inplace=True)

                    required_cols = ['Incident', 'Occurrence Date', 'Impact']
                    if not all(col in df.columns for col in required_cols):
                        st.error(f"ไฟล์อัปโหลดขาดคอลัมน์ที่จำเป็น: {', '.join(required_cols)}.")
                        st.stop()

                    df['Impact'] = df['Impact'].astype(str).str.strip()
                    df['รหัส'] = df['Incident'].astype(str).str.slice(0, 6).str.strip()

                    if not df2.empty:
                        df = pd.merge(df, df2, on='รหัส', how='left')
                    for col in ["ชื่ออุบัติการณ์ความเสี่ยง", "กลุ่ม", "หมวด"]:
                        if col not in df.columns:
                            df[col] = 'N/A'
                        else:
                            df[col].fillna('N/A', inplace=True)

                    df['Occurrence Date'] = pd.to_datetime(df['Occurrence Date'], errors='coerce')
                    df.dropna(subset=['Occurrence Date'], inplace=True)
                    if df.empty:
                        st.error("ไม่พบข้อมูลวันที่ที่ถูกต้อง")
                        st.stop()

                    impact_level_map = {('A', 'B', '1'): '1', ('C', 'D', '2'): '2', ('E', 'F', '3'): '3',
                                        ('G', 'H', '4'): '4', ('I', '5'): '5'}

                    def map_impact_level_func(val):
                        s_val = str(val)
                        for k, v in impact_level_map.items():
                            if s_val in k: return v
                        return 'N/A'

                    df['Impact Level'] = df['Impact'].apply(map_impact_level_func)

                    max_p = df['Occurrence Date'].max().to_period('M')
                    min_p = df['Occurrence Date'].min().to_period('M')
                    total_month_calc = (max_p.year - min_p.year) * 12 + (max_p.month - min_p.month) + 1

                    df_freq_temp = df['Incident'].value_counts().reset_index()
                    df_freq_temp.columns = ['Incident', 'count']
                    df_freq_temp['Incident Rate/mth'] = (df_freq_temp['count'] / max(1, total_month_calc)).round(1)
                    df = pd.merge(df, df_freq_temp, on="Incident", how='left')

                    conditions_freq = [(df['Incident Rate/mth'] < 2.0), (df['Incident Rate/mth'] < 3.9),
                                       (df['Incident Rate/mth'] < 6.9), (df['Incident Rate/mth'] < 29.9)]
                    choices_freq = ['1', '2', '3', '4']
                    df['Frequency Level'] = np.select(conditions_freq, choices_freq, default='5')

                    df['Risk Level'] = df.apply(
                        lambda row: f"{row['Impact Level']}{row['Frequency Level']}" if pd.notna(
                            row['Impact Level']) and row['Impact Level'] != 'N/A' else 'N/A', axis=1)

                    df = pd.merge(df, risk_color_df, on='Risk Level', how='left')
                    df['Category Color'].fillna('Undefined', inplace=True)

                    df['Incident Type'] = df['Incident'].astype(str).str[:3]
                    df['Month'] = df['Occurrence Date'].dt.month
                    df['เดือน'] = df['Month'].map(month_label)
                    df['Year'] = df['Occurrence Date'].dt.year.astype(str)

                    PSG9_ID_COL = 'PSG_ID'
                    if 'PSG9code_df_master' in globals() and not PSG9code_df_master.empty and PSG9_ID_COL in PSG9code_df_master.columns:
                        standards_to_merge = PSG9code_df_master[['รหัส', PSG9_ID_COL]].copy().drop_duplicates(
                            subset=['รหัส'])
                        standards_to_merge['รหัส'] = standards_to_merge['รหัส'].astype(str).str.strip()
                        df = pd.merge(df, standards_to_merge, on='รหัส', how='left')
                        df['หมวดหมู่มาตรฐานสำคัญ'] = df[PSG9_ID_COL].map(PSG9_label_dict).fillna(
                            "ไม่จัดอยู่ใน PSG9 Catalog")
                    else:
                        df['หมวดหมู่มาตรฐานสำคัญ'] = "ไม่สามารถระบุ (PSG9code.xlsx ไม่ได้โหลด)"

                    for col in df.select_dtypes(include=['object']).columns:
                        df[col] = df[col].astype(str)

                    df.to_parquet(PERSISTED_DATA_PATH, index=False)
                    st.success(f"ประมวลผลสำเร็จ! ข้อมูล {len(df)} รายการถูกบันทึกแล้ว")
    elif password:
        st.error("รหัสผ่านไม่ถูกต้อง!")


def display_executive_dashboard():
    # --- 1. โหลดข้อมูล ---
    try:
        df = pd.read_parquet(PERSISTED_DATA_PATH)
        df['Occurrence Date'] = pd.to_datetime(df['Occurrence Date'])
    except FileNotFoundError:
        st.warning("ยังไม่มีข้อมูลในระบบ กรุณาติดต่อผู้ดูแลระบบเพื่ออัปโหลดข้อมูล")
        st.markdown(f'<p>หากคุณเป็น Admin กรุณาไปที่ URL <a href="/?page=admin">?page=admin</a> เพื่ออัปโหลดข้อมูล</p>',
                    unsafe_allow_html=True)
        return

    # --- 2. สร้าง Sidebar และเมนูกรองข้อมูล ---
    st.sidebar.markdown(
        f"""<div style="display: flex; align-items: center; margin-bottom: 1rem;"><img src="{LOGO_URL}" style="height: 32px; margin-right: 10px;"><h2 style="margin: 0; font-size: 1.7rem; color: #001f3f; font-weight: bold;">HOIA-RR Menu</h2></div>""",
        unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.header("Filter by Date")

    min_date_in_data = df['Occurrence Date'].min().date()
    max_date_in_data = df['Occurrence Date'].max().date()
    today = datetime.now().date()

    filter_option = st.sidebar.selectbox("เลือกช่วงเวลา:",
                                         ["ทั้งหมด", "ปีนี้", "ไตรมาสนี้", "เดือนนี้", "ปีที่แล้ว", "กำหนดเอง..."])

    start_date, end_date = min_date_in_data, max_date_in_data

    if filter_option == "ปีนี้":
        start_date = today.replace(month=1, day=1)
        end_date = today
    elif filter_option == "ไตรมาสนี้":
        current_quarter = (today.month - 1) // 3 + 1
        start_date = datetime(today.year, 3 * current_quarter - 2, 1).date()
        end_date = today
    elif filter_option == "เดือนนี้":
        start_date = today.replace(day=1)
        end_date = today
    elif filter_option == "ปีที่แล้ว":
        last_year = today.year - 1
        start_date = datetime(last_year, 1, 1).date()
        end_date = datetime(last_year, 12, 31).date()
    elif filter_option == "กำหนดเอง...":
        start_date, end_date = st.sidebar.date_input(
            "เลือกระหว่างวันที่:",
            [min_date_in_data, max_date_in_data],
            min_value=min_date_in_data,
            max_value=max_date_in_data
        )

    df_filtered = df[(df['Occurrence Date'].dt.date >= start_date) & (df['Occurrence Date'].dt.date <= end_date)].copy()
    # ✅ *** เพิ่มบรรทัดนี้เพื่อสร้างคอลัมน์ชื่อเต็มสำหรับทุกการวิเคราะห์ ***
    df_filtered['Incident Type Name'] = df_filtered['Incident Type'].map(type_name).fillna(df_filtered['Incident Type'])
    if df_filtered.empty:
        st.sidebar.warning("ไม่พบข้อมูลในช่วงเวลาที่ท่านเลือก")
        st.warning("ไม่พบข้อมูลในช่วงเวลาที่ท่านเลือก กรุณาเลือกช่วงเวลาอื่น")
        return

    min_date_str = df_filtered['Occurrence Date'].min().strftime('%d/%m/%Y')
    max_date_str = df_filtered['Occurrence Date'].max().strftime('%d/%m/%Y')

    max_p = df_filtered['Occurrence Date'].max().to_period('M')
    min_p = df_filtered['Occurrence Date'].min().to_period('M')
    total_month = (max_p.year - min_p.year) * 12 + (max_p.month - min_p.month) + 1
    total_month = max(1, total_month)

    st.sidebar.markdown(f"**ช่วงข้อมูล:** {min_date_str} ถึง {max_date_str}")
    st.sidebar.markdown(f"**จำนวนเดือน:** {total_month} เดือน")
    st.sidebar.markdown(f"**จำนวนอุบัติการณ์:** {df_filtered.shape[0]:,} รายการ")

    st.sidebar.markdown("---")
    st.sidebar.markdown("เลือกส่วนที่ต้องการแสดงผล:")

    analysis_options_list = [
        "แดชบอร์ดสรุปภาพรวม", "Heatmap: อุบัติการณ์รายเดือน", "รายการ Sentinel Events",
        "Risk Matrix (Interactive)", "กราฟสรุปอุบัติการณ์ (รายมิติ)",
        "Sankey: ภาพรวม", "Sankey: มาตรฐานสำคัญจำเป็นต่อความปลอดภัย 9 ข้อ",
        "Scatter Plot & Top 10", "สรุปอุบัติการณ์ตาม Safety Goals", "วิเคราะห์ตามหมวดหมู่และสถานะการแก้ไข",
        "Persistence Risk Index", "Early Warning: อุบัติการณ์ที่มีแนวโน้มสูงขึ้น", "บทสรุปสำหรับผู้บริหาร",
        "คุยกับ AI Assistant",
    ]
    if 'selected_analysis' not in st.session_state:
        st.session_state.selected_analysis = analysis_options_list[0]

    for option in analysis_options_list:
        if st.sidebar.button(option, key=f"btn_{option}",
                             type="primary" if st.session_state.selected_analysis == option else "secondary",
                             use_container_width=True):
            st.session_state.selected_analysis = option
            st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
            **กิตติกรรมประกาศ:** ขอขอบพระคุณ 
            - Prof. Shin Ushiro
            - นพ.อนุวัฒน์ ศุภชุติกุล 
            - นพ.ก้องเกียรติ เกษเพ็ชร์ 
            - พญ.ปิยวรรณ ลิ้มปัญญาเลิศ 
            - ภก.ปรมินทร์ วีระอนันตวัฒน์    
            - ผศ.ดร.นิเวศน์ กุลวงค์ (อ.ที่ปรึกษา)

            เป็นอย่างสูง สำหรับการริเริ่ม เติมเต็ม สนับสนุน และสร้างแรงบันดาลใจ อันเป็นรากฐานสำคัญในการพัฒนาเครื่องมือนี้
            """)

    # ✅ --- เพิ่มโค้ดส่วนนี้เข้าไปที่ท้ายสุดของ Sidebar ---
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<p style="font-size:12px; color:gray;">*เครื่องมือนี้เป็นส่วนหนึ่งของวิทยานิพนธ์ IMPLEMENTING THE  HOSPITAL OCCURRENCE/INCIDENT ANALYSIS & RISK REGISTER (HOIA-RR TOOL) IN THAI HOSPITALS: A STUDY ON EFFECTIVE ADOPTION โดย นางสาววิลาศินี  เขื่อนแก้ว นักศึกษาปริญญาโท สำนักวิชาวิทยาศาสตร์สุขภาพ มหาวิทยาลัยแม่ฟ้าหลวง</p>',
        unsafe_allow_html=True)
    # ✅ --- จบส่วนที่เพิ่ม ---

    # --- 3. คำนวณ Metrics สำหรับ Dashboard ---
    metrics_data = {}
    metrics_data['total_processed_incidents'] = df_filtered.shape[0]
    metrics_data['total_psg9_incidents_for_metric1'] = \
    df_filtered[df_filtered['รหัส'].isin(psg9_r_codes_for_counting)].shape[
        0] if 'psg9_r_codes_for_counting' in globals() else 0
    if 'sentinel_composite_keys' in globals() and sentinel_composite_keys:
        df_filtered['Sentinel code for check'] = df_filtered['รหัส'].astype(str).str.strip() + '-' + df_filtered[
            'Impact'].astype(str).str.strip()
        metrics_data['total_sentinel_incidents_for_metric1'] = \
        df_filtered[df_filtered['Sentinel code for check'].isin(sentinel_composite_keys)].shape[0]
    else:
        metrics_data['total_sentinel_incidents_for_metric1'] = 0

    severe_impact_levels_list = ['3', '4', '5']
    df_severe_incidents_calc = df_filtered[df_filtered['Impact Level'].isin(severe_impact_levels_list)].copy()
    metrics_data['total_severe_incidents'] = df_severe_incidents_calc.shape[0]
    if 'Resulting Actions' in df_filtered.columns:
        unresolved_conditions = df_severe_incidents_calc['Resulting Actions'].astype(str).isin(['None', '', 'nan'])
        df_severe_unresolved_calc = df_severe_incidents_calc[unresolved_conditions].copy()
        metrics_data['total_severe_unresolved_incidents_val'] = df_severe_unresolved_calc.shape[0]
        metrics_data['total_severe_unresolved_psg9_incidents_val'] = \
        df_severe_unresolved_calc[df_severe_unresolved_calc['รหัส'].isin(psg9_r_codes_for_counting)].shape[
            0] if 'psg9_r_codes_for_counting' in globals() else 0
    else:
        metrics_data['total_severe_unresolved_incidents_val'] = "N/A"
        metrics_data['total_severe_unresolved_psg9_incidents_val'] = "N/A"
    metrics_data['total_month'] = total_month

    df_freq = df_filtered['Incident'].value_counts().reset_index()
    df_freq.columns = ['Incident', 'count']

    # --- 4. PAGE CONTENT ROUTING ---
    selected_analysis = st.session_state.selected_analysis

    if selected_analysis == "แดชบอร์ดสรุปภาพรวม":
        st.markdown("<h4 style='color: #001f3f;'>สรุปภาพรวมอุบัติการณ์:</h4>", unsafe_allow_html=True)

        with st.expander("แสดง/ซ่อน ตารางข้อมูลอุบัติการณ์ทั้งหมด (Full Data Table)"):
            st.dataframe(df_filtered, hide_index=True, use_container_width=True, column_config={
                "Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด", format="DD/MM/YYYY")
            })

        dashboard_expander_cols = ['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด', 'Resulting Actions']
        date_format_config = {"Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด", format="DD/MM/YYYY")}

        total_processed_incidents = metrics_data.get("total_processed_incidents", 0)
        total_psg9_incidents_for_metric1 = metrics_data.get("total_psg9_incidents_for_metric1", 0)
        total_sentinel_incidents_for_metric1 = metrics_data.get("total_sentinel_incidents_for_metric1", 0)
        total_severe_incidents = metrics_data.get("total_severe_incidents", 0)
        total_severe_unresolved_incidents_val = metrics_data.get("total_severe_unresolved_incidents_val", "N/A")
        total_severe_unresolved_psg9_incidents_val = metrics_data.get("total_severe_unresolved_psg9_incidents_val",
                                                                      "N/A")

        df_severe_incidents = df_filtered[df_filtered['Impact Level'].isin(['3', '4', '5'])].copy()
        total_severe_psg9_incidents = \
        df_severe_incidents[df_severe_incidents['รหัส'].isin(psg9_r_codes_for_counting)].shape[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", f"{total_processed_incidents:,}")
        with col2:
            st.metric("PSG9", f"{total_psg9_incidents_for_metric1:,}")
            with st.expander(f"ดูรายละเอียด ({total_psg9_incidents_for_metric1} รายการ)"):
                psg9_df = df_filtered[df_filtered['รหัส'].isin(psg9_r_codes_for_counting)]
                st.dataframe(psg9_df[dashboard_expander_cols], use_container_width=True, hide_index=True,
                             column_config=date_format_config)
        with col3:
            st.metric("Sentinel", f"{total_sentinel_incidents_for_metric1:,}")
            with st.expander(f"ดูรายละเอียด ({total_sentinel_incidents_for_metric1} รายการ)"):
                if 'Sentinel code for check' in df_filtered.columns:
                    sentinel_df = df_filtered[df_filtered['Sentinel code for check'].isin(sentinel_composite_keys)]
                    st.dataframe(sentinel_df[dashboard_expander_cols], use_container_width=True, hide_index=True,
                                 column_config=date_format_config)

        col4, col5, col6, col7 = st.columns(4)
        with col4:
            st.metric("E-I & 3-5 [all]", f"{total_severe_incidents:,}")
            with st.expander(f"ดูรายละเอียด ({total_severe_incidents} รายการ)"):
                st.dataframe(df_severe_incidents[dashboard_expander_cols], use_container_width=True, hide_index=True,
                             column_config=date_format_config)
        with col5:
            st.metric("E-I & 3-5 [PSG9]", f"{total_severe_psg9_incidents:,}")
            with st.expander(f"ดูรายละเอียด ({total_severe_psg9_incidents} รายการ)"):
                severe_psg9_df = df_severe_incidents[df_severe_incidents['รหัส'].isin(psg9_r_codes_for_counting)]
                st.dataframe(severe_psg9_df[dashboard_expander_cols], use_container_width=True, hide_index=True,
                             column_config=date_format_config)
        with col6:
            val_unresolved_all = f"{total_severe_unresolved_incidents_val:,}" if isinstance(
                total_severe_unresolved_incidents_val, int) else "N/A"
            st.metric(f"E-I & 3-5 [all] ที่ยังไม่ถูกแก้ไข", val_unresolved_all)
            if isinstance(total_severe_unresolved_incidents_val, int) and total_severe_unresolved_incidents_val > 0:
                with st.expander(f"ดูรายละเอียด ({total_severe_unresolved_incidents_val} รายการ)"):
                    unresolved_df_all = df_filtered[
                        df_filtered['Impact Level'].isin(['3', '4', '5']) & df_filtered['Resulting Actions'].astype(
                            str).isin(['None', '', 'nan'])]
                    st.dataframe(unresolved_df_all[dashboard_expander_cols], use_container_width=True, hide_index=True,
                                 column_config=date_format_config)
        with col7:
            val_unresolved_psg9 = f"{total_severe_unresolved_psg9_incidents_val:,}" if isinstance(
                total_severe_unresolved_psg9_incidents_val, int) else "N/A"
            st.metric(f"E-I & 3-5 [PSG9] ที่ยังไม่ถูกแก้ไข", val_unresolved_psg9)
            if isinstance(total_severe_unresolved_psg9_incidents_val,
                          int) and total_severe_unresolved_psg9_incidents_val > 0:
                with st.expander(f"ดูรายละเอียด ({total_severe_unresolved_psg9_incidents_val} รายการ)"):
                    unresolved_df_all = df_filtered[
                        df_filtered['Impact Level'].isin(['3', '4', '5']) & df_filtered['Resulting Actions'].astype(
                            str).isin(['None', '', 'nan'])]
                    unresolved_df_psg9 = unresolved_df_all[unresolved_df_all['รหัส'].isin(psg9_r_codes_for_counting)]
                    st.dataframe(unresolved_df_psg9[dashboard_expander_cols], use_container_width=True, hide_index=True,
                                 column_config=date_format_config)
        st.markdown("---")
        st.subheader("📈 แนวโน้มอุบัติการณ์รายเดือน")

        # เตรียมข้อมูล: จัดกลุ่มข้อมูลตามปี-เดือน แล้วนับจำนวน
        monthly_counts = df_filtered.copy()
        monthly_counts['เดือน-ปี'] = monthly_counts['Occurrence Date'].dt.strftime('%Y-%m')

        incident_trend = monthly_counts.groupby('เดือน-ปี').size().reset_index(name='จำนวนอุบัติการณ์')
        incident_trend = incident_trend.sort_values(by='เดือน-ปี')

        # สร้างกราฟเส้น
        fig_trend = px.line(
            incident_trend,
            x='เดือน-ปี',
            y='จำนวนอุบัติการณ์',
            title='จำนวนอุบัติการณ์ทั้งหมดที่เกิดขึ้นในแต่ละเดือน',
            markers=True,  # เพิ่มจุดบนเส้นเพื่อให้เห็นข้อมูลแต่ละเดือนชัดขึ้น
            labels={'เดือน-ปี': 'เดือน', 'จำนวนอุบัติการณ์': 'จำนวนครั้ง'},
            line_shape = 'spline'
        )
        fig_trend.update_traces(line=dict(width=3))
        st.plotly_chart(fig_trend, use_container_width=True)
    elif selected_analysis == "Heatmap: อุบัติการณ์รายเดือน":
        st.markdown("<h4 style='color: #001f3f;'>Heatmap: จำนวนอุบัติการณ์รายเดือน</h4>", unsafe_allow_html=True)
        st.info(
            "💡 Heatmap นี้แสดงความถี่ของการเกิดอุบัติการณ์แต่ละรหัสในแต่ละเดือน สีที่เข้มกว่าหมายถึงจำนวนครั้งที่เกิดสูงกว่า ช่วยให้มองเห็นรูปแบบหรืออุบัติการณ์ที่เกิดบ่อยในช่วงเวลาต่างๆ ได้ง่ายขึ้น")

        st.markdown("<h5 style='color: #003366;'>ภาพรวมอุบัติการณ์ทั้งหมด (เลือกจำนวนที่แสดงได้)</h5>",
                    unsafe_allow_html=True)
        heatmap_req_cols = ['รหัส', 'เดือน', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Month', 'หมวด']
        if not all(col in df_filtered.columns for col in heatmap_req_cols):
            st.warning(f"ไม่สามารถสร้าง Heatmap ได้ เนื่องจากขาดคอลัมน์ที่จำเป็น: {', '.join(heatmap_req_cols)}")
        else:
            df_heat = df_filtered.copy()
            df_heat['incident_label'] = df_heat['รหัส'] + " | " + df_heat['ชื่ออุบัติการณ์ความเสี่ยง'].fillna('')

            total_counts = df_heat['incident_label'].value_counts().reset_index()
            total_counts.columns = ['incident_label', 'total_count']

            top_n = st.slider(
                "เลือกจำนวนอุบัติการณ์ (ตามความถี่) ที่ต้องการแสดงบน Heatmap รวม:",
                min_value=5, max_value=min(50, len(total_counts)),
                value=min(20, len(total_counts)), step=5, key="top_n_slider"
            )
            top_incident_labels = total_counts.nlargest(top_n, 'total_count')['incident_label']
            df_heat_filtered_view = df_heat[df_heat['incident_label'].isin(top_incident_labels)]
            try:
                heatmap_pivot = pd.pivot_table(df_heat_filtered_view, values='Incident', index='incident_label',
                                               columns='เดือน', aggfunc='count', fill_value=0)
                sorted_month_names = [v for k, v in sorted(month_label.items())]
                available_months = [m for m in sorted_month_names if m in heatmap_pivot.columns]
                if available_months:
                    heatmap_pivot = heatmap_pivot[available_months]
                    heatmap_pivot = heatmap_pivot.reindex(top_incident_labels).dropna()
                    if not heatmap_pivot.empty:
                        fig_heatmap = px.imshow(heatmap_pivot,
                                                labels=dict(x="เดือน", y="อุบัติการณ์", color="จำนวนครั้ง"),
                                                text_auto=True, aspect="auto", color_continuous_scale='Reds')
                        fig_heatmap.update_layout(title_text=f"Heatmap ของอุบัติการณ์ Top {top_n} ที่เกิดบ่อยที่สุด",
                                                  height=max(600, len(heatmap_pivot.index) * 25), xaxis_title="เดือน",
                                                  yaxis_title="รหัส | ชื่ออุบัติการณ์")
                        fig_heatmap.update_xaxes(side="top")
                        st.plotly_chart(fig_heatmap, use_container_width=True)
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการสร้าง Heatmap รวม: {e}")

            st.markdown("---")

            st.markdown("<h5 style='color: #003366;'>Heatmap แยกตามเป้าหมายความปลอดภัย (Safety Goal)</h5>",
                        unsafe_allow_html=True)
            goal_search_terms = {
                "Patient Safety/ Common Clinical Risk": "Patient Safety", "Specific Clinical Risk": "Specific Clinical",
                "Personnel Safety": "Personnel Safety", "Organization Safety": "Organization Safety"
            }

            for display_name, search_term in goal_search_terms.items():
                df_goal_filtered = df_heat[df_heat['หมวด'].str.contains(search_term, na=False, case=False)].copy()
                if df_goal_filtered.empty:
                    st.markdown(f"**{display_name}**")
                    st.info(f"ไม่พบข้อมูลสำหรับเป้าหมายนี้")
                    st.markdown("---")
                    continue
                try:
                    goal_pivot = pd.pivot_table(df_goal_filtered, values='Incident', index='incident_label',
                                                columns='เดือน', aggfunc='count', fill_value=0)
                    if not goal_pivot.empty:
                        sorted_month_names = [v for k, v in sorted(month_label.items())]
                        available_months_goal = [m for m in sorted_month_names if m in goal_pivot.columns]
                        if available_months_goal:
                            goal_pivot = goal_pivot[available_months_goal]
                            incident_counts_in_goal = df_goal_filtered['incident_label'].value_counts()
                            sorted_incidents = incident_counts_in_goal.index.tolist()
                            goal_pivot = goal_pivot.reindex(sorted_incidents).dropna(how='all')

                    if goal_pivot.empty:
                        st.markdown(f"**{display_name}**")
                        st.info(f"ไม่มีข้อมูลอุบัติการณ์ที่บันทึกเป็นรายเดือนสำหรับเป้าหมายนี้")
                        st.markdown("---")
                        continue

                    fig_goal = px.imshow(goal_pivot, labels=dict(x="เดือน", y="อุบัติการณ์", color="จำนวน"),
                                         text_auto=True, aspect="auto", color_continuous_scale='Oranges')
                    fig_goal.update_layout(title_text=f"<b>{display_name}</b>",
                                           height=max(500, len(goal_pivot.index) * 28), xaxis_title="เดือน",
                                           yaxis_title="รหัส | ชื่ออุบัติการณ์")
                    fig_goal.update_xaxes(side="top")
                    st.plotly_chart(fig_goal, use_container_width=True)
                    st.markdown("---")
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการสร้าง Heatmap สำหรับ '{display_name}': {e}")
    elif selected_analysis == "รายการ Sentinel Events":
        st.markdown("<h4 style='color: #001f3f;'>รายการ Sentinel Events ที่ตรวจพบ</h4>", unsafe_allow_html=True)
        # ✅ แก้ไข: ใช้ df_filtered ที่ผ่านการกรองตามช่วงเวลาแล้ว
        if 'sentinel_composite_keys' in globals() and sentinel_composite_keys and 'Sentinel code for check' in df_filtered.columns:
            sentinel_events = df_filtered[df_filtered['Sentinel code for check'].isin(sentinel_composite_keys)].copy()

            if not sentinel_events.empty:
                if 'Sentinel2024_df' in globals() and not Sentinel2024_df.empty and 'ชื่ออุบัติการณ์ความเสี่ยง' in Sentinel2024_df.columns:
                    sentinel_events = pd.merge(sentinel_events,
                                               Sentinel2024_df[['รหัส', 'Impact', 'ชื่ออุบัติการณ์ความเสี่ยง']].rename(
                                                   columns={'ชื่ออุบัติการณ์ความเสี่ยง': 'Sentinel Event Name'}),
                                               on=['รหัส', 'Impact'], how='left')
                display_sentinel_cols = ['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด',
                                         'Resulting Actions']
                if 'Sentinel Event Name' in sentinel_events.columns:
                    display_sentinel_cols.insert(2, 'Sentinel Event Name')
                final_display_cols = [col for col in display_sentinel_cols if col in sentinel_events.columns]
                st.dataframe(sentinel_events[final_display_cols], use_container_width=True, hide_index=True,
                             column_config={
                                 "Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด", format="DD/MM/YYYY")})
            else:
                st.info("ไม่พบ Sentinel Events ในช่วงเวลาที่เลือก")
        else:
            st.warning("ไม่สามารถตรวจสอบ Sentinel Events ได้ (ไฟล์ Sentinel2024.xlsx อาจไม่มีข้อมูล)")
    elif selected_analysis == "Risk Matrix (Interactive)":
        st.subheader("Risk Matrix (Interactive)")

        matrix_data_counts = np.zeros((5, 5), dtype=int)
        impact_level_keys = ['5', '4', '3', '2', '1']
        freq_level_keys = ['1', '2', '3', '4', '5']

        # ✅ --- ส่วนที่แก้ไข ---
        # 1. สร้าง DataFrame ที่สะอาด โดยกรองเอาเฉพาะข้อมูลที่มี Impact และ Frequency Level ที่ถูกต้อง
        matrix_df = df_filtered[
            df_filtered['Impact Level'].isin(impact_level_keys) &
            df_filtered['Frequency Level'].isin(freq_level_keys)
            ].copy()

        # 2. ทำการนับจาก DataFrame ที่สะอาดแล้วเท่านั้น
        if not matrix_df.empty:
            risk_counts_df = matrix_df.groupby(['Impact Level', 'Frequency Level']).size().reset_index(name='counts')
            for _, row in risk_counts_df.iterrows():
                il_key, fl_key = str(row['Impact Level']), str(row['Frequency Level'])
                # ไม่จำเป็นต้องเช็คซ้ำ เพราะเรากรองข้อมูลมาแล้ว
                row_idx, col_idx = impact_level_keys.index(il_key), freq_level_keys.index(fl_key)
                matrix_data_counts[row_idx, col_idx] = row['counts']
        # ✅ --- จบส่วนที่แก้ไข ---

        impact_labels_display = {
            '5': "I / 5<br>Extreme / Death", '4': "G-H / 4<br>Major / Severe",
            '3': "E-F / 3<br>Moderate", '2': "C-D / 2<br>Minor / Low", '1': "A-B / 1<br>Insignificant / No Harm"
        }
        freq_labels_display_short = {"1": "F1", "2": "F2", "3": "F3", "4": "F4", "5": "F5"}
        freq_labels_display_long = {
            "1": "Remote<br>(<2/mth)", "2": "Uncommon<br>(2-3/mth)", "3": "Occasional<br>(4-6/mth)",
            "4": "Probable<br>(7-29/mth)", "5": "Frequent<br>(>=30/mth)"
        }
        impact_to_color_row = {'5': 0, '4': 1, '3': 2, '2': 3, '1': 4}
        freq_to_color_col = {'1': 2, '2': 3, '3': 4, '4': 5, '5': 6}

        cols_header = st.columns([2.2, 1, 1, 1, 1, 1])
        with cols_header[0]:
            st.markdown(
                f"<div style='background-color:{colors2[6, 0]}; color:{get_text_color_for_bg(colors2[6, 0])}; padding:8px; text-align:center; font-weight:bold; border-radius:3px; margin:1px; height:60px; display:flex; align-items:center; justify-content:center;'>Impact / Frequency</div>",
                unsafe_allow_html=True)
        for i, fl_key in enumerate(freq_level_keys):
            with cols_header[i + 1]:
                header_freq_bg_color = colors2[5, freq_to_color_col.get(fl_key, 2) - 1]
                header_freq_text_color = get_text_color_for_bg(header_freq_bg_color)
                st.markdown(
                    f"<div style='background-color:{header_freq_bg_color}; color:{header_freq_text_color}; padding:8px; text-align:center; font-weight:bold; border-radius:3px; margin:1px; height:60px; display:flex; flex-direction: column; align-items:center; justify-content:center;'><div>{freq_labels_display_short.get(fl_key, '')}</div><div style='font-size:0.7em;'>{freq_labels_display_long.get(fl_key, '')}</div></div>",
                    unsafe_allow_html=True)

        for il_key in impact_level_keys:
            cols_data_row = st.columns([2.2, 1, 1, 1, 1, 1])
            row_idx_color = impact_to_color_row[il_key]
            with cols_data_row[0]:
                header_impact_bg_color = colors2[row_idx_color, 1]
                header_impact_text_color = get_text_color_for_bg(header_impact_bg_color)
                st.markdown(
                    f"<div style='background-color:{header_impact_bg_color}; color:{header_impact_text_color}; padding:8px; text-align:center; font-weight:bold; border-radius:3px; margin:1px; height:70px; display:flex; align-items:center; justify-content:center;'>{impact_labels_display[il_key]}</div>",
                    unsafe_allow_html=True)
            for i, fl_key in enumerate(freq_level_keys):
                with cols_data_row[i + 1]:
                    count = matrix_data_counts[impact_level_keys.index(il_key), freq_level_keys.index(fl_key)]
                    cell_bg_color = colors2[row_idx_color, freq_to_color_col[fl_key]]
                    text_color = get_text_color_for_bg(cell_bg_color)
                    st.markdown(
                        f"<div style='background-color:{cell_bg_color}; color:{text_color}; padding:5px; margin:1px; border-radius:3px; text-align:center; font-weight:bold; min-height:40px; display:flex; align-items:center; justify-content:center;'>{count}</div>",
                        unsafe_allow_html=True)
                    if count > 0:
                        button_key = f"view_risk_{il_key}_{fl_key}"
                        if st.button("👁️", key=button_key, help=f"ดูรายการ - {count} รายการ", use_container_width=True):
                            st.session_state.clicked_risk_impact = il_key
                            st.session_state.clicked_risk_freq = fl_key
                            st.session_state.show_incident_table = True
                            st.rerun()
                    else:
                        st.markdown("<div style='height:38px; margin-top:5px;'></div>", unsafe_allow_html=True)

        if st.session_state.get('show_incident_table', False) and st.session_state.clicked_risk_impact is not None:
            il_selected = st.session_state.clicked_risk_impact
            fl_selected = st.session_state.clicked_risk_freq

            df_incidents_in_cell = df_filtered[(df_filtered['Impact Level'].astype(str) == il_selected) & (
                    df_filtered['Frequency Level'].astype(str) == fl_selected)].copy()
            expander_title = f"รายการอุบัติการณ์: Impact Level {il_selected}, Frequency Level {fl_selected} - พบ {len(df_incidents_in_cell)} รายการ"
            with st.expander(expander_title, expanded=True):
                st.dataframe(df_incidents_in_cell[display_cols_common], use_container_width=True, hide_index=True)
                if st.button("ปิดรายการ", key="clear_risk_selection_button"):
                    st.session_state.show_incident_table = False
                    st.session_state.clicked_risk_impact = None
                    st.session_state.clicked_risk_freq = None
                    st.rerun()

        st.write("---")
        st.subheader("ตารางสรุปสีตามระดับความเสี่ยงสูงสุดของแต่ละอุบัติการณ์")
        st.info("สีและป้ายกำกับ (I: Impact, F: Frequency) มาจากช่องที่มีความเสี่ยงสูงสุดของอุบัติการณ์ประเภทนั้นๆ")

        if 'Impact Level' in df_filtered.columns and 'Frequency Level' in df_filtered.columns:
            incident_risk_summary = df_filtered.groupby(['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง']).agg(
                max_impact_level=('Impact Level', 'max'),
                frequency_level=('Frequency Level', 'first'),
                total_occurrences=('Incident Rate/mth', 'first')
            ).reset_index()

            def get_color_for_incident(row):
                il_key, fl_key = str(row['max_impact_level']), str(row['frequency_level'])
                if il_key in impact_to_color_row and fl_key in freq_to_color_col:
                    return colors2[impact_to_color_row[il_key], freq_to_color_col[fl_key]]
                return '#808080'

            incident_risk_summary['risk_color_hex'] = incident_risk_summary.apply(get_color_for_incident, axis=1)
            incident_risk_summary = incident_risk_summary.sort_values(by='total_occurrences', ascending=False)

            st.write("---")
            for _, row in incident_risk_summary.iterrows():
                color, text_color = row['risk_color_hex'], get_text_color_for_bg(row['risk_color_hex'])
                risk_label = f"I: {row['max_impact_level']} | F: {row['frequency_level']}"
                col1, col2 = st.columns([1, 6])
                with col1:
                    st.markdown(
                        f'<div style="background-color: {color}; color: {text_color}; font-weight: bold; text-align: center; padding: 8px; border-radius: 5px; height: 100%; display: flex; align-items: center; justify-content: center;">{risk_label}</div>',
                        unsafe_allow_html=True)
                with col2:
                    st.markdown(
                        f"**{row['รหัส']} | {row['ชื่ออุบัติการณ์ความเสี่ยง']}** (อัตราการเกิด: {row.get('total_occurrences', 0):.2f} ครั้ง/เดือน)")
        else:
            st.warning("ไม่พบคอลัมน์ 'Impact Level' หรือ 'Frequency Level' ที่จำเป็นสำหรับการสร้างตารางสรุปสี")

    elif selected_analysis == "กราฟสรุปอุบัติการณ์ (รายมิติ)":
        st.markdown("<h4 style='color: #001f3f;'>กราฟสรุปอุบัติการณ์ (แบ่งตามมิติต่างๆ)</h4>", unsafe_allow_html=True)
        pastel_color_discrete_map_dimensions = {'Critical': '#FF9999', 'High': '#FFCC99', 'Medium': '#FFFF99',
                                                'Low': '#99FF99', 'Undefined': '#D3D3D3'}
        tab1_v, tab2_v, tab3_v, tab4_v = st.tabs(
            ["👁️By Goals (หมวด)", "👁️By Group (กลุ่ม)", "👁️By Shift (เวร)", "👁️By Place (สถานที่)"])

        # ✅ แก้ไข: ใช้ df_filtered
        df_charts = df_filtered.copy()
        df_charts['Count'] = 1

        with tab1_v:
            # ✅ แก้ไข: ใช้ total_month ที่คำนวณใหม่
            st.markdown(f"Incidents by Safety Goals ({total_month} เดือน)")
            if 'หมวด' in df_charts.columns:
                df_c1 = df_charts[~df_charts['หมวด'].isin(
                    ['N/A', 'N/A (ข้อมูลจาก AllCode ไม่พร้อมใช้งาน)', 'N/A (ไม่พบรหัสใน AllCode)'])]
                if not df_c1.empty:
                    fig_c1 = px.bar(df_c1.groupby(['หมวด', 'Category Color']).size().reset_index(name='Count'),
                                    x='หมวด', y='Count', color='Category Color',
                                    color_discrete_map=pastel_color_discrete_map_dimensions)
                    st.plotly_chart(fig_c1, use_container_width=True)
        with tab2_v:
            st.markdown(f"Incidents by Group ({total_month} เดือน)")
            if 'กลุ่ม' in df_charts.columns:
                df_c2 = df_charts[~df_charts['กลุ่ม'].isin(
                    ['N/A', 'N/A (ข้อมูลจาก AllCode ไม่พร้อมใช้งาน)', 'N/A (ไม่พบรหัสใน AllCode)'])]
                if not df_c2.empty:
                    fig_c2 = px.bar(df_c2.groupby(['กลุ่ม', 'Category Color']).size().reset_index(name='Count'),
                                    x='กลุ่ม', y='Count', color='Category Color',
                                    color_discrete_map=pastel_color_discrete_map_dimensions)
                    st.plotly_chart(fig_c2, use_container_width=True)
        with tab3_v:
            st.markdown(f"Incidents by Shift ({total_month} เดือน)")
            if 'ช่วงเวลา/เวร' in df_charts.columns:
                df_c3 = df_charts[df_charts['ช่วงเวลา/เวร'].notna() & ~df_charts['ช่วงเวลา/เวร'].isin(['None', 'N/A'])]
                if not df_c3.empty:
                    fig_c3 = px.bar(df_c3.groupby(['ช่วงเวลา/เวร', 'Category Color']).size().reset_index(name='Count'),
                                    x='ช่วงเวลา/เวร', y='Count', color='Category Color',
                                    color_discrete_map=pastel_color_discrete_map_dimensions)
                    st.plotly_chart(fig_c3, use_container_width=True)
        with tab4_v:
            st.markdown(f"Incidents by Place ({total_month} เดือน)")
            if 'ชนิดสถานที่' in df_charts.columns:
                df_c4 = df_charts[df_charts['ชนิดสถานที่'].notna() & ~df_charts['ชนิดสถานที่'].isin(['None', 'N/A'])]
                if not df_c4.empty:
                    fig_c4 = px.bar(df_c4.groupby(['ชนิดสถานที่', 'Category Color']).size().reset_index(name='Count'),
                                    x='ชนิดสถานที่', y='Count', color='Category Color',
                                    color_discrete_map=pastel_color_discrete_map_dimensions)
                    st.plotly_chart(fig_c4, use_container_width=True)
    elif selected_analysis == "Sankey: ภาพรวม":
        st.markdown("<h4 style='color: #001f3f;'>Sankey Diagram: ภาพรวม</h4>", unsafe_allow_html=True)
        st.markdown("""
        <style>
            .plot-container .svg-container .sankey-node text {
                stroke-width: 0 !important; text-shadow: none !important; paint-order: stroke fill;
            }
        </style>
        """, unsafe_allow_html=True)

        req_cols = ['หมวด', 'Impact', 'Impact Level', 'Category Color']
        if not all(col in df_filtered.columns for col in req_cols):
            st.warning(f"ไม่พบคอลัมน์ที่จำเป็น ({', '.join(req_cols)}) สำหรับการสร้าง Sankey diagram")
        else:
            # ✅ ใช้ df_filtered เป็นข้อมูลตั้งต้น
            sankey_df = df_filtered.copy()

            placeholders = ['None', '', 'N/A', 'ไม่ระบุ',
                            'N/A (ข้อมูลจาก AllCode ไม่พร้อมใช้งาน)',
                            'N/A (ไม่พบรหัสใน AllCode หรือค่าว่างใน AllCode)']

            # กรองข้อมูลที่มี 'หมวด' ที่ถูกต้องออกมาเพื่อใช้งาน
            sankey_df = sankey_df[~sankey_df['หมวด'].astype(str).isin(placeholders)]

            if sankey_df.empty:
                st.warning("ไม่สามารถสร้าง Sankey Diagram ได้ เนื่องจากไม่มีข้อมูล 'หมวด' ที่ถูกต้องในช่วงเวลาที่เลือก")
            else:
                sankey_df['หมวด_Node'] = "หมวด: " + sankey_df['หมวด'].astype(str).str.strip()
                sankey_df['Impact_AI_Node'] = "Impact: " + sankey_df['Impact'].astype(str).str.strip()
                sankey_df.loc[
                    sankey_df['Impact'].astype(str).isin(placeholders), 'Impact_AI_Node'] = "Impact: ไม่ระบุ A-I"

                impact_level_display_map = {'1': "Level: 1 (A-B)", '2': "Level: 2 (C-D)", '3': "Level: 3 (E-F)",
                                            '4': "Level: 4 (G-H)", '5': "Level: 5 (I)", 'N/A': "Level: ไม่ระบุ"}
                sankey_df['Impact_Level_Node'] = sankey_df['Impact Level'].astype(str).str.strip().map(
                    impact_level_display_map).fillna("Level: ไม่ระบุ")
                sankey_df['Risk_Category_Node'] = "Risk: " + sankey_df['Category Color'].astype(str).str.strip()

                node_cols = ['หมวด_Node', 'Impact_AI_Node', 'Impact_Level_Node', 'Risk_Category_Node']
                sankey_df.dropna(subset=node_cols, inplace=True)

                labels_muad = sorted(list(sankey_df['หมวด_Node'].unique()))
                impact_ai_order = [f"Impact: {i}" for i in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']] + [
                    "Impact: ไม่ระบุ A-I"]
                labels_impact_ai = sorted(list(sankey_df['Impact_AI_Node'].unique()),
                                          key=lambda x: impact_ai_order.index(x) if x in impact_ai_order else 999)
                level_order_map = {"Level: 1 (A-B)": 1, "Level: 2 (C-D)": 2, "Level: 3 (E-F)": 3, "Level: 4 (G-H)": 4,
                                   "Level: 5 (I)": 5, "Level: ไม่ระบุ": 6}
                labels_impact_level = sorted(list(sankey_df['Impact_Level_Node'].unique()),
                                             key=lambda x: level_order_map.get(x, 999))
                risk_order = ["Risk: Critical", "Risk: High", "Risk: Medium", "Risk: Low", "Risk: Undefined"]
                labels_risk_cat = sorted(list(sankey_df['Risk_Category_Node'].unique()),
                                         key=lambda x: risk_order.index(x) if x in risk_order else 999)

                all_labels_ordered = labels_muad + labels_impact_ai + labels_impact_level + labels_risk_cat
                all_labels = list(pd.Series(all_labels_ordered).unique())
                label_to_idx = {label: i for i, label in enumerate(all_labels)}

                source_indices, target_indices, values = [], [], []
                links1 = sankey_df.groupby(['หมวด_Node', 'Impact_AI_Node']).size().reset_index(name='value')
                for _, row in links1.iterrows():
                    if row['หมวด_Node'] in label_to_idx and row['Impact_AI_Node'] in label_to_idx:
                        source_indices.append(label_to_idx[row['หมวด_Node']])
                        target_indices.append(label_to_idx[row['Impact_AI_Node']])
                        values.append(row['value'])

                links2 = sankey_df.groupby(['Impact_AI_Node', 'Impact_Level_Node']).size().reset_index(name='value')
                for _, row in links2.iterrows():
                    if row['Impact_AI_Node'] in label_to_idx and row['Impact_Level_Node'] in label_to_idx:
                        source_indices.append(label_to_idx[row['Impact_AI_Node']])
                        target_indices.append(label_to_idx[row['Impact_Level_Node']])
                        values.append(row['value'])

                links3 = sankey_df.groupby(['Impact_Level_Node', 'Risk_Category_Node']).size().reset_index(name='value')
                for _, row in links3.iterrows():
                    if row['Impact_Level_Node'] in label_to_idx and row['Risk_Category_Node'] in label_to_idx:
                        source_indices.append(label_to_idx[row['Impact_Level_Node']])
                        target_indices.append(label_to_idx[row['Risk_Category_Node']])
                        values.append(row['value'])

                if source_indices:
                    node_colors = []
                    palette1, palette2, palette3 = px.colors.qualitative.Pastel1, px.colors.qualitative.Pastel2, px.colors.qualitative.Set3
                    risk_color_map = {"Risk: Critical": "red", "Risk: High": "orange", "Risk: Medium": "#F7DC6F",
                                      "Risk: Low": "green", "Risk: Undefined": "grey"}
                    for label in all_labels:
                        if label.startswith("หมวด:"):
                            node_colors.append(palette1[labels_muad.index(label) % len(palette1)])
                        elif label.startswith("Impact:"):
                            node_colors.append(palette2[labels_impact_ai.index(label) % len(palette2)])
                        elif label.startswith("Level:"):
                            node_colors.append(palette3[labels_impact_level.index(label) % len(palette3)])
                        elif label.startswith("Risk:"):
                            node_colors.append(risk_color_map.get(label, 'grey'))
                        else:
                            node_colors.append('rgba(200,200,200,0.8)')

                    link_colors_rgba = [
                        f'rgba({int(c.lstrip("#")[0:2], 16)},{int(c.lstrip("#")[2:4], 16)},{int(c.lstrip("#")[4:6], 16)},0.3)' if c.startswith(
                            '#') else 'rgba(200,200,200,0.3)' for c in [node_colors[s] for s in source_indices]]

                    fig = go.Figure(data=[go.Sankey(
                        arrangement='snap',
                        node=dict(pad=15, thickness=18, line=dict(color="rgba(0,0,0,0.6)", width=0.75),
                                  label=all_labels, color=node_colors,
                                  hovertemplate='%{label} มีจำนวน: %{value}<extra></extra>'),
                        link=dict(source=source_indices, target=target_indices, value=values, color=link_colors_rgba,
                                  hovertemplate='จาก %{source.label}<br />ไปยัง %{target.label}<br />จำนวน: %{value}<extra></extra>')
                    )])
                    fig.update_layout(
                        title_text="<b>แผนภาพ Sankey:</b> หมวด -> Impact (A-I) -> Impact Level (1-5) -> Risk Category",
                        font_size=12, height=max(700, len(all_labels) * 18), template='plotly_white',
                        margin=dict(t=60, l=10, r=10, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("ไม่สามารถสร้างลิงก์สำหรับ Sankey diagram ได้ (อาจไม่มีความเชื่อมโยงของข้อมูล)")

    elif selected_analysis == "Sankey: มาตรฐานสำคัญจำเป็นต่อความปลอดภัย 9 ข้อ":
        st.markdown("<h4 style='color: #001f3f;'>Sankey Diagram: มาตรฐานสำคัญจำเป็นต่อความปลอดภัย 9 ข้อ</h4>",
                    unsafe_allow_html=True)
        st.markdown("""
        <style>
            .plot-container .svg-container .sankey-node text {
                stroke-width: 0 !important; text-shadow: none !important; paint-order: stroke fill;
            }
        </style>
        """, unsafe_allow_html=True)

        required_cols = ['หมวดหมู่มาตรฐานสำคัญ', 'รหัส', 'Impact', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Category Color']
        if not all(col in df_filtered.columns for col in required_cols):
            st.warning(f"ไม่พบคอลัมน์ที่จำเป็น ({', '.join(required_cols)}) สำหรับการสร้าง Sankey diagram")
        else:
            # ✅ แก้ไข: ใช้ df_filtered เป็นข้อมูลตั้งต้น
            sankey_df_new = df_filtered.copy()

            placeholders_to_filter = ["ไม่จัดอยู่ใน PSG9 Catalog", "ไม่สามารถระบุ (Merge PSG9 ล้มเหลว)",
                                      "ไม่สามารถระบุ (PSG9code.xlsx ไม่ได้โหลด)"]
            sankey_df_new = sankey_df_new[
                ~sankey_df_new['หมวดหมู่มาตรฐานสำคัญ'].astype(str).isin(placeholders_to_filter)]

            if sankey_df_new.empty:
                st.warning("ไม่พบข้อมูลที่เกี่ยวข้องกับ PSG9 ในช่วงเวลาที่เลือก")
            else:
                psg9_to_cp_gp_map = {PSG9_label_dict[num].strip(): 'CP (หมวดตาม PSG9)' for num in
                                     [1, 3, 4, 5, 6, 7, 8, 9] if num in PSG9_label_dict}
                psg9_to_cp_gp_map.update(
                    {PSG9_label_dict[num].strip(): 'GP (หมวดตาม PSG9)' for num in [2] if num in PSG9_label_dict})

                sankey_df_new['หมวด_CP_GP_Node'] = sankey_df_new['หมวดหมู่มาตรฐานสำคัญ'].map(psg9_to_cp_gp_map)
                sankey_df_new['หมวดหมู่PSG_Node'] = "PSG9: " + sankey_df_new['หมวดหมู่มาตรฐานสำคัญ']
                sankey_df_new['รหัส_Node'] = "รหัส: " + sankey_df_new['รหัส']
                sankey_df_new['Impact_AI_Node'] = "Impact: " + sankey_df_new['Impact']
                sankey_df_new['Risk_Category_Node'] = "Risk: " + sankey_df_new['Category Color']
                sankey_df_new['ชื่ออุบัติการณ์ความเสี่ยง_for_hover'] = sankey_df_new[
                    'ชื่ออุบัติการณ์ความเสี่ยง'].fillna('ไม่มีคำอธิบาย')

                cols_for_dropna = ['หมวด_CP_GP_Node', 'หมวดหมู่PSG_Node', 'รหัส_Node', 'Impact_AI_Node',
                                   'Risk_Category_Node']
                sankey_df_new.dropna(subset=cols_for_dropna, inplace=True)

                labels_muad_cp_gp_simp = sorted(list(sankey_df_new['หมวด_CP_GP_Node'].unique()))
                labels_psg9_cat_simp = sorted(list(sankey_df_new['หมวดหมู่PSG_Node'].unique()))
                rh_node_to_desc_map = sankey_df_new.drop_duplicates(subset=['รหัส_Node']).set_index('รหัส_Node')[
                    'ชื่ออุบัติการณ์ความเสี่ยง_for_hover'].to_dict()
                labels_rh_simp = sorted(list(sankey_df_new['รหัส_Node'].unique()))
                labels_impact_ai_simp = sorted(list(sankey_df_new['Impact_AI_Node'].unique()))
                risk_order = ["Risk: Critical", "Risk: High", "Risk: Medium", "Risk: Low", "Risk: Undefined"]
                labels_risk_category = sorted(list(sankey_df_new['Risk_Category_Node'].unique()),
                                              key=lambda x: risk_order.index(x) if x in risk_order else 99)

                all_labels_ordered_simp = labels_muad_cp_gp_simp + labels_psg9_cat_simp + labels_rh_simp + labels_impact_ai_simp + labels_risk_category
                all_labels_simp = list(pd.Series(all_labels_ordered_simp).unique())
                label_to_idx_simp = {label: i for i, label in enumerate(all_labels_simp)}
                customdata_for_nodes_simp = [
                    f"<br>คำอธิบาย: {str(rh_node_to_desc_map.get(label_node, ''))}" if label_node in rh_node_to_desc_map else ""
                    for label_node in all_labels_simp]

                source_indices_simp, target_indices_simp, values_simp = [], [], []
                links_l1 = sankey_df_new.groupby(['หมวด_CP_GP_Node', 'หมวดหมู่PSG_Node']).size().reset_index(
                    name='value')
                for _, row in links_l1.iterrows():
                    if row['หมวด_CP_GP_Node'] in label_to_idx_simp and row['หมวดหมู่PSG_Node'] in label_to_idx_simp:
                        source_indices_simp.append(label_to_idx_simp[row['หมวด_CP_GP_Node']])
                        target_indices_simp.append(label_to_idx_simp[row['หมวดหมู่PSG_Node']])
                        values_simp.append(row['value'])

                links_l2 = sankey_df_new.groupby(['หมวดหมู่PSG_Node', 'รหัส_Node']).size().reset_index(name='value')
                for _, row in links_l2.iterrows():
                    if row['หมวดหมู่PSG_Node'] in label_to_idx_simp and row['รหัส_Node'] in label_to_idx_simp:
                        source_indices_simp.append(label_to_idx_simp[row['หมวดหมู่PSG_Node']])
                        target_indices_simp.append(label_to_idx_simp[row['รหัส_Node']])
                        values_simp.append(row['value'])

                links_l3 = sankey_df_new.groupby(['รหัส_Node', 'Impact_AI_Node']).size().reset_index(name='value')
                for _, row in links_l3.iterrows():
                    if row['รหัส_Node'] in label_to_idx_simp and row['Impact_AI_Node'] in label_to_idx_simp:
                        source_indices_simp.append(label_to_idx_simp[row['รหัส_Node']])
                        target_indices_simp.append(label_to_idx_simp[row['Impact_AI_Node']])
                        values_simp.append(row['value'])

                links_l4 = sankey_df_new.groupby(['Impact_AI_Node', 'Risk_Category_Node']).size().reset_index(
                    name='value')
                for _, row in links_l4.iterrows():
                    if row['Impact_AI_Node'] in label_to_idx_simp and row['Risk_Category_Node'] in label_to_idx_simp:
                        source_indices_simp.append(label_to_idx_simp[row['Impact_AI_Node']])
                        target_indices_simp.append(label_to_idx_simp[row['Risk_Category_Node']])
                        values_simp.append(row['value'])

                if source_indices_simp:
                    node_colors_simp = []
                    palette_l1, palette_l2, palette_l3, palette_l4 = px.colors.qualitative.Bold, px.colors.qualitative.Pastel, px.colors.qualitative.Vivid, px.colors.qualitative.Set3
                    risk_cat_color_map = {"Risk: Critical": "red", "Risk: High": "orange", "Risk: Medium": "#F7DC6F",
                                          "Risk: Low": "green", "Risk: Undefined": "grey"}
                    for label_node in all_labels_simp:
                        if label_node in labels_muad_cp_gp_simp:
                            node_colors_simp.append(
                                palette_l1[labels_muad_cp_gp_simp.index(label_node) % len(palette_l1)])
                        elif label_node in labels_psg9_cat_simp:
                            node_colors_simp.append(
                                palette_l2[labels_psg9_cat_simp.index(label_node) % len(palette_l2)])
                        elif label_node in labels_rh_simp:
                            node_colors_simp.append(palette_l3[labels_rh_simp.index(label_node) % len(palette_l3)])
                        elif label_node in labels_impact_ai_simp:
                            node_colors_simp.append(
                                palette_l4[labels_impact_ai_simp.index(label_node) % len(palette_l4)])
                        elif label_node in labels_risk_category:
                            node_colors_simp.append(risk_cat_color_map.get(label_node, 'grey'))
                        else:
                            node_colors_simp.append('rgba(200,200,200,0.8)')

                    link_colors_simp = []
                    default_link_color_simp = 'rgba(200,200,200,0.35)'
                    for s_idx in source_indices_simp:
                        try:
                            hex_color = node_colors_simp[s_idx]
                            h = hex_color.lstrip('#')
                            rgb_tuple = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
                            link_colors_simp.append(f'rgba({rgb_tuple[0]},{rgb_tuple[1]},{rgb_tuple[2]},0.3)')
                        except:
                            link_colors_simp.append(default_link_color_simp)

                    fig_sankey_psg9_simplified = go.Figure(data=[go.Sankey(
                        arrangement='snap',
                        node=dict(pad=10, thickness=15, line=dict(color="rgba(0,0,0,0.4)", width=0.4),
                                  label=all_labels_simp, color=node_colors_simp, customdata=customdata_for_nodes_simp,
                                  hovertemplate='<b>%{label}</b><br>จำนวน: %{value}%{customdata}<extra></extra>'),
                        link=dict(source=source_indices_simp, target=target_indices_simp, value=values_simp,
                                  color=link_colors_simp,
                                  hovertemplate='จาก %{source.label}<br />ไปยัง %{target.label}<br />จำนวน: %{value}<extra></extra>')
                    )])
                    fig_sankey_psg9_simplified.update_layout(
                        title_text="<b>แผนภาพ SANKEY:</b> CP/GP -> หมวดหมู่ PSG9 -> รหัส -> Impact -> Risk Category",
                        font_size=11, height=max(800, len(all_labels_simp) * 12 + 200),
                        template='plotly_white', margin=dict(t=70, l=10, r=10, b=20)
                    )
                    st.plotly_chart(fig_sankey_psg9_simplified, use_container_width=True)
                else:
                    st.warning("ไม่สามารถสร้างลิงก์สำหรับ Sankey diagram (PSG9) ได้")

    elif selected_analysis == "Scatter Plot & Top 10":
        st.markdown("<h4 style='color: #001f3f;'>ความสัมพันธ์ระหว่างรหัสอุบัติการณ์และหมวดหมู่ (Scatter Plot)</h4>",
                    unsafe_allow_html=True)

        scatter_cols = ['รหัส', 'หมวด', 'Category Color', 'Incident Rate/mth']
        if all(col in df_filtered.columns for col in scatter_cols):
            # ✅ แก้ไข: ใช้ df_filtered เป็นข้อมูลตั้งต้น
            df_sc = df_filtered.dropna(subset=scatter_cols, how='any')
            if not df_sc.empty:
                fig_sc = px.scatter(
                    df_sc,
                    x='รหัส',
                    y='หมวด',
                    color='Category Color',
                    size='Incident Rate/mth',
                    hover_data=['Incident', 'ชื่ออุบัติการณ์ความเสี่ยง'],
                    size_max=30
                )
                st.plotly_chart(fig_sc, theme="streamlit", use_container_width=True)
            else:
                st.info("ไม่มีข้อมูลเพียงพอสำหรับสร้าง Scatter Plot ในช่วงเวลาที่เลือก")
        else:
            st.warning(f"ไม่พบคอลัมน์ที่จำเป็นสำหรับ Scatter Plot: {', '.join(scatter_cols)}")

        st.markdown("---")
        st.subheader("Top 10 อุบัติการณ์ (ตามความถี่)")

        # ✅ แก้ไข: df_freq ถูกคำนวณจาก df_filtered อยู่แล้ว และเพิ่มการ join ชื่อเต็มเข้ามา
        if not df_freq.empty:
            df_freq_top10 = df_freq.nlargest(10, 'count')

            # นำชื่อเต็มของอุบัติการณ์มาแสดงในตารางด้วย
            incident_names = df_filtered[['Incident', 'ชื่ออุบัติการณ์ความเสี่ยง']].drop_duplicates()
            df_freq_top10 = pd.merge(df_freq_top10, incident_names, on='Incident', how='left')

            st.dataframe(
                df_freq_top10[['Incident', 'count']],
                column_config={
                    "Incident": "รหัส Incident",
                    "count": "จำนวนครั้ง"
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("ไม่สามารถแสดง Top 10 อุบัติการณ์ได้")

    elif selected_analysis == "สรุปอุบัติการณ์ตาม Safety Goals":
        st.markdown("<h4 style='color: #001f3f;'>สรุปอุบัติการณ์ตามเป้าหมาย (Safety Goals)</h4>",
                    unsafe_allow_html=True)

        goal_definitions = {
            "Patient Safety/ Common Clinical Risk": "P:Patient Safety Goals หรือ Common Clinical Risk Incident",
            "Specific Clinical Risk": "S:Specific Clinical Risk Incident",
            "Personnel Safety": "P:Personnel Safety Goals",
            "Organization Safety": "O:Organization Safety Goals"
        }

        # --- ส่วนที่ 1: แสดงตารางสรุปเหมือนเดิม ---
        for display_name, cat_name in goal_definitions.items():
            st.markdown(f"##### {display_name}")

            is_org_safety = (display_name == "Organization Safety")

            summary_table = create_goal_summary_table(
                df_filtered,
                cat_name,
                e_up_non_numeric_levels_param=[] if is_org_safety else ['A', 'B', 'C', 'D'],
                e_up_numeric_levels_param=['1', '2'] if is_org_safety else None,
                is_org_safety_table=is_org_safety
            )

            if summary_table is not None and not summary_table.empty:
                st.dataframe(summary_table, use_container_width=True)
            else:
                st.info(f"ไม่มีข้อมูลสำหรับ '{display_name}' ในช่วงเวลาที่เลือก")

        # --- ส่วนที่ 2: กราฟแท่งเปรียบเทียบ % E-up (ที่เพิ่มเข้ามา) ---
        st.markdown("---")
        st.subheader("📊 เจาะลึกสัดส่วนอุบัติการณ์รุนแรง (% E-up) ในแต่ละหัวข้อ")

        severe_levels = ['E', 'F', 'G', 'H', 'I', '3', '4', '5']
        valid_goals = [goal for goal in df_filtered['หมวด'].unique() if
                       goal and goal not in ['N/A', 'N/A (ข้อมูลจาก AllCode ไม่พร้อมใช้งาน)',
                                             'N/A (ไม่พบรหัสใน AllCode)']]

        for goal in valid_goals:
            st.markdown(f"#### {goal}")

            goal_df = df_filtered[df_filtered['หมวด'] == goal].copy()

            summary = goal_df.groupby('Incident Type Name').apply(
                lambda x: (x['Impact'].isin(severe_levels).sum() / len(x) * 100) if len(x) > 0 else 0
            ).reset_index(name='ร้อยละ E-up')

            summary = summary[summary['ร้อยละ E-up'] > 0].sort_values(by='ร้อยละ E-up', ascending=True)

            if summary.empty:
                st.info("ไม่พบอุบัติการณ์รุนแรง (E-up) ในหมวดหมู่นี้")
                continue

            fig = px.bar(
                summary,
                x='ร้อยละ E-up',
                y='Incident Type Name',
                orientation='h',
                title=f"สัดส่วนอุบัติการณ์รุนแรงในหมวด {goal}",
                labels={'Incident Type Name': 'หัวข้ออุบัติการณ์', 'ร้อยละ E-up': 'ร้อยละของอุบัติการณ์รุนแรง (%)'},
                text_auto='.2f',
                color='ร้อยละ E-up',
                color_continuous_scale='Reds'
            )
            fig.update_layout(yaxis_title=None, xaxis_ticksuffix="%")
            st.plotly_chart(fig, use_container_width=True)

        # --- ส่วนที่ 3: กราฟ Sunburst (ที่เพิ่มเข้ามา) ---
        st.markdown("---")
        st.subheader("☀️ ภาพรวมสัดส่วนอุบัติการณ์รุนแรงแบบ Sunburst")

        total_counts = df_filtered.groupby(['หมวด', 'Incident Type Name']).size().reset_index(name='จำนวนทั้งหมด')
        severe_df = df_filtered[df_filtered['Impact'].isin(severe_levels)]
        severe_counts = severe_df.groupby(['หมวด', 'Incident Type Name']).size().reset_index(name='จำนวน E-up')
        summary_df = pd.merge(total_counts, severe_counts, on=['หมวด', 'Incident Type Name'], how='left').fillna(0)
        summary_df['ร้อยละ E-up'] = (summary_df['จำนวน E-up'] / summary_df['จำนวนทั้งหมด'] * 100)
        summary_df = summary_df[summary_df['จำนวนทั้งหมด'] > 0]

        if summary_df.empty:
            st.info("ไม่มีข้อมูลเพียงพอสำหรับสร้างกราฟ Sunburst")
        else:
            fig_sunburst = px.sunburst(
                summary_df,
                path=['หมวด', 'Incident Type Name'],
                values='จำนวนทั้งหมด',
                color='ร้อยละ E-up',
                color_continuous_scale='YlOrRd',
                hover_data={'ร้อยละ E-up': ':.2f'},
                title="ภาพรวมสัดส่วนอุบัติการณ์รุนแรง (ขนาด = จำนวนรวม, สี = % E-up)"
            )
            fig_sunburst.update_traces(textinfo="label+percent entry")
            st.plotly_chart(fig_sunburst, use_container_width=True)

    elif selected_analysis == "วิเคราะห์ตามหมวดหมู่และสถานะการแก้ไข":
        st.markdown("<h4 style='color: #001f3f;'>วิเคราะห์ตามหมวดหมู่และสถานะการแก้ไข</h4>", unsafe_allow_html=True)

        if 'Resulting Actions' not in df_filtered.columns or 'หมวดหมู่มาตรฐานสำคัญ' not in df_filtered.columns:
            st.error(
                "ไม่สามารถแสดงข้อมูลได้ เนื่องจากไม่พบคอลัมน์ 'Resulting Actions' หรือ 'หมวดหมู่มาตรฐานสำคัญ' ในข้อมูล")
        else:
            tab_psg9, tab_groups, tab_summary, tab_waitlist = st.tabs(
                ["👁️ วิเคราะห์ตามหมวดหมู่ PSG9", "👁️ วิเคราะห์ตามกลุ่มหลัก (C/G)",
                 "👁️ สรุปเปอร์เซ็นต์การแก้ไขอุบัติการณ์รุนแรง (E-I & 3-5)","👁️ อุบัติการณ์ที่รอการแก้ไข(ตามความรุนแรง)"])

            # --- Tab ที่ 1: วิเคราะห์ตามหมวดหมู่ PSG9 ---
            with tab_psg9:
                st.subheader("ภาพรวมอุบัติการณ์ตามมาตรฐานสำคัญจำเป็นต่อความปลอดภัย (PSG9)")
                # ✅ แก้ไข: ใช้ df_filtered
                psg9_summary_table = create_psg9_summary_table(df_filtered)
                if psg9_summary_table is not None and not psg9_summary_table.empty:
                    st.dataframe(psg9_summary_table, use_container_width=True)
                else:
                    st.info("ไม่พบข้อมูลอุบัติการณ์ที่เกี่ยวข้องกับมาตรฐานสำคัญ 9 ข้อในช่วงเวลานี้")

                st.markdown("---")
                st.subheader("สถานะการแก้ไขในแต่ละหมวดหมู่ PSG9")

                # ✅ แก้ไข: ใช้ df_filtered
                psg9_categories = {k: v for k, v in PSG9_label_dict.items() if
                                   v in df_filtered['หมวดหมู่มาตรฐานสำคัญ'].unique()}

                for psg9_id, psg9_name in psg9_categories.items():
                    # ✅ แก้ไข: ใช้ df_filtered
                    psg9_df = df_filtered[df_filtered['หมวดหมู่มาตรฐานสำคัญ'] == psg9_name]
                    total_count = len(psg9_df)
                    resolved_df = psg9_df[~psg9_df['Resulting Actions'].astype(str).isin(['None', '', 'nan'])]
                    resolved_count = len(resolved_df)
                    unresolved_count = total_count - resolved_count

                    expander_title = f"{psg9_name} (ทั้งหมด: {total_count} | แก้ไขแล้ว: {resolved_count} | รอแก้ไข: {unresolved_count})"
                    with st.expander(expander_title):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("จำนวนทั้งหมด", f"{total_count:,}")
                        c2.metric("ดำเนินการแก้ไขแล้ว", f"{resolved_count:,}")
                        c3.metric("รอการแก้ไข", f"{unresolved_count:,}")

                        if total_count > 0:
                            tab_resolved, tab_unresolved = st.tabs(
                                [f"รายการที่แก้ไขแล้ว ({resolved_count})", f"รายการที่รอการแก้ไข ({unresolved_count})"])
                            with tab_resolved:
                                if resolved_count > 0:
                                    st.dataframe(
                                        resolved_df[['Occurrence Date', 'Incident', 'Impact', 'Resulting Actions']],
                                        hide_index=True, use_container_width=True, column_config={
                                            "Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด",
                                                                                               format="DD/MM/YYYY")})
                                else:
                                    st.info("ไม่มีรายการที่แก้ไขแล้วในหมวดนี้")
                            with tab_unresolved:
                                if unresolved_count > 0:
                                    st.dataframe(
                                        psg9_df[psg9_df['Resulting Actions'].astype(str).isin(['None', '', 'nan'])][
                                            ['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด']],
                                        hide_index=True, use_container_width=True, column_config={
                                            "Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด",
                                                                                               format="DD/MM/YYYY")})
                                else:
                                    st.success("อุบัติการณ์ทั้งหมดในหมวดนี้ได้รับการแก้ไขแล้ว")

            # --- Tab ที่ 2: วิเคราะห์ตามกลุ่มหลัก (C/G) ---
            with tab_groups:
                st.subheader("เจาะลึกสถานะการแก้ไขตามกลุ่มหลักและหมวดย่อย")
                st.markdown("#### กลุ่มอุบัติการณ์ทางคลินิก (รหัสขึ้นต้นด้วย C)")

                # ✅ แก้ไข: ใช้ df_filtered
                df_clinical = df_filtered[df_filtered['รหัส'].str.startswith('C', na=False)].copy()

                if df_clinical.empty:
                    st.info("ไม่พบข้อมูลอุบัติการณ์กลุ่ม Clinical")
                else:
                    clinical_categories = sorted([cat for cat in df_clinical['หมวด'].unique() if cat and pd.notna(cat)])
                    for category in clinical_categories:
                        category_df = df_clinical[df_clinical['หมวด'] == category]
                        total_count = len(category_df)
                        resolved_df = category_df[
                            ~category_df['Resulting Actions'].astype(str).isin(['None', '', 'nan'])]
                        resolved_count = len(resolved_df)
                        unresolved_count = total_count - resolved_count

                        expander_title = f"{category} (ทั้งหมด: {total_count} | แก้ไขแล้ว: {resolved_count} | รอแก้ไข: {unresolved_count})"
                        with st.expander(expander_title):
                            tab_resolved, tab_unresolved = st.tabs(
                                [f"รายการที่แก้ไขแล้ว ({resolved_count})", f"รายการที่รอการแก้ไข ({unresolved_count})"])
                            with tab_resolved:
                                if resolved_count > 0:
                                    st.dataframe(
                                        resolved_df[['Occurrence Date', 'Incident', 'Impact', 'Resulting Actions']],
                                        hide_index=True, use_container_width=True, column_config={
                                            "Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด",
                                                                                               format="DD/MM/YYYY")})
                                else:
                                    st.info("ไม่มีรายการที่แก้ไขแล้วในหมวดนี้")
                            with tab_unresolved:
                                if unresolved_count > 0:
                                    st.dataframe(category_df[category_df['Resulting Actions'].astype(str).isin(
                                        ['None', '', 'nan'])][
                                                     ['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด']],
                                                 hide_index=True, use_container_width=True, column_config={
                                            "Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด",
                                                                                               format="DD/MM/YYYY")})
                                else:
                                    st.success("อุบัติการณ์ทั้งหมดในหมวดนี้ได้รับการแก้ไขแล้ว")

                st.markdown("---")
                st.markdown("#### กลุ่มอุบัติการณ์ทั่วไป (รหัสขึ้นต้นด้วย G)")

                # ✅ แก้ไข: ใช้ df_filtered
                df_general = df_filtered[df_filtered['รหัส'].str.startswith('G', na=False)].copy()

                if df_general.empty:
                    st.info("ไม่พบข้อมูลอุบัติการณ์กลุ่ม General")
                else:
                    general_categories = sorted([cat for cat in df_general['หมวด'].unique() if cat and pd.notna(cat)])
                    for category in general_categories:
                        category_df = df_general[df_general['หมวด'] == category]
                        total_count = len(category_df)
                        resolved_df = category_df[
                            ~category_df['Resulting Actions'].astype(str).isin(['None', '', 'nan'])]
                        resolved_count = len(resolved_df)
                        unresolved_count = total_count - resolved_count

                        expander_title = f"{category} (ทั้งหมด: {total_count} | แก้ไขแล้ว: {resolved_count} | รอแก้ไข: {unresolved_count})"
                        with st.expander(expander_title):
                            tab_resolved, tab_unresolved = st.tabs(
                                [f"รายการที่แก้ไขแล้ว ({resolved_count})", f"รายการที่รอการแก้ไข ({unresolved_count})"])
                            with tab_resolved:
                                if resolved_count > 0:
                                    st.dataframe(
                                        resolved_df[['Occurrence Date', 'Incident', 'Impact', 'Resulting Actions']],
                                        hide_index=True, use_container_width=True, column_config={
                                            "Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด",
                                                                                               format="DD/MM/YYYY")})
                                else:
                                    st.info("ไม่มีรายการที่แก้ไขแล้วในหมวดนี้")
                            with tab_unresolved:
                                if unresolved_count > 0:
                                    st.dataframe(category_df[category_df['Resulting Actions'].astype(str).isin(
                                        ['None', '', 'nan'])][
                                                     ['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด']],
                                                 hide_index=True, use_container_width=True, column_config={
                                            "Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด",
                                                                                               format="DD/MM/YYYY")})
                                else:
                                    st.success("อุบัติการณ์ทั้งหมดในหมวดนี้ได้รับการแก้ไขแล้ว")

            # --- Tab ที่ 3: สรุปเปอร์เซ็นต์การแก้ไข ---
            with tab_summary:
                st.subheader("สรุปเปอร์เซ็นต์การแก้ไขอุบัติการณ์รุนแรง (E-I & 3-5)")

                # ✅ หมายเหตุ: ค่าเหล่านี้ถูกคำนวณจาก df_filtered ที่ด้านบนของฟังก์ชัน display_executive_dashboard() แล้ว
                total_severe_incidents = metrics_data.get("total_severe_incidents", 0)
                total_severe_unresolved_incidents_val = metrics_data.get("total_severe_unresolved_incidents_val", 0)

                # คำนวณเฉพาะส่วนของ PSG9 จาก df_filtered
                severe_df = df_filtered[df_filtered['Impact Level'].isin(['3', '4', '5'])]
                total_severe_psg9_incidents = severe_df[severe_df['รหัส'].isin(psg9_r_codes_for_counting)].shape[0]
                total_severe_unresolved_psg9_incidents_val = metrics_data.get(
                    "total_severe_unresolved_psg9_incidents_val", 0)

                val_row3_total_pct = (
                            total_severe_unresolved_incidents_val / total_severe_incidents * 100) if total_severe_incidents > 0 else 0
                val_row3_psg9_pct = (
                            total_severe_unresolved_psg9_incidents_val / total_severe_psg9_incidents * 100) if total_severe_psg9_incidents > 0 else 0

                summary_action_data = [
                    {"รายละเอียด": "1. จำนวนอุบัติการณ์รุนแรง E-I & 3-5", "ทั้งหมด": f"{total_severe_incidents:,}",
                     "เฉพาะ PSG9": f"{total_severe_psg9_incidents:,}"},
                    {"รายละเอียด": "2. อุบัติการณ์ E-I & 3-5 ที่ยังไม่ได้รับการแก้ไข",
                     "ทั้งหมด": f"{total_severe_unresolved_incidents_val:,}",
                     "เฉพาะ PSG9": f"{total_severe_unresolved_psg9_incidents_val:,}"},
                    {"รายละเอียด": "3. % อุบัติการณ์ E-I & 3-5 ที่ยังไม่ได้รับการแก้ไข",
                     "ทั้งหมด": f"{val_row3_total_pct:.2f}%", "เฉพาะ PSG9": f"{val_row3_psg9_pct:.2f}%"}
                ]
                st.dataframe(pd.DataFrame(summary_action_data).set_index('รายละเอียด'), use_container_width=True)

            # --- Tab ที่ 4: รายการอุบัติการณ์ที่รอการแก้ไข ---
            with tab_waitlist:
                st.subheader("รายการอุบัติการณ์ที่รอการแก้ไข (ตามความรุนแรง)")
                unresolved_df = df_filtered[df_filtered['Resulting Actions'].astype(str).isin(['None', '', 'nan'])].copy()

                if unresolved_df.empty:
                    st.success("🎉 ไม่พบรายการที่รอการแก้ไขในช่วงเวลานี้ ยอดเยี่ยมมากครับ!")
                else:
                    st.metric("จำนวนรายการที่รอการแก้ไขทั้งหมด", f"{len(unresolved_df):,} รายการ")
                    severity_order = ['Critical', 'High', 'Medium', 'Low', 'Undefined']
                    for severity in severity_order:
                        severity_df = unresolved_df[unresolved_df['Category Color'] == severity]
                        if not severity_df.empty:
                            with st.expander(f"ระดับความรุนแรง: {severity} ({len(severity_df)} รายการ)"):
                                display_cols = ['Occurrence Date', 'Incident', 'Impact',
                                                'รายละเอียดการเกิด']

                                st.dataframe(severity_df[display_cols], use_container_width=True, hide_index=True,
                                             column_config={"Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด",
                                                                                                               format="DD/MM/YYYY")})
    elif selected_analysis == "Persistence Risk Index":
        st.markdown("<h4 style='color: #001f3f;'>ดัชนีความเสี่ยงเรื้อรัง (Persistence Risk Index)</h4>", unsafe_allow_html=True)
        st.info(
            "ตารางนี้ให้คะแนนอุบัติการณ์ที่เกิดขึ้นซ้ำและมีความเสี่ยงโดยเฉลี่ยสูง ซึ่งเป็นปัญหาเรื้อรังที่ควรได้รับการทบทวนเชิงระบบ")

        # ✅ แก้ไข: เรียกใช้ฟังก์ชันด้วย df_filtered และ total_month ที่ผ่านการกรองตามช่วงเวลาแล้ว
        persistence_df = calculate_persistence_risk_score(df_filtered, total_month)

        if not persistence_df.empty:
            display_df_persistence = persistence_df.rename(columns={
                'Persistence_Risk_Score': 'ดัชนีความเรื้อรัง',
                'Average_Ordinal_Risk_Score': 'คะแนนเสี่ยงเฉลี่ย',
                'Incident_Rate_Per_Month': 'อัตราการเกิด (ครั้ง/เดือน)',
                'Total_Occurrences': 'จำนวนครั้งทั้งหมด'
            })
            st.dataframe(
                display_df_persistence[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'คะแนนเสี่ยงเฉลี่ย', 'ดัชนีความเรื้อรัง',
                                        'อัตราการเกิด (ครั้ง/เดือน)', 'จำนวนครั้งทั้งหมด']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "คะแนนเสี่ยงเฉลี่ย": st.column_config.NumberColumn(format="%.2f"),
                    "อัตราการเกิด (ครั้ง/เดือน)": st.column_config.NumberColumn(format="%.2f"),
                    "ดัชนีความเรื้อรัง": st.column_config.ProgressColumn(
                        "ดัชนีความเสี่ยงเรื้อรัง",
                        help="คำนวณจากความถี่และความรุนแรงเฉลี่ย ยิ่งสูงยิ่งเป็นปัญหาเรื้อรัง",
                        min_value=0,
                        max_value=2,  # ค่าสูงสุดทางทฤษฎีคือ 2 (Frequency Score = 1, Severity Score = 1)
                        format="%.2f"
                    )
                }
            )
            st.markdown("---")
            st.markdown("##### กราฟวิเคราะห์ลักษณะของปัญหาเรื้อรัง")
            fig = px.scatter(
                persistence_df,
                x="Average_Ordinal_Risk_Score",
                y="Incident_Rate_Per_Month",
                size="Total_Occurrences",
                color="Persistence_Risk_Score",
                hover_name="ชื่ออุบัติการณ์ความเสี่ยง",
                color_continuous_scale=px.colors.sequential.Reds,
                size_max=60,
                labels={
                    "Average_Ordinal_Risk_Score": "คะแนนความเสี่ยงเฉลี่ย (ยิ่งขวายิ่งรุนแรง)",
                    "Incident_Rate_Per_Month": "อัตราการเกิดต่อเดือน (ยิ่งสูงยิ่งบ่อย)",
                    "Persistence_Risk_Score": "ดัชนีความเรื้อรัง",
                    "Total_Occurrences": "จำนวนครั้งทั้งหมด"
                },
                title="การกระจายตัวของปัญหาเรื้อรัง: ความถี่ vs ความรุนแรง"
            )
            fig.update_layout(xaxis_title="ความรุนแรงเฉลี่ย", yaxis_title="ความถี่เฉลี่ย")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("ไม่มีข้อมูลเพียงพอสำหรับวิเคราะห์ความเสี่ยงเรื้อรังในช่วงเวลานี้")

    elif selected_analysis == "Early Warning: อุบัติการณ์ที่มีแนวโน้มสูงขึ้น":
        st.markdown("<h4 style='color:#001f3f;'>Early Warning: อุบัติการณ์ที่มีแนวโน้มสูงขึ้น</h4>", unsafe_allow_html=True)

        # ตรวจสอบว่ามีฟังก์ชันที่จำเป็นอยู่หรือไม่
        if 'prioritize_incidents_nb_logit_v2' not in globals():
            st.error("ไม่พบฟังก์ชัน `prioritize_incidents_nb_logit_v2` ในโค้ด")
        else:
            # ส่วนควบคุมการวิเคราะห์
            c1, c2, c3 = st.columns(3)
            with c1:
                horizon = st.slider("พยากรณ์ล่วงหน้า (เดือน):", 1, 12, 3, 1, key="ew_horizon")
            with c2:
                min_months = st.slider("ขั้นต่ำเดือนที่ใช้วิเคราะห์:", 3, 12, 4, 1, key="ew_min_months")
            with c3:
                min_total = st.slider("ขั้นต่ำจำนวนครั้งสะสม/รหัส:", 3, 200, 5, 1, key="ew_min_total")

            st.markdown("**น้ำหนักคะแนน (รวมกัน = 1 อัตโนมัติ)**")
            c4, c5, c6 = st.columns(3)
            with c4:
                w1 = st.slider("คาดการณ์เหตุรุนแรง (ฐาน 0.7)", 0.0, 1.0, 0.7, 0.05, key="ew_w1")
            with c5:
                w2 = st.slider("การเติบโตความถี่ (ฐาน 0.2)", 0.0, 1.0, 0.2, 0.05, key="ew_w2")
            with c6:
                w3 = st.slider("การเติบโตความรุนแรง (ฐาน 0.1)", 0.0, 1.0, 0.1, 0.05, key="ew_w3")

            # Normalize น้ำหนักให้รวมเท่ากับ 1
            _sumw = max(w1 + w2 + w3, 1e-9)
            w1n, w2n, w3n = w1 / _sumw, w2 / _sumw, w3 / _sumw

            try:
                # ✅ แก้ไข: เรียกใช้ฟังก์ชันด้วย df_filtered
                res = prioritize_incidents_nb_logit_v2(
                    df_filtered,
                    horizon=horizon,
                    min_months=min_months,
                    min_total=min_total,
                    w_expected_severe=w1n,
                    w_freq_growth=w2n,
                    w_sev_growth=w3n
                )

            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดระหว่างคำนวณลำดับความสำคัญ: {e}")
                res = pd.DataFrame()  # สร้าง DataFrame ว่างเพื่อไม่ให้โค้ดส่วนล่าง error

            if res.empty:
                st.info("ไม่มีข้อมูลเพียงพอสำหรับวิเคราะห์ Early Warning ในช่วงเวลานี้")
            else:
                topn = st.slider("แสดง Top-N:", 5, 50, 10, 5, key="ew_topn")
                only_sig = st.checkbox("แสดงเฉพาะรหัสที่มีนัยสำคัญ (ถี่↑ และ/หรือ รุนแรง↑)", value=False, key="ew_only_sig")

                show = res.copy()
                if only_sig:
                    show = show[
                        (show['Freq_p_value'].notna() & (show['Freq_p_value'] < 0.05)) |
                        (show['Severity_p_value'].notna() & (show['Severity_p_value'] < 0.05))
                        ]

                st.dataframe(
                    show.head(topn),
                    use_container_width=True, hide_index=True,
                    column_config={
                        'รหัส': st.column_config.Column("รหัส"),
                        'ชื่ออุบัติการณ์ความเสี่ยง': st.column_config.Column("ชื่ออุบัติการณ์", width="large"),
                        'Months_Observed': st.column_config.NumberColumn("เดือนที่วิเคราะห์", format="%d"),
                        'Total_Occurrences': st.column_config.NumberColumn("ครั้งสะสม", format="%d"),
                        'Expected_Severe_nextH': st.column_config.NumberColumn(f"คาดการณ์ 'รุนแรง' (H={horizon})",
                                                                               format="%.1f"),
                        'Freq_Factor_per_month': st.column_config.NumberColumn("เท่าความถี่/เดือน", format="%.2f"),
                        'Freq_p_value': st.column_config.NumberColumn("p(ถี่↑)", format="%.3f"),
                        'Severe_OR_per_month': st.column_config.NumberColumn("Odds รุนแรง/เดือน", format="%.2f"),
                        'Severity_p_value': st.column_config.NumberColumn("p(รุนแรง↑)", format="%.3f"),
                        'Priority_Score': st.column_config.ProgressColumn("Priority", min_value=0,
                                                                          max_value=show['Priority_Score'].max(),
                                                                          format="%.3f"),
                    }
                )

                with st.expander("เกณฑ์ที่ใช้จัดอันดับ (อธิบายย่อ)"):
                    st.markdown(f"""
                        - **คาดการณ์ 'รุนแรง' (H={horizon})**: คาดการณ์จำนวนเหตุการณ์รุนแรง (ระดับ 3–5) ที่จะเกิดขึ้นในอีก {horizon} เดือนข้างหน้า
                        - **Priority Score**: คะแนนรวมที่ถ่วงน้ำหนักระหว่าง 'การคาดการณ์เหตุรุนแรง', 'แนวโน้มความถี่ที่เพิ่มขึ้น', และ 'แนวโน้มสัดส่วนความรุนแรงที่เพิ่มขึ้น'
                        - **p(ถี่↑)** และ **p(รุนแรง↑)**: ค่า p-value ยิ่งน้อย (เช่น < 0.05) ยิ่งหมายความว่าแนวโน้มที่เพิ่มขึ้นนั้นมีนัยสำคัญทางสถิติ
                        """)

    elif selected_analysis == "บทสรุปสำหรับผู้บริหาร":

        st.markdown("<h4 style='color: #001f3f;'>บทสรุปสำหรับผู้บริหาร</h4>", unsafe_allow_html=True)
        st.markdown(f"**เรื่อง:** รายงานสรุปอุบัติการณ์โรงพยาบาล")
        st.markdown(f"**ช่วงข้อมูลที่วิเคราะห์:** {min_date_str} ถึง {max_date_str} (รวม {total_month} เดือน)")
        st.markdown(f"**จำนวนอุบัติการณ์ที่พบทั้งหมด:** {metrics_data.get('total_processed_incidents', 0):,} รายการ")
        st.markdown("---")

        # --- 1. แดชบอร์ดสรุปภาพรวม ---
        st.subheader("1. แดชบอร์ดสรุปภาพรวม")
        col1_m, col2_m, col3_m, col4_m, col5_m = st.columns(5)
        with col1_m:
            st.metric("อุบัติการณ์ทั้งหมด", f"{metrics_data.get('total_processed_incidents', 0):,}")
        with col2_m:
            st.metric("Sentinel Events", f"{metrics_data.get('total_sentinel_incidents_for_metric1', 0):,}")
        with col3_m:
            st.metric("มาตรฐานสำคัญฯ 9 ข้อ", f"{metrics_data.get('total_psg9_incidents_for_metric1', 0):,}")
        with col4_m:
            st.metric("ความรุนแรงสูง (E-I & 3-5)", f"{metrics_data.get('total_severe_incidents', 0):,}")
        with col5_m:
            val_unresolved = metrics_data.get('total_severe_unresolved_incidents_val', 'N/A')
            st.metric("รุนแรงสูง & ยังไม่แก้ไข",
                      f"{val_unresolved:,}" if isinstance(val_unresolved, int) else val_unresolved)
        st.markdown("---")

        # --- 2. Risk Matrix และ Top 10 อุบัติการณ์ ---
        st.subheader("2. Risk Matrix และ Top 10 อุบัติการณ์")
        col_matrix, col_top10 = st.columns(2)
        with col_matrix:
            st.markdown("##### Risk Matrix")
            impact_level_keys = ['5', '4', '3', '2', '1']
            freq_level_keys = ['1', '2', '3', '4', '5']
            matrix_df = df_filtered[
                df_filtered['Impact Level'].isin(impact_level_keys) & df_filtered['Frequency Level'].isin(
                    freq_level_keys)]
            if not matrix_df.empty:
                matrix_data = pd.crosstab(matrix_df['Impact Level'], matrix_df['Frequency Level'])
                matrix_data = matrix_data.reindex(index=impact_level_keys, columns=freq_level_keys, fill_value=0)
                impact_labels = {'5': "5 (Extreme)", '4': "4 (Major)", '3': "3 (Moderate)", '2': "2 (Minor)",
                                 '1': "1 (Insignificant)"}
                freq_labels = {'1': "F1", '2': "F2", '3': "F3", '4': "F4", '5': "F5"}
                st.table(matrix_data.rename(index=impact_labels, columns=freq_labels))
        with col_top10:
            st.markdown("##### Top 10 อุบัติการณ์ (ตามความถี่)")
            if not df_freq.empty:
                df_freq_top10 = df_freq.nlargest(10, 'count').copy()
                display_top10 = pd.merge(df_freq_top10,
                                         df_filtered[['Incident', 'ชื่ออุบัติการณ์ความเสี่ยง']].drop_duplicates(),
                                         on='Incident', how='left')
                st.dataframe(display_top10[['Incident', 'count']], hide_index=True,
                             use_container_width=True,
                             column_config={"Incident": "รหัส",
                                            "count": "จำนวน"})
            else:
                st.info("ไม่มีข้อมูล Top 10")
        st.markdown("---")

        # --- 3. รายการ Sentinel Events ---
        st.subheader("3. รายการ Sentinel Events")
        if 'Sentinel code for check' in df_filtered.columns:
            sentinel_events_df = df_filtered[df_filtered['Sentinel code for check'].isin(sentinel_composite_keys)]
            if not sentinel_events_df.empty:
                st.dataframe(sentinel_events_df[['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด']],
                             hide_index=True, use_container_width=True,
                             column_config={"Occurrence Date": "วันที่เกิด", "Incident": "รหัส", "Impact": "ระดับ",
                                            "รายละเอียดการเกิด": "รายละเอียด"})
            else:
                st.info("ไม่พบ Sentinel Events ในช่วงเวลาที่เลือก")
        st.markdown("---")

        # --- 4. PSG9 Summary ---
        st.subheader("4. วิเคราะห์ตามหมวดหมู่ มาตรฐานสำคัญจำเป็นต่อความปลอดภัย 9 ข้อ")
        psg9_summary_table = create_psg9_summary_table(df_filtered)
        if psg9_summary_table is not None and not psg9_summary_table.empty:
            st.table(psg9_summary_table)
        else:
            st.info("ไม่พบข้อมูลอุบัติการณ์ที่เกี่ยวข้องกับ PSG9 ในช่วงเวลานี้")
        st.markdown("---")

        # --- 5. รายการอุบัติการณ์รุนแรงที่ยังไม่ถูกแก้ไข ---
        st.subheader("5. รายการอุบัติการณ์รุนแรง (E-I & 3-5) ที่ยังไม่ถูกแก้ไข")
        if 'Resulting Actions' in df_filtered.columns:
            unresolved_severe_df = df_filtered[
                df_filtered['Impact Level'].isin(['3', '4', '5']) &
                df_filtered['Resulting Actions'].astype(str).isin(['None', '', 'nan'])
                ]
            if not unresolved_severe_df.empty:
                display_cols_unresolved = ['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด']
                st.dataframe(
                    unresolved_severe_df[display_cols_unresolved],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Occurrence Date": st.column_config.DatetimeColumn(
                    "วันที่เกิด",
                    format="DD/MM/YYYY",
                ),
                        "Incident": "รหัส",
                        "Impact": "ระดับ",
                        "รายละเอียดการเกิด": "รายละเอียด"
                    }
                )
            else:
                st.info("ไม่พบอุบัติการณ์รุนแรงที่ยังไม่ถูกแก้ไขในช่วงเวลานี้")

        # --- 6. สรุปอุบัติการณ์ตามเป้าหมาย Safety Goals ---
        st.subheader("6. สรุปอุบัติการณ์ตามเป้าหมาย Safety Goals")
        goal_definitions = {
            "Patient Safety/ Common Clinical Risk": "P:Patient Safety Goals หรือ Common Clinical Risk Incident",
            "Specific Clinical Risk": "S:Specific Clinical Risk Incident",
            "Personnel Safety": "P:Personnel Safety Goals", "Organization Safety": "O:Organization Safety Goals"}
        for display_name, cat_name in goal_definitions.items():
            st.markdown(f"##### {display_name}")
            is_org_safety = (display_name == "Organization Safety")
            summary_table = create_goal_summary_table(df_filtered, cat_name,
                                                      e_up_non_numeric_levels_param=[] if is_org_safety else ['A', 'B',
                                                                                                              'C', 'D'],
                                                      e_up_numeric_levels_param=['1', '2'] if is_org_safety else None,
                                                      is_org_safety_table=is_org_safety)
            if summary_table is not None and not summary_table.empty:
                st.table(summary_table)
            else:
                st.info(f"ไม่มีข้อมูลสำหรับ '{display_name}'")
        st.markdown("---")

        # --- 7. Early Warning (Top 5) ---
        st.subheader("7. Early Warning: อุบัติการณ์ที่มีแนวโน้มสูงขึ้น ใน 3 เดือนข้างหน้า (Top 5)")
        st.write(
            "แสดง Top 5 อุบัติการณ์ที่ถูกจัดลำดับความสำคัญสูงสุด โดยพิจารณาจากแนวโน้มความถี่, ความรุนแรง, และจำนวนที่คาดการณ์ว่าจะเกิดในอนาคต")
        if 'prioritize_incidents_nb_logit_v2' in globals():
            early_warning_df = prioritize_incidents_nb_logit_v2(df_filtered, horizon=3, min_months=4, min_total=5)
            if not early_warning_df.empty:
                top_ew_incidents = early_warning_df.head(5).copy()
                display_ew_df = top_ew_incidents.rename(
                    columns={'ชื่ออุบัติการณ์ความเสี่ยง': 'ชื่ออุบัติการณ์', 'Priority_Score': 'คะแนนความสำคัญ',
                             'Expected_Severe_nextH': 'คาดการณ์เหตุรุนแรง (3 ด.)'})
                st.dataframe(
                    display_ew_df[['รหัส', 'ชื่ออุบัติการณ์', 'คะแนนความสำคัญ', 'คาดการณ์เหตุรุนแรง (3 ด.)']],
                    column_config={
                        "คะแนนความสำคัญ": st.column_config.ProgressColumn("คะแนนความสำคัญ", format="%.3f", min_value=0,
                                                                          max_value=float(
                                                                              display_ew_df['คะแนนความสำคัญ'].max())),
                        "คาดการณ์เหตุรุนแรง (3 ด.)": st.column_config.NumberColumn(format="%.2f")
                    },
                    use_container_width=True, hide_index=True
                )
            else:
                st.info("ไม่มีข้อมูลเพียงพอสำหรับวิเคราะห์ Early Warning")
        else:
            st.warning("ไม่พบฟังก์ชันสำหรับวิเคราะห์ Early Warning")
        st.markdown("---")

        # --- 8. สรุปอุบัติการณ์ที่เป็นปัญหาเรื้อรัง (Top 5) ---
        st.subheader("8. สรุปอุบัติการณ์ที่เป็นปัญหาเรื้อรัง (Persistence Risk - Top 5)")
        st.write("แสดง Top 5 อุบัติการณ์ที่เกิดขึ้นบ่อยและมีความรุนแรงเฉลี่ยสูง ซึ่งควรทบทวนเชิงระบบ")
        persistence_df_exec = calculate_persistence_risk_score(df_filtered, total_month)
        if not persistence_df_exec.empty:
            top_persistence_incidents = persistence_df_exec.head(5)
            display_df_persistence = top_persistence_incidents.rename(
                columns={'Persistence_Risk_Score': 'ดัชนีความเรื้อรัง',
                         'Average_Ordinal_Risk_Score': 'คะแนนเสี่ยงเฉลี่ย'})
            st.dataframe(
                display_df_persistence[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'คะแนนเสี่ยงเฉลี่ย', 'ดัชนีความเรื้อรัง']],
                column_config={
                    "คะแนนเสี่ยงเฉลี่ย": st.column_config.NumberColumn(format="%.2f"),
                    "ดัชนีความเรื้อรัง": st.column_config.ProgressColumn("ดัชนีความเรื้อรัง", min_value=0, max_value=2,
                                                                         format="%.2f")
                },
                use_container_width=True
            )
        else:
            st.info("ไม่มีข้อมูลเพียงพอสำหรับวิเคราะห์ความเสี่ยงเรื้อรัง")

        st.markdown("---")
        st.subheader("ดาวน์โหลดรายงานสรุปเป็นไฟล์ PDF")

        if st.button("สร้างไฟล์ PDF สำหรับดาวน์โหลด"):
            with st.spinner("กำลังสร้าง PDF... (อาจใช้เวลาสักครู่)"):
                # ✅ เรียกใช้ฟังก์ชันเวอร์ชันใหม่ โดยส่งข้อมูลที่จำเป็นทั้งหมดเข้าไป
                pdf_file = generate_executive_summary_pdf(
                    df_filtered=df_filtered,
                    metrics_data=metrics_data,
                    total_month=total_month,
                    df_freq=df_freq,
                    min_date_str=min_date_str,
                    max_date_str=max_date_str
                )

                st.download_button(
                    label="📥 ดาวน์โหลด PDF ที่สร้างเสร็จแล้ว",
                    data=pdf_file,
                    file_name=f"Executive_Summary_{date.today().strftime('%Y-%m-%d')}.pdf",
                    mime="application/pdf"
                )

    elif selected_analysis == "คุยกับ AI Assistant":
        st.markdown("<h4 style='color: #001f3f;'>AI Assistant (ผู้ช่วย AI)</h4>", unsafe_allow_html=True)
        st.info(
            "ถามเกี่ยวกับข้อมูลในช่วงเวลาที่เลือกได้หลากหลาย เช่น 'อุบัติการณ์ใดรุนแรงที่สุด', 'วิเคราะห์รหัส CPP101', 'ปัญหาเรื้อรังคืออะไร'")

        # --- การตั้งค่า AI ---
        AI_IS_CONFIGURED = False
        if genai:
            try:
                genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
                model = genai.GenerativeModel('gemini-1.5-flash')
                AI_IS_CONFIGURED = True
            except Exception as e:
                st.error(
                    f"⚠️ ไม่สามารถตั้งค่า AI Assistant ได้ กรุณาตรวจสอบว่าคุณได้ตั้งค่า GOOGLE_API_KEY ใน secrets.toml ถูกต้องแล้ว")
                st.caption(f"รายละเอียดข้อผิดพลาด: {e}")
        else:
            st.error("ไม่ได้ติดตั้งไลบรารี google-generativeai กรุณาติดตั้งด้วยคำสั่ง: pip install google-generativeai")

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        chat_history_container = st.container(height=400, border=True)
        with chat_history_container:
            for message in st.session_state.chat_messages:
                avatar = LOGO_URL if message["role"] == "assistant" else "❓"
                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])

        if prompt := st.chat_input("ถามเกี่ยวกับข้อมูลความเสี่ยง..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="❓"):
                st.markdown(prompt)

            if AI_IS_CONFIGURED:
                with st.spinner("AI กำลังวิเคราะห์ข้อมูล..."):
                    df_for_ai = df_filtered.copy()
                    total_months_for_ai = total_month
                    prompt_lower = prompt.lower()
                    ai_prompt_for_gemini = ""
                    final_ai_response_text = ""

                    incident_code_match = re.search(r'[A-Z]{3}\d{3}', prompt.upper())

                    if incident_code_match:
                        extracted_code = incident_code_match.group(0)
                        incident_df = df_for_ai[df_for_ai['รหัส'] == extracted_code]
                        if incident_df.empty:
                            final_ai_response_text = f"ขออภัยครับ ผมไม่พบข้อมูลเกี่ยวกับรหัส '{extracted_code}' ในช่วงเวลาที่เลือกนี้ครับ"
                        else:
                            # ✅ แก้ไข: นำตรรกะการสร้าง prompt กลับมาใส่แทนที่ pass
                            incident_name = incident_df['ชื่ออุบัติการณ์ความเสี่ยง'].iloc[0]
                            total_count = len(incident_df)
                            severity_dist = incident_df['Impact'].value_counts().to_dict()
                            severity_text = ", ".join(
                                [f"ระดับ {k}: {v} ครั้ง" for k, v in severity_dist.items()]) or "ไม่มีข้อมูล"

                            poisson_df = calculate_frequency_trend_poisson(df_for_ai)
                            trend_row = poisson_df[poisson_df['รหัส'] == extracted_code]
                            trend_text = f"{trend_row['Poisson_Trend_Slope'].iloc[0]:.4f}" if not trend_row.empty else "ข้อมูลไม่พอคำนวณ"

                            persistence_df = calculate_persistence_risk_score(df_for_ai, total_months_for_ai)
                            persistence_row = persistence_df[persistence_df['รหัส'] == extracted_code]
                            persistence_text = f"{persistence_row['Persistence_Risk_Score'].iloc[0]:.2f}" if not persistence_row.empty else "ข้อมูลไม่พอคำนวณ"

                            ai_prompt_for_gemini = f"""
                                คุณคือ AI ผู้ช่วยวิเคราะห์ความเสี่ยงในโรงพยาบาลครับ
                                โปรดสรุปและวิเคราะห์ข้อมูลของอุบัติการณ์รหัส '{extracted_code}' จากข้อมูลต่อไปนี้:
                                - **ชื่ออุบัติการณ์:** {incident_name}
                                - **จำนวนครั้งที่เกิดทั้งหมดในช่วงเวลานี้:** {total_count} ครั้ง
                                - **การกระจายความรุนแรง:** {severity_text}
                                - **แนวโน้มการเกิด (Poisson Trend Slope):** {trend_text} (ค่าบวก = แนวโน้มเพิ่มขึ้น, ค่าลบ = แนวโน้มลดลง)
                                - **ดัชนีความเสี่ยงเรื้อรัง (Persistence Score):** {persistence_text} (ยิ่งสูง ยิ่งเป็นปัญหาเรื้อรัง)
        
                                ช่วยเขียนสรุปข้อมูลข้างต้นให้เป็น **รายงานสั้นๆ ที่อ่านเข้าใจง่าย** โดยใช้ Markdown สำหรับจัดรูปแบบหัวข้อ และอธิบายความหมายของค่าต่างๆ ด้วยครับ
                                """
                    elif 'รุนแรงที่สุด' in prompt_lower or 'เสี่ยงที่สุด' in prompt_lower:
                        # กำหนดลำดับความรุนแรงจากสูงไปต่ำ
                        severity_order = ['I', 'H', 'G', 'F', 'E', 'D', 'C', 'B', 'A', '5', '4', '3', '2', '1']

                        df_sev = df_for_ai.copy()

                        # แปลงคอลัมน์ Impact ให้เป็นประเภทที่เรียงลำดับได้
                        df_sev['Impact_cat'] = pd.Categorical(df_sev['Impact'].astype(str), categories=severity_order,
                                                              ordered=True)

                        # เรียงลำดับตามความรุนแรง (สูงสุดก่อน) และตามวันที่ (ล่าสุดก่อน)
                        df_sorted = df_sev.sort_values(by=['Impact_cat', 'Occurrence Date'], ascending=[True, False])

                        if df_sorted.empty or pd.isna(df_sorted['Impact_cat'].iloc[0]):
                            final_ai_response_text = "ขออภัยครับ ผมไม่สามารถหาอุบัติการณ์ที่รุนแรงที่สุดได้ อาจเนื่องจากไม่มีข้อมูลระดับความรุนแรงที่ชัดเจนครับ"
                        else:
                            most_severe_incident = df_sorted.iloc[0]
                            incident_name = most_severe_incident.get('ชื่ออุบัติการณ์ความเสี่ยง', 'N/A')
                            incident_code = most_severe_incident.get('รหัส', 'N/A')
                            impact_level = most_severe_incident.get('Impact', 'N/A')
                            occ_date = most_severe_incident.get('Occurrence Date').strftime('%d/%m/%Y') if pd.notna(
                                most_severe_incident.get('Occurrence Date')) else 'N/A'
                            details = most_severe_incident.get('รายละเอียดการเกิด', 'ไม่มีคำอธิบายเพิ่มเติม')

                            final_ai_response_text = f"""อุบัติการณ์ที่มีระดับความรุนแรงสูงที่สุดที่พบในช่วงเวลานี้คือ:
                        - **รหัส:** {incident_code}
                        - **ชื่อ:** {incident_name}
                        - **ระดับความรุนแรง:** {impact_level}
                        - **วันที่เกิด (ล่าสุด):** {occ_date}
                        - **รายละเอียด:** {details}
                        """
                    elif ('เดือนไหน' in prompt_lower and ('มากที่สุด' in prompt_lower or 'เยอะที่สุด' in prompt_lower)):
                        # สร้างคอลัมน์ 'ปี-เดือน' เพื่อใช้จัดกลุ่ม
                        df_for_ai['MonthYear'] = df_for_ai['Occurrence Date'].dt.strftime('%B %Y')
                        monthly_counts = df_for_ai['MonthYear'].value_counts()

                        if monthly_counts.empty:
                            final_ai_response_text = "ขออภัยครับ ไม่สามารถสรุปข้อมูลเป็นรายเดือนได้"
                        else:
                            max_count = monthly_counts.iloc[0]
                            top_months = monthly_counts[monthly_counts == max_count]

                            if len(top_months) > 1:
                                months_str = ", ".join(top_months.index)
                                final_ai_response_text = f"เดือนที่มีอุบัติการณ์เกิดขึ้นมากที่สุดคือเดือน {months_str} โดยเกิดเดือนละ {max_count} ครั้งเท่ากันครับ"
                            else:
                                max_month = monthly_counts.index[0]
                                final_ai_response_text = f"เดือนที่มีอุบัติการณ์เกิดขึ้นมากที่สุดคือเดือน {max_month} โดยเกิดขึ้นทั้งหมด {max_count} ครั้งครับ"
                    # ✅ --- จบส่วนที่แก้ไข ---
                    elif ('เดือนไหน' in prompt_lower and 'น้อยที่สุด' in prompt_lower):
                        # กรองข้อมูลเฉพาะปีปัจจุบัน หากมีการระบุ "ในปีนี้"
                        current_year = datetime.now().year
                        if 'ปีนี้' in prompt_lower:
                            df_to_analyze = df_for_ai[df_for_ai['Occurrence Date'].dt.year == current_year].copy()
                            year_context = f"สำหรับปี {current_year} นี้"
                        else:
                            df_to_analyze = df_for_ai.copy()
                            year_context = "จากข้อมูลทั้งหมด"

                        if df_to_analyze.empty:
                            final_ai_response_text = f"ขออภัยครับ ผมไม่พบข้อมูลอุบัติการณ์{year_context.replace('สำหรับ', ' ')}เลยครับ"
                        else:
                            df_to_analyze['MonthYear'] = df_to_analyze['Occurrence Date'].dt.strftime('%B %Y')
                            monthly_counts = df_to_analyze['MonthYear'].value_counts()

                            if monthly_counts.empty:
                                final_ai_response_text = "ขออภัยครับ ไม่สามารถสรุปข้อมูลเป็นรายเดือนได้"
                            else:
                                # หาจำนวนน้อยที่สุด
                                min_count = monthly_counts.iloc[-1]
                                bottom_months = monthly_counts[monthly_counts == min_count]

                                if len(bottom_months) > 1:
                                    months_str = ", ".join(bottom_months.index)
                                    final_ai_response_text = f"{year_context}, เดือนที่มีอุบัติการณ์เกิดขึ้นน้อยที่สุดคือเดือน {months_str} โดยเกิดเดือนละ {min_count} ครั้งเท่ากันครับ"
                                else:
                                    min_month = monthly_counts.index[-1]
                                    final_ai_response_text = f"{year_context}, เดือนที่มีอุบัติการณ์เกิดขึ้นน้อยที่สุดคือเดือน {min_month} โดยเกิดขึ้นทั้งหมด {min_count} ครั้งครับ"
                    # ✅ --- จบส่วนที่แก้ไข ---
                    elif 'บ่อยที่สุด' in prompt_lower or 'ความถี่สูงสุด' in prompt_lower:
                        most_frequent = df_freq.iloc[0]
                        incident_name = \
                        df_filtered[df_filtered['Incident'] == most_frequent['Incident']]['ชื่ออุบัติการณ์ความเสี่ยง'].iloc[0]
                        final_ai_response_text = f"อุบัติการณ์ที่เกิดบ่อยที่สุดในช่วงเวลานี้คือ:\n- **รหัส:** {most_frequent['Incident']}\n- **ชื่อ:** {incident_name}\n- **จำนวน:** {most_frequent['count']} ครั้งครับ"
                    # ✅ --- ส่วนที่แก้ไขและเพิ่มเข้ามา ---
                    elif ('เปรียบเทียบ' in prompt_lower or 'สัดส่วน' in prompt_lower) and (
                            'แก้ไข' in prompt_lower) and ('safety goal' in prompt_lower or 'เป้าหมาย' in prompt_lower):

                        unresolved_conditions = df_for_ai['Resulting Actions'].astype(str).isin(['None', '', 'nan'])

                        # กรองข้อมูลเฉพาะที่มีหมวดหมู่ Safety Goal ที่ถูกต้อง
                        placeholders = ['N/A', 'N/A (ข้อมูลจาก AllCode ไม่พร้อมใช้งาน)', 'N/A (ไม่พบรหัสใน AllCode)']
                        df_goals = df_for_ai[~df_for_ai['หมวด'].isin(placeholders)]

                        if df_goals.empty:
                            final_ai_response_text = "ขออภัยครับ ไม่พบข้อมูลอุบัติการณ์ที่สามารถจัดกลุ่มตาม Safety Goal ได้ในช่วงเวลานี้ครับ"
                        else:
                            results = []
                            # จัดกลุ่มตาม 'หมวด' ซึ่งก็คือ Safety Goal
                            for category, group_df in df_goals.groupby('หมวด'):
                                total_count = len(group_df)
                                unresolved_count = len(group_df[unresolved_conditions])
                                percentage = (unresolved_count / total_count * 100) if total_count > 0 else 0
                                results.append({
                                    'หมวด': category,
                                    'สัดส่วนที่ยังไม่แก้ไข': percentage,
                                    'รายละเอียด': f"({unresolved_count} จาก {total_count} รายการ)"
                                })

                            if not results:
                                final_ai_response_text = "ไม่สามารถคำนวณสัดส่วนการแก้ไขของ Safety Goals ได้ครับ"
                            else:
                                sorted_results = sorted(results, key=lambda x: x['สัดส่วนที่ยังไม่แก้ไข'], reverse=True)

                                summary_list = [
                                    "ผมได้ทำการวิเคราะห์สัดส่วนปัญหาที่ยังไม่ได้รับการแก้ไขในแต่ละ Safety Goal แล้วครับ:"]
                                for res in sorted_results:
                                    summary_list.append(
                                        f"- **{res['หมวด']}**: มีสัดส่วนที่ยังไม่แก้ไข **{res['สัดส่วนที่ยังไม่แก้ไข']:.2f}%** {res['รายละเอียด']}")

                                highest = sorted_results[0]
                                summary_list.append(
                                    f"\n**สรุป:** '{highest['หมวด']}' เป็น Safety Goal ที่มีสัดส่วนปัญหาคงค้างสูงที่สุดครับ")

                                final_ai_response_text = "\n".join(summary_list)
                    elif ('psg9' in prompt_lower and ('เปรียบเทียบ' in prompt_lower or 'สัดส่วน' in prompt_lower) and (
                            'แก้ไข' in prompt_lower or 'ค้าง' in prompt_lower)):

                        unresolved_conditions = df_for_ai['Resulting Actions'].astype(str).isin(['None', '', 'nan'])

                        # กรองเฉพาะข้อมูลที่มีหมวดหมู่ PSG9
                        psg9_placeholders = ["ไม่จัดอยู่ใน PSG9 Catalog", "ไม่สามารถระบุ (Merge PSG9 ล้มเหลว)",
                                             "ไม่สามารถระบุ (PSG9code.xlsx ไม่ได้โหลด)"]
                        df_psg9 = df_for_ai[~df_for_ai['หมวดหมู่มาตรฐานสำคัญ'].isin(psg9_placeholders)]

                        if df_psg9.empty:
                            final_ai_response_text = "ขออภัยครับ ไม่พบข้อมูลอุบัติการณ์ที่เกี่ยวข้องกับ PSG9 ในช่วงเวลานี้ครับ"
                        else:
                            # คำนวณสัดส่วนของแต่ละหมวดหมู่
                            results = []
                            for category, group_df in df_psg9.groupby('หมวดหมู่มาตรฐานสำคัญ'):
                                total_count = len(group_df)
                                unresolved_count = len(group_df[unresolved_conditions])
                                percentage = (unresolved_count / total_count * 100) if total_count > 0 else 0
                                results.append({
                                    'หมวดหมู่': category,
                                    'สัดส่วนที่ยังไม่แก้ไข': percentage,
                                    'รายละเอียด': f"({unresolved_count} จาก {total_count} รายการ)"
                                })

                            # สร้างข้อความสรุป
                            if not results:
                                final_ai_response_text = "ไม่สามารถคำนวณสัดส่วนการแก้ไขของหมวดหมู่ PSG9 ได้ครับ"
                            else:
                                # เรียงลำดับตามสัดส่วนจากมากไปน้อย
                                sorted_results = sorted(results, key=lambda x: x['สัดส่วนที่ยังไม่แก้ไข'], reverse=True)

                                summary_list = [
                                    "ผมได้ทำการวิเคราะห์สัดส่วนปัญหาที่ยังไม่ได้รับการแก้ไขในแต่ละหมวดหมู่ PSG9 แล้วครับ:"]
                                for res in sorted_results:
                                    summary_list.append(
                                        f"- **{res['หมวดหมู่']}**: มีสัดส่วนที่ยังไม่แก้ไข **{res['สัดส่วนที่ยังไม่แก้ไข']:.2f}%** {res['รายละเอียด']}")

                                # เพิ่มบทสรุป
                                highest = sorted_results[0]
                                summary_list.append(
                                    f"\n**สรุป:** '{highest['หมวดหมู่']}' เป็นหมวดหมู่ที่มีสัดส่วนปัญหาคงค้างสูงที่สุดครับ")

                                final_ai_response_text = "\n".join(summary_list)
                    # ✅ --- จบส่วนที่แก้ไข ---
                    elif all(k in prompt_lower for k in ['สัดส่วน', 'ยังไม่', 'สูงกว่า']) and (
                            'c' in prompt_lower or 'g' in prompt_lower or 'คลินิก' in prompt_lower or 'ทั่วไป' in prompt_lower):

                        unresolved_conditions = df_for_ai['Resulting Actions'].astype(str).isin(['None', '', 'nan'])

                        # กลุ่ม Clinical (C)
                        df_c = df_for_ai[df_for_ai['รหัส'].str.startswith('C', na=False)]
                        total_c = len(df_c)
                        unresolved_c = len(df_c[unresolved_conditions])
                        prop_c = (unresolved_c / total_c * 100) if total_c > 0 else 0

                        # กลุ่ม General (G)
                        df_g = df_for_ai[df_for_ai['รหัส'].str.startswith('G', na=False)]
                        total_g = len(df_g)
                        unresolved_g = len(df_g[unresolved_conditions])
                        prop_g = (unresolved_g / total_g * 100) if total_g > 0 else 0

                        # สร้างข้อความสรุป
                        summary_text = (
                            f"ผมได้ทำการวิเคราะห์สัดส่วนปัญหาที่ยังไม่ได้รับการแก้ไขแล้วครับ:\n\n"
                            f"- **กลุ่มอุบัติการณ์ทางคลินิก (C):**\n"
                            f"  - มีปัญหาที่ยังไม่แก้ไข {unresolved_c} จากทั้งหมด {total_c} รายการ\n"
                            f"  - คิดเป็นสัดส่วน **{prop_c:.2f}%**\n\n"
                            f"- **กลุ่มอุบัติการณ์ทั่วไป (G):**\n"
                            f"  - มีปัญหาที่ยังไม่แก้ไข {unresolved_g} จากทั้งหมด {total_g} รายการ\n"
                            f"  - คิดเป็นสัดส่วน **{prop_g:.2f}%**\n\n"
                        )

                        # เปรียบเทียบและสรุปผล
                        if prop_c > prop_g:
                            conclusion = f"**สรุป:** กลุ่มอุบัติการณ์ทางคลินิก (C) มีสัดส่วนปัญหาที่ยังไม่ได้รับการแก้ไขสูงกว่าครับ"
                        elif prop_g > prop_c:
                            conclusion = f"**สรุป:** กลุ่มอุบัติการณ์ทั่วไป (G) มีสัดส่วนปัญหาที่ยังไม่ได้รับการแก้ไขสูงกว่าครับ"
                        else:
                            conclusion = f"**สรุป:** ทั้งสองกลุ่มมีสัดส่วนปัญหาที่ยังไม่ได้รับการแก้ไขเท่ากันครับ"

                        final_ai_response_text = summary_text + conclusion
                    # ✅ --- จบส่วนที่แก้ไข ---
                    elif 'sentinel' in prompt_lower:
                        sentinel_df = df_for_ai[df_for_ai['Sentinel code for check'].isin(sentinel_composite_keys)]
                        if sentinel_df.empty:
                            final_ai_response_text = "ในช่วงเวลานี้ ไม่พบข้อมูล Sentinel Event เลยครับ"
                        else:
                            # สร้างคอลัมน์ 'ปี-เดือน' เพื่อใช้จัดกลุ่ม
                            sentinel_df['MonthYear'] = sentinel_df['Occurrence Date'].dt.strftime('%B %Y')
                            monthly_counts = sentinel_df['MonthYear'].value_counts()

                            if monthly_counts.empty:
                                final_ai_response_text = "พบ Sentinel Event แต่ไม่สามารถสรุปเป็นรายเดือนได้ครับ"
                            else:
                                # หาเดือนที่มีจำนวนสูงสุด
                                max_count = monthly_counts.iloc[0]
                                top_months = monthly_counts[monthly_counts == max_count]

                                if len(top_months) > 1:
                                    # กรณีมีเดือนที่จำนวนเท่ากัน
                                    months_str = ", ".join(top_months.index)
                                    final_ai_response_text = f"เดือนที่มี Sentinel Event เกิดขึ้นมากที่สุดคือเดือน {months_str} โดยเกิดเดือนละ {max_count} ครั้งเท่ากันครับ"
                                else:
                                    # กรณีมีเดือนเดียว
                                    max_month = monthly_counts.index[0]
                                    final_ai_response_text = f"เดือนที่มี Sentinel Event เกิดขึ้นมากที่สุดคือเดือน {max_month} โดยเกิดขึ้นทั้งหมด {max_count} ครั้งครับ"
                    # ✅ --- จบส่วนที่แก้ไข ---
                    else:
                        # สร้าง context จากข้อมูลสรุป
                        summary_context = f"""
                            ข้อมูลสรุปในช่วงเวลานี้:
                            - จำนวนอุบัติการณ์ทั้งหมด: {metrics_data.get('total_processed_incidents', 0)}
                            - จำนวน Sentinel Events: {metrics_data.get('total_sentinel_incidents_for_metric1', 0)}
                            - จำนวนอุบัติการณ์รุนแรง (E-I & 3-5): {metrics_data.get('total_severe_incidents', 0)}
                            - อุบัติการณ์รุนแรงที่ยังไม่แก้ไข: {metrics_data.get('total_severe_unresolved_incidents_val', 'N/A')}
                            - ข้อมูล Top 5 ที่เกิดบ่อยสุด: {df_freq.head(5).to_string()}
                            """
                        ai_prompt_for_gemini = f"ในฐานะผู้ช่วยวิเคราะห์ความเสี่ยง โปรดตอบคำถามต่อไปนี้ โดยใช้ข้อมูลสรุปที่ให้มาเป็นบริบทหลัก:\n\n{summary_context}\n\nคำถามจากผู้ใช้: '{prompt}'\n\nโปรดตอบเป็นภาษาไทย ใช้สรรพนามแทนตัวเองว่า 'ผม' และลงท้ายด้วย 'ครับ'"

                    # --- ส่วนสร้างและแสดงผลคำตอบ ---
                    with st.chat_message("assistant", avatar=LOGO_URL):
                        final_response = ""
                        if final_ai_response_text:
                            final_response = final_ai_response_text
                            st.markdown(final_response)
                        elif ai_prompt_for_gemini:
                            try:
                                response = model.generate_content(ai_prompt_for_gemini)
                                final_response = response.text
                                st.markdown(final_response)
                            except Exception as e:
                                final_response = f"ขออภัยครับ เกิดข้อผิดพลาดในการเชื่อมต่อกับ AI: {e}"
                                st.error(final_response)
                        else:
                            final_response = "ขออภัยครับ ผมไม่เข้าใจคำถาม กรุณาลองถามใหม่อีกครั้งครับ"
                            st.markdown(final_response)

                    st.session_state.chat_messages.append({"role": "assistant", "content": final_response})

def main():
    page = st.query_params.get("page", "executive")
    if page == "admin":
        display_admin_page()
    else:
        display_executive_dashboard()


if __name__ == "__main__":
    main()