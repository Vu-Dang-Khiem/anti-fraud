import streamlit as st
import pandas as pd
import numpy as np
st.set_page_config(page_title="Radar Gian Lận - Demo", layout="wide", page_icon="🚨")

# ===== Đọc dữ liệu (cache để không đọc lại mỗi lần bấm) =====
@st.cache_data
def load_data():
    pair = pd.read_parquet("pair_anomaly_scores_all.parquet")
    driver = pd.read_parquet("driver_anomaly_scores_all.parquet")
    customer = pd.read_parquet("customer_anomaly_scores_all.parquet")
    lift_vertical = pd.read_csv("fraud_lift_by_vertical.csv")
    lift_province = pd.read_csv("fraud_lift_by_province.csv")
    return pair, driver, customer, lift_vertical, lift_province


pair, driver, customer, lift_vertical, lift_province = load_data()

st.title("🚨 Radar Gian Lận — Ride Trip")
st.caption("Hệ thống phát hiện gian lận: 5 rule + 3 model (Isolation Forest, VAE, LOF)")

tab1, tab2, tab3, tab4 = st.tabs([
    "📉 Tổng quan", "🔍 Điều tra case", "📈 Dashboard", "✅ Vì sao tin được"
])

# ========================================================================
# TAB 1: TỔNG QUAN
# ========================================================================
with tab1:
    st.subheader("Bức tranh tổng quan")

    n_total_orders = 15_699_432

    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng đơn hàng đã phân tích", f"{n_total_orders:,}".replace(",", "."))
    col2.metric("Case bị gắn cờ nghi vấn", "9.493",
                help="Tổng số case bị 5 rule gắn cờ (978 + 6.613 + 1.244 + 455 + 203)")
    col3.metric("Đơn hàng liên quan", "99.969",
                help="Đơn thuộc các case bị gắn cờ (cặp + tài xế + khách hàng, đã khử trùng) "
                     "— chiếm 0,64% tổng số đơn")

    st.subheader("Chi tiết theo từng loại hành vi")
    breakdown = pd.DataFrame({
        "Loại": ["Behavior 1 (arbitrage_promo)", "Behavior 2 (internal_exploit)",
                 "Collusion (ghost/tạo đơn nhanh)", "Story 8.3 (khách lạm dụng KM)",
                 "Story 8.4 (tài xế đi vòng)"],
        "Số case": [978, 6613, 1244, 455, 203],
    })
    st.bar_chart(breakdown.set_index("Loại"))
    st.caption("Collusion và Story 8.4 còn có tầng lọc chặt hơn: **107** và **63** case đạt mức "
               "'bằng chứng cao' sau khi qua đủ các lớp kiểm chứng độc lập.")


# ========================================================================
# TAB 2: TRA CỨU CASE (phần demo chính)
# ========================================================================
with tab2:
    @st.cache_data
    def load_all_evidence():
        return (pd.read_parquet("evidence_orders.parquet"),
                pd.read_parquet("evidence_orders_driver.parquet"),
                pd.read_parquet("evidence_orders_customer.parquet"))
    ev_pair, ev_drv, ev_cus = load_all_evidence()

    cap_do = st.radio("Cấp độ điều tra:",
                      ["👥 Cặp tài xế–khách", "🚗 Tài xế (Story 8.4)", "🙋 Khách hàng (Story 8.3)"],
                      horizontal=True)

    # ===== Cau hinh theo tung cap do =====
    if cap_do.startswith("👥"):
        base, ev, id_cols = pair, ev_pair, ["driver_id", "customer_id"]
        col_flag, ten_rule = "n_rules_flagged", "Behavior 1 / 2 / Collusion"
        sort_don = (["so_bang_chung", "ty_le_km", "gap_min"], [False, True, True])
        cot_bang = ["n_rules_flagged", "n_trips", "so_don_bang_chung", "if_score_percentile"]
    elif cap_do.startswith("🚗"):
        base, ev, id_cols = driver, ev_drv, ["driver_id"]
        col_flag, ten_rule = "story84_flagged", "Story 8.4"
        sort_don = (["so_bang_chung", "km_ratio"], [False, False])
        cot_bang = ["n_orders", "pct_high", "so_don_bang_chung", "if_score_percentile"]
    else:
        base, ev, id_cols = customer, ev_cus, ["customer_id"]
        col_flag, ten_rule = "story83_flagged", "Story 8.3"
        sort_don = (["so_bang_chung", "discount_ratio"], [False, False])
        cot_bang = ["n_orders", "total_discount", "promo_rate", "so_don_bang_chung", "if_score_percentile"]

    # ===== Gan tom tat bang chung vao bang thuc the =====
    @st.cache_data
    def gan_bang_chung(_ev, _base, _id_cols, ten):
        s = _ev[_ev["so_bang_chung"] > 0].groupby(_id_cols).size().rename("so_don_bang_chung").reset_index()
        return _base.merge(s, on=_id_cols, how="left").fillna({"so_don_bang_chung": 0})
    full = gan_bang_chung(ev, base, id_cols, cap_do)

    full["_bi_flag"] = full[col_flag].astype(bool) if full[col_flag].dtype == bool else full[col_flag] > 0

    # ===== Ham hien chi tiet =====
    def hien_chi_tiet(row):
        khoa = {c: row[c] for c in id_cols}
        mask = np.ones(len(ev), dtype=bool)
        for c, v in khoa.items():
            mask &= (ev[c] == v)
        don = ev[mask].copy()

        n_flag = int(row[col_flag]) if not isinstance(row[col_flag], bool) else int(row[col_flag])
        if n_flag >= 3:   muc, icon = "RẤT CAO", "🔴"
        elif n_flag == 2: muc, icon = "CAO", "🟠"
        elif n_flag == 1: muc, icon = "TRUNG BÌNH", "🟡"
        else:             muc, icon = "MODEL PHÁT HIỆN", "🔵"

        with st.container(border=True):
            h1, h2 = st.columns([3, 1])
            h1.markdown("**" + " · ".join(f"`{row[c]}`" for c in id_cols) + "**")
            h2.markdown(f"### {icon} {muc}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Rule ({ten_rule})**")
                if col_flag == "n_rules_flagged":
                    for ten, cot in [("Behavior 1","b1_flagged"),("Behavior 2","b2_flagged"),
                                     ("Collusion","collusion_flagged")]:
                        if cot in row.index:
                            st.write(f"{'✅' if row[cot] else '⬜'} {ten}")
                else:
                    st.write(f"{'✅' if row[col_flag] else '⬜'} {ten_rule}")
            with c2:
                st.markdown("**Mức độ bất thường theo model**")
                for ten, cot in [("Isolation Forest","if_score_percentile"),
                                 ("VAE","vae_rank_pct"), ("LOF","lof_score_percentile")]:
                    if cot in row.index and pd.notna(row[cot]):
                        st.progress(float(1 - row[cot]), text=f"{ten} — top {row[cot]*100:.2f}%")
            if len(don):
                n_bc = int((don["so_bang_chung"] > 0).sum())
                st.markdown(f"**Truy vết cấp đơn:** {n_bc}/{len(don)} đơn có bằng chứng "
                            f"({n_bc/len(don)*100:.1f}%)")

        if len(don) == 0:
            st.warning("Chưa có dữ liệu đơn chi tiết cho trường hợp này.")
            return
        cols_sort = [c for c in sort_don[0] if c in don.columns]
        asc = [sort_don[1][sort_don[0].index(c)] for c in cols_sort]
        don = don.sort_values(cols_sort, ascending=asc)
        st.dataframe(don.drop(columns=id_cols, errors="ignore"),
                     use_container_width=True, hide_index=True)

    # ==================================================================
    # PHAN 1 - RULE
    # ==================================================================
    st.subheader(f"1️⃣ Case do RULE phát hiện — {ten_rule}")
    ds_r = full[full["_bi_flag"]].sort_values(
        [col_flag, "so_don_bang_chung"], ascending=[False, False]).head(50).reset_index(drop=True)
    st.caption(f"**{int(full['_bi_flag'].sum()):,}** trường hợp bị rule gắn cờ · hiển thị 50 case mạnh nhất")

    cot_r = [c for c in id_cols + cot_bang if c in ds_r.columns]
    sel_r = st.dataframe(ds_r[cot_r], use_container_width=True, hide_index=True,
                         on_select="rerun", selection_mode="single-row", key=f"tbl_r_{cap_do}")
    if sel_r.selection.rows:
        hien_chi_tiet(ds_r.iloc[sel_r.selection.rows[0]])
    else:
        st.caption("👆 Bấm vào 1 dòng để xem chi tiết")

    st.divider()

    # ==================================================================
    # PHAN 2 - MODEL BAT THEM
    # ==================================================================
    st.subheader("2️⃣ Case MODEL bắt thêm (rule chưa từng bắt)")
    c1, c2 = st.columns(2)
    N = c1.selectbox("Xét top bao nhiêu case bất thường nhất?", [500, 1000, 2000], index=1,
                     key=f"N_{cap_do}")
    ten_model = c2.selectbox("Xếp hạng theo model nào?",
                             ["Isolation Forest", "VAE", "Đồng thuận (IF+VAE)"], key=f"M_{cap_do}")

    if ten_model == "Isolation Forest":   cot_diem = "if_score_percentile"
    elif ten_model == "VAE":              cot_diem = "vae_rank_pct"
    else:
        full["_dt"] = (full["if_score_percentile"] + full["vae_rank_pct"]) / 2
        cot_diem = "_dt"

    top_n = full.nsmallest(N, cot_diem)
    n_trung = int(top_n["_bi_flag"].sum())
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Top {N:,} theo model", f"{N:,}")
    m2.metric("Trùng với rule", f"{n_trung:,}")
    m3.metric("Case MỚI", f"{N-n_trung:,}")
    st.caption(f"Model **không tự đặt ngưỡng flag** — ngưỡng do người dùng chọn. Ở top {N:,} theo "
               f"{ten_model}: **{n_trung:,} case trùng rule**, **{N-n_trung:,} case mới**.")

    ds_m = top_n[~top_n["_bi_flag"]].head(50).reset_index(drop=True)
    cot_m = list(dict.fromkeys(id_cols + [cot_diem] + cot_bang))
    cot_m = [c for c in cot_m if c in ds_m.columns]
    sel_m = st.dataframe(ds_m[cot_m], use_container_width=True, hide_index=True,
                         on_select="rerun", selection_mode="single-row", key=f"tbl_m_{cap_do}")
    if sel_m.selection.rows:
        hien_chi_tiet(ds_m.iloc[sel_m.selection.rows[0]])
    else:
        st.caption("👆 Bấm vào 1 dòng để xem chi tiết")

# ========================================================================
# TAB 3: BIỂU ĐỒ — đầy đủ toàn bộ Dashboard Lãnh đạo
# ========================================================================
with tab3:
    @st.cache_data
    def load_charts():
        return (pd.read_csv("daily_trend_v2.csv"),
                pd.read_csv("daily_trend_by_rule_v2.csv"),
                pd.read_csv("story84_funnel_v2.csv"))
    daily_trend, daily_by_rule, funnel = load_charts()

    st.markdown("## Bức tranh rủi ro — nhìn theo thời gian và theo nhóm")

    # ==================================================================
    # A. THEO THOI GIAN
    # ==================================================================
    st.subheader("📅 Diễn biến theo ngày")

    dt = daily_trend.set_index("order_date")
    cot_tin_hieu = [c for c in ["pct_high_discount", "pct_km_ratio_outlier"] if c in dt.columns]
    st.line_chart(dt[cot_tin_hieu])

    if cot_tin_hieu:
        c0 = cot_tin_hieu[0]
        lo, hi = dt[c0].min(), dt[c0].max()
        bien_dong = (hi - lo) / dt[c0].mean() * 100
        st.markdown(
            f"**Kết luận:** tín hiệu bất thường ở cấp đơn dao động trong khoảng "
            f"**{lo:.1f}% – {hi:.1f}%**, biên độ chỉ {bien_dong:.0f}% quanh mức trung bình — "
            "**không có ngày nào đột biến**. Đây là mức nền ổn định, cho thấy gian lận diễn ra "
            "liên tục chứ không phải sự cố nhất thời."
        )
    st.caption("Khoảng trống 18/07–23/07 là do cấu trúc dữ liệu (2 đợt lấy mẫu), không phải ngày không có đơn.")

    st.markdown("")
    st.subheader("📅 Số case bị gắn cờ theo ngày, tách theo từng rule")
    dbr = daily_by_rule.set_index("order_date")
    st.line_chart(dbr)

    rule_cao_nhat = dbr.mean().idxmax()
    st.markdown(
        f"**Kết luận:** **{rule_cao_nhat}** chiếm tỷ trọng lớn nhất mỗi ngày "
        f"(trung bình {dbr[rule_cao_nhat].mean():.0f} case/ngày) — phù hợp với việc đây là hành vi "
        "mang tính hệ thống, không phải cá biệt."
    )

    st.divider()

    # ==================================================================
    # B. RUI RO TAP TRUNG O DAU
    # ==================================================================
    st.subheader("🚗 Mức độ tập trung nghi vấn theo hạng xe")
    lv = lift_vertical.sort_values("lift", ascending=False)
    st.bar_chart(lv.set_index("hang_xe")["lift"])

    top_v = lv.iloc[0]
    bot_v = lv.iloc[-1]
    st.markdown(
        f"**Kết luận:** **{top_v['hang_xe']}** tập trung nghi vấn gấp **{top_v['lift']:.2f} lần** mức bình thường "
        f"(chiếm {top_v['flagged_pct']:.1f}% trong nhóm nghi vấn, trong khi chỉ chiếm {top_v['baseline_pct']:.1f}% "
        f"tổng số đơn). Ngược lại **{bot_v['hang_xe']}** gần như không xuất hiện ({bot_v['lift']:.2f} lần) → "
        f"**ưu tiên giám sát {top_v['hang_xe']}**."
    )
    st.caption("Lift = tỷ lệ trong nhóm nghi vấn ÷ tỷ lệ trong toàn quần thể. Lift = 1 nghĩa là không có gì bất thường.")

    st.markdown("")
    st.subheader("📍 Mức độ tập trung nghi vấn theo khu vực")
    lp = lift_province.sort_values("lift", ascending=False).head(15)
    st.bar_chart(lp.set_index("khu_vuc")["lift"])

    top_p = lp.iloc[0]
    st.markdown(
        f"**Kết luận:** **{top_p['khu_vuc']}** dẫn đầu với lift **{top_p['lift']:.2f} lần** — "
        f"chỉ chiếm {top_p['baseline_pct']:.2f}% tổng số đơn nhưng chiếm tới {top_p['flagged_pct']:.2f}% "
        "trong nhóm nghi vấn. Các điểm nóng đều là **tỉnh nhỏ ngoài TP.HCM/Hà Nội** — nơi mật độ giám sát "
        "thường thấp hơn."
    )
    st.info(
        "🔍 **Đã kiểm chứng không phải do vài cặp cực đoan gây nhiễu:** 3 tỉnh dẫn đầu có 191–271 cặp "
        "tài xế–khách hàng độc lập cùng đóng góp, cặp nhiều nhất cũng chỉ chiếm dưới 2,5% số đơn của tỉnh."
    )

    st.divider()

    # ==================================================================
    # C. CHAT LUONG LOC
    # ==================================================================
    st.subheader("🔻 Quy trình lọc 3 tầng — Story 8.4 (tài xế đi vòng tăng cước)")
    st.bar_chart(funnel.set_index("tang")["so_tai_xe"])

    t1, t3 = int(funnel["so_tai_xe"].iloc[0]), int(funnel["so_tai_xe"].iloc[-1])
    st.markdown(
        f"**Kết luận:** từ **{t1} tài xế** nghi vấn ban đầu (chỉ dựa 1 tín hiệu thống kê), sau khi lọc qua "
        f"đủ 3 lớp bằng chứng độc lập chỉ còn **{t3} tài xế** — loại bỏ **{(1-t3/t1)*100:.0f}%** trường hợp "
        "không đủ cơ sở. Đây là minh chứng hệ thống **không kết luận vội từ 1 tín hiệu đơn lẻ**."
    )

# ========================================================================
# TAB 4: CASE MỚI (model tìm ra, rule chưa từng flag)
# ========================================================================
with tab4:
    st.markdown("## Vì sao có thể tin vào kết quả này?")
    st.caption("Dự án chưa có nhãn gian lận thật — nên mọi kết luận đều được kiểm chứng bằng "
               "nhiều kỹ thuật độc lập, không dựa vào một nguồn duy nhất.")

    # ==================================================================
    # 1. KIEM CHUNG CHEO BANG TIN HIEU DOC LAP
    # ==================================================================
    st.subheader("1️⃣ Kiểm chứng bằng tín hiệu hoàn toàn độc lập")
    st.markdown(
        "Cột **đánh giá 5 sao** không hề được dùng để xây bất kỳ rule nào. "
        "Nếu rule bắt đúng hành vi thật, nhóm bị gắn cờ phải khác biệt rõ ở tín hiệu này."
    )

    @st.cache_data
    def load_rating():
        return pd.read_csv("rating_crosscheck_summary.csv")
    rating = load_rating()

    rc = rating.pivot(index="rule", columns="nhom", values="ty_le_pct")
    st.bar_chart(rc)

    cot_flag = [c for c in rc.columns if "flag" in c.lower()]
    cot_base = [c for c in rc.columns if "baseline" in c.lower() or "ngau" in c.lower()]
    if cot_flag and cot_base:
        tb_flag = rc[cot_flag[0]].mean()
        tb_base = rc[cot_base[0]].mean()
        st.success(
            f"**Kết quả:** nhóm bị gắn cờ có tỷ lệ 2 bên cùng chấm 5 sao tuyệt đối trung bình "
            f"**{tb_flag:.1f}%**, so với nhóm ngẫu nhiên chỉ **{tb_base:.1f}%** — "
            f"**cao gấp {tb_flag/tb_base:.1f} lần**. Vì đây là tín hiệu độc lập, đây là bằng chứng "
            "khách quan cho thấy rule bắt đúng hành vi bất thường thật."
        )
    st.caption("Trong quan hệ bình thường, đánh giá luôn có dao động tự nhiên — việc 2 bên liên tục "
               "chấm 5 sao tuyệt đối cho nhau là dấu hiệu đáng ngờ của hành vi thông đồng.")

    st.divider()

    # ==================================================================
    # 2. MODEL CO THUC SU HOC DUOC GI KHONG
    # ==================================================================
    st.subheader("2️⃣ Model có thực sự học được gì, hay chỉ đoán ngẫu nhiên?")
    st.markdown("Nếu model vô dụng, case bị rule gắn cờ sẽ nằm ở mức trung bình **50%** như mọi case khác. "
                "Thực tế chúng nằm ở nhóm bất thường nhất:")

    lift_df = pd.DataFrame({
        "Nhóm case (rule đã gắn cờ)": ["Behavior 1", "Behavior 2", "Collusion",
                                        "Cả 3 rule cùng xác nhận", "Story 8.4 (tài xế)", "Story 8.3 (khách)"],
        "Isolation Forest": [0.53, 2.83, 0.14, 0.043, 0.81, 0.14],
        "VAE": [1.46, 2.85, 0.27, 0.049, 0.65, 2.77],
        "LOF": [17.4, 41.4, 1.8, 1.2, 6.8, 5.6],
    })
    st.dataframe(lift_df, use_container_width=True, hide_index=True)
    st.success(
        "**Kết quả:** case bị rule gắn cờ nằm ở **top 0,04% – 2,8%** bất thường nhất theo Isolation Forest — "
        "so với mức trung bình 50% của quần thể. Nghĩa là model **độc lập tìm ra đúng những case rule đã bắt**, "
        "dù chưa từng được cho biết rule là gì."
    )
    st.caption("Đơn vị: percentile — số càng thấp nghĩa là model càng coi case đó là bất thường. "
               "Baseline toàn quần thể = 50%.")

    st.divider()

    # ==================================================================
    # 3. CAC MODEL CO DONG THUAN KHONG
    # ==================================================================
    st.subheader("3️⃣ Ba model có đồng thuận với nhau không?")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Tương quan xếp hạng (Spearman)**")
        st.dataframe(pd.DataFrame({
            "Cấp độ": ["Cặp", "Tài xế", "Khách hàng"],
            "IF ↔ VAE": [0.674, 0.819, 0.242],
            "IF ↔ LOF": [0.347, 0.449, 0.043],
            "VAE ↔ LOF": [0.416, 0.450, 0.276],
        }), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Tỷ lệ bắt lại case rule đã gắn cờ (top-N)**")
        st.dataframe(pd.DataFrame({
            "Rule": ["Behavior 1", "Behavior 2", "Collusion"],
            "IF": ["30,2%", "33,6%", "24,3%"],
            "VAE": ["10,7%", "33,5%", "23,4%"],
            "Đồng thuận": ["26,1%", "39,3%", "30,8%"],
        }), use_container_width=True, hide_index=True)

    st.success(
        "**Kết quả:** IF và VAE đồng thuận mạnh ở cấp cặp và tài xế (0,67–0,82). Ở Behavior 2 và Collusion, "
        "**kết hợp 2 model cho kết quả tốt hơn từng model riêng lẻ** (39,3% > 33,6%/33,5%) — bằng chứng "
        "2 model bổ sung cho nhau chứ không lặp lại nhau."
    )
    st.info("**Trung thực về hạn chế:** LOF yếu hơn hẳn ở mọi cấp độ, gần như thất bại ở Behavior 2 "
            "(41,4% — sát mức ngẫu nhiên). Nguyên nhân đã xác định: dữ liệu đếm rời rạc nhiều giá trị trùng "
            "lặp phá vỡ giả định mật độ cục bộ của LOF. Vì vậy LOF **không được dùng làm tín hiệu chính**, "
            "chỉ giữ vai trò đối chiếu độc lập.")

    st.divider()

    # ==================================================================
    # 4. TAP THAM CHIEU TIN CAY NHAT
    # ==================================================================
    st.subheader("4️⃣ Model có bắt được những case chắc chắn nhất không?")
    st.markdown("**26 cặp** bị **cả 3 rule độc lập** cùng xác nhận — đây là tập gần với 'đã kiểm chứng' "
                "nhất hiện có, vì 3 tiêu chí hoàn toàn khác nhau khó cùng sai một lúc.")

    m1, m2, m3 = st.columns(3)
    m1.metric("IF — trong top 500", "26/26", "100%")
    m2.metric("VAE — trong top 500", "16/26", "62%")
    m3.metric("Đồng thuận — trong top 100", "16/26", "62%")

    st.success(
        "**Kết quả:** Isolation Forest xếp **trọn vẹn 26/26 cặp** vào top 500 — tức chỉ **0,18%** "
        "trong tổng số 279.353 cặp. Đây là chỉ số mạnh nhất trong toàn bộ phần đánh giá."
    )

    st.divider()

    # ==================================================================
    # 5. TU TIM RA VA SUA LOI CUA CHINH MINH
    # ==================================================================
    st.subheader("5️⃣ Những lỗi đã tự phát hiện và sửa")
    st.markdown("Độ tin cậy không đến từ việc không có lỗi, mà từ việc **chủ động tìm ra lỗi của chính mình**:")

    st.dataframe(pd.DataFrame({
        "Lỗi phát hiện": [
            "Trung bình bị outlier chi phối (Collusion)",
            "Chỉ số không phản ánh lợi ích tài chính thật (Story 8.4)",
            "Điều kiện lọc bị hiểu nhầm thành bằng chứng (gating artifact)",
            "Tiền xử lý không đồng nhất giữa các model (log1p)",
            "Kết quả có thể do vài cặp cực đoan gây nhiễu",
        ],
        "Cách xử lý": [
            "Đổi min_gap_min → pct_close_gap (tỷ lệ thay vì giá trị nhỏ nhất)",
            "Thêm tầng kiểm chứng lợi ích tài chính → 203 còn 63 tài xế",
            "So sánh trong nhóm đã cùng vượt ngưỡng (fair pool)",
            "Xác định rõ vì sao VAE cần log1p còn IF thì không",
            "Đếm số cặp độc lập đóng góp — 191–271 cặp/tỉnh, cặp cao nhất <2,5%",
        ],
    }), use_container_width=True, hide_index=True)

