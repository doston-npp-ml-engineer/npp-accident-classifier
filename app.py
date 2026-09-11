import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================
# SAHIFA SOZLAMALARI
# ============================================
st.set_page_config(page_title="AES Avariya Bashoratchisi", page_icon="☢️", layout="wide")

# ============================================
# FON RASMI QO'SHISH
# CSS orqali butun sahifaning orqa foniga rasm qo'yamiz
# ============================================
def add_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: 
                linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)),
                url("{image_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
add_background("https://images.unsplash.com/photo-1751453875319-660527493daa?q=80&w=1074&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")

# ============================================
# MODEL VA YORDAMCHI FAYLLARNI YUKLASH
# @st.cache_resource -- bu Streamlit'ga "bu funksiyani faqat BIR MARTA
# ishga tushir, natijani xotirada saqla" deydi, aks holda har safar
# foydalanuvchi biror tugma bossa, model qaytadan yuklanib, sekinlashadi
# ============================================
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    class_names = joblib.load('class_names.pkl')
    feature_importance = joblib.load('feature_importance.pkl')
    feature_means = joblib.load('feature_means.pkl')
    return model, scaler, feature_names, class_names, feature_importance, feature_means

model, scaler, feature_names, class_names, feature_importance, feature_means = load_artifacts()

# Foydalanuvchiga qo'lda kiritish uchun eng muhim 20 ta feature nomi
TOP_FEATURES = feature_importance.head(20)['feature'].tolist()

# ============================================
# SARLAVHA
# ============================================
st.title("☢️ AES Avariya Turini Bashorat Qilish")
st.markdown("""
Bu ilova **Random Forest** modeli yordamida, atom elektr stansiyasi (AES) 
sensor ko'rsatkichlari asosida yuz berayotgan avariya turini bashorat qiladi.

Dataset: [NPPAD — Nuclear Power Plant Accident Data](https://github.com/thu-inet/NuclearPowerPlantAccidentData) 
(Tsinghua universiteti, PCTRAN simulyatori)
""")

# ============================================
# YORDAMCHI FUNKSIYA: bashorat qilish
# ============================================
def predict(input_dict):
    """
    input_dict -- foydalanuvchi kiritgan {feature_nomi: qiymat} lug'ati.
    Kiritilmagan feature'lar uchun avtomatik o'rtacha (mean) qiymat ishlatiladi.
    """
    # Barcha 331 ta feature uchun to'liq qator yasaymiz
    row = []
    for fname in feature_names:
        if fname in input_dict:
            row.append(input_dict[fname])
        else:
            row.append(feature_means[fname])  # kiritilmagan -> o'rtacha qiymat

    row_df = pd.DataFrame([row], columns=feature_names)

    # Xuddi Train paytida qilganimizdek, scaling qo'llaymiz
    row_scaled = scaler.transform(row_df)

    # Bashorat va ehtimolliklar
    pred_class_idx = model.predict(row_scaled)[0]
    pred_proba = model.predict_proba(row_scaled)[0]

    pred_class_name = class_names[pred_class_idx]
    return pred_class_name, pred_proba

# ============================================
# IKKITA REJIM: Tab orqali ajratamiz
# ============================================
tab1, tab2 = st.tabs(["✍️ Qo'lda kiritish", "📄 CSV fayl yuklash"])

# --------------------------------------------
# REJIM 1: Qo'lda kiritish
# --------------------------------------------
with tab1:
    st.subheader("Eng muhim sensor ko'rsatkichlarini kiriting")
    st.caption(f"Qolgan {len(feature_names) - len(TOP_FEATURES)} ta feature uchun avtomatik o'rtacha qiymat ishlatiladi.")

    input_dict = {}
    cols = st.columns(3)  # 3 ustunli chiroyli joylashuv uchun
    for i, fname in enumerate(TOP_FEATURES):
        default_val = float(feature_means[fname])
        with cols[i % 3]:
            input_dict[fname] = st.number_input(
                fname, value=default_val, format="%.4f", key=f"manual_{fname}"
            )

    if st.button("🔍 Bashorat qilish", type="primary", key="manual_predict"):
        pred_class, pred_proba = predict(input_dict)
        st.success(f"### Bashorat qilingan avariya turi: **{pred_class}**")

        # Ehtimolliklarni chiroyli jadval qilib ko'rsatamiz
        proba_df = pd.DataFrame({
            'Avariya turi': class_names,
            'Ehtimollik (%)': (pred_proba * 100).round(2)
        }).sort_values('Ehtimollik (%)', ascending=False).head(5)
        st.write("**Eng yuqori 5 ta ehtimollik:**")
        st.dataframe(proba_df, use_container_width=True, hide_index=True)

# --------------------------------------------
# REJIM 2: CSV fayl yuklash
# --------------------------------------------
with tab2:
    st.subheader("CSV fayl yuklang (bir nechta qator uchun)")
    st.caption(f"Fayl {len(feature_names)} ta ustunga ega bo'lishi kerak, nomlari model kutgan feature nomlari bilan mos kelishi kerak.")

    uploaded_file = st.file_uploader("CSV faylni tanlang", type=['csv'])

    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)

        # Ustun nomlari mosligini tekshiramiz
        missing_cols = set(feature_names) - set(user_df.columns)
        if missing_cols:
            st.warning(f"Faylda {len(missing_cols)} ta ustun yo'q, ular uchun o'rtacha qiymat ishlatiladi.")
            for col in missing_cols:
                user_df[col] = feature_means[col]

        # To'g'ri tartibda qayta joylaymiz
        user_df_ordered = user_df[feature_names]

        if st.button("🔍 Barchasini bashorat qilish", type="primary", key="csv_predict"):
            scaled_data = scaler.transform(user_df_ordered)
            predictions = model.predict(scaled_data)
            pred_names = [class_names[p] for p in predictions]

            result_df = user_df.copy()
            result_df['Bashorat_qilingan_avariya'] = pred_names
            st.dataframe(result_df, use_container_width=True)

            # Natijani yuklab olish imkoni
            csv_out = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Natijani yuklab olish (CSV)", csv_out, "natijalar.csv", "text/csv")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("Toshkent davlat texnika universiteti | Energetika fakulteti | Atom elektr stansiyalari yo'nalishi")
