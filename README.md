# ☢️ AES Avariya Turini Bashorat Qilish (NPP Accident Classifier)

Bu loyiha — atom elektr stansiyasi (AES) sensor ko'rsatkichlari asosida, 
yuz berayotgan avariya turini avtomatik aniqlaydigan Machine Learning modeli.

**Muallif:** Doston, TDTU Energetika fakulteti, Atom elektr stansiyalari yo'nalishi, 4-kurs

## 📊 Dataset

[NPPAD (Nuclear Power Plant Accident Data)](https://github.com/thu-inet/NuclearPowerPlantAccidentData) 
— Tsinghua universiteti tomonidan PCTRAN simulyatori yordamida yaratilgan, 
PWR reaktori uchun 18 xil holat (17 avariya turi + Normal) bo'yicha vaqt qatori ma'lumotlari.

## 🔧 Jarayon

1. **Ma'lumotni yig'ish**: 1200+ CSV fayl, har biri statistik xususiyatlarga (mean, std, min, max) aylantirildi
2. **EDA**: sinflar nomutanosibligi va bo'sh qiymatlar tahlil qilindi
3. **Preprocessing**: 793 → 331 ustunga tozalandi, encoding qilindi
4. **Modellashtirish**: Logistic Regression, Decision Tree, Random Forest solishtirildi
5. **Baholash**: Cross-validation, Confusion Matrix, Precision/Recall/F1

## 🏆 Natija

- **Yakuniy model:** Random Forest
- **Test accuracy:** 97.25%
- **Macro F1:** 0.86 (kam sonli sinflarda past support tufayli)
- **Weighted F1:** 0.97

## ⚠️ Cheklovlar

- Ba'zi avariya turlari (ATWS, LOF, Normal, SP, TT) uchun juda kam namuna 
  bor edi (1-3 ta test misoli), shuning uchun bu sinflarda ishonchlilik pastroq.
- Bu — o'quv/tadqiqot loyihasi, real AES nazorat tizimida ishlatish uchun emas.

## 🚀 Ilovani ishga tushirish

🔗 **Jonli havola:** https://npp-accident-classifier-kvvxnnyg2cb4erv6cc9eot.streamlit.app/

## 📁 Fayllar

- `app.py` — Streamlit ilovasi
- `model.pkl`, `scaler.pkl`, `feature_names.pkl`, `class_names.pkl`, 
  `feature_importance.pkl`, `feature_means.pkl` — o'qitilgan model va yordamchi fayllar
- `requirements.txt` — kerakli Python kutubxonalari
