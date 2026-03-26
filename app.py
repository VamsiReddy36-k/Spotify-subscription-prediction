import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ================= LOAD =================
model  = pickle.load(open('model_new.pkl',  'rb'))
enc    = pickle.load(open('enc_new.pkl',    'rb'))
scaler = pickle.load(open('scaler_new.pkl', 'rb'))

if not hasattr(model, "multi_class"):
    model.multi_class = "auto"

# ================= CONFIG =================
st.set_page_config(page_title="Spotify AI Dashboard", layout="wide")

# ================= GLOBAL CSS =================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #1DB954;
}
.stButton>button {
    background: linear-gradient(90deg,#1DB954,#1ed760);
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div style='background: linear-gradient(90deg,#1DB954,#121212);
            padding:25px;
            border-radius:15px;
            text-align:center;
            color:white;'>
<h1>🎧 Spotify AI Intelligence</h1>
<p>Predict user churn • Improve engagement • Boost retention</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================= OPTIONS =================
countries = ['Australia','Brazil','Canada','France','Germany','India','Indonesia','Italy','Mexico','Spain','UK','USA']
subs = ['Free','Premium Duo','Premium Family','Premium Individual','Student']
devices = ['Car System','Desktop','Mobile','Smart Speaker','Tablet']

# ================= USER INFO =================
st.markdown("""
<div style='background: linear-gradient(135deg,#1DB95420,#ffffff05);
            padding:20px;
            border-radius:15px;
            border:1px solid #2a2a2a;'>
<h3>👤 User Information</h3>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    country = st.selectbox("🌍 Country", countries)
    age = st.slider("🎂 Age", 13, 75, 25)

with col2:
    signup_date = st.text_input("📅 Signup Date", "2022-01-15")
    sub_type = st.selectbox("💳 Subscription", subs)

with col3:
    device = st.selectbox("📱 Device", devices)

# ================= BEHAVIOR =================
st.markdown("""
<div style='background: linear-gradient(135deg,#3b82f620,#ffffff05);
            padding:20px;
            border-radius:15px;
            border:1px solid #2a2a2a;
            margin-top:15px;'>
<h3>📊 Listening Behavior</h3>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    months_inac = st.slider("📉 Months Inactive", 0, 24, 2)
    inac_flag = st.selectbox("⚠️ Inactive 3+ Months?", [0,1])

with col2:
    avg_hrs = st.slider("⏱ Listening Hours", 0.0, 100.0, 12.0)
    avg_skips = st.slider("⏭ Skips per Day", 0, 100, 8)

with col3:
    playlists = st.number_input("🎶 Playlists Created", 0, 500, 5)

# ================= PREFERENCES =================
st.markdown("""
<div style='background: linear-gradient(135deg,#9333ea20,#ffffff05);
            padding:20px;
            border-radius:15px;
            border:1px solid #2a2a2a;
            margin-top:15px;'>
<h3>🎵 Music Preferences</h3>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    fav_genre = st.selectbox("🎵 Genre", ['Bollywood','Classical','Country','Electronic','Hip-Hop','Indie','Jazz','K-Pop','Latin','Pop','R&B','Rock'])

with col2:
    liked_feat = st.selectbox("⭐ Liked Feature", ['AI DJ','Daily Mix','Discover Weekly','Lyrics','Offline Mode','Playlists','Podcasts','Radio'])
    music_rtg = st.slider("🎧 Music Rating", 1, 5, 3)

with col3:
    future_feat = st.selectbox("🚀 Future Feature", ['Better AI Recommendations','Concert Alerts','HiFi Audio','Lyrics Translation','Mood-based Auto Playlists','Social Listening'])
    ad_inter = st.radio("📢 Ad Interaction", ["No", "Yes"])
    ad_conv = st.radio("💰 Ad Conversion", ["No", "Yes"])

# ================= BUTTON =================
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Predict User Status", use_container_width=True):

    CAT = ['country','signup_date','subscription_type','ad_interaction',
           'ad_conversion_to_subscription','favorite_genre',
           'most_liked_feature','desired_future_feature','primary_device']

    NUM = ['age','months_inactive','inactive_3_months_flag',
           'music_suggestion_rating_1_to_5','avg_listening_hours_per_week',
           'playlists_created','avg_skips_per_day']

    row = pd.DataFrame([{
        'country': country,
        'signup_date': signup_date,
        'subscription_type': sub_type,
        'ad_interaction': ad_inter,
        'ad_conversion_to_subscription': ad_conv,
        'favorite_genre': fav_genre,
        'most_liked_feature': liked_feat,
        'desired_future_feature': future_feat,
        'primary_device': device,
        'age': age,
        'months_inactive': months_inac,
        'inactive_3_months_flag': inac_flag,
        'music_suggestion_rating_1_to_5': music_rtg,
        'avg_listening_hours_per_week': avg_hrs,
        'playlists_created': playlists,
        'avg_skips_per_day': avg_skips
    }])

    try:
        X_cat = enc.transform(row[CAT])
        X_num = scaler.transform(row[NUM].astype(float))
        X = np.hstack([X_cat, X_num])

        pred = model.predict(X)[0]

        proba = model.predict_proba(X)[0]
        classes = model.classes_

        active_prob   = proba[list(classes).index("Active")]
        inactive_prob = proba[list(classes).index("Inactive")]

        # ================= RESULT =================
        st.markdown("## 🎯 Prediction Insights")

        col1, col2, col3 = st.columns(3)

        col1.markdown(f"""
        <div style='background:#181818;padding:20px;border-radius:12px;text-align:center;'>
        <h4>Status</h4><h2>{pred}</h2></div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
        <div style='background:#181818;padding:20px;border-radius:12px;text-align:center;'>
        <h4>Active %</h4><h2>{active_prob*100:.1f}%</h2></div>
        """, unsafe_allow_html=True)

        col3.markdown(f"""
        <div style='background:#181818;padding:20px;border-radius:12px;text-align:center;'>
        <h4>Churn Risk</h4><h2 style='color:#ff4b4b;'>{inactive_prob*100:.1f}%</h2></div>
        """, unsafe_allow_html=True)

        st.progress(int(inactive_prob * 100))

        if pred == "Active":
            st.success("✅ Strong retention user")
        else:
            st.error("⚠️ High churn risk detected")

    except Exception as e:
        st.error(f"Error: {e}")