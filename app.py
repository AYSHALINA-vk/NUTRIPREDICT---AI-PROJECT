import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from pathlib import Path
import base64


st.set_page_config(page_title="NutriPredict", page_icon="🥗", layout="wide")

DAILY_TARGETS = {
    "Calories (kcal)": 1433.0,
    "Protein (g)": 38.3,
    "Fiber (g)": 14.7,
}

NOTEBOOK_RDA_THRESHOLDS = {
    "Protein (g)": {"threshold": 46.0, "direction": "min", "unit": "g"},
    "Fiber (g)": {"threshold": 25.0, "direction": "min", "unit": "g"},
    "Sodium (mg)": {"threshold": 2300.0, "direction": "max", "unit": "mg"},
    "Cholesterol (mg)": {"threshold": 300.0, "direction": "max", "unit": "mg"},
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

            :root {
                --bg: #050607;
                --panel: rgba(11, 15, 16, 0.88);
                --panel-soft: rgba(18, 24, 25, 0.9);
                --line: rgba(255, 255, 255, 0.09);
                --text: #f3f5f3;
                --muted: #97a09a;
                --green: #37d69b;
                --sand: #e1b36d;
                --blue: #9fc7ff;
                --danger: #ff9b70;
                --glow: rgba(55, 214, 155, 0.18);
            }

            .stApp {
                background:
                    radial-gradient(circle at 0% 0%, rgba(55, 214, 155, 0.12), transparent 24%),
                    radial-gradient(circle at 100% 10%, rgba(225, 179, 109, 0.10), transparent 26%),
                    radial-gradient(circle at 50% 100%, rgba(159, 199, 255, 0.06), transparent 24%),
                    linear-gradient(180deg, #040506 0%, #090c0d 50%, #050607 100%);
                color: var(--text);
            }

            .block-container {
                max-width: 1220px;
                padding-top: 1.35rem;
                padding-bottom: 3rem;
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            h1, h2, h3, h4, p, div, span, label {
                font-family: "Manrope", sans-serif !important;
            }

            .hero {
                text-align: center;
                padding: 1.25rem 0 2.4rem;
                position: relative;
            }

            .hero-badge {
                display: inline-block;
                padding: 0.58rem 1rem;
                border-radius: 999px;
                background: rgba(55, 214, 155, 0.1);
                border: 1px solid rgba(55, 214, 155, 0.2);
                color: var(--green);
                font-size: 0.9rem;
                margin-bottom: 1.2rem;
                letter-spacing: 0.03em;
                box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset;
            }

            .hero-title {
                font-size: clamp(3.2rem, 7vw, 5.8rem);
                line-height: 0.9;
                font-weight: 800;
                letter-spacing: -0.06em;
                margin: 0;
                text-shadow: 0 10px 40px rgba(0, 0, 0, 0.26);
            }

            .hero-accent {
                display: block;
                background: linear-gradient(90deg, #37d69b 0%, #9aba7a 50%, #e2a160 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                color: transparent;
            }

            .hero-copy {
                max-width: 760px;
                margin: 1.25rem auto 0;
                color: var(--muted);
                font-size: 1.06rem;
                line-height: 1.75;
            }

            .glass-card {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 28px;
                padding: 1.3rem;
                box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
                backdrop-filter: blur(14px);
                position: relative;
                overflow: hidden;
            }

            .glass-card::before {
                content: "";
                position: absolute;
                inset: 0;
                background: linear-gradient(180deg, rgba(255,255,255,0.035), transparent 32%);
                pointer-events: none;
            }

            .section-title {
                font-size: 1.12rem;
                font-weight: 700;
                color: var(--text);
                margin-bottom: 1rem;
                position: relative;
                z-index: 1;
            }

            .visual-card {
                position: relative;
                min-height: 290px;
                border-radius: 30px;
                overflow: hidden;
                border: 1px solid rgba(255, 255, 255, 0.09);
                background:
                    linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)),
                    linear-gradient(135deg, #16221f 0%, #1c2430 55%, #36241d 100%);
                background-size: cover;
                background-position: center;
                margin-bottom: 1.15rem;
                box-shadow: 0 18px 42px rgba(0,0,0,0.22);
                transform: translateY(0);
                transition: transform 0.18s ease, box-shadow 0.18s ease;
            }

            .visual-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 28px 52px rgba(0,0,0,0.28);
            }

            .visual-card::after {
                content: "";
                position: absolute;
                inset: 0;
                background: linear-gradient(180deg, rgba(4, 6, 7, 0.04) 16%, rgba(4, 6, 7, 0.86) 100%);
            }

            .visual-content {
                position: absolute;
                left: 1.1rem;
                right: 1.1rem;
                bottom: 1.05rem;
                z-index: 1;
            }

            .visual-tag {
                display: inline-block;
                padding: 0.35rem 0.7rem;
                border-radius: 999px;
                background: rgba(46, 207, 143, 0.14);
                color: #8aeebe;
                font-size: 0.75rem;
                font-weight: 800;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                margin-bottom: 0.7rem;
            }

            .visual-title {
                color: var(--text);
                font-size: 1.15rem;
                font-weight: 800;
                margin-bottom: 0.3rem;
            }

            .visual-copy {
                color: #d4dad6;
                font-size: 0.9rem;
                line-height: 1.6;
                max-width: 92%;
            }

            .meal-block {
                background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.02));
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 22px;
                padding: 1rem 1rem 0.95rem;
                margin-bottom: 1rem;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
            }

            .meal-title {
                color: var(--text);
                font-size: 1rem;
                font-weight: 700;
                margin-bottom: 0.2rem;
            }

            .meal-copy {
                color: var(--muted);
                font-size: 0.87rem;
                margin-bottom: 0.8rem;
            }

            .metric-card {
                background:
                    radial-gradient(circle at top right, rgba(55,214,155,0.08), transparent 30%),
                    linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.02));
                border: 1px solid rgba(255, 255, 255, 0.07);
                border-radius: 22px;
                padding: 1rem;
                min-height: 142px;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
            }

            .metric-label {
                color: var(--muted);
                font-size: 0.88rem;
                margin-bottom: 0.4rem;
            }

            .metric-value {
                color: var(--text);
                font-size: 2rem;
                font-weight: 800;
                letter-spacing: -0.04em;
            }

            .metric-copy {
                color: #a7b2ac;
                font-size: 0.88rem;
                margin-top: 0.55rem;
                line-height: 1.5;
            }

            .summary-strip {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.9rem;
                margin-top: 1rem;
            }

            .summary-pill {
                padding: 0.85rem 0.95rem;
                border-radius: 18px;
                background: rgba(255,255,255,0.035);
                border: 1px solid rgba(255,255,255,0.06);
                color: var(--muted);
                font-size: 0.88rem;
                line-height: 1.5;
            }

            .summary-pill strong {
                display: block;
                color: var(--text);
                font-size: 1rem;
                margin-bottom: 0.18rem;
            }

            .status-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 1rem;
                background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.018));
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 18px;
                padding: 0.95rem 1rem;
                margin-bottom: 0.75rem;
            }

            .status-label {
                color: var(--text);
                font-weight: 700;
            }

            .status-copy {
                color: var(--muted);
                font-size: 0.87rem;
                margin-top: 0.2rem;
            }

            .status-warning {
                color: #f5c088;
                font-size: 0.84rem;
                margin-top: 0.45rem;
                line-height: 1.45;
            }

            .pill-ok, .pill-risk {
                border-radius: 999px;
                padding: 0.42rem 0.78rem;
                font-size: 0.82rem;
                font-weight: 800;
            }

            .pill-ok {
                background: rgba(46, 207, 143, 0.14);
                color: #88efc1;
            }

            .pill-risk {
                background: rgba(255, 155, 112, 0.14);
                color: #ffc09c;
            }

            .tip-card {
                background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.018));
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 18px;
                padding: 1rem;
                margin-bottom: 0.8rem;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
            }

            .tip-title {
                color: var(--text);
                font-weight: 700;
                margin-bottom: 0.3rem;
            }

            .tip-copy {
                color: var(--muted);
                font-size: 0.9rem;
                line-height: 1.6;
            }

            .target-note {
                color: var(--muted);
                font-size: 0.86rem;
                line-height: 1.6;
                margin-top: 0.85rem;
            }

            .footer-note {
                color: #76807a;
                text-align: center;
                padding-top: 1.2rem;
                font-size: 0.9rem;
            }

            .stMultiSelect div[data-baseweb="select"],
            .stNumberInput div[data-baseweb="input"] {
                background: rgba(255, 255, 255, 0.045);
                border-radius: 16px;
            }

            .stMultiSelect label, .stNumberInput label {
                color: #dfe8e2 !important;
                font-weight: 600;
            }

            .stButton > button {
                width: 100%;
                min-height: 3.25rem;
                border-radius: 18px;
                border: none;
                color: #08110d;
                font-weight: 800;
                font-size: 1rem;
                background: linear-gradient(90deg, #37d69b 0%, #81c485 48%, #e2a160 100%);
                box-shadow: 0 18px 30px rgba(55, 214, 155, 0.18);
                transition: transform 0.18s ease, box-shadow 0.18s ease;
            }

            .stButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 24px 34px rgba(55, 214, 155, 0.22);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_row(name: str, probability: float, high_risk: bool, warning_text: str = "") -> None:
    pill_class = "pill-risk" if high_risk else "pill-ok"
    pill_text = "High Risk" if high_risk else "Normal"
    st.markdown(
        f"""
        <div class="status-row">
            <div>
                <div class="status-label">{name}</div>
                <div class="status-copy">{probability:.1%} estimated probability</div>
                {"<div class='status-warning'>" + warning_text + "</div>" if warning_text else ""}
            </div>
            <div class="{pill_class}">{pill_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_data_uri(path: str) -> str | None:
    image_path = Path(path)
    if not image_path.exists():
        return None
    suffix = image_path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix)
    if mime is None:
        return None
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def render_visual_card(title: str, tag: str, copy: str, image_path: str) -> None:
    image_uri = image_data_uri(image_path)
    style = f"background-image: url('{image_uri}');" if image_uri else ""
    st.markdown(
        f"""
        <div class="visual-card" style="{style}">
            <div class="visual-content">
                <div class="visual-tag">{tag}</div>
                <div class="visual-title">{title}</div>
                <div class="visual-copy">{copy}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_strip() -> None:
    st.markdown(
        """
        <div class="summary-strip">
            <div class="summary-pill"><strong>Daily Insight</strong>Monitor how each meal shapes the full day.</div>
            <div class="summary-pill"><strong>Threshold Check</strong>See warnings based on the notebook RDA cutoffs.</div>
            <div class="summary-pill"><strong>Smart Additions</strong>Get realistic foods to improve the weakest nutrients.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_food_suggestions(
    food_df: pd.DataFrame,
    nutrient: str,
    current_value: float,
    target_value: float,
    selected_foods: set[str],
    limit: int = 3,
) -> list[dict[str, float | str]]:
    gap = max(target_value - current_value, 0.0)
    if gap <= 0 or nutrient not in food_df.columns:
        return []

    candidate_df = food_df.copy()
    if selected_foods:
        candidate_df = candidate_df[~candidate_df["Food_Item"].isin(selected_foods)]

    candidate_df = candidate_df[candidate_df[nutrient].fillna(0) > 0].copy()
    if candidate_df.empty:
        return []

    ideal_addition = min(gap, 12.0 if nutrient == "Protein (g)" else 4.0)
    candidate_df["coverage"] = candidate_df[nutrient].astype(float) / gap
    candidate_df["density"] = candidate_df[nutrient].astype(float) / candidate_df["Calories (kcal)"].replace(0, np.nan)
    candidate_df["density"] = candidate_df["density"].fillna(0)
    candidate_df["portion_fit"] = (candidate_df[nutrient].astype(float) - ideal_addition).abs()
    ranked = candidate_df.sort_values(
        by=["portion_fit", "density", nutrient],
        ascending=[True, False, False],
    ).head(limit)

    suggestions = []
    for _, row in ranked.iterrows():
        suggestions.append(
            {
                "food": str(row["Food_Item"]),
                "category": str(row.get("Category", "Food")),
                "amount": float(row[nutrient]),
                "coverage_pct": min(float(row["coverage"]) * 100, 100.0),
            }
        )
    return suggestions


def get_limit_relief_suggestions(
    food_df: pd.DataFrame,
    nutrient: str,
    selected_foods: set[str],
    limit: int = 3,
) -> list[dict[str, float | str]]:
    if nutrient not in food_df.columns:
        return []

    candidate_df = food_df.copy()
    if selected_foods:
        candidate_df = candidate_df[~candidate_df["Food_Item"].isin(selected_foods)]

    keep_cols = ["Food_Item", "Category", nutrient, "Calories (kcal)"]
    candidate_df = candidate_df[[col for col in keep_cols if col in candidate_df.columns]].copy()
    candidate_df = candidate_df.sort_values(
        by=[nutrient, "Calories (kcal)"],
        ascending=[True, True],
    ).head(limit)

    suggestions = []
    for _, row in candidate_df.iterrows():
        suggestions.append(
            {
                "food": str(row["Food_Item"]),
                "category": str(row.get("Category", "Food")),
                "amount": float(row.get(nutrient, 0.0)),
            }
        )
    return suggestions


inject_styles()


@st.cache_data
@st.cache_data
@st.cache_data
def load_food_data():
    try:
        df = pd.read_csv(
            'data/daily_food.csv',
            on_bad_lines='skip',      # Skip bad rows
            engine='python',          # More tolerant parser
            quotechar='"',            # Handle commas inside quotes
            encoding='utf-8'
        )
        df.columns = df.columns.str.strip()
        st.success(f"✅ Food dataset loaded! ({df.shape[0]} foods)")
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        st.info("Some rows had formatting issues and were skipped.")
        return pd.DataFrame(columns=['Food_Item', 'Calories (kcal)', 'Protein (g)', 'Fiber (g)'])


food_df = load_food_data()


@st.cache_resource
def load_model():
    try:
        df = pd.read_csv("preprocessed_nutri_data.csv")
        risk_cols = [col for col in df.columns if "_risk" in col]

        if len(risk_cols) < 2:
            rda = {"Protein (g)": 16.7, "Fiber (g)": 8.3}
            for nutrient, threshold in rda.items():
                if nutrient in df.columns:
                    df[f"{nutrient}_risk"] = (df[nutrient] < threshold).astype(int)
            df.to_csv("preprocessed_nutri_data.csv", index=False)
            risk_cols = [col for col in df.columns if "_risk" in col]

        features = [
            "Calories (kcal)",
            "Protein (g)",
            "Carbohydrates (g)",
            "Fat (g)",
            "Fiber (g)",
            "Sugars (g)",
            "Sodium (mg) (g)",
            "Cholesterol (mg) (g)",
        ]
        available_features = [col for col in features if col in df.columns]

        x_train = df[available_features].fillna(0)
        y_train = df[risk_cols].astype(int)

        model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
        model.fit(x_train, y_train)
        return model, risk_cols, available_features
    except Exception:
        dummy_features = [
            "Calories (kcal)",
            "Protein (g)",
            "Carbohydrates (g)",
            "Fat (g)",
            "Fiber (g)",
            "Sugars (g)",
            "Sodium (mg) (g)",
            "Cholesterol (mg) (g)",
        ]
        dummy_risk = ["Protein (g)_risk", "Fiber (g)_risk"]
        dummy_x = pd.DataFrame(np.random.rand(200, 8), columns=dummy_features)
        dummy_y = pd.DataFrame(np.random.randint(0, 2, (200, 2)), columns=dummy_risk)
        model = MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state=42))
        model.fit(dummy_x, dummy_y)
        return model, dummy_risk, dummy_features


model, risk_cols, feature_cols = load_model()

st.markdown(
    """
    <section class="hero">
        <div class="hero-badge">🥗 Lifestyle Nutrition Dashboard</div>
        <h1 class="hero-title">
            Nutri Predict
            <span class="hero-accent">Predict your health before it fails</span>
        </h1>
        <p class="hero-copy">
            Track how your meals shape energy, protein, and fiber across the day, then use the analysis to keep your routine more balanced.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)
render_summary_strip()

visual_cols = st.columns(4, gap="large")
visual_items = [
    ("Balanced Plate", "Lifestyle", "A colorful plate helps the day feel lighter and more intentional.", "assets/balanced_plate.jpg"),
    ("Protein Focus", "Recovery", "Build steadier meals around protein-rich foods for strength and satiety.", "assets/protein_focus.jpg"),
    ("Fresh Produce", "Fiber", "Vegetables and fruits add volume, micronutrients, and better daily rhythm.", "assets/fresh_produce.jpg"),
    ("Limit Often", "Awareness", "Use highly processed choices more mindfully when shaping a balanced routine.", "assets/processed_foods.jpg"),
]
for column, (title, tag, copy, path) in zip(visual_cols, visual_items):
    with column:
        render_visual_card(title, tag, copy, path)

meals = ["Breakfast", "Lunch", "Evening Snack", "Dinner"]
daily_nutrition = {
    "Calories (kcal)": 0.0,
    "Protein (g)": 0.0,
    "Fiber (g)": 0.0,
    "Sodium (mg)": 0.0,
    "Cholesterol (mg)": 0.0,
}
selected_food_names = set()

top_left, top_right = st.columns([1.15, 0.85], gap="large")

with top_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Plan Your Meals</div>', unsafe_allow_html=True)

    for meal in meals:
        st.markdown(
            f"""
            <div class="meal-block">
                <div class="meal-title">{meal}</div>
                <div class="meal-copy">Select the foods that fit this part of your routine.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected_foods = st.multiselect(
            f"Choose foods for {meal}",
            options=food_df["Food_Item"].unique() if not food_df.empty else [],
            key=f"select_{meal}",
            label_visibility="collapsed",
            placeholder=f"Choose foods for {meal.lower()}",
        )

        for food in selected_foods:
            selected_food_names.add(food)
            qty = st.number_input(
                f"Quantity (g) - {food}",
                min_value=50,
                max_value=500,
                value=100,
                step=50,
                key=f"qty_{meal}_{food}",
            )

            if not food_df.empty:
                food_data = food_df[food_df["Food_Item"] == food].iloc[0]
                daily_nutrition["Calories (kcal)"] += food_data.get("Calories (kcal)", 0) * (qty / 100)
                daily_nutrition["Protein (g)"] += food_data.get("Protein (g)", 0) * (qty / 100)
                daily_nutrition["Fiber (g)"] += food_data.get("Fiber (g)", 0) * (qty / 100)
                daily_nutrition["Sodium (mg)"] += food_data.get("Sodium (mg)", 0) * (qty / 100)
                daily_nutrition["Cholesterol (mg)"] += food_data.get("Cholesterol (mg)", 0) * (qty / 100)

    analyze = st.button("Analyze My Day & Get Suggestions", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with top_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Daily Snapshot</div>', unsafe_allow_html=True)
    metric_cols = st.columns(3)
    with metric_cols[0]:
        render_metric_card("Calories", f"{daily_nutrition['Calories (kcal)']:.0f} kcal", "Energy from today’s selected meals.")
    with metric_cols[1]:
        render_metric_card("Protein", f"{daily_nutrition['Protein (g)']:.1f} g", "Supports recovery and satiety.")
    with metric_cols[2]:
        render_metric_card("Fiber", f"{daily_nutrition['Fiber (g)']:.1f} g", "Helps daily balance and digestion.")
    st.markdown("</div>", unsafe_allow_html=True)


if analyze:
    if daily_nutrition["Calories (kcal)"] == 0:
        st.error("Please select at least some foods!")
    else:
        user_input = pd.DataFrame(
            {
                "Calories (kcal)": [daily_nutrition["Calories (kcal)"]],
                "Protein (g)": [daily_nutrition["Protein (g)"]],
                "Carbohydrates (g)": [daily_nutrition["Calories (kcal)"] * 0.55 / 4],
                "Fat (g)": [daily_nutrition["Calories (kcal)"] * 0.30 / 9],
                "Fiber (g)": [daily_nutrition["Fiber (g)"]],
                "Sugars (g)": [daily_nutrition["Calories (kcal)"] * 0.10 / 4],
                "Sodium (mg) (g)": [daily_nutrition["Sodium (mg)"] / 1000],
                "Cholesterol (mg) (g)": [daily_nutrition["Cholesterol (mg)"] / 1000],
            }
        )
        user_input = user_input.reindex(columns=feature_cols, fill_value=0)

        deficits = model.predict(user_input)[0]
        probs = model.predict_proba(user_input)

        result_left, result_right = st.columns(2, gap="large")

        with result_left:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Deficiency Risk Prediction</div>', unsafe_allow_html=True)
            for i, col in enumerate(risk_cols):
                nutrient = col.replace("_risk", "")
                prob = probs[i][1] if len(probs[i]) > 1 else 0.5
                warning_text = ""
                if nutrient in ("Protein (g)", "Fiber (g)"):
                    threshold = NOTEBOOK_RDA_THRESHOLDS[nutrient]["threshold"]
                    current_value = daily_nutrition.get(nutrient, 0.0)
                    unit = NOTEBOOK_RDA_THRESHOLDS[nutrient]["unit"]
                    difference = current_value - threshold
                    if difference < 0:
                        warning_text = f"Warning: intake is {abs(difference):.1f} {unit} below the required threshold of {threshold:.1f} {unit}."
                    elif difference > 0:
                        warning_text = f"Warning: intake is {difference:.1f} {unit} above the required threshold of {threshold:.1f} {unit}."
                    else:
                        warning_text = f"Intake matches the required threshold of {threshold:.1f} {unit}."
                render_status_row(nutrient, prob, deficits[i] == 1, warning_text)

            st.markdown('<div class="section-title">RDA Threshold Warnings</div>', unsafe_allow_html=True)
            warnings_found = False
            for nutrient, config in NOTEBOOK_RDA_THRESHOLDS.items():
                current_value = daily_nutrition.get(nutrient, 0.0)
                threshold = config["threshold"]
                unit = config["unit"]

                if config["direction"] == "min" and current_value < threshold:
                    warnings_found = True
                    gap = threshold - current_value
                    st.markdown(
                        f"""
                        <div class="tip-card">
                            <div class="tip-title">{nutrient} is below required level</div>
                            <div class="tip-copy">Current intake: {current_value:.1f} {unit} | Required threshold: {threshold:.1f} {unit} | You still need about {gap:.1f} {unit}.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if config["direction"] == "max" and current_value > threshold:
                    warnings_found = True
                    excess = current_value - threshold
                    st.markdown(
                        f"""
                        <div class="tip-card">
                            <div class="tip-title">{nutrient} is above recommended limit</div>
                            <div class="tip-copy">Current intake: {current_value:.1f} {unit} | Limit: {threshold:.1f} {unit} | You are about {excess:.1f} {unit} above the threshold.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if not warnings_found:
                st.markdown(
                    """
                    <div class="tip-card">
                        <div class="tip-title">RDA threshold check</div>
                        <div class="tip-copy">Your current intake is within the notebook-defined thresholds for protein, fiber, sodium, and cholesterol.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with result_right:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Personalized Recommendations</div>', unsafe_allow_html=True)
            recommendation_found = False

            for nutrient, config in NOTEBOOK_RDA_THRESHOLDS.items():
                current_value = daily_nutrition.get(nutrient, 0.0)
                threshold = config["threshold"]
                unit = config["unit"]

                if config["direction"] == "min" and current_value < threshold:
                    recommendation_found = True
                    gap = threshold - current_value
                    suggestions = get_food_suggestions(
                        food_df,
                        nutrient,
                        current_value,
                        threshold,
                        selected_food_names,
                    )
                    suggestion_lines = "".join(
                        [
                            f"<div class='tip-copy'>Add <strong>{item['food']}</strong> ({item['category']}) for about {item['amount']:.1f} {unit} {nutrient.split(' ')[0].lower()} per serving.</div>"
                            for item in suggestions
                        ]
                    )
                    st.markdown(
                        f"""
                        <div class="tip-card">
                            <div class="tip-title">Increase {nutrient}</div>
                            <div class="tip-copy">Your intake is {current_value:.1f} {unit}. According to <em>recommandations.ipynb</em>, the RDA threshold is {threshold:.1f} {unit}, so you are short by about {gap:.1f} {unit}.</div>
                            {suggestion_lines if suggestion_lines else "<div class='tip-copy'>No suitable nutrient-rich foods were available to suggest from the dataset.</div>"}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if config["direction"] == "max" and current_value > threshold:
                    recommendation_found = True
                    excess = current_value - threshold
                    suggestions = get_limit_relief_suggestions(
                        food_df,
                        nutrient,
                        selected_food_names,
                    )
                    suggestion_lines = "".join(
                        [
                            f"<div class='tip-copy'>Choose <strong>{item['food']}</strong> ({item['category']}) instead. It has about {item['amount']:.1f} {unit} {nutrient.split(' ')[0].lower()} per serving in your dataset.</div>"
                            for item in suggestions
                        ]
                    )
                    st.markdown(
                        f"""
                        <div class="tip-card">
                            <div class="tip-title">Reduce {nutrient}</div>
                            <div class="tip-copy">Your intake is {current_value:.1f} {unit}. According to <em>recommandations.ipynb</em>, the limit is {threshold:.1f} {unit}, so you are over by about {excess:.1f} {unit}.</div>
                            {suggestion_lines if suggestion_lines else "<div class='tip-copy'>No lower-{nutrient.split(' ')[0].lower()} foods were available to suggest from the dataset.</div>"}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if not recommendation_found:
                st.success("Excellent! Your intake is within the notebook-defined RDA thresholds.")

            st.markdown(
                """
                <div class="target-note">
                    Recommendations now follow the thresholds defined in <em>recommandations.ipynb</em>: protein 46 g, fiber 25 g, sodium 2300 mg, and cholesterol 300 mg.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="footer-note">NutriPredict | One day | Four meals | Lifestyle-first nutrition view</div>', unsafe_allow_html=True)
