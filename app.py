import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

DATA_URL = r".\data\conversion_data_train.csv"
MODEL_DF_URL = r".\data\models_summary.csv"
MODEL_URL = r".\model\model.joblib"
LOGO_URL = r".\images\news-letter.png"

### Config
st.set_page_config(
    page_title="News letter conversion app",
    page_icon="📰",
    layout="wide"
)

### Data
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    data = data[data['age'] < 65]
    model_df = pd.read_csv(MODEL_DF_URL)

    return data, model_df

data, model_df = load_data()

### Load model
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_URL)
    return model

model = load_model()

### Streamlit pages

def title_page():
    st.title("News letter conversion app")
    st.image(LOGO_URL, width="stretch")

def dataset_page():
    st.title("The dataset")
    st.write(f"The dataset have: **{data.shape[0]} rows** and **{data.shape[1]} columns**, below is a summary:")
    
    meta_data = pd.DataFrame({
        "features": [
            "country",
            "age",
            "news_user",
            "source",
            "total_pages_visited",
            "converted"
            ],

        "description": [
            "User country",
            "User age",
            "User is a news user",
            "User source",
            "Number of pages visited",
            "User subscribed to the newsletter"
            ]
        })

    with st.expander("Show metadata"):
        st.dataframe(meta_data, hide_index=True, width='stretch')

    with st.expander("Show raw data preview"):
        st.dataframe(data.head(), hide_index=True, width='stretch')

    with st.expander("Show data types"):
        data_types = data.dtypes
        st.dataframe(pd.DataFrame(data_types).T, hide_index=True, width='stretch')

    with st.expander("Show descriptive statistics"):
        st.write(data.describe())
        st.caption("Records with age > 65 are removed (to remove users with 111 and 123 years old)")
    
    with st.expander("Show missing values"):
        null_table = data.isnull().sum()
        st.dataframe(pd.DataFrame(null_table).T, hide_index=True, width='stretch')

def conversion_rate_analysis():
    st.title("Conversion rate analysis")
    converted_perc_df = (data.value_counts('converted')/data.shape[0]).round(2)
    converted_perc_df.index = ['Not converted','Converted']
    converted_perc_df.rename('percentage', inplace=True)
    fig = px.pie(converted_perc_df, values='percentage', names=converted_perc_df.index, title='Conversion rate')
    fig.update_traces(textposition='inside', textinfo='percent+label', title_text='')
    st.plotly_chart(fig, width='stretch')

    st.badge("**Conversion rate is generally very low**", icon="📉", color="blue")

def country_analysis():
    st.title("Conversion rate per country")
    converted_perc_df = data.groupby('country')['converted'].mean().sort_values(ascending=False).round(2)
    fig = px.bar(x=converted_perc_df.index, y=converted_perc_df.values, color=converted_perc_df.index)
    fig.update_layout(yaxis_title='Conversion rate', xaxis_title='Country', title='Conversion rate by country')
    st.plotly_chart(fig, width='stretch')

    st.badge("**German users are more likely to subscribe to the newsletter**", icon="🥨", color="blue")

def age_visits_analysis():
    st.title("Conversion rate per age")
    age_converted = data.groupby('age')['converted'].mean().sort_values(ascending=False)
    fig = px.bar(x=age_converted.index, y=age_converted.values)
    fig.update_layout(yaxis_title='Conversion rate', xaxis_title='Age', title='Conversion rate by age')
    st.plotly_chart(fig, width='stretch')

    st.badge("**Young users are more likely to subscribe to the newsletter**", icon="👶", color="blue")

def visit_analysis():
    st.title("Visits and subscription")

    page_visited_converted = data.groupby('total_pages_visited')['converted'].mean().sort_values(ascending=False)
    fig = px.bar(x=page_visited_converted.index, y=page_visited_converted.values)
    fig.update_layout(yaxis_title='Conversion rate', xaxis_title='Pages visited', title='Conversion rate by pages visited')
    st.plotly_chart(fig, width='stretch')

    st.badge("**Users with a high engagement are more likely to subscribe to the newsletter**", icon="🌐", color="blue")

def model_comparison():
    st.title("Models comparison")

    models_scores= model_df.groupby('model')[['f1', 'precision', 'recall']].max().reset_index()
    models_scores = models_scores.sort_values('f1', ascending=False)
    fig = px.bar(models_scores, x='model', y=['f1', 'precision', 'recall'], barmode='group')
    fig.update_layout(yaxis_title='Scores', xaxis_title='Models', yaxis_range=[0.65, 0.88])
    st.plotly_chart(fig, width='stretch')

def optimizer_comparison():
    st.title("Optimizers comparison")

    optimizers_scores= model_df.groupby('optimizer')[['f1', 'precision', 'recall']].max().reset_index()
    optimizers_scores = optimizers_scores.sort_values('f1', ascending=False)
    fig = px.bar(optimizers_scores, x='optimizer', y=['f1', 'precision', 'recall'], barmode='group')
    fig.update_layout(yaxis_title='Scores', xaxis_title='Optimizers', yaxis_range=[0.65, 0.88])
    st.plotly_chart(fig, width='stretch')
    
def conversion_prediction_app():
    st.title("Conversion prediction app")

## Inputs
    pred_country = st.selectbox("Select a country", data["country"].unique())
    pred_age = st.number_input("Select an age", value=30, min_value=17, max_value=62, step=1)
    pred_visit = st.number_input("Select a number of visits", value=5, min_value=1, step=1)
    pred_new = st.checkbox("Is the person a new visitor ?")

## Prediction
    if st.button("Make a prediction"):
        data_to_predict = pd.DataFrame({
            "total_pages_visited": [pred_visit],
            "country": [pred_country],
            "new_user": [1 if pred_new else 0],
            "age": [pred_age]
            })
        
        with st.spinner("Predicting..."):
            prediction = model.predict(data_to_predict)
            if prediction == 1:
                st.success("The person **will subscribe** to the newsletter")
            else:
                st.error("The person **will not subscribe** to the newsletter")

    ### Pages layout

pages = {
    "Context": [
    st.Page(title_page, title="Welcome", icon="👋"),
    st.Page(dataset_page, title="Dataset", icon="📜")
    ],
    "Insights": [
    st.Page(conversion_rate_analysis, title="Conversion rate", icon="💌"),
    st.Page(country_analysis, title="Country", icon="🗺️"),
    st.Page(age_visits_analysis, title="Age and visits", icon="👤"),
    st.Page(visit_analysis, title="visits and conversion", icon="📈"),
    ],
    "Prediction App": [
    st.Page(model_comparison, title="Model comparison", icon="⚙️"),
    st.Page(optimizer_comparison, title="Optimizer comparison", icon="🦾"),
    st.Page(conversion_prediction_app, title="Predictions", icon="📑"),
    ]
    }

pg = st.navigation(pages)

pg.run()