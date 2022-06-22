import streamlit as st
import pandas as pd
import joblib

# 页面内容设置
# 页面名称
st.set_page_config(page_title="MINS", layout="wide")
# 标题
st.title('The machine-learning based model to predict MINS')
# 文本
st.write('This is a web app to predict the prob of MINS based on\
         several features that you can see in the sidebar. Please adjust the\
         value of each feature. After that, click on the Predict button at the bottom to\
         see the prediction.')

st.markdown('## Input Data:')
# 隐藏底部水印
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            <![]()yle>
            """

st.markdown(hide_st_style, unsafe_allow_html=True)

def option_name(x):
    if x == 0:
        return "no"
    if x == 1:
        return "yes"
@st.cache
def predict_quality(model, df):
    y_pred = model.predict(df, prediction_type='Probability')
    return y_pred[:, 1]

# 导入模型
model = joblib.load('cb_25fea.pkl')
st.sidebar.title("Features")

# 设置各项特征的输入范围和选项
age = st.sidebar.number_input(label='age', min_value=1.0,
                                  max_value=100.0,
                                  value=70.0,
                                  step=1.0)

ASA = st.sidebar.selectbox(label='ASA', options=[1,2,3,4], index=1)

surgeryduration = st.sidebar.number_input(label='surgeryduration', min_value=1.00,
                                max_value=1000.00,
                                value=205.0,
                                step=1.0)

anesthesiaduration = st.sidebar.number_input(label='anesthesiaduration', min_value=1.0,
                                   max_value=5000.0,
                                   value=240.0,
                                   step=1.0)

hypertension = st.sidebar.selectbox(label='hypertension', options=[0, 1], format_func=lambda x: option_name(x), index=0)

cerebrovascular_disease = st.sidebar.selectbox(label='cerebrovascular disease', options=[0, 1], format_func=lambda x: option_name(x), index=0)

coronary_heart_disease = st.sidebar.selectbox(label='coronary heart disease', options=[0, 1], format_func=lambda x: option_name(x), index=0)

renal_insufficiency = st.sidebar.selectbox(label='renal insufficiency', options=[0, 1], format_func=lambda x: option_name(x), index=0)

RBC = st.sidebar.number_input(label='RBC', min_value=1.00,
                       max_value=10.00,
                       value=4.67,
                       step=0.01)

lymphocyte = st.sidebar.number_input(label='lymphocyte', min_value=0.01,
                              max_value=50.00,
                              value=0.31,
                              step=0.01)

RDW = st.sidebar.number_input(label='RDW', min_value=10.0,
                            max_value=50.0,
                            value=12.4,
                            step=0.1)

HGB = st.sidebar.slider(label='HGB', min_value=1.0,
                            max_value=500.0,
                            value=138.0,
                            step=1.0)

SCR = st.sidebar.slider(label='SCR', min_value=1.0,
                            max_value=1000.0,
                            value=68.7,
                            step=0.1)

Albumin = st.sidebar.slider(label='Albumin', min_value=1.0,
                            max_value=100.0,
                            value=44.2,
                            step=0.1)

GLU = st.sidebar.slider(label='GLU', min_value=1.0,
                            max_value=50.0,
                            value=4.4,
                            step=0.01)

Na = st.sidebar.slider(label='Na', min_value=100.0,
                            max_value=200.0,
                            value=142.2,
                            step=0.1)

Diuretic = st.sidebar.selectbox(label='Diuretic', options=[0, 1], format_func=lambda x: option_name(x), index=0)

Anticoagulants = st.sidebar.selectbox(label='Anticoagulants', options=[0, 1], format_func=lambda x: option_name(x), index=0)

Beta_blockers = st.sidebar.selectbox(label='Beta_blockers', options=[0, 1], format_func=lambda x: option_name(x), index=0)

colloid = st.sidebar.number_input(label='colloid', min_value=0.0,
                            max_value=10000.0,
                            value=1000.0,
                            step=10.0)

crystalloid = st.sidebar.slider(label='crystalloid', min_value=0.0,
                            max_value=10000.0,
                            value=3100.0,
                            step=50.0)
blood_loss = st.sidebar.slider(label='blood loss', min_value=0.0,
                            max_value=10000.0,
                            value=1100.0,
                            step=1.0)

BP_monitoring = st.sidebar.selectbox(label='BP_monitoring', options=[0, 1], format_func=lambda x: option_name(x), index=0)

MAP_65_time = st.sidebar.slider(label='MAP.65.time', min_value=0.0,
                            max_value=1000.0,
                            value=44.0,
                            step=0.5)
Blood_transfusion = st.sidebar.selectbox(label='Blood_transfusion', options=[0, 1], format_func=lambda x: option_name(x), index=0)

features = {'age': age, 'ASA': ASA,
            'surgeryduration': surgeryduration, 'anesthesiaduration': anesthesiaduration,
            'hypertension': hypertension, 'cerebrovascular disease': cerebrovascular_disease,
            'coronary heart disease': coronary_heart_disease, 'renal insufficiency': renal_insufficiency,
            'RBC': RBC, ' lymphocyte': lymphocyte,
            'RDW': RDW, 'HGB': HGB,
            'SCR': SCR, 'Albumin': Albumin,
            'GLU': GLU, 'Na': Na,
            'Diuretic': Diuretic, 'Anticoagulants': Anticoagulants,
            'Beta blockers': Beta_blockers, 'colloid': colloid,
            'crystalloid': crystalloid, 'blood loss': blood_loss,
            'BP monitoring ': BP_monitoring, 'MAP.65.time': MAP_65_time,
            'Blood transfusion ': Blood_transfusion
            }

features_df = pd.DataFrame([features])
# show_features_df = pd.DataFrame(features, index=None)

# features_df.style.format("{:0.2f}").hidden_index()
print(features_df)
st.table(features_df)
# from st_aggrid import AgGrid, GridOptionsBuilder
# # options_builder = GridOptionsBuilder.from_dataframe(features_df)
# # options_builder.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=False, wrapText=True, autoHeight=True)
# # grid_options = options_builder.build()
# grid_return = AgGrid(features_df, height=70, theme='blue')
import shap
import matplotlib.pyplot as plt
if st.button('Predict'):
    prediction = predict_quality(model, features_df)
    st.write("the probability of MINS:")
    st.success(round(prediction[0], 4))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df)
    shap.force_plot(explainer.expected_value, shap_values[0], features_df, matplotlib=True, show=False)
    plt.subplots_adjust(top=0.67,
                        bottom=0.0,
                        left=0.1,
                        right=0.9,
                        hspace=0.2,
                        wspace=0.2)
    plt.savefig('test_shap.png')
    st.image('test_shap.png', caption='Individual prediction explaination', use_column_width=True)
    #st.metric("the probability of MINS:", round(prediction[0], 4))
    #st.write(' Based on feature values, the probability of MINS is ' + str(prediction))


