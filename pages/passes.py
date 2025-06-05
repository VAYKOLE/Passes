# Library imports
from pathlib import Path
import sys

#importing necessary libraries
from mplsoccer import Sbopen
import pandas as pd
import numpy as np
import json
import warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os
import random as rn
#warnings not visible on the course webpage
pd.options.mode.chained_assignment = None
warnings.filterwarnings('ignore')


#setting random seeds so that the results are reproducible on the webpage
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(1)
rn.seed(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import streamlit as st
import pandas as pd
import numpy as np
import argparse
import tiktoken
import os



#from classes.visual import PassVisual_logistic as PassVisual
from classes.data_source import Passes
from classes.visual import  PassContributionPlot_XGBoost
from classes.description import PassDescription_xgboost
from classes.visual import  PassContributionPlot_XGBoost
from classes.visual import PassVisual
from classes.description import PassDescription_xgboost
from classes.data_source import Passes
from classes.visual import PassVisual,PassContributionPlot_XGBoost
from classes.description import PassDescription_xgboost
from classes.chat import Chat




#from classes.data_source import show_mimic_tree_in_streamlit
#from classes.data_source import generate_pass_counterfactuals_by_id
#from classes.visual import CounterfactualContributionPlot_XGBoost


    
# Function to load and inject custom CSS from an external file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



from utils.page_components import (
    add_common_page_elements
)


from classes.chat import PlayerChat

from utils.page_components import add_common_page_elements
from utils.utils import select_player, create_chat

sidebar_container = add_common_page_elements()
page_container = st.sidebar.container()
sidebar_container = st.sidebar.container()

st.divider()

st.markdown("## Passes commentator")

competitions = {
    "Allsevenskan 2022": "data/matches_2022.json",
    "Allsevenskan 2023": "data/matches_2023.json"
}

# Select a competition
selected_competition = st.sidebar.selectbox("Select a Competition", options=competitions.keys())

# Load the JSON file corresponding to the selected competition
file_path = competitions[selected_competition]

with open(file_path, 'r') as f:
    id_to_match_name = json.load(f)


selected_match_name = st.sidebar.selectbox(
    "Select a Match", 
    options=id_to_match_name.values())

match_name_to_id = {v: k for k, v in id_to_match_name.items()}
selected_match_id = match_name_to_id[selected_match_name]

# Create a dropdown to select a shot ID from the available shot IDs in shots.df_shots['id']

pass_data = Passes(selected_competition,selected_match_id)

pass_df = pass_data.df_pass
tracking_df = pass_data.df_tracking
pass_df = pass_df[[col for col in pass_df.columns if "_contribution" not in col and col != "xT"]]
pass_df_xgboost = pass_data.pass_df_xgboost



# Dropdown showing actual pass IDs
selected_pass_id = st.sidebar.selectbox("Select a pass id:", options=pass_df['id'].tolist())

pass_id = selected_pass_id

# Define the tabs
tab1, = st.tabs([ "XGBoost"])


 
with tab1:
    st.header("XGBoost")

   # model = Passes.load_xgboost_model(selected_competition)
    st.write(pass_df_xgboost.astype(str))
    st.markdown("<h3 style='font-size:24px; color:black;'>Feature contribution from model</h3>", unsafe_allow_html=True)
    #feature_contrib_df = Passes.get_feature_contributions(pass_df_xgboost, model)
    feature_contrib_df = pass_data.feature_contrib_df
    
    st.write(feature_contrib_df.astype(str))

    #xgboost_contribution_describe = feature_contrib_df.describe()
    #xgboost_contribution_describe.to_csv("xgboost_contribution_describe.csv")

    # Show the XGBoost feature contribution plot
    st.markdown("<h3 style='font-size:24px; color:black;'>XGBoost contribution plot</h3>", unsafe_allow_html=True)

    excluded_columns = ['xT_predicted','id', 'match_id']
    metrics = [col for col in feature_contrib_df.columns if col not in excluded_columns]

    visuals_xgboost = PassContributionPlot_XGBoost(feature_contrib_df=feature_contrib_df,pass_df_xgboost=pass_df_xgboost,metrics=metrics)
    visuals_xgboost.add_passes(pass_df_xgboost, metrics, selected_pass_id=selected_pass_id)
    visuals_xgboost.annotate = True
    visuals_xgboost.add_pass(feature_contrib_df=feature_contrib_df,pass_df_xgboost=pass_df_xgboost,
    pass_id=selected_pass_id,metrics=metrics,selected_pass_id=selected_pass_id)

    visuals_xgboost.show()

    # Show the XGBoost counterfactual plot
    # st.markdown("<h3 style='font-size:24px; color:black;'>XGBoost counterfactual plot</h3>", unsafe_allow_html=True)

    # # Explanation for the slider
    # st.markdown(
    # """
    # Use the slider below to set a threshold for expected threat (xT).
    # Counterfactual examples where the predicted xT exceeds this value will be shown.
    # """,
    # unsafe_allow_html=True
    # )

    # # Add a slider to the sidebar or main app
    # threshold = st.slider(
    # label="Set expected threat (xT) threshold",
    # min_value=0.04,
    # max_value=0.65,
    # value=0.5,      # default value shown initially
    # step=0.01       # how much the slider moves with each step
    # )

    # # Display the selected value to the user
    # st.write(f"You selected xT threshold: {threshold}")

    # passes_instance = Passes(competition=selected_competition, match_id=selected_match_id)
    # xGB_model = passes_instance.load_xgboost_model(selected_competition)


    # # Run counterfactual generation
    # result_df, pred_prob, shap_df_cf =passes_instance.generate_pass_counterfactuals_by_id(selected_pass_id=selected_pass_id,pass_df_xgboost=pass_df_xgboost,xGB_model=xGB_model,
    #                                                 threshold=threshold,total_CFs=1)

    # st.info(f"xT for selected pass ({selected_pass_id}) = {pred_prob:.3f}")

    #if result_df.empty:
    #    st.warning(f"Original xT ({pred_prob:.3f}) already exceeds threshold ({threshold}) — skipping counterfactual generation.")
    #else:
    #    st.subheader("Counterfactuals for this pass")
    #    st.dataframe(result_df)

    # if result_df.empty:
    #     if pred_prob > threshold:
    #         st.warning(f"Original xT ({pred_prob:.3f}) already exceeds threshold ({threshold}) — skipping counterfactual generation.")
    #     else:
    #         st.warning("No counterfactuals could be generated for this pass — the model couldn’t find a better option.")
    # else:
    #     st.subheader("Counterfactuals for this pass")
    #     st.dataframe(result_df)

    # if not shap_df_cf.empty:
    #     st.subheader("SHAP Contributions for Counterfactual Pass")
    #     st.dataframe(shap_df_cf)
    #     shap_df_cf_long = shap_df_cf.T.reset_index()
    #     shap_df_cf_long.columns = ["feature", "shap_value"]
    #     shap_df_cf_long["feature_value"] = shap_df_cf.iloc[0].values

    
        # plotter = CounterfactualContributionPlot_XGBoost(shap_df_cf_long)
        # fig = plotter.plot()
        # st.plotly_chart(fig, use_container_width=True)
    # Show results
    #st.subheader("Counterfactuals for this pass")
    #st.dataframe(result_df)



    xt_value_xgboost = feature_contrib_df[feature_contrib_df['id'] == pass_id]['xT_predicted']
    xt_value_xgboost = xt_value_xgboost.iloc[0] if not xt_value_xgboost.empty else "N/A"


    descriptions = PassDescription_xgboost(pass_data,feature_contrib_df,pass_id, selected_competition)
    
    to_hash = ("xgBoost",selected_match_id, pass_id)
    summaries = descriptions.stream_gpt()
    chat = create_chat(to_hash, Chat)

    st.markdown(
    f"<h5 style='font-size:18px; color:green;'>Pass ID: {pass_id} | Match Name : {selected_match_name} | xT : {xt_value_xgboost}</h5>",
    unsafe_allow_html=True
    )
    visuals = PassVisual(metric=None)
    visuals.add_pass(pass_data,pass_id,home_team_color = "green" , away_team_color = "red")
    visuals.show()
    
    if summaries:
        chat.add_message(summaries)

    chat.display_messages()





    



