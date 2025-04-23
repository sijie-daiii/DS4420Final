import streamlit as st
from PIL import Image

def landing_page():
    st.title("Bird Population Decline Prediction")
    st.markdown("""
    ### Project Overview  
    **Idea:**  
    Predicting bird population decline based on sightings data.
    
    **GitHub Repository (Arihant Pant & Sijie Dai):**  
    [https://github.com/pantari76/ds4420-bird-model](https://github.com/pantari76/ds4420-bird-model)
    
    **Datasets:**  
    - **GBIF:** Contains bird sighting records ([GBIF Dataset](https://www.gbif.org/dataset/4fa7b334-ce0d-4e88-aaae-2e0c138d049e/activity)).  
    - **IUCN Red List:** Provides conservation statuses ([IUCN Red List](https://www.iucnredlist.org/)).  
    - **NABBS (1966-2021):** Long-term North American breeding bird survey data.
    
    **Methods Employed:**  
    - **Time Series Analysis:** To forecast population changes over time.  
    - **Collaborative Filtering:** To detect similarities between species’ sighting patterns.  
    - **Neural Networks (MLP):** To extract features (e.g., trends, location, density) and predict population fluctuations.
    
    **Literature References:**  
    - [IEEE Document](https://ieeexplore.ieee.org/abstract/document/10450141)  
    - [ScienceDirect Article](https://www.sciencedirect.com/science/article/pii/S0048969721011608)  
    - [The Auk Article](https://academic.oup.com/auk/article/124/3/1047/5562747)
    """)
    st.markdown("---")
    st.subheader("Data Description & Completeness")
    st.markdown("""
    The North American Breeding Bird Survey (NABBS) provides data from 1966 to 2021, including:
    
    1. **Data Collection:** Annual route-based surveys.  
    2. **Sampling Bias:** Road-based observations may under-sample remote areas.  
    3. **Trend Analysis:** A long period enables detection of sustained population changes.  
    4. **Observer Variability:** Variations in observer skill affect detection rates.  
    5. **Habitat Coverage:** Limited to areas near roads.
    
    **Advice:**  
    - Leverage robust statistical models and machine learning techniques to extract meaningful patterns.  
    - Continuously validate models using multiple data sources.  
    - Use interactive visualizations to share insights and support conservation decisions.
    """)
    
    img1 = Image.open("run_data/bird_decline.png")
    st.image(img1, caption="Bird Population Visualization")

def mlp_visualization():
    st.title("Neural Network (MLP) Visualization")
    st.markdown("### MLP Training Loss Curve")

    img_loss = Image.open("run_data/mlp_loss.png")
    st.image(img_loss, caption="MLP Training Loss Curve")
        
    st.markdown("---")
    st.markdown("### MLP Test Set Confusion Matrix")

    img_cm_test = Image.open("run_data/confusion_matrix_test.png")
    st.image(img_cm_test, caption="MLP Test Set Confusion Matrix")
        
    st.markdown("---")
    st.markdown("### MLP Training Set Confusion Matrix")

    img_cm_train = Image.open("run_data/confusion_matrix_train.png")
    st.image(img_cm_train, caption="MLP Training Set Confusion Matrix")

def time_series_visualization():
    st.title("Time Series Forecast Visualization")
    st.markdown("#### Naïve Forecast (BlackSwiftPop.csv)")

    img_ts = Image.open("run_data/time_series_black_swift.png")
    st.image(img_ts, caption="Black Swift Population Trend (Naïve Forecast)")
        
    st.markdown("---")
    st.markdown("#### AR(2)/ARMA(2,1) Forecast (Black Swift)")

    img_forecast_bs = Image.open("run_data/forecast_black_swift.png")
    st.image(img_forecast_bs, caption="Black Swift Forecast: AR(2)/ARMA(2,1)")
 
    st.markdown("---")
    st.markdown("#### AR(6)/ARMA(6,1) Forecast (Marbled Murrelet)")

    img_forecast_mm = Image.open("run_data/forecast_murrelet.png")
    st.image(img_forecast_mm, caption="Marbled Murrelet Forecast: AR(6)/ARMA(6,1)")

def collaborative_filtering():
    st.title("Collaborative Filtering Insights")
    st.markdown("""
    **Concept:**  
    Collaborative filtering identifies species with similar observational patterns.  
    Species similar to those already labeled as critically endangered may require further conservation action.
    """)
    img_cf = Image.open("run_data/collaborative_filtering_real.png")
    st.image(img_cf, caption="Collaborative Filtering Similarity Matrix")

def conclusion():
    st.title("Conclusion")
    st.markdown(""" 
    **Analysis Summary:**  
    - **Neural Network (MLP):**  
      The MLP training loss curve shows that the model consistently reduced its error over 1000 epochs, converging to a low loss value. The test set confusion matrix indicates that most predictions aligned well with the true IUCN categories; however, misclassifications between adjacent classes suggest that further refinements (e.g., feature engineering or hyperparameter tuning) could improve accuracy.
      
    - **Time Series Forecasting:**  
      The naïve forecast for Black Swift offers a simple baseline by extending the most recent population value. More sophisticated forecasts using AR(2)/ARMA(2,1) capture underlying trends better, while AR(6)/ARMA(6,1) forecasts for Marbled Murrelet reveal more complex dynamics—indicating that species display varying temporal behaviors.
      
    - **Collaborative Filtering:**  
      The similarity matrix computed from the sightings data provides insights into which species share similar observational patterns. These insights can help identify species that, while not yet critically endangered, show warning signs similar to those that are.
      
    **Overall Conclusion:**  
    By integrating neural network classification, time series forecasting, and collaborative filtering, the project provides a multifaceted, data-driven framework for assessing bird population trends and conservation statuses. These analyses can inform early conservation interventions and improve our understanding of biodiversity changes.

    """)

    img_con = Image.open("run_data/bird.jpg")
    st.image(img_con, caption="Additional Visual Insight")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page:", ("Landing", "MLP Visualization", "Time Series Visualization", "Collaborative Filtering", "Conclusion"))
    if page == "Landing":
        landing_page()
    elif page == "MLP Visualization":
        mlp_visualization()
    elif page == "Time Series Visualization":
        time_series_visualization()
    elif page == "Collaborative Filtering":
        collaborative_filtering()
    elif page == "Conclusion":
        conclusion()

if __name__ == '__main__':
    main()
