"""
UI app for Mercari Price Suggestion case study using streamlit.
"""


import re
import os
import joblib
import gdown

import nltk
import streamlit as st
import pandas as pd
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from scipy.sparse import hstack
from bs4 import BeautifulSoup
from tensorflow import keras


def split_cat(category_name):
    """
    Split Category into 3 different levels.
    """
    categories = ["no_main_cat", "no_subcat_1", "no_subcat_2"]
    if isinstance(category_name, str):
        split_values = category_name.split("/")
        for i, cat in enumerate(split_values):
            if cat and (i < 3):
                categories[i] = cat
    return categories

def preprocess_brand_name(txt):
    """
    Preprocess brand name text.
    """
    # remove html tags
    sent = re.sub(r"http\S+", "", txt)
    sent = BeautifulSoup(sent, 'lxml').get_text()
    # clean up certain special symbols in text
    sent = sent.replace("′", "'").replace("’", "'")
    sent = re.sub('[^A-Za-z0-9_\-]+', ' ', sent)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\n', ' ')
    # lower the text
    sent = str(sent).lower()
    # removing any extra white spaces
    sent = ' '.join(word for word in sent.split())
    sent = sent.strip()
    return sent

def decontracted(phrase):
    """
    decontract shorthand words used in the data.
    """
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"cannot", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# removing the following words from the stop words list: 'no', 'nor', 'not'
STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS = STOP_WORDS - {'no', 'nor', 'not'}
# let's add "br" because after removing html tags such as "<br />" using 
# Beatiful Soup we may still be left with "br".
# instead of <br /> if we had <br/> these tags would have been removed with 
# Beatiful Soup.
STOP_WORDS.update("br")

# initialize stemmer
stemmer = SnowballStemmer("english")

# let's keep ? and ! marks in the item name and desc without removing them, 
# since it could mean seller was trying to convey something in a different way 
# to grab the buyer's attention and it could have had a first impression on the 
# buyer when the name or desc was read. 
def preprocess_item_name_and_desc(txt, remove_stopwords=True):
    """
    Preprocess item_name and item_description.
    """
    # remove html tags
    sent = re.sub(r"http\S+", "", txt)
    sent = BeautifulSoup(sent, 'lxml').get_text()
    # lower the text and clean up certain special symbols in text
    sent = sent.replace("′", "'").replace("’", "'")
    sent = str(sent).lower()
    sent = decontracted(sent)
    sent = re.sub('[^A-Za-z0-9_\!\?]+', ' ', sent)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\n', ' ')
    sent = sent.replace('\\"', ' ')
    # remove stopwords if required and remove any extra white spaces.
    if remove_stopwords:
        sent = ' '.join(
                   stemmer.stem(e) for e in sent.split() if e not in STOP_WORDS)
    else:
        sent = " ".join(stemmer.stem(e) for e in sent.split())
    sent = sent.strip()
    return sent

def tokenize(txt):
    """
    Tokenize text using nltk word_tokenize and sent_tokenize.
    """
    tokens = []
    for s in sent_tokenize(txt):
        tokens.extend(word_tokenize(s))
    return tokens

def preprocess_data(df):
    """
    Preprocess input data as necessary by the models for prediction.

    Parameters
    ----------
    df: pandas.DataFrame
        Input product data whose price prediction is to be made. 
        This can be single point or a set of points. 
    
    Returns
    -------
    preprocessed_df: pandas.DataFrame
        Dataframe with preprocessed data.
    """
    # Fill empty string("") for Null Values:
    df.fillna('', inplace=True)
    df.loc[df["item_description"].str.lower() == 
            "no description yet", "item_description"] = ""
    # ==========================================================================

    # Creating indicators "missing_brand_name" and "missing_item_desc":
    df["missing_brand_name"] = (df["brand_name"]=="").map(int)
    df["missing_item_desc"] = (df["item_description"]=="").map(int)
    # ==========================================================================

    # Split Category:
    df["main_cat"], df["subcat_1"], df["subcat_2"] = zip(
                                        *df["category_name"].apply(split_cat))
    # drop the original "category_name" column as it's not used anymore.
    df.drop("category_name", inplace=True, axis=1)
    # ==========================================================================

    # Clean up text columns - "brand_name", "item_name" and "item_description":=
    df["preproc_brand_name"] = df["brand_name"].apply(preprocess_brand_name)
    # drop the original brand_name data as it's not used anymore.
    df.drop("brand_name", axis=1, inplace=True)
    # preprocess item name without removing stop words
    df["preproc_name"] = df["name"].apply(preprocess_item_name_and_desc,
                                                    remove_stopwords=False)
    # preprocess item name by removing stop words
    df["preproc_desc"] = df["item_description"].apply(
                                                preprocess_item_name_and_desc)
    # ==========================================================================

    # Combining "item_name", "brand_name" and "item_description" into "text" 
    # column and combining "item_name" and "brand_name" into "name_and_brand" 
    # column:
    df["preproc_text"] = (df["preproc_name"]+" "+df["preproc_desc"]+" "+
                            df["preproc_brand_name"])
    df["preproc_name_and_brand"] = (df["preproc_name"]+" "+
                                    df["preproc_brand_name"])
    # replace multiple whitespaces with a single space
    df['preproc_text'] = df['preproc_text'].replace(r'\s+', ' ', regex=True)
    df['preproc_name_and_brand'] = df['preproc_name_and_brand'].replace(
        r'\s+', ' ', regex=True)
    # ==========================================================================

    # Calculating the length and word count for the below:
    # item_name:
    # "name_length", "preproc_name_length", "name_word_count".
    # item_description:
    # "log_desc_length". "log_preproc_desc_word_count".

    # "item_name"
    df["name_length"] = df["name"].apply(len)
    df["preproc_name_length"] = df["preproc_name"].apply(len)
    df["name_word_count"] = df["name"].apply(lambda x: len(word_tokenize(x)))
    # "item_description"
    df["log_desc_length"] = df["item_description"].apply(
        lambda x: np.log1p(len(x)))
    df["log_preproc_desc_word_count"] = df["preproc_desc"].apply(
        lambda x: np.log1p(len(tokenize(x))))
    # drop the columns that are not needed anymore "item_name", 
    # "item_description" and "preproc_desc".
    df.drop("name", axis=1, inplace=True)
    df.drop("item_description", axis=1, inplace=True)
    df.drop(["preproc_desc"], axis=1, inplace=True)
    df.drop(["preproc_name"], axis=1, inplace=True)
    # ==========================================================================

    # Assign the brand count values calculated from train data for test data:
    df["brand_count"] = df["preproc_brand_name"].apply(
                                     lambda x: data["brand_counts"].get(x, 0))
    # ==========================================================================

    # Assign group-by statistics obtained from train data to test data:
    # group_by stats for cv data
    for grp_name, grp_combo in data["group_by_combos"].items():
        df = df.merge(data["grp_by_stats"][grp_name], how='left', on=grp_combo)
    # fill 0 to any nan values that may have resulted when a particular combo 
    # value present in the main dataframe is not found in any of the dataframes 
    # containing group_by stats.
    df.fillna(0, inplace=True)
    # this is not necessary for train set but is needed for test set as it may 
    # contain unseen values for a group_by combo.
    # dropping "preproc_brand_name", since its already merged into "preproc_text"
    df.drop("preproc_brand_name", axis=1, inplace=True)
    # ==========================================================================
    
    cols = set()
    for key, vals in data["final_features"].items():
        cols.update(set(vals))
    assert cols == (set(df.columns) - {"test_id", "train_id", "bins", "price"})
    return df

def create_dataset_1(row_df):
    """
    Return features pertaining to Dataset-1(One-Hot encoded categories and 
    shipping_id, Ordinal encoded item_condition_id + TFIDF encoded text with 
    bigrams and other numerical features obtained from feature engineering).

    Parameters:
    -----------
    row_df: pandas.DataFrame
        Input product data for which features pertaining to Dataset-1 is to be 
        returned. 

    Returns:
    --------
    features: scipy.sparse.csr.csr_matrix
        Features pertaining to dataset-1
    """
    # Seperating Features:
    
    # indicators
    ind_cols_te = row_df.loc[:, data["final_features"]['ind_cols']].astype(int)
    # Categories and Shipping ID
    categories_shipping_te = row_df.loc[:, data["final_features"]['cats_and_shipping']]
    # Preproc name and brand
    preproc_name_and_brand_te = row_df.loc[:, ['preproc_name_and_brand']]
    # Preproc text
    preproc_text_te = row_df.loc[:, ['preproc_text']]
    # Other Engineered features:
    count_feats_te = row_df.loc[:, data["final_features"]['count_feats']]
    # group_by features
    group_by_feat_te = row_df.loc[:, data["final_features"]['grp_by_features']]

    # TFIDF:
    tfidf_text_te = data["tfv_text"].transform(preproc_text_te['preproc_text'])
    tfidf_name_and_brand_te = data["tfv_name_and_brand"].transform(
        preproc_name_and_brand_te['preproc_name_and_brand'])

    # One-Hot Encoding Categories and Shipping Id:
    data["one_hot_encoder"].handle_unknown = "ignore"
    enc_categories_shipping_te = data["one_hot_encoder"].transform(categories_shipping_te)

    # scale test data
    scaled_features_te = data["min_max_scaler_1"].transform(
        np.hstack((ind_cols_te.values, count_feats_te.values, 
                group_by_feat_te.values))
    )

    features = hstack((scaled_features_te, enc_categories_shipping_te, 
                       tfidf_name_and_brand_te, tfidf_text_te), format="csr")
    assert(features.shape[1] == 171883)
    
    return features


def create_dataset_2(row_df):
    """
    Return features pertaining to Dataset-2(Target encoded categories and 
    shipping_id, Ordinal encoded item_condition_id + BOW encoded 
    text with bigrams and other numerical features obtained from feature 
    engineering).

    Parameters:
    -----------
    row_df: pandas.DataFrame
        Input product data for which features pertaining to Dataset-2 is to be 
        returned. 

    Returns:
    --------
    features: scipy.sparse.csr.csr_matrix
        Features pertaining to dataset-2
    """
    # Seperating Features:
    
    # indicators
    ind_cols_te = row_df.loc[:, data["final_features"]['ind_cols']].astype(int)
    # Categories and Shipping ID
    categories_shipping_te = row_df.loc[:, data["final_features"]['cats_and_shipping']]
    # Preproc name and brand
    preproc_name_and_brand_te = row_df.loc[:, ['preproc_name_and_brand']]
    # Preproc text
    preproc_text_te = row_df.loc[:, ['preproc_text']]
    # Other Engineered features:
    count_feats_te = row_df.loc[:, data["final_features"]['count_feats']]
    # group_by features
    group_by_feat_te = row_df.loc[:, data["final_features"]['grp_by_features']]

    # BOW:
    bin_bow_text_te = data["ctv_text"].transform(
        preproc_text_te['preproc_text'])
    bin_bow_name_and_brand_te = data["ctv_name_and_brand"].transform(
        preproc_name_and_brand_te['preproc_name_and_brand'])

    # Target Encoding Categories and Shipping Id:
    enc_categories_shipping_te = data["target_encoder"].transform(
        categories_shipping_te)

    # scale test data
    scaled_features_te = data["min_max_scaler_2"].transform(
        np.hstack((ind_cols_te.values, count_feats_te.values, 
                enc_categories_shipping_te.values, group_by_feat_te.values))
    )

    features = hstack(
        (scaled_features_te, bin_bow_name_and_brand_te, bin_bow_text_te), 
                      format="csr")
    assert(features.shape[1] == 170890)
    return features

def final_predict_prices(X):
    """
    Predict price of the input product data.
    
    Parameters:
    -----------
    X: pandas.DataFrame
        Input product data whose price prediction is to be made. 
        This can be single point or a set of points. 
    
    Returns:
    --------
    Pred_df: pandas.DataFrame
        predicted prices for the product data.
        If the input data had "test_id" column, then it will be used as the index.
    """
    test_id_in_cols = "test_id" in X.columns
    pred_df_index = X["test_id"] if test_id_in_cols else range(len(X))
    pred_df = pd.DataFrame(columns=["price"], index=pred_df_index)
    
    # iterate through each row and generate predictions
    for i in tqdm(range(len(X))):
        row_df = X.iloc[i: i+1]
        # preprocess data
        row_df = preprocess_data(row_df)
        # create datasets with different featurizations
        features_1 = create_dataset_1(row_df)
        features_2 = create_dataset_2(row_df)
        # make prediction
        pred_1 = data["mlp_ds1"].predict(features_1)
        pred_2 = data["mlp_ds2"].predict(features_2)
        final_pred = np.abs(np.expm1(data["meta_lr"].predict(
            np.hstack((pred_1, pred_2)))))[0]
        # store prediction
        if test_id_in_cols:
            pred_df.loc[row_df["test_id"], "price"] = final_pred
        else:
            pred_df.loc[i, "price"] = final_pred
            
    return pred_df


# creating folder to store featurizers and models
if not os.path.exists('featurizers'):
    os.mkdir('featurizers')
if not os.path.exists('models'):
    os.mkdir('models')

# download featurizers and models that is stored in drive
if len(os.listdir('featurizers')) <= 1:
    url = "https://drive.google.com/drive/folders/14tLKBd1PezOfWYjkDQIQ7xop6sbHgCcX"
    gdown.download_folder(url, quiet=False, use_cookies=False)

if len(os.listdir('models')) <= 1:
    url = "https://drive.google.com/drive/folders/1LM71ZoPUV5xaPHRCPHqd0aDGg8Gyyk31"
    gdown.download_folder(url, quiet=False, use_cookies=False)


st.title("Mercari Price Suggestion")  # title

@st.cache(show_spinner=False, allow_output_mutation=True)
def load_model_and_featurizers():
    """
    Load trained model and data feturizer objects for prediction. 
    """
    data = {}

    data["final_features"] = joblib.load("featurizers/final_features.joblib")
    data["brand_counts"] = joblib.load("featurizers/brand_counts.joblib")
    data["grp_by_stats"] = joblib.load("featurizers/grp_by_stats.joblib")
    data["group_by_combos"] = joblib.load("featurizers/group_by_combos.joblib")

    # Featurizers
    data["target_encoder"] = joblib.load("featurizers/target_encoder.joblib")
    data["one_hot_encoder"] = joblib.load("featurizers/one_hot_encoder.joblib")
    data["ctv_text"] = joblib.load("featurizers/ctv_text.joblib")
    data["tfv_text"] = joblib.load("featurizers/tfv_text.joblib")
    data["ctv_name_and_brand"] = joblib.load("featurizers/ctv_name_and_brand.joblib")
    data["tfv_name_and_brand"] = joblib.load("featurizers/tfv_name_and_brand.joblib")

    # feature scalers
    data["min_max_scaler_1"] = joblib.load("featurizers/min_max_scaler_1.joblib")
    data["min_max_scaler_2"] = joblib.load("featurizers/min_max_scaler_2.joblib")

    # models
    data["mlp_ds1"] = keras.models.load_model("models/mlp_ds1")
    data["mlp_ds2"] = keras.models.load_model("models/mlp_ds2")
    data["meta_lr"] = joblib.load("models/meta_lr.pkl")

    return data


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading necessary stuff...')
data = load_model_and_featurizers()
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading necessary stuff...done!')


st.header("Product Details: ")

col1, col2 = st.columns(2, gap="medium")
with col1:
    item_name = st.text_input("Name: ")
with col2:
    item_brand = st.text_input("Brand: ")

col3, col4, col5 = st.columns(3, gap="medium")
item_main_cats = data["one_hot_encoder"].categories[0]
item_sub_cat_1s = data["one_hot_encoder"].categories[1]
item_sub_cat_2s = data["one_hot_encoder"].categories[2]
if 'no_main_cat' in item_main_cats:
    item_main_cats.remove('no_main_cat')
if 'no_subcat_1' in item_sub_cat_1s:
    item_sub_cat_1s.remove('no_subcat_1')
if 'no_subcat_2' in item_sub_cat_2s:
    item_sub_cat_2s.remove('no_subcat_2')
with col3:
    item_main_cat = st.selectbox("Main Category", item_main_cats)
with col4:
    item_sub_cat_1 = st.selectbox("Sub-Category-1", item_sub_cat_1s)
with col5:
    item_sub_cat_2 = st.selectbox("Sub-Category-2", item_sub_cat_2s)

col6, col7 = st.columns(2, gap="medium")
item_condition_options = ["New", "Like New", "Good", "Fair", "Poor"]
item_shipping_options = ['paid by buyer', 'paid by seller']
with col6:
    item_shipping = st.radio("Shipping", item_shipping_options, horizontal=True)
with col7:
    item_condition_id = st.select_slider(
        'Item Condition: ', options=item_condition_options)

item_description = st.text_area('Product Description',)

submit_btn = st.button('Submit')
if submit_btn:
    df = pd.DataFrame({
        "name": [str(item_name)],
        "item_condition_id": [int(item_condition_options.index(
            str(item_condition_id)))],
        "category_name": [
            str(item_main_cat)+"/"+str(item_sub_cat_1)+"/"+str(item_sub_cat_2)],
        "brand_name": [str(item_brand)],
        "shipping": [int(item_shipping_options.index(str(item_shipping)))],
        "item_description": [str(item_description)]
    })
    pred_df = final_predict_prices(df)
    st.subheader("Price Suggestion: $ "+str(pred_df["price"].values[0].round(3)))
