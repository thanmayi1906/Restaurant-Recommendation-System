from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from difflib import get_close_matches

app = Flask(__name__)

# Load model data
try:
    with open('restaurant.pkl', 'rb') as file:
        model_data = pickle.load(file)

    zomato_df = model_data['zomato_df']
    vectorizer = model_data['tfidf']
    tfidf_matrix = model_data['tfidf_matrix']
    indices = model_data['indices']

    nn = NearestNeighbors(metric='cosine', algorithm='brute')
    nn.fit(tfidf_matrix)

    print("✅ Model data loaded successfully!")
    print(f"Total restaurants: {len(zomato_df)}")

except Exception as e:
    print(f"❌ Error loading model data: {str(e)}")
    raise

def find_restaurant_indices(search_term):
    all_names = list(indices.keys())

    exact_matches = [name for name in all_names if search_term.lower() == name.lower()]
    if exact_matches:
        return [indices[exact_matches[0]]], exact_matches[0]

    contains_matches = [name for name in all_names if search_term.lower() in name.lower()]
    if contains_matches:
        return [indices[contains_matches[0]]], contains_matches[0]

    fuzzy_matches = get_close_matches(search_term.lower(), [name.lower() for name in all_names], n=1, cutoff=0.6)
    if fuzzy_matches:
        matched_name = [name for name in all_names if name.lower() == fuzzy_matches[0]][0]
        return [indices[matched_name]], matched_name

    return None, None

def recommend(restaurant_name, top_n=10, df=zomato_df, scenario='User'):
    try:
        df_subset = df[df['rate'] > 1.5].copy() if len(df) > 10000 else df.copy()

        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df_subset['cuisines'].fillna(''))

        nn = NearestNeighbors(n_neighbors=top_n + 1, algorithm='brute', metric='cosine')
        nn.fit(tfidf_matrix)

        indices = pd.Series(df_subset.index, index=df_subset['name']).drop_duplicates()
        idx = indices.get(restaurant_name)

        if idx is None:
            suggestions = df[df['name'].str.contains(restaurant_name.split()[0], case=False)]['name'].unique()[:5]
            return None, suggestions.tolist() if hasattr(suggestions, 'tolist') else list(suggestions)

        distances, neighbor_indices = nn.kneighbors(tfidf_matrix[idx])
        neighbor_indices = neighbor_indices[0][1:min(top_n + 1, len(df_subset))]

        columns = ['name', 'cuisines', 'rate', 'cost']
        if scenario == 'Owner':
            columns += ['reviews_list', 'location']

        recommendations = df_subset.iloc[neighbor_indices][columns]

        rename_map = {
            'name': 'Restaurant',
            'cuisines': 'Cuisines',
            'rate': 'Mean Rating',
            'cost': 'Cost'
        }
        if 'reviews_list' in recommendations.columns:
            rename_map['reviews_list'] = 'Reviews'
        if 'location' in recommendations.columns:
            rename_map['location'] = 'Location'

        recommendations = recommendations.rename(columns=rename_map)

        if len(recommendations) > 0:
            return recommendations.to_dict('records'), None
        else:
            return None, None

    except Exception as e:
        print(f"⚠ Error in recommend function: {str(e)}")
        return None, None

@app.route('/')
def home():
    try:
        restaurant_list = list(indices.keys())
        return render_template('index.html', restaurant_list=restaurant_list[:200])
    except Exception as e:
        return render_template('web.html', error_message=str(e)), 500

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    try:
        restaurant_name = request.form.get('restaurant', '').strip()
        scenario = request.form.get('scenario', 'User')

        if not restaurant_name:
            return render_template('web.html',
                                   restaurant_name='',
                                   scenario=scenario,
                                   recommendations=None,
                                   suggestions=None,
                                   error="Please enter a restaurant name")

        recommendations, suggestions = recommend(restaurant_name, scenario=scenario)

        if recommendations:
            return render_template('web.html',restaurant_name=restaurant_name,scenario=scenario,
                                   recommendations=recommendations,suggestions=None,error=None)

        elif suggestions:
            return render_template('web.html',
                                   restaurant_name=restaurant_name,
                                   scenario=scenario,
                                   recommendations=None,
                                   suggestions=suggestions,
                                   error="Showing similar restaurants:")

        else:
            return render_template('web.html',
                                   restaurant_name=restaurant_name,
                                   scenario=scenario,
                                   recommendations=None,
                                   suggestions=None,
                                   error=f"No results found for '{restaurant_name}'")

    except Exception as e:
        print(f"❌ Error in recommendation route: {str(e)}")
        return render_template('web.html',
                               restaurant_name=request.form.get('restaurant', ''),
                               scenario=request.form.get('scenario', 'User'),
                               recommendations=None,
                               suggestions=None,
                               error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
