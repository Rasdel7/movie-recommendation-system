import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))


movies = {
    'title': [
        'The Dark Knight', 'Inception', 'Interstellar', 'The Matrix',
        'Fight Club', 'Pulp Fiction', 'The Godfather', 'Goodfellas',
        'The Shawshank Redemption', 'Forrest Gump', 'The Lion King',
        'Toy Story', 'Avatar', 'Titanic', 'Jurassic Park',
        'The Avengers', 'Iron Man', 'Captain America', 'Thor',
        'Black Panther', 'Spider-Man', 'Doctor Strange',
        'The Silence of the Lambs', 'Se7en', 'Gone Girl',
        'Parasite', 'Spirited Away', 'Your Name', 'Akira',
        '3 Idiots', 'Dangal', 'PK', 'Sholay', 'Dil Chahta Hai'
    ],
    'genre': [
        'action crime thriller', 'action scifi thriller', 'scifi drama adventure',
        'action scifi thriller', 'drama thriller', 'crime drama thriller',
        'crime drama', 'crime drama', 'drama', 'drama romance',
        'animation adventure drama', 'animation adventure comedy',
        'action scifi adventure', 'romance drama', 'adventure scifi thriller',
        'action adventure scifi', 'action adventure scifi', 'action adventure',
        'action adventure fantasy', 'action adventure scifi',
        'action adventure', 'action adventure fantasy',
        'thriller crime horror', 'thriller crime drama', 'thriller drama mystery',
        'thriller drama', 'animation adventure fantasy', 'animation romance drama',
        'animation scifi action', 'comedy drama', 'drama sport',
        'comedy drama scifi', 'action adventure drama', 'comedy drama romance'
    ],
    'description': [
        'Batman fights Joker in Gotham crime dark superhero',
        'Dreams within dreams heist mind bending reality',
        'Space time travel black hole love science',
        'Virtual reality simulation hacker robots rebellion',
        'Identity split personality soap underground boxing',
        'Crime stories Los Angeles hitman nonlinear',
        'Mafia family power crime loyalty betrayal',
        'Mob gangster crime New York loyalty violence',
        'Prison friendship hope escape injustice',
        'Simple man extraordinary journey love history',
        'Pride kingdom animals Africa journey courage',
        'Toys friendship adventure imagination loyalty',
        'Alien world nature technology war ecology',
        'Ship sinking love story class difference ocean',
        'Dinosaurs island theme park chaos science',
        'Superheroes team earth save aliens battle',
        'Billionaire genius suit armor hero technology',
        'Soldier super serum war loyalty friendship',
        'God thunder hammer lightning Asgard earth',
        'King Africa technology vibranium identity pride',
        'Teen hero city crime spider powers school',
        'Surgeon magic spells multiverse time reality',
        'FBI serial killer cannibal psychological horror',
        'Detective sins murder psychological thriller dark',
        'Marriage deception media crime psychological',
        'Class inequality family crime thriller Korea',
        'Spirit world bath house girl journey magic',
        'Comet body swap love distance Japan anime',
        'Motorcycle gang future Neo Tokyo destruction',
        'Engineering college friendship pressure India comedy',
        'Wrestling daughters champion father India sports',
        'Alien satire religion India comedy',
        'Friendship revenge dacoit India classic action',
        'Friends road trip Goa India coming of age'
    ],
    'rating': [
        9.0, 8.8, 8.6, 8.7, 8.8, 8.9, 9.2, 8.7,
        9.3, 8.8, 8.5, 8.3, 7.9, 7.9, 8.1,
        8.0, 7.9, 7.8, 7.9, 7.3, 7.4, 7.5,
        8.6, 8.6, 8.1, 8.5, 8.6, 8.4, 8.0,
        8.4, 8.4, 8.1, 8.2, 8.1
    ]
}

df = pd.DataFrame(movies)
print(f"Movie database loaded! {len(df)} movies available.")


df['combined'] = df['genre'] + ' ' + df['description']


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("Similarity matrix built!")


def get_recommendations(movie_title, n=5):
    if movie_title not in df['title'].values:
        print(f"Movie '{movie_title}' not found in database.")
        return []

    idx = df[df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]

    recommendations = []
    for i, score in sim_scores:
        recommendations.append({
            'title':      df.iloc[i]['title'],
            'genre':      df.iloc[i]['genre'],
            'rating':     df.iloc[i]['rating'],
            'similarity': round(score * 100, 1)
        })
    return recommendations


test_movies = ['The Dark Knight', 'Inception', '3 Idiots', 'Parasite']

for movie in test_movies:
    print(f"\n🎬 Because you watched: {movie}")
    print("-" * 45)
    recs = get_recommendations(movie, n=5)
    for i, r in enumerate(recs, 1):
        print(f"  {i}. {r['title']:<30} ⭐{r['rating']}  ({r['similarity']}% match)")


plt.figure(figsize=(12, 6))
top_movies = df.nlargest(10, 'rating')
colors = plt.cm.RdYlGn(np.linspace(0.4, 1.0, len(top_movies)))
bars = plt.barh(top_movies['title'], top_movies['rating'],
                color=colors, edgecolor='black')
for bar, val in zip(bars, top_movies['rating']):
    plt.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
             f'{val}', va='center', fontsize=10)
plt.title('Top 10 Rated Movies in Database', fontsize=14)
plt.xlabel('IMDb Rating')
plt.xlim(8.0, 9.8)
plt.tight_layout()
plt.savefig('top_rated_movies.png')
print("\nTop rated chart saved!")


selected = ['The Dark Knight', 'Inception', 'The Avengers',
            'Parasite', '3 Idiots', 'Spirited Away', 'Pulp Fiction']
idx_list = [df[df['title'] == m].index[0] for m in selected]
sim_subset = cosine_sim[np.ix_(idx_list, idx_list)]

plt.figure(figsize=(8, 6))
sns.heatmap(sim_subset, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=selected, yticklabels=selected)
plt.title('Movie Similarity Heatmap', fontsize=13)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('similarity_heatmap.png')
print("Similarity heatmap saved!")

print("\nDone! Check your folder for both charts.")