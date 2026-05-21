🎬 Movie Matcher: AI-Powered Recommendation System
Movie Matcher is a content-based recommendation engine that suggests films based on their "DNA"—a combination of genres, plot summaries, lead actors, and directors. It transforms text data into mathematical vectors to find the closest matches to your favorite movies.

🎯 Project Overview
Most recommendation systems either use Collaborative Filtering (User-to-User) or Content-Based Filtering. This project focuses on the latter, making it perfect for discovering movies similar in theme and style to what you already love.

🚀 Key Features
Intelligent Tagging: Merges multiple data points into a single "searchable" fingerprint.

NLTK Stemming: Simplifies words (e.g., "Action" and "Actions" become the same) to improve accuracy.

Cosine Similarity: Uses vector geometry to calculate "closeness" between 4,800+ movies.

Robust Search: Handles case-insensitive inputs and provides error handling for missing titles.

🧠 The "Under the Hood" Logic
The system follows a 4-step Machine Learning pipeline:

Data Cleaning: We extract the Director and top 3 actors from the TMDB 5,000 Dataset.

Text Processing: We remove spaces from names (e.g., Johnny Depp becomes JohnnyDepp) so the AI doesn't confuse different people with the same first name.

Vectorization: We convert the text into a 5,000-dimensional space using CountVectorizer.

Similarity Matrix: We calculate the cosine distance between every movie vector.

🛠️ Installation & Setup
1. Clone the Repository
Bash
git clone https://github.com/VyomM005/Movie-Recommender-System.git
cd Movie-Recommender-System
2. Create Virtual Environment
Bash
python -m venv venv
# Activate on Windows:
.\venv\Scripts\activate
# Activate on Mac/Linux:
source venv/bin/activate
3. Install Dependencies
Bash
pip install pandas scikit-learn nltk
4. Add the Data
Download the TMDB 5000 Movie Dataset and place the .csv files in the root folder.

🖥️ Usage
Run the interactive script:

Bash
python main.py
Example:

Input: The Avengers

Output: > 1. Avengers: Age of Ultron

2. Iron Man 3

3. Captain America: Civil War

4. Thor: The Dark World

5. Ant-Man

📁 Project Structure
main.py: The full pipeline from data cleaning to user interface.

movie_dict.pkl: Pre-processed movie data (serialized).

similarity.pkl: The calculated similarity scores (serialized).

.gitignore: Prevents large datasets and cache files from cluttering your GitHub.
