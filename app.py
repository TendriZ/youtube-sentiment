# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import pandas as pd
# from scipy.sparse import csr_matrix, lil_matrix
# from collections import Counter
# import re
# import io
# import time
# from datetime import datetime
# import requests

# app = Flask(__name__)
# CORS(app)

# # Configuration for Vercel Serverless
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# def allowed_file(filename):
#     ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # Simplified YouTube Scraping (tanpa yt-dlp karena tidak supported di Vercel)
# class YouTubeScraper:
#     def __init__(self, api_key=None):
#         self.api_key = api_key
#         self.base_url = "https://www.googleapis.com/youtube/v3"
        
#     def extract_video_id(self, url):
#         patterns = [
#             r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
#             r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
#         ]
        
#         for pattern in patterns:
#             match = re.search(pattern, url)
#             if match:
#                 return match.group(1)
#         return None
    
#     def get_comments_api(self, video_id, max_comments=1000):
#         if not self.api_key:
#             return [], "YouTube API key diperlukan untuk scraping di Vercel. Dapatkan gratis di Google Cloud Console."
        
#         comments = []
#         next_page_token = None
        
#         while len(comments) < max_comments:
#             try:
#                 url = f"{self.base_url}/commentThreads"
#                 params = {
#                     'part': 'snippet',
#                     'videoId': video_id,
#                     'key': self.api_key,
#                     'maxResults': min(100, max_comments - len(comments)),
#                     'order': 'relevance'
#                 }
                
#                 if next_page_token:
#                     params['pageToken'] = next_page_token
                
#                 response = requests.get(url, params=params, timeout=10)
                
#                 if response.status_code == 403:
#                     return [], "API quota exceeded atau API key tidak valid"
#                 elif response.status_code == 404:
#                     return [], "Video tidak ditemukan atau komentar dinonaktifkan"
#                 elif response.status_code != 200:
#                     return [], f"YouTube API Error: {response.status_code}"
                
#                 data = response.json()
                
#                 for item in data.get('items', []):
#                     comment_data = item['snippet']['topLevelComment']['snippet']
#                     comments.append({
#                         'comment': comment_data.get('textDisplay', ''),
#                         'author': comment_data.get('authorDisplayName', ''),
#                         'like_count': comment_data.get('likeCount', 0),
#                         'published_at': comment_data.get('publishedAt', '')
#                     })
                
#                 next_page_token = data.get('nextPageToken')
#                 if not next_page_token:
#                     break
                    
#                 time.sleep(0.1)  # Rate limiting
                
#             except requests.RequestException as e:
#                 return [], f"Network error: {str(e)}"
#             except Exception as e:
#                 return [], f"Error fetching comments: {str(e)}"
        
#         return comments, None

# # Sentiment analysis functions (optimized untuk serverless)
# def preprocess(text):
#     if pd.isna(text) or not text:
#         return []
    
#     text = str(text).lower()
#     # Remove URLs, mentions, hashtags
#     text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
#     text = re.sub(r'@\w+', '', text)
#     text = re.sub(r'#\w+', '', text)
#     text = re.sub(r'[^\w\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
    
#     words = [word for word in text.split() if len(word) > 2]
#     return words

# def calculate_tfidf(processed_comments, max_features=2000, min_df=2):
#     """Optimized TF-IDF for serverless environment"""
#     if not processed_comments:
#         return csr_matrix((0, 0)), {}
    
#     word_freq = Counter()
#     for doc in processed_comments:
#         if doc:
#             word_freq.update(set(doc))
    
#     valid_words = [word for word, freq in word_freq.items() if freq >= min_df]
    
#     if len(valid_words) > max_features:
#         valid_words = [word for word, freq in word_freq.most_common(max_features)]
    
#     if not valid_words:
#         return csr_matrix((len(processed_comments), 0)), {}
    
#     word_to_idx = {word: i for i, word in enumerate(valid_words)}
    
#     N = len(processed_comments)
#     tfidf = lil_matrix((N, len(valid_words)), dtype=np.float32)
    
#     for i, doc in enumerate(processed_comments):
#         if not doc:
#             continue
                
#         word_count = Counter(doc)
#         doc_length = len(doc)
        
#         for word, count in word_count.items():
#             if word not in word_to_idx:
#                 continue
                
#             tf = count / doc_length
#             df = word_freq[word]
#             idf = np.log(N / (1 + df))
#             tfidf[i, word_to_idx[word]] = tf * idf
    
#     return tfidf.tocsr(), word_to_idx

# def kmeans(X, k=3, max_iter=30, random_state=42):
#     """Optimized K-means for serverless"""
#     np.random.seed(random_state)
    
#     if X.shape[0] == 0 or X.shape[1] == 0:
#         return np.array([]), np.array([])
    
#     if X.shape[0] < k:
#         k = max(1, X.shape[0])
    
#     if hasattr(X, 'toarray'):
#         X_dense = X.toarray().astype(np.float32)
#     else:
#         X_dense = X.astype(np.float32)
    
#     # Initialize centroids
#     initial_centroids = X_dense[np.random.choice(X_dense.shape[0], k, replace=False)]
#     centroids = initial_centroids
    
#     for iteration in range(max_iter):
#         # Calculate distances
#         distances = np.linalg.norm(X_dense[:, np.newaxis] - centroids, axis=2)
#         labels = np.argmin(distances, axis=1)
        
#         # Update centroids
#         new_centroids = []
#         for i in range(k):
#             mask = labels == i
#             if np.any(mask):
#                 new_centroids.append(X_dense[mask].mean(axis=0))
#             else:
#                 new_centroids.append(centroids[i])
        
#         new_centroids = np.array(new_centroids)
        
#         if np.allclose(centroids, new_centroids, atol=1e-4):
#             break
            
#         centroids = new_centroids
    
#     return labels, centroids

# def label_sentiments(centroids, word_to_idx):
#     """Indonesian sentiment labeling"""
#     neg_words = {
#         'hancur', 'bakar', 'bubar', 'korup', 'marah', 'sengsara', 'rusak', 
#         'anarkis', 'jahat', 'bodoh', 'tolol', 'benci', 'kecewa', 'buruk', 
#         'jelek', 'gagal', 'sedih', 'stress', 'mampus', 'parah', 'anjing'
#     }
    
#     pos_words = {
#         'semangat', 'dukung', 'mantap', 'hebat', 'merdeka', 'bersatu', 
#         'lindungi', 'bagus', 'baik', 'senang', 'bangga', 'optimis',
#         'sukses', 'berhasil', 'luar', 'biasa', 'keren', 'amazing', 'love'
#     }
    
#     cluster_labels = []
    
#     for i, centroid in enumerate(centroids):
#         if len(centroid) == 0:
#             cluster_labels.append('Netral')
#             continue
            
#         neg_score = sum(centroid[word_to_idx[w]] for w in neg_words if w in word_to_idx)
#         pos_score = sum(centroid[word_to_idx[w]] for w in pos_words if w in word_to_idx)
        
#         if neg_score > pos_score and neg_score > 0:
#             cluster_labels.append('Negatif')
#         elif pos_score > neg_score and pos_score > 0:
#             cluster_labels.append('Positif')
#         else:
#             cluster_labels.append('Netral')
    
#     return cluster_labels

# def analyze_sentiment_backend(comments, k_clusters=3):
#     """Main sentiment analysis function"""
#     if not comments:
#         return None, "Tidak ada komentar untuk dianalisis"
    
#     processed_comments = [preprocess(c) for c in comments]
    
#     valid_indices = [i for i, doc in enumerate(processed_comments) if len(doc) > 0]
#     if not valid_indices:
#         return None, "Tidak ada komentar valid setelah preprocessing"
    
#     comments = [comments[i] for i in valid_indices]
#     processed_comments = [processed_comments[i] for i in valid_indices]
    
#     tfidf_matrix, word_to_idx = calculate_tfidf(processed_comments)
    
#     if tfidf_matrix.shape[1] == 0:
#         return None, "Tidak dapat mengekstrak fitur dari komentar"
    
#     labels, centroids = kmeans(tfidf_matrix, k=k_clusters)
#     cluster_sentiments = label_sentiments(centroids, word_to_idx)
#     sentiments = [cluster_sentiments[label] for label in labels]
    
#     df_result = pd.DataFrame({
#         'Comment': comments,
#         'Sentiment': sentiments,
#         'Cluster': labels
#     })
    
#     return df_result, None

# def load_comments_from_file_stream(file_obj):
#     """Load comments from uploaded Excel file"""
#     try:
#         file_obj.seek(0)
        
#         # Try reading with different engines
#         try:
#             df = pd.read_excel(file_obj, engine='openpyxl')
#         except:
#             try:
#                 df = pd.read_excel(file_obj, engine='xlrd')
#             except:
#                 return [], "Format file Excel tidak didukung"
        
#         if df.empty:
#             return [], "File Excel kosong"
        
#         comment_columns = ['comment', 'Comment', 'AuthorComment', 'text', 'Text', 'Komentar', 'comments']
#         comment_col = None
        
#         for col in comment_columns:
#             if col in df.columns:
#                 comment_col = col
#                 break
        
#         if comment_col is None:
#             available_cols = ', '.join(df.columns.tolist())
#             return [], f"Kolom komentar tidak ditemukan. Kolom yang tersedia: {available_cols}. Gunakan salah satu nama: {', '.join(comment_columns)}"
        
#         comments = df[comment_col].dropna().astype(str).tolist()
        
#         if not comments:
#             return [], f"Tidak ada data di kolom '{comment_col}'"
        
#         return comments, None
        
#     except Exception as e:
#         return [], f"Error membaca file: {str(e)}"

# # Routes (with /api prefix)
# @app.route('/api/scrape', methods=['POST'])
# def scrape_youtube():
#     """Endpoint untuk scraping YouTube comments"""
#     try:
#         data = request.get_json()
#         if not data:
#             return jsonify({'error': 'Invalid JSON data'}), 400
            
#         video_url = data.get('video_url', '').strip()
#         max_comments = min(int(data.get('max_comments', 500)), 1000)  # Limit for serverless
#         api_key = data.get('api_key', '').strip()
        
#         if not video_url:
#             return jsonify({'error': 'URL video YouTube diperlukan'}), 400
        
#         if not api_key:
#             return jsonify({
#                 'error': 'YouTube API key diperlukan untuk scraping di Vercel Serverless', 
#                 'info': 'Dapatkan API key gratis di Google Cloud Console > YouTube Data API v3',
#                 'tutorial': 'https://developers.google.com/youtube/v3/getting-started'
#             }), 400
        
#         scraper = YouTubeScraper(api_key=api_key)
        
#         video_id = scraper.extract_video_id(video_url)
#         if not video_id:
#             return jsonify({'error': 'URL YouTube tidak valid'}), 400
        
#         print(f"Scraping video ID: {video_id}")
        
#         comments_data, error = scraper.get_comments_api(video_id, max_comments)
        
#         if error:
#             return jsonify({'error': error}), 400
        
#         if not comments_data:
#             return jsonify({'error': 'Tidak dapat mengambil komentar dari video. Pastikan video publik dan komentar tidak dinonaktifkan.'}), 400
        
#         comments = [item['comment'] for item in comments_data if item.get('comment')]
        
#         if len(comments) < 3:
#             return jsonify({'error': 'Terlalu sedikit komentar untuk dianalisis (minimal 3)'}), 400
        
#         df_result, error = analyze_sentiment_backend(comments)
        
#         if error:
#             return jsonify({'error': error}), 400
        
#         sentiment_counts = df_result['Sentiment'].value_counts()
#         cluster_counts = df_result['Cluster'].value_counts()
        
#         sample_data = df_result.head(50).to_dict('records')
        
#         response_data = {
#             'totalComments': len(df_result),
#             'videoId': video_id,
#             'scrapedAt': datetime.now().isoformat(),
#             'sentiments': sentiment_counts.to_dict(),
#             'clusters': {f'Cluster {k}': v for k, v in cluster_counts.to_dict().items()},
#             'sampleData': [
#                 {
#                     'comment': row['Comment'][:100] + '...' if len(row['Comment']) > 100 else row['Comment'],
#                     'sentiment': row['Sentiment'],
#                     'cluster': row['Cluster']
#                 }
#                 for row in sample_data
#             ]
#         }
        
#         return jsonify(response_data)
        
#     except Exception as e:
#         print(f"Scrape error: {e}")
#         return jsonify({'error': f'Server error: {str(e)}'}), 500

# @app.route('/api/upload', methods=['POST'])
# def upload_file():
#     """Endpoint untuk upload dan analisis file Excel"""
#     try:
#         if 'file' not in request.files:
#             return jsonify({'error': 'Tidak ada file yang diupload'}), 400
        
#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
        
#         if not file or not allowed_file(file.filename):
#             return jsonify({'error': 'Format file tidak didukung. Gunakan file .xlsx atau .xls'}), 400
        
#         # Load comments from file
#         comments, error = load_comments_from_file_stream(file)
        
#         if error:
#             return jsonify({'error': error}), 400
        
#         if not comments:
#             return jsonify({'error': 'Tidak dapat membaca komentar dari file'}), 400
        
#         if len(comments) < 3:
#             return jsonify({'error': f'Terlalu sedikit komentar untuk dianalisis ({len(comments)} komentar, minimal 3)'}), 400
        
#         # Perform sentiment analysis
#         df_result, error = analyze_sentiment_backend(comments)
        
#         if error:
#             return jsonify({'error': error}), 400
        
#         sentiment_counts = df_result['Sentiment'].value_counts()
#         cluster_counts = df_result['Cluster'].value_counts()
        
#         sample_data = df_result.head(50).to_dict('records')
        
#         response_data = {
#             'totalComments': len(df_result),
#             'sentiments': sentiment_counts.to_dict(),
#             'clusters': {f'Cluster {k}': v for k, v in cluster_counts.to_dict().items()},
#             'sampleData': [
#                 {
#                     'comment': row['Comment'][:100] + '...' if len(row['Comment']) > 100 else row['Comment'],
#                     'sentiment': row['Sentiment'],
#                     'cluster': row['Cluster']
#                 }
#                 for row in sample_data
#             ]
#         }
        
#         return jsonify(response_data)
        
#     except Exception as e:
#         print(f"Upload error: {e}")
#         return jsonify({'error': f'Server error: {str(e)}'}), 500

# @app.route('/api/health')
# def health_check():
#     """Health check endpoint"""
#     return jsonify({
#         'status': 'healthy',
#         'message': 'YouTube Sentiment Analysis API is running on Vercel',
#         'timestamp': datetime.now().isoformat(),
#         'version': '1.0.0',
#         'endpoints': {
#             'scrape': 'POST /api/scrape - YouTube comment scraping',
#             'upload': 'POST /api/upload - Excel file analysis', 
#             'health': 'GET /api/health - Health check'
#         }
#     })

# if __name__ == '__main__':
#     # For local development
#     app.run(debug=True, host='0.0.0.0', port=5000)
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Basic Flask API is running',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)