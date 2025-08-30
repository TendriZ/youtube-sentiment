from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from collections import Counter
import re
import io
import json
import time
import uuid
from werkzeug.utils import secure_filename
import os
import openpyxl
import requests
import threading
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# YouTube Scraping Functions
class YouTubeScraper:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
        
    def extract_video_id(self, url):
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def get_comments_api(self, video_id, max_comments=1000):
        if not self.api_key:
            raise ValueError("API key required")
        
        comments = []
        next_page_token = None
        
        while len(comments) < max_comments:
            try:
                url = f"{self.base_url}/commentThreads"
                params = {
                    'part': 'snippet',
                    'videoId': video_id,
                    'key': self.api_key,
                    'maxResults': min(100, max_comments - len(comments)),
                    'order': 'relevance'
                }
                
                if next_page_token:
                    params['pageToken'] = next_page_token
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 403:
                    return [], "API quota exceeded or invalid API key"
                elif response.status_code != 200:
                    return [], f"API Error: {response.status_code}"
                
                data = response.json()
                
                for item in data.get('items', []):
                    comment_data = item['snippet']['topLevelComment']['snippet']
                    comments.append({
                        'comment': comment_data.get('textDisplay', ''),
                        'author': comment_data.get('authorDisplayName', ''),
                        'like_count': comment_data.get('likeCount', 0),
                        'published_at': comment_data.get('publishedAt', '')
                    })
                
                next_page_token = data.get('nextPageToken')
                if not next_page_token:
                    break
                    
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                return [], f"Error fetching comments: {str(e)}"
        
        return comments, None

    def get_comments_yt_dlp(self, video_url, max_comments=1000):
        try:
            import yt_dlp
        except ImportError:
            return [], "yt-dlp not installed. Please install with: pip install yt-dlp"
        
        try:
            ydl_opts = {
                'getcomments': True,
                'extractor_args': {
                    'youtube': {
                        'max_comments': f'{max_comments},{max_comments},0,0'  # max top-level, max parents, no replies, no per-thread
                    }
                },
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                raw_comments = info.get('comments', [])
            
            # Map to standard format (only top-level)
            comments = []
            for c in raw_comments:
                published_at = ''
                if 'timestamp' in c and c['timestamp']:
                    try:
                        published_at = datetime.fromtimestamp(c['timestamp']).isoformat()
                    except:
                        pass
                elif 'time_text' in c:
                    published_at = c['time_text']
                
                comments.append({
                    'comment': c.get('text', ''),
                    'author': c.get('author', ''),
                    'like_count': c.get('like_count', 0),
                    'published_at': published_at
                })
            
            return comments, None
            
        except Exception as e:
            return [], f"yt-dlp error: {str(e)}"

# Sentiment analysis functions (sama seperti sebelumnya)
def preprocess(text):
    if pd.isna(text):
        return []
    
    text = str(text).lower()
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = [word for word in text.split() if len(word) > 2]
    return words

def calculate_tfidf(processed_comments, max_features=3000, min_df=2):
    word_freq = Counter()
    for doc in processed_comments:
        word_freq.update(set(doc))
    
    valid_words = [word for word, freq in word_freq.items() if freq >= min_df]
    
    if len(valid_words) > max_features:
        valid_words = [word for word, freq in word_freq.most_common(max_features)]
    
    word_to_idx = {word: i for i, word in enumerate(valid_words)}
    
    N = len(processed_comments)
    tfidf = lil_matrix((N, len(valid_words)), dtype=np.float32)
    
    for i, doc in enumerate(processed_comments):
        if not doc:
            continue
                
        word_count = Counter(doc)
        doc_length = len(doc)
        
        for word, count in word_count.items():
            if word not in word_to_idx:
                continue
                
            tf = count / doc_length
            df = word_freq[word]
            idf = np.log(N / (1 + df))
            tfidf[i, word_to_idx[word]] = tf * idf
    
    return tfidf.tocsr(), word_to_idx

def kmeans(X, k=3, max_iter=50, random_state=42):
    np.random.seed(random_state)
    
    if hasattr(X, 'toarray'):
        if X.shape[0] > 5000:
            sample_indices = np.random.choice(X.shape[0], min(1000, X.shape[0]), replace=False)
            X_sample = X[sample_indices].toarray()
            initial_centroids = X_sample[np.random.choice(X_sample.shape[0], k, replace=False)]
        else:
            X_dense = X.toarray()
            initial_centroids = X_dense[np.random.choice(X_dense.shape[0], k, replace=False)]
    else:
        initial_centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    centroids = initial_centroids.astype(np.float32)
    
    for iteration in range(max_iter):
        if hasattr(X, 'toarray'):
            distances = []
            batch_size = 1000
            for i in range(0, X.shape[0], batch_size):
                batch = X[i:i+batch_size].toarray().astype(np.float32)
                batch_distances = np.linalg.norm(batch[:, np.newaxis] - centroids, axis=2)
                distances.append(batch_distances)
            distances = np.vstack(distances)
        else:
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        
        labels = np.argmin(distances, axis=1)
        
        new_centroids = []
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                if hasattr(X, 'toarray'):
                    cluster_points = X[mask].toarray().astype(np.float32)
                else:
                    cluster_points = X[mask].astype(np.float32)
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                new_centroids.append(centroids[i])
        
        new_centroids = np.array(new_centroids)
        
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
            
        centroids = new_centroids
    
    return labels, centroids

def label_sentiments(centroids, word_to_idx):
    neg_words = {
        'hancur', 'bakar', 'bubar', 'korup', 'marah', 'sengsara', 'rusak', 
        'anarkis', 'jahat', 'bodoh', 'tolol', 'benci', 'kecewa', 'buruk', 
        'jelek', 'gagal', 'sedih', 'stress', 'mampus'
    }
    
    pos_words = {
        'semangat', 'dukung', 'mantap', 'hebat', 'merdeka', 'bersatu', 
        'lindungi', 'bagus', 'baik', 'senang', 'bangga', 'optimis',
        'sukses', 'berhasil', 'luar', 'biasa', 'keren', 'amazing'
    }
    
    cluster_labels = []
    
    for i, centroid in enumerate(centroids):
        neg_score = sum(centroid[word_to_idx[w]] for w in neg_words if w in word_to_idx)
        pos_score = sum(centroid[word_to_idx[w]] for w in pos_words if w in word_to_idx)
        
        if neg_score > pos_score and neg_score > 0:
            cluster_labels.append('Negatif')
        elif pos_score > neg_score and pos_score > 0:
            cluster_labels.append('Positif')
        else:
            cluster_labels.append('Netral')
    
    return cluster_labels

def analyze_sentiment_backend(comments, k_clusters=3):
    processed_comments = [preprocess(c) for c in comments]
    
    valid_indices = [i for i, doc in enumerate(processed_comments) if len(doc) > 0]
    comments = [comments[i] for i in valid_indices]
    processed_comments = [processed_comments[i] for i in valid_indices]
    
    if len(comments) == 0:
        return None, "Tidak ada komentar valid setelah preprocessing"
    
    tfidf_matrix, word_to_idx = calculate_tfidf(processed_comments)
    labels, centroids = kmeans(tfidf_matrix, k=k_clusters)
    cluster_sentiments = label_sentiments(centroids, word_to_idx)
    sentiments = [cluster_sentiments[label] for label in labels]
    
    df_result = pd.DataFrame({
        'Comment': comments,
        'Sentiment': sentiments,
        'Cluster': labels
    })
    
    return df_result, None

# File handling functions (yang sudah diperbaiki)
def load_comments_from_file_stream(file_obj):
    try:
        file_obj.seek(0)
        df = pd.read_excel(file_obj.stream)
        
        comment_columns = ['comment', 'Comment', 'AuthorComment', 'text', 'Text', 'Komentar']
        comment_col = None
        
        for col in comment_columns:
            if col in df.columns:
                comment_col = col
                break
        
        if comment_col is None:
            return []
        
        comments = df[comment_col].dropna().astype(str).tolist()
        return comments
        
    except Exception as e:
        print(f"Error loading from stream: {e}")
        return []

def cleanup_file_with_retry(file_path, max_retries=5):
    for attempt in range(max_retries):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except PermissionError:
            time.sleep(2)
        except Exception as e:
            print(f"Cleanup error: {e}")
            break
    return False

# Routes
@app.route('/')
def index():
    with open('index.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content

@app.route('/scrape', methods=['POST'])
def scrape_youtube():
    """Endpoint untuk scraping YouTube comments"""
    try:
        data = request.get_json()
        video_url = data.get('video_url', '')
        max_comments = data.get('max_comments', 500)
        api_key = data.get('api_key', '')
        method = data.get('method', 'yt-dlp')  # Default ke yt-dlp
        
        if not video_url:
            return jsonify({'error': 'URL video YouTube diperlukan'}), 400
        
        scraper = YouTubeScraper(api_key=api_key if api_key else None)
        
        # Extract video ID
        video_id = scraper.extract_video_id(video_url)
        if not video_id:
            return jsonify({'error': 'URL YouTube tidak valid'}), 400
        
        print(f"Scraping video ID: {video_id} with method: {method}")
        
        # Scrape based on method
        if method == 'api' and api_key:
            comments_data, error = scraper.get_comments_api(video_id, max_comments)
        elif method == 'yt-dlp':
            comments_data, error = scraper.get_comments_yt_dlp(video_url, max_comments)
        else:
            return jsonify({'error': 'Method tidak didukung atau API key hilang untuk method API'}), 400
        
        if error:
            return jsonify({'error': error}), 400
        
        if not comments_data:
            return jsonify({'error': 'Tidak dapat mengambil komentar dari video'}), 400
        
        # Extract comment text
        comments = [item['comment'] for item in comments_data if item.get('comment')]
        
        # Perform sentiment analysis
        df_result, error = analyze_sentiment_backend(comments)
        
        if error:
            return jsonify({'error': error}), 400
        
        # Prepare response data
        sentiment_counts = df_result['Sentiment'].value_counts()
        cluster_counts = df_result['Cluster'].value_counts()
        
        sample_data = df_result.head(50).to_dict('records')
        
        response_data = {
            'totalComments': len(df_result),
            'videoId': video_id,
            'scrapedAt': datetime.now().isoformat(),
            'sentiments': sentiment_counts.to_dict(),
            'clusters': {f'Cluster {k}': v for k, v in cluster_counts.to_dict().items()},
            'sampleData': [
                {
                    'comment': row['Comment'][:100] + '...' if len(row['Comment']) > 100 else row['Comment'],
                    'sentiment': row['Sentiment'],
                    'cluster': row['Cluster']
                }
                for row in sample_data
            ]
        }
        
        # Save scraped data to Excel
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scraped_comments_{video_id}_{timestamp}.xlsx"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            # Combine original data with sentiment results
            full_df = pd.DataFrame(comments_data)
            full_df = full_df.merge(
                df_result.reset_index().rename(columns={'index': 'comment_index'}),
                left_index=True,
                right_on='comment_index',
                how='left'
            )
            
            full_df.to_excel(filepath, index=False)
            response_data['savedFile'] = filename
            
        except Exception as e:
            print(f"Error saving scraped data: {e}")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Scrape error: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    filepath = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            unique_filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Try reading from stream first
            comments = load_comments_from_file_stream(file)
            
            if not comments:
                return jsonify({'error': 'Tidak dapat membaca komentar dari file'}), 400
            
            df_result, error = analyze_sentiment_backend(comments)
            
            if error:
                return jsonify({'error': error}), 400
            
            sentiment_counts = df_result['Sentiment'].value_counts()
            cluster_counts = df_result['Cluster'].value_counts()
            
            sample_data = df_result.head(50).to_dict('records')
            
            response_data = {
                'totalComments': len(df_result),
                'sentiments': sentiment_counts.to_dict(),
                'clusters': {f'Cluster {k}': v for k, v in cluster_counts.to_dict().items()},
                'sampleData': [
                    {
                        'comment': row['Comment'][:100] + '...' if len(row['Comment']) > 100 else row['Comment'],
                        'sentiment': row['Sentiment'],
                        'cluster': row['Cluster']
                    }
                    for row in sample_data
                ]
            }
            
            return jsonify(response_data)
        
        else:
            return jsonify({'error': 'File type not allowed'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    finally:
        if filepath and os.path.exists(filepath):
            cleanup_file_with_retry(filepath)

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("YouTube Sentiment Analysis Server")
    print("Required packages:")
    print("pip install flask flask-cors pandas numpy scipy openpyxl requests")
    print("\nOptional for advanced scraping:")
    print("pip install yt-dlp selenium")
    print("\nServer starting on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)