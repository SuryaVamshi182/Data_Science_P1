# Convert a Youtube video URL -> Clean text transcript
# Youtube already provides subtitles , so fetch the text transcript
# Store it in a python structure -> list of chunks with timestamps      also legal since we are using public captions
# Youtube-transcript-api    given a yt video id -> return transcript text
import os
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract video-id from the url
def extract_video_id(url):
    return url.split("v=")[-1].split("&")[0]        # v= preceeds the video id

# Fetch transcript
def get_transcript(video_id):                  # return a list of dictionaries which contians text,start,duration
    yt = YouTubeTranscriptApi()                 # create an instance
    transcript = yt.fetch(video_id)             # call fetch on the instance
    return transcript

# Test the url
if __name__=="__main__":
    url = "https://www.youtube.com/watch?v=ULvplwBTbQk"
    video_id = extract_video_id(url)
    transcript_segments = get_transcript(video_id)
    
    print("Total transcript segments:", len(transcript_segments))
    print("\n First 3 segments:\n")
    
    for segment in transcript_segments[:3]:
        print(segment)

# Now we want one string or a list of strings to generate embeddings
# Combine all trnascript segments to a single string
full_text = " ".join([segment.text for segment in transcript_segments]) #extracts text from each segment and combines to one large string

print("Total words in transcript:", len(full_text.split()))     #len -> gives idea of how long the text is (number of words)
print("First 500 characters:\n")
print(full_text[:500])                  # prints the first 500 characters to check

# Since LLMS have token limit 4,000 for gpt 3.5, we split text into manageable chunks
# Chunking function
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into chunks of 'chunk_size' words with 'overlap' words overlapping 
    """
    
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
        
    return chunks

# Apply chunking
chunks = chunk_text(full_text, chunk_size=500, overlap=50)      #produces a list of strings, each string is a chunk
print("Number of chunks created:", len(chunks))
print("\n First chunk preview:\n")
print(chunks[0][:500])

# Embeddings -> numerical representations of text  -- allow llm to search for relavant parts of the transcript
# openai's text-embedding-3-small or text-embedding-3-large
# Generate Embeddings
def get_embedding(text, model="text-embedding-3-small"):
    return embedding_model.encode(text)         # a vector(list of numbers) representing the chunk

# Generate embeddings for all chunks
chunk_embeddings = embedding_model.encode(chunks)       # NumPy array

dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

# print("Number of embeddings created:", len(chunk_embeddings))
print("Total chunks indexed:", index.ntotal)

# Retrieval function
def retrieve_chunks(question, k=3):
    question_embedding = embedding_model.encode([question])
    distances, indices = index.search(question_embedding, k)
    return [chunks[i] for i in indices[0]]

# Test this function
question = "How does India look from space?"
retrieved = retrieve_chunks(question)

print("Retrieved chunks:")
for i, c in enumerate(retrieved, 1):
    print(f"\nChunk {i}:\n{c[:200]}")








        
