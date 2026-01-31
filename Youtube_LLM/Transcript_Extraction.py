# Convert a Youtube video URL -> Clean text transcript
# Youtube already provides subtitles , so fetch the text transcript
# Store it in a python structure -> list of chunks with timestamps      also legal since we are using public captions
# Youtube-transcript-api    given a yt video id -> return transcript text
import os

from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi     # no api-key-required # gives access to yt-caption-data


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
    response = client.embeddings.create(             # sends chunk to the embedding model
        input = text,
        model = model
    )
    return response.data[0].embedding        # a vector(list of numbers) representing the chunk

# Generate embeddings for all chunks
chunk_embeddings = [get_embedding(chunk) for chunk in chunks]       # list of embeddings

# print("Number of embeddings created:", len(chunk_embeddings))
print("Dimension of first embedding:", len(chunk_embeddings[0]))
print("First 5 values of first embedding:", chunk_embeddings[0][:5])





        
