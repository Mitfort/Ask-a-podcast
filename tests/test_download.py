"""
Test cases for the download_and_transcribe function.
"""

def test_download_and_transcribe():
    """
    Test the download_and_transcribe function with a sample YouTube URL.
    """
    from src.transcriber import download_and_transcribe
    import os

    # Sample YouTube URL (replace with a valid URL for actual testing)
    yt_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    # Call the function
    result = download_and_transcribe(yt_url)

    # Check if the result is a list of segments
    assert isinstance(result, list), "Result should be a list of segments."
    assert all(isinstance(segment, dict) for segment in result), "Each segment should be a dictionary."
    assert os.path.exists('downloads/audio.mp3'), "Audio file should exist after download."