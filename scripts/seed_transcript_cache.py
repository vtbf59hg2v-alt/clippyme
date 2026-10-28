"""Put an existing transcript in the cache for a video file.

A job started from an uploaded file reuses a cached transcript when the same
file is uploaded again. Jobs that ran before that cache existed left their
transcript only in the job's checkpoint; this copies it into the cache so a
re-cut with other settings (e.g. Clip style) skips transcription:

    docker compose exec backend python scripts/seed_transcript_cache.py \\
        <video file> <job dir>/.clippyme_checkpoint/transcript.json
"""
import json
import sys

from clippyme.pipeline.transcription.transcribe_cache import (
    file_cache_key,
    save_transcript_cache,
)


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    video, transcript_path = argv[1], argv[2]
    key = file_cache_key(video)
    if key is None:
        print(f"❌ Can't read {video}")
        return 1
    with open(transcript_path, encoding="utf-8") as f:
        transcript = json.load(f)
    if not isinstance(transcript.get("segments"), list) or not transcript["segments"]:
        print(f"❌ {transcript_path} has no transcript segments")
        return 1
    save_transcript_cache(key, transcript)
    print(f"✅ {len(transcript['segments'])} segments cached for {video}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
