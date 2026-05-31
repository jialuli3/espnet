from google import genai
import os
import glob

client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))
minute=5
prompt = f"""You are given an audio recording.

Goal: Produce a speaker-attributed transcript.

Step 1 (ASR): Transcribe the audio with approximate timestamps.
Step 2 (Speaker attribution): Assign consistent speaker labels based on the transcript and timing cues.

Constraints:
- Do NOT invent content. Transcript text must be verbatim (as heard).
- Timestamps should be rounded to the nearest 0.1s when possible, but may be coarser if uncertain.
- Use labels "Speaker A", "Speaker B", ... and reuse them consistently.

Output Format:
[start_time, end_time] Speaker X: utterance
Example:
[0.0, 2.3] Speaker A: Hello, how are you?
[2.4, 4.1] Speaker B: I'm good, thank you!
"""

all_ami_wavs = sorted(glob.glob("/work/nvme/bbjs/jialuli3/test_wav/ami_wav_segments/EN2002b*.wav"))
output_prefix="/work/nvme/bbjs/jialuli3/espnet/egs2/ami/speechlm1/genai_test_output"
for wav_file in all_ami_wavs[1:]:
    print(wav_file)
    myfile = client.files.upload(file=wav_file)

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[prompt, myfile],
        config=genai.types.GenerateContentConfig(
            temperature=0.0,
            candidate_count=1,
            top_p=1.0,
            top_k=1,
            # optional: best-effort repeatability (still not guaranteed)
            # seed=1234,
        ),
    )
    f=open(f"{output_prefix}/{os.path.basename(wav_file)}_temp0.txt", "w")
    f.write(response.text)
    f.close()

# all_ami_wavs = sorted(glob.glob("/work/nvme/bbjs/jialuli3/test_wav/ami_wav_segments/EN2002b*.wav"))
# output_prefix="/work/nvme/bbjs/jialuli3/espnet/egs2/ami/speechlm1/genai_test_output"

# print(wav_file)
# myfile = client.files.upload(file=wav_file)

# response = client.models.generate_content(
#     model="gemini-3-flash-preview",
#     contents=[prompt, myfile],
#     config=genai.types.GenerateContentConfig(
#         temperature=0.0,
#         candidate_count=1,
#         top_p=1.0,
#         top_k=1,
#         # optional: best-effort repeatability (still not guaranteed)
#         # seed=1234,
#     ),
# )
# f=open(f"{output_prefix}/{os.path.basename(wav_file)}_temp0.txt", "w")
# f.write(response.text)
# f.close()
