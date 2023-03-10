from pydub import AudioSegment
from pyannote.audio import Pipeline
from pydub import AudioSegment
import re 
import gc
import torch
import webvtt
import os
from datetime import timedelta

audio_path = 'F:\\Videos\\audio2.wav'
newAudio = AudioSegment.from_wav(audio_path)

spacermilli = 0
spacer = AudioSegment.silent(duration=spacermilli)
audio = spacer.append(newAudio, crossfade=0)

pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token='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')

DEMO_FILE = {'uri': 'blabal', 'audio': audio_path}
dz = pipeline(DEMO_FILE, num_speakers=7)  

with open("diarization.txt", "w") as text_file:
    text_file.write(str(dz)),

print(*list(dz.itertracks(yield_label = True))[:10], sep="\n")

def millisec(timeStr):
  spl = timeStr.split(":")
  s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
  return s

dz = open('diarization.txt').read().splitlines()
dzList = []
for l in dz:
  start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
  start = millisec(start) - spacermilli
  end = millisec(end)  - spacermilli
  lex = re.search('SPEAKER_\d\d', string=l).group()
  dzList.append([start, end, lex])

print(*dzList[:10], sep='\n')

spacer = AudioSegment.silent(duration=spacermilli)
sounds = spacer
segments = []

dz = open('diarization.txt').read().splitlines()
for l in dz:
  start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
  start = int(millisec(start)) #milliseconds
  end = int(millisec(end))  #milliseconds
  
  segments.append(len(sounds))
  sounds = sounds.append(audio[start:end], crossfade=0)
  sounds = sounds.append(spacer, crossfade=0)

sounds.export("dz.wav", format="wav") #Exports to a wav file in the current path.

print(segments[:8])


del   sounds, DEMO_FILE, pipeline, spacer,  audio, dz, newAudio

gc.collect()
torch.cuda.empty_cache()

os.system('whisper dz.wav --language ru --device cuda --model medium')

captions = [[(int)(millisec(caption.start)), (int)(millisec(caption.end)),  caption.text] for caption in webvtt.read('dz.wav.vtt')]
print(*captions[:8], sep='\n')

html = list()

for i in range(len(segments)):
  idx = 0
  for idx in range(len(captions)):
    if captions[idx][0] >= (segments[i] - spacermilli):
      break;
  
  while (idx < (len(captions))) and ((i == len(segments) - 1) or (captions[idx][1] < segments[i+1])):
    c = captions[idx]  
    
    start = dzList[i][0] + (c[0] -segments[i])

    if start < 0: 
      start = 0
    idx += 1

    html.append(f'{dzList[i][2]}: {c[2]}\n')
    
s = "".join(html)

with open("lexicap.html", "w") as text_file:
    text_file.write(s)
print(s)
