from pydub import AudioSegment
from pyannote.audio import Pipeline
from pydub import AudioSegment
import re 
import gc
import torch
import webvtt
import os
from datetime import timedelta
from fastapi import FastAPI
from pydantic import BaseModel
from pytube import YouTube
import time
import uuid

#{
#  "url": "https://www.youtube.com/watch?v=vQKci8SEE6Q",
#  "speakers": 2
#}

class Video(BaseModel):
    url: str
    speakers: int

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/describe_video/")
async def describe_video(item: Video):
    return ml(item)

def ml(item: Video):
  yt = YouTube(item.url)

  new_downloaded_name = str(uuid.uuid4())
  out_file = yt.streams.filter(only_audio=True).first().download('C:\\Users\\Eatric\\Downloads', filename=new_downloaded_name + ".mp4")
  base, ext = os.path.splitext(out_file)

  new_file = base + '.wav'
  print(new_file)
  time.sleep(5)
  os.system("ffmpeg -i " + out_file + " " + new_file)

  audio_path = new_file
  newAudio = AudioSegment.from_wav(audio_path)

  spacermilli = 0
  spacer = AudioSegment.silent(duration=spacermilli)
  audio = spacer.append(newAudio, crossfade=0)

  pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization', use_auth_token='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')

  DEMO_FILE = {'uri': 'blabal', 'audio': audio_path}
  dz = pipeline(DEMO_FILE, num_speakers=item.speakers)  

  with open('diarization-'+ new_downloaded_name + '.txt', "w") as text_file:
      text_file.write(str(dz)),

  def millisec(timeStr):
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
    return s

  dz = open('diarization-'+ new_downloaded_name + '.txt').read().splitlines()
  dzList = []
  for l in dz:
    start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
    start = millisec(start) - spacermilli
    end = millisec(end)  - spacermilli
    lex = re.search('SPEAKER_\d\d', string=l).group()
    dzList.append([start, end, lex])

  spacer = AudioSegment.silent(duration=spacermilli)
  sounds = spacer
  segments = []

  dz = open('diarization-'+ new_downloaded_name + '.txt').read().splitlines()
  for l in dz:
    start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
    start = int(millisec(start)) #milliseconds
    end = int(millisec(end))  #milliseconds
    
    segments.append(len(sounds))
    sounds = sounds.append(audio[start:end], crossfade=0)
    sounds = sounds.append(spacer, crossfade=0)

  sounds.export("dz" + new_downloaded_name + ".wav", format="wav") #Exports to a wav file in the current path.


  del   sounds, DEMO_FILE, pipeline, spacer,  audio, dz, newAudio

  gc.collect()
  torch.cuda.empty_cache()

  os.system('whisper dz' + new_downloaded_name + '.wav --language ru --device cuda --model medium')

  captions = [[(int)(millisec(caption.start)), (int)(millisec(caption.end)),  caption.text] for caption in webvtt.read('dz' + new_downloaded_name + '.wav.vtt')]

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

  return s
