#!/usr/bin/env python3

import sys
import os
import subprocess
import uuid

from flask import Flask
from flask import request
from flask import render_template
from flask import redirect
from flask import url_for
from flask import flash

import werkzeug.utils

import youtube_dl as ytdl

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads/'
app.secret_key = 'OMEGALUL'

def midi_transcribe(x):
	p = os.path.join(app.config['UPLOAD_FOLDER'], str(x) + '.wav')

def youtube_dl_hook(d):
	print('foo')
	print(d)

def youtube_dl(url):
	x = uuid.uuid4()
	opts = {
		'outtmpl': os.path.join(app.config['UPLOAD_FOLDER'], str(x) + '.%(ext)s'),
		'format': 'bestaudio/best',
		'postprocessors': [{
			'key': 'FFmpegExtractAudio',
			'preferredcodec': 'wav',
		}],
		'progress_hooks': [ youtube_dl_hook ],
	}
	with ytdl.YoutubeDL(opts) as yt:
		ret = yt.download([ url ])
		if ret == 0:
			return x
	return None

def wave_convert(f, x):
	dst = os.path.join(app.config['UPLOAD_FOLDER'], str(x) + '.wav')
	cmd = [
		'ffmpeg',
		'-y',
		'-i', f,
		dst,
	]
	p = subprocess.Popen(cmd)
	try:
		p.wait(4)
	except subprocess.TimeoutExpired:
		return None
	if p.returncode != 0:
		return None
	return dst

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/about/')
def about():
	return render_template('about.html')

@app.route('/upload/', methods=[ 'POST' ])
def upload():
	if ('wav' in request.files):
		f = request.files['wav']
		if f.filename == '':
			flash('No selected file.')
			return redirect(url_for('index'))
		p = werkzeug.utils.secure_filename(f.filename)
		ext = 'dat' if (len(p) < 3) else p[-3:]
		x = uuid.uuid4()
		p = os.path.join(app.config['UPLOAD_FOLDER'], str(x) + '.' + ext)
		f.save(p)
		wav = wave_convert(p, x)
		if not (wav is None):
			midi = midi_transcribe(x)
			return redirect(url_for('download', x=str(x)))
	elif ('url' in request.form) and len(request.form['url'].strip()) > 0:
		yt_url = request.form['url']
		x = youtube_dl(yt_url)
		if not (x is None):
			midi = midi_transcribe(x)
			return redirect(url_for('download', x=str(x)))

	flash('Invalid request.')
	return redirect(url_for('index'))

@app.route('/download/<x>/')
def download(x):
	return render_template('download.html', x=x)

def main(args):
	print('hmmm')
	wav = youtube_dl('https://www.youtube.com/watch?v=ibJhcheHdyE')
	print(wav)
	midi = midi_transcribe(wav)

if __name__ == '__main__':
	main(sys.argv[1:])
