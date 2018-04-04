#!/usr/bin/env python3

import sys
import os
import subprocess
import uuid
import shutil
from collections import deque

from flask import Flask
from flask import request
from flask import render_template
from flask import redirect
from flask import url_for
from flask import flash
from flask import send_from_directory

import werkzeug.utils

from celery import Celery

import youtube_dl as ytdl

if os.environ.get('foo') == 'bar':
	import level1

history = deque()

def make_celery(app):
	celery = Celery(app.import_name, backend=app.config['CELERY_RESULT_BACKEND'], broker=app.config['CELERY_BROKER_URL'])
	celery.conf.update(app.config)
	TaskBase = celery.Task
	class ContextTask(TaskBase):
		abstract = True
		def __call__(self, *args, **kwargs):
			with app.app_context():
				return TaskBase.__call__(self, *args, **kwargs)
	celery.Task = ContextTask
	return celery

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads/'
app.secret_key = 'OMEGALUL'
app.config.update(
#	CELERY_BROKER_URL='redis://192.168.56.1:6379',
#	CELERY_RESULT_BACKEND='redis://192.168.56.1:6379'
	CELERY_BROKER_URL='redis://127.0.0.1:6379',
	CELERY_RESULT_BACKEND='redis://127.0.0.1:6379'
)
celery = make_celery(app)

@celery.task()
def transcribe_youtube(url, x):
	try:
		transcription_status_update(x, 'downloading')
		ret = youtube_dl(url, x)
		if not ret:
			transcription_status_update(x, 'error')
			return
		transcription_status_update(x, 'transcribing')
		midi_transcribe(x)
		history.append(x)
		if len(history) > 8:
			history.popleft()
		transcription_status_update(x, 'done')
	except Exception as e:
		transcription_status_update(x, 'error')
		raise e

@celery.task()
def transcribe_file(p, x):
	try:
		transcription_status_update(x, 'loading')
		ret = wave_convert(p, x)
		transcription_status_update(x, 'transcribing')
		midi_transcribe(x)
		transcription_status_update(x, 'done')
	except Exception as e:
		transcription_status_update(x, 'error')
		raise e

def file_path(x, ext):
	return os.path.join(app.config['UPLOAD_FOLDER'], str(x) + '.' + ext)

def transcription_status_update(x, status):
	with open(file_path(x, 'txt'), 'w') as f:
		f.write(status)

def transcription_status(x):
	try:
		with open(file_path(x, 'txt')) as f:
			return f.readline().strip()
	except OSError:
		pass
	return None

def youtube_write_title(x, title):
	with open(file_path(x, 'youtube.txt'), 'w') as f:
		f.write(title)

def youtube_title(x):
	try:
		with open(file_path(x, 'youtube.txt')) as f:
			return f.readline().strip()
	except OSError:
		pass
	return None

def midi_transcribe(x):
	p = file_path(x, 'wav')
	level1.predict(p)

def youtube_dl_hook(x, d):
	d['percent'] = x.get('_percent_str')
	d['filename'] = x.get('filename')

def youtube_dl(url, x):
	data = {}
	opts = {
		'outtmpl': file_path('%(title)s', '%(ext)s'),
		'format': 'bestaudio/best',
		'postprocessors': [{
			'key': 'FFmpegExtractAudio',
			'preferredcodec': 'wav',
		}],
		'progress_hooks': [ lambda x: youtube_dl_hook(x, data) ],
	}
	with ytdl.YoutubeDL(opts) as yt:
		ret = yt.download([ url ])
		s = data['filename']
		pos = s.rindex('.')
		title = s[s.rindex('/') + 1:pos]
		p = s[:pos] + '.wav'
		youtube_write_title(x, title)
		shutil.move(p, file_path(x, 'wav'))
		if ret == 0:
			return True
	return False

def wave_convert(f, x):
	tmp = f[:-3] + 'tmp.' + f[-3:]
	shutil.copyfile(f, tmp)
	dst = file_path(x, 'wav')
	cmd = [
		'ffmpeg',
		'-y',
		'-i', tmp,
		dst,
	]
	p = subprocess.Popen(cmd)
	try:
		p.wait(64)
	except subprocess.TimeoutExpired:
		return False
	if p.returncode != 0:
		return False
	return True

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/about/')
def about():
	return render_template('about.html')

@app.route('/upload/', methods=[ 'POST' ])
def upload():
	if ('url' in request.form) and len(request.form['url'].strip()) > 0:
		url = request.form['url']
		x = uuid.uuid4()
		transcribe_youtube.delay(url, x)
		transcription_status_update(x, 'queued')
		return redirect(url_for('download', x=str(x)))
	elif ('wav' in request.files):
		f = request.files['wav']
		if f.filename == '':
			flash('No selected file.')
			return redirect(url_for('index'))
		p = werkzeug.utils.secure_filename(f.filename)
		ext = 'dat' if (len(p) < 3) else p[-3:]
		x = uuid.uuid4()
		p = file_path(x, ext)
		f.save(p)
		transcribe_file.delay(p, x)
		transcription_status_update(x, 'queued')
		return redirect(url_for('download', x=str(x)))

	flash('Invalid request.')
	return redirect(url_for('index'))

@app.route('/download/<x>/')
def download(x):
	status = transcription_status(x)
	title = youtube_title(x)
	return render_template('download.html', x=x, status=status, file_title='' if title is None else title
		, history=list(map(lambda x: (x, youutbe_title(x), history))))

@app.route('/download/<x>/midi')
def download_midi(x):
	p = file_path(x, 'mid')
	if os.path.isfile(p):
		return send_from_directory('.', p, as_attachment=True)
	flash('Invalid request.')
	return redirect(url_for('index'))

@app.route('/download/<x>/wav')
def download_wav(x):
	p = file_path(x, 'grand.wav')
	if os.path.isfile(p):
		return send_from_directory('.', p, as_attachment=True)
	flash('Invalid request.')
	return redirect(url_for('index'))

def main(args):
	print('hmmm')
	wav = youtube_dl('https://www.youtube.com/watch?v=ibJhcheHdyE', 'foo')
	print(wav)

if __name__ == '__main__':
	main(sys.argv[1:])
