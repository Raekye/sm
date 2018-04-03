#!/usr/bin/env bash

case "$1" in
	's')
		export FLASK_APP=hayate.py
		export FLASK_DEBUG=1
		flask run
		;;
	'w')
		celery -A hayate.celery worker
		;;
	*)
		echo 'Hmmmm.'
		exit 1
		;;
esac
