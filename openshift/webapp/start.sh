#!/bin/bash

echo "Downloading static files...\n"
manage_mibios collectstatic

echo "Starting application...\n"
uwsgi --ini /etc/uwsgi.ini