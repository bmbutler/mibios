# The Microbiome data project

A database for microbiome data sets.

## Deployment

### Requirements

* Apache web server with wsgi, remote authentication 
* python3 (to set up venv)
* git (to install from local copy of repoisitory)

### Steps

1. Get a git repo clone on server
2. Copy upgrade script to deployment directory, e.g. `/var/www/mibios`
3. run `cd /var/www/mibios && ./upgrade` to set up venv and install mibios and dependencies
4. set up apache webserver, e.g. include something like

    WSGIScriptAlias /data /var/www/mibios/venv/lib/python3.7/site-packages/mibios-0.0.1-py3.7.egg/mibios/ops/wsgi.py
    WSGIDaemonProcess mibios user=www-data group=www-data threads=25 python-home=/var/www/mibios/venv home=/var/www/mibios

  in your apache conf file.
5. Ensure deployment directory and database file has read/write permissions for the web server's system user
