FROM debian:bookworm-slim
ARG DEBIAN_FRONTEND=noninteractive 
RUN ln -sf /usr/share/zoneinfo/America/Detroit /etc/localtime
RUN apt-get update && apt-get install --assume-yes --no-install-recommends \
    git \
    procps \
    python3-biopython \
    python3-defusedxml \
    python3-django \
    python3-djangorestframework \
    python3-django-tables2 \
    python3-django-extensions \
    python3-matplotlib \
    python3-pandas \
    python3-pip \
    python3-psycopg2 \
    python3-xlrd \
    python3-zipstream \
    uwsgi \
    uwsgi-plugin-python3

# this is Debian's pip so this will install under /usr/local
# just "pip install" will install for python3.10, hence invoke via "python3.11 -m pip"
RUN python3.11 -m pip install --no-index --no-deps --python /usr/bin/python3.11 git+https://github.com/Geo-omics/mibios.git

RUN apt-get update && apt-get install --assume-yes --no-install-recommends \
    iputils-ping \
    bind9-dnsutils

COPY uwsgi.ini /etc/uwsgi.ini
RUN mkdir /glamr && chown www-data:www-data /glamr

USER www-data:www-data
WORKDIR /glamr

RUN mkdir static_var && chown www-data:www-data static_var
COPY settings.py ./settings.py
RUN manage_mibios collectstatic
VOLUME ["/glamr/static"]

EXPOSE 3032/tcp
# HEALTHCHECK
CMD ["uwsgi", "--ini", "/etc/uwsgi.ini"]
#CMD [ \
#  "uwsgi", \
#  "--plugins", "python3", \
#  "--uwsgi-socket", "0.0.0.0:3023", \
#  "--chdir", "/glamr", \
#]
