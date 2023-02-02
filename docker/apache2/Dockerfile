FROM debian:bookworm-slim
ARG DEBIAN_FRONTEND=noninteractive 
RUN ln -sf /usr/share/zoneinfo/America/Detroit /etc/localtime

RUN apt-get update && apt-get install --assume-yes --no-install-recommends \
    curl \
    apache2 \
    libapache2-mod-uwsgi

COPY apache24.conf /etc/apache2/conf-available/glamr.conf
RUN a2enmod proxy
RUN a2enmod proxy_uwsgi
RUN a2enconf glamr

ENV APACHE_RUN_USER www-data
ENV APACHE_RUN_GROUP www-data
ENV APACHE_LOG_DIR /var/log/apache2
ENV APACHE_RUN_DIR /run
ENV APACHE_PID_FILE /run/apache2.pid

# static content from webapp
VOLUME /glamr/static/

EXPOSE 80/tcp
HEALTHCHECK CMD curl --silent --fail http://localhost:80/server-status/ || exit 1

CMD ["apache2", "-k", "start", "-DFOREGROUND"]
