FROM debian:bookworm-slim
ARG DEBIAN_FRONTEND=noninteractive

USER root

RUN ln -sf /usr/share/zoneinfo/America/Detroit /etc/localtime

RUN apt-get update && apt-get install --assume-yes --no-install-recommends \
    curl \
    apache2 \
    libapache2-mod-uwsgi \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/apache2 \
             /var/run/apache2/lock

COPY start.sh /start.sh
COPY apache24.conf /etc/apache2/conf-available/glamr.conf

RUN    a2enmod proxy \
    && a2enmod proxy_uwsgi \
    && a2enconf glamr \
    && chmod +x /start.sh

RUN      sed -i 's/80/8080/' /etc/apache2/sites-enabled/000-default.conf \
      && sed -i 's/80/8080/' /etc/apache2/ports.conf \
      && sed -i 's/ErrorLog\s*.*/ErrorLog \/dev\/stderr/' /etc/apache2/apache2.conf \
      && echo "CustomLog /dev/stdout combined" >> /etc/apache2/apache2.conf \
      && sed -i 's/APACHE_LOCK_DIR=\/var\/lock\/apache2$SUFFIX/APACHE_LOCK_DIR=\/var\/run\/apache2\/lock\/apache2$SUFFIX/' /etc/apache2/envvars

USER 1001

EXPOSE 8080/tcp

CMD ["/bin/bash", "/start.sh"]
