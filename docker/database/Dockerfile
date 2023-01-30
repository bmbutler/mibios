FROM debian:bookworm-slim
ARG POSTGRES_BASE=/var/local/postgresql
ARG POSTGRES_UID=121
ARG POSTGRES_GID=130
ARG DEBIAN_FRONTEND=noninteractive 

RUN ln -sf /usr/share/zoneinfo/America/Detroit /etc/localtime

# locale setup
RUN apt-get update && apt-get install --assume-yes --no-install-recommends \
    locales \
    adduser
RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && locale-gen && update-locale
ENV LANG=en_US.utf8

# assign postgreq user/group corresponding to external storage
RUN addgroup --gid $POSTGRES_GID postgres
RUN adduser --system --home=$POSTGRES_BASE --no-create-home --uid $POSTGRES_UID --ingroup postgres postgres

RUN apt-get update && apt-get install --assume-yes --no-install-recommends \
    postgresql-15

# replace data directory, normally that is /var/lib/postgres/15/main, which
# already has a initialized DB from package installation, we leave that in
# place and just substitute a new data directory, but leave the rest of the
# configuration alone.  The new place needs to have something mounted via
# docker-compose's volumes directive or docker-run's --volume option that
# contains a ready-to-use database or have initdb run manually.
ENV PGDATA=$POSTGRES_BASE/data
USER postgres
RUN echo "data_directory = '$PGDATA'" >> /etc/postgresql/15/main/conf.d/local.conf
RUN echo "listen_addresses = '0.0.0.0'" >> /etc/postgresql/15/main/conf.d/local.conf
RUN echo "host all all  samenet trust" >> /etc/postgresql/15/main/pg_hba.conf
EXPOSE 5432/tcp
HEALTHCHECK CMD ["pg_isready"]
STOPSIGNAL SIGINT
CMD ["/usr/lib/postgresql/15/bin/postgres", "--config-file=/etc/postgresql/15/main/postgresql.conf"]
