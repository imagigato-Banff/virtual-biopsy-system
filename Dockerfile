FROM rocker/shiny-verse:latest

RUN R -e "install.packages(c('shiny','plotly','DT','dplyr','tidyr','tibble','purrr','stringr','caret','caretEnsemble','randomForest','gbm','xgboost','nnet','MASS'), repos='https://cloud.r-project.org', dependencies=TRUE)"

WORKDIR /srv/shiny-server

COPY . /srv/shiny-server

ENV PORT=10000

EXPOSE 10000

CMD ["R", "-e", "shiny::runApp('/srv/shiny-server/app.R', host = '0.0.0.0', port = 10000)"]