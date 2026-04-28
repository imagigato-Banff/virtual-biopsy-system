FROM rocker/shiny-verse:latest

RUN R -e "install.packages(c( \
  'shiny', \
  'plotly', \
  'DT', \
  'dplyr', \
  'tidyr', \
  'tibble', \
  'purrr', \
  'stringr', \
  'caret', \
  'randomForest', \
  'gbm', \
  'xgboost', \
  'nnet', \
  'MASS', \
  'missForest', \
  'readxl' \
), repos='https://cloud.r-project.org')"

WORKDIR /srv/shiny-server

COPY . /srv/shiny-server

EXPOSE 10000

CMD R -e "shiny::runApp('/srv/shiny-server/app.R', host = '0.0.0.0', port = as.numeric(Sys.getenv('PORT', '10000')))"