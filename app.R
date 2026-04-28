library(shiny)
library(plotly)
library(DT)
library(dplyr)
library(tidyr)
library(tibble)
library(purrr)
library(stringr)
library(caret)
library(caretEnsemble)
library(randomForest)
library(gbm)
library(xgboost)
library(nnet)
library(MASS)

options(shiny.maxRequestSize = 500 * 1024^2)

model_dir <- "models"
model_paths <- list(
  cv   = file.path(model_dir, "cv_finalround_list_forSynapse.rds"),
  ah   = file.path(model_dir, "ah_finalround_list_forSynapse.rds"),
  IFTA = file.path(model_dir, "IFTA_finalround_list_forSynapse.rds"),
  glo  = file.path(model_dir, "Glo_finalround_list_forSynapse.rds")
)

models_available <- all(file.exists(unlist(model_paths)))
loaded_models <- NULL
if (models_available) {
  loaded_models <- list(
    cv   = readRDS(model_paths$cv),
    ah   = readRDS(model_paths$ah),
    IFTA = readRDS(model_paths$IFTA),
    glo  = readRDS(model_paths$glo)
  )
}

collect_train_models <- function(x) {
  out <- list()
  walk <- function(obj, path = "model") {
    if (is.null(obj)) return(NULL)
    if (inherits(obj, "train")) {
      out[[path]] <<- obj
      return(NULL)
    }
    if (is.list(obj) && !inherits(obj, "data.frame")) {
      nms <- names(obj)
      if (is.null(nms)) nms <- paste0("item", seq_along(obj))
      for (i in seq_along(obj)) {
        nm <- nms[[i]]
        if (is.na(nm) || nm == "") nm <- paste0("item", i)
        walk(obj[[i]], paste0(path, "/", nm))
      }
    }
    NULL
  }
  walk(x)
  out
}

make_patient_values <- function(input) {
  deceased <- input$donor_type == "Fallecido"
  gender <- ifelse(input$sex == "Mujer", 1, 0)
  donor_type <- ifelse(deceased, 1, 0)
  vascular_death <- ifelse(deceased && input$vascular_death == "Sí", 1, 0)
  dcd <- ifelse(deceased && input$dcd == "Sí", 1, 0)
  hypertension <- ifelse(input$hypertension == "Sí", 1, 0)
  diabetes <- ifelse(input$diabetes == "Sí", 1, 0)
  hcv <- ifelse(input$hcv == "Sí", 1, 0)
  proteinuria <- ifelse(input$proteinuria == "Sí", 1, 0)

  list(
    Age = as.numeric(input$age),
    Gender = gender,
    Gender1 = gender,
    Sex = gender,
    Sex1 = gender,
    Donor_type = donor_type,
    Donor_type1 = donor_type,
    Hypertension = hypertension,
    Hypertension1 = hypertension,
    Diabetes = diabetes,
    Diabetes1 = diabetes,
    Creatinine = as.numeric(input$creatinine),
    Proteinuria = proteinuria,
    Proteinuria1 = proteinuria,
    HCV_status = hcv,
    HCV_status1 = hcv,
    DCD = dcd,
    DCD1 = dcd,
    bmi = as.numeric(input$bmi),
    BMI = as.numeric(input$bmi),
    vascular_death = vascular_death,
    vascular_death1 = vascular_death
  )
}

resolve_value <- function(var_name, patient_values) {
  if (var_name %in% names(patient_values)) return(patient_values[[var_name]])
  if (grepl("1$", var_name)) {
    base_name <- sub("1$", "", var_name)
    if (base_name %in% names(patient_values)) return(patient_values[[base_name]])
  }
  alt_name <- paste0(var_name, "1")
  if (alt_name %in% names(patient_values)) return(patient_values[[alt_name]])
  NA
}

choose_factor_level <- function(var_name, value01, lvls) {
  lvls_chr <- as.character(lvls)
  lvls_low <- tolower(lvls_chr)
  value01 <- as.integer(value01)

  if (length(lvls_chr) == 0) return(as.character(value01))

  if (var_name %in% c("Gender", "Gender1", "Sex", "Sex1")) {
    if (value01 == 1) {
      idx <- grep("female|woman|mujer|femenino|^f$|^1$", lvls_low)
      if (length(idx) > 0) return(lvls_chr[idx[1]])
    } else {
      idx <- grep("male|man|hombre|masculino|^m$|^0$", lvls_low)
      if (length(idx) > 0) return(lvls_chr[idx[1]])
    }
  }

  if (var_name %in% c("Donor_type", "Donor_type1")) {
    if (value01 == 1) {
      idx <- grep("deceased|dead|cadaver|fallecido|^1$", lvls_low)
      if (length(idx) > 0) return(lvls_chr[idx[1]])
    } else {
      idx <- grep("living|live|vivo|^0$", lvls_low)
      if (length(idx) > 0) return(lvls_chr[idx[1]])
    }
  }

  if (value01 == 1) {
    idx <- grep("yes|si|sí|true|positive|pos|^1$", lvls_low)
    if (length(idx) > 0) return(lvls_chr[idx[1]])
  } else {
    idx <- grep("no|false|negative|neg|^0$", lvls_low)
    if (length(idx) > 0) return(lvls_chr[idx[1]])
  }

  if (length(lvls_chr) == 2) return(ifelse(value01 == 1, lvls_chr[2], lvls_chr[1]))
  lvls_chr[1]
}

expected_variables_from_model <- function(train_model) {
  dc <- NULL
  if (!is.null(train_model$terms)) {
    dc <- attr(train_model$terms, "dataClasses")
  }
  if (!is.null(dc)) {
    dc <- dc[names(dc) != ".outcome"]
    dc <- dc[names(dc) != "(response)"]
    return(dc)
  }
  if (!is.null(train_model$trainingData) && is.data.frame(train_model$trainingData)) {
    td <- train_model$trainingData
    td <- td[, setdiff(names(td), ".outcome"), drop = FALSE]
    dc <- vapply(td, function(z) {
      if (is.factor(z)) "factor" else if (is.integer(z)) "integer" else if (is.numeric(z)) "numeric" else if (is.logical(z)) "logical" else if (is.character(z)) "character" else class(z)[1]
    }, character(1))
    return(dc)
  }
  c(Age = "numeric", Gender = "numeric", Donor_type = "numeric", Hypertension = "numeric", Diabetes = "numeric", Creatinine = "numeric", Proteinuria = "numeric", HCV_status = "numeric", DCD = "numeric", bmi = "numeric", vascular_death = "numeric")
}

build_newdata_for_train <- function(train_model, patient_values) {
  expected <- expected_variables_from_model(train_model)
  predictor_names <- names(expected)
  nd <- vector("list", length(predictor_names))
  names(nd) <- predictor_names

  for (nm in predictor_names) {
    cls <- unname(expected[[nm]])
    raw_value <- resolve_value(nm, patient_values)

    if (cls %in% c("factor", "ordered")) {
      lvls <- NULL
      if (!is.null(train_model$xlevels) && nm %in% names(train_model$xlevels)) lvls <- train_model$xlevels[[nm]]
      if (is.null(lvls) && !is.null(train_model$trainingData) && nm %in% names(train_model$trainingData) && is.factor(train_model$trainingData[[nm]])) lvls <- levels(train_model$trainingData[[nm]])
      if (is.null(lvls)) lvls <- c("0", "1")
      selected <- choose_factor_level(nm, raw_value, lvls)
      nd[[nm]] <- if (cls == "ordered") ordered(selected, levels = lvls) else factor(selected, levels = lvls)
    } else if (cls %in% c("numeric", "double", "nmatrix.1", "integer")) {
      nd[[nm]] <- as.numeric(raw_value)
    } else if (cls == "logical") {
      nd[[nm]] <- as.logical(as.integer(raw_value))
    } else if (cls == "character") {
      nd[[nm]] <- as.character(raw_value)
    } else {
      nd[[nm]] <- as.numeric(raw_value)
    }
  }

  as.data.frame(nd, check.names = FALSE)
}

clean_probability_output <- function(pred) {
  p <- as.data.frame(pred, check.names = FALSE)
  if (nrow(p) < 1) stop("Predicción vacía.")

  names(p) <- gsub("^X", "", names(p))
  names(p) <- gsub("[^0-9]", "", names(p))

  for (k in as.character(0:3)) {
    if (!k %in% names(p)) p[[k]] <- 0
  }

  p <- p[, as.character(0:3), drop = FALSE]
  p <- as.data.frame(lapply(p, as.numeric), check.names = FALSE)
  p[is.na(p)] <- 0
  rs <- rowSums(p)
  if (any(rs <= 0)) p[,] <- 0.25 else p <- p / rs
  names(p) <- paste0("X", 0:3)
  p[1, , drop = FALSE]
}

predict_probability_ensemble <- function(model_container, patient_values) {
  train_models <- collect_train_models(model_container)
  if (length(train_models) == 0 && inherits(model_container, "train")) train_models <- list(model_container)
  if (length(train_models) == 0) stop("No se encontraron modelos internos de tipo caret::train.")

  prob_list <- list()
  errors <- character()

  for (model_name in names(train_models)) {
    m <- train_models[[model_name]]
    nd <- build_newdata_for_train(m, patient_values)
    pred <- tryCatch(
      predict(m, newdata = nd, type = "prob"),
      error = function(e) {
        errors <<- c(errors, paste0(model_name, ": ", conditionMessage(e)))
        NULL
      }
    )
    if (!is.null(pred)) {
      cp <- tryCatch(clean_probability_output(pred), error = function(e) NULL)
      if (!is.null(cp)) prob_list[[length(prob_list) + 1]] <- cp
    }
  }

  if (length(prob_list) == 0) {
    stop(paste0("No se pudieron calcular probabilidades con los modelos internos. Primeros errores: ", paste(head(errors, 4), collapse = " | ")))
  }

  arr <- simplify2array(lapply(prob_list, as.matrix))
  mean_prob <- apply(arr, c(1, 2), mean)
  clean_probability_output(as.data.frame(mean_prob, check.names = FALSE))
}

predict_regression_ensemble <- function(model_container, patient_values) {
  train_models <- collect_train_models(model_container)
  if (length(train_models) == 0 && inherits(model_container, "train")) train_models <- list(model_container)
  if (length(train_models) == 0) stop("No se encontraron modelos internos de tipo caret::train.")

  values <- numeric(0)
  errors <- character()

  for (model_name in names(train_models)) {
    m <- train_models[[model_name]]
    nd <- build_newdata_for_train(m, patient_values)
    pred <- tryCatch(
      predict(m, newdata = nd),
      error = function(e) {
        errors <<- c(errors, paste0(model_name, ": ", conditionMessage(e)))
        NULL
      }
    )
    if (!is.null(pred)) {
      val <- suppressWarnings(as.numeric(pred[1]))
      if (!is.na(val)) values <- c(values, val)
    }
  }

  if (length(values) == 0) stop(paste0("No se pudo calcular la glomeruloesclerosis. Primeros errores: ", paste(head(errors, 4), collapse = " | ")))
  max(0, min(100, mean(values, na.rm = TRUE)))
}

predicted_class <- function(prob_row) which.max(as.numeric(prob_row[1, paste0("X", 0:3)])) - 1
prob_ge2 <- function(prob_row) as.numeric(prob_row$X2 + prob_row$X3)
lesion_name <- function(code) switch(code, cv = "Arteriosclerosis (cv)", ah = "Hialinosis arteriolar (ah)", IFTA = "Fibrosis intersticial y atrofia tubular (IFTA)")
grade_label <- function(x) unname(c("0" = "0 - ausente", "1" = "1 - leve", "2" = "2 - moderada", "3" = "3 - grave")[as.character(x)])

clinical_interpretation <- function(probs, glo) {
  txt <- character(0)
  for (nm in names(probs)) {
    p <- probs[[nm]]
    cls <- predicted_class(p)
    psev <- prob_ge2(p)
    if (psev >= 0.50 || cls >= 2) {
      txt <- c(txt, paste0(lesion_name(nm), ": alta probabilidad de lesión moderada o grave (grado ≥2: ", round(100 * psev, 1), "%). Considerar seguimiento estrecho y contextualización con la evolución clínica y biopsias posteriores."))
    } else if (psev >= 0.25) {
      txt <- c(txt, paste0(lesion_name(nm), ": probabilidad intermedia de lesión moderada o grave (grado ≥2: ", round(100 * psev, 1), "%). Interpretar junto con edad, comorbilidad, función renal del donante y contexto de asignación."))
    } else {
      txt <- c(txt, paste0(lesion_name(nm), ": predominan grados ausente o leve. Clase predicha: ", grade_label(cls), "."))
    }
  }
  if (glo >= 20) txt <- c(txt, paste0("Glomeruloesclerosis estimada elevada: ", round(glo, 1), "%. Sugiere mayor carga crónica virtual y requiere valoración clínica cuidadosa."))
  else if (glo >= 10) txt <- c(txt, paste0("Glomeruloesclerosis estimada intermedia: ", round(glo, 1), "%. Puede justificar vigilancia adicional según el contexto del donante y receptor."))
  else txt <- c(txt, paste0("Glomeruloesclerosis estimada baja: ", round(glo, 1), "% ."))
  paste(txt, collapse = "\n\n")
}

ui <- fluidPage(
  tags$head(tags$style(HTML("
      body { background-color: #f6f8fb; color: #1f2937; }
      .main-title { background: linear-gradient(135deg, #19324a, #295c7a); color: white; padding: 22px; border-radius: 16px; margin-bottom: 20px; }
      .main-title h1 { margin-top: 0; font-weight: 700; }
      .panel-card { background: white; border-radius: 16px; padding: 18px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-bottom: 16px; }
      .warning-card { background: #fff3cd; border-left: 6px solid #f0ad4e; padding: 14px; border-radius: 10px; margin-bottom: 16px; }
      .ok-card { background: #eaf7ee; border-left: 6px solid #2e8b57; padding: 14px; border-radius: 10px; margin-bottom: 16px; }
      .result-number { font-size: 30px; font-weight: 700; color: #19324a; }
      .small-muted { color: #6b7280; font-size: 13px; }
      h3 { font-weight: 700; color: #19324a; }
  "))),
  div(class = "main-title", h1("The Virtual Biopsy System"), h4("Biopsia virtual día cero para trasplante renal"), p("Predicción de lesiones Banff y porcentaje de glomeruloesclerosis a partir de parámetros clínicos del donante.")),
  if (!models_available) {
    div(class = "warning-card", h4("Modelos no encontrados"), p("La aplicación necesita estos cuatro archivos dentro de la carpeta models/:"), tags$ul(tags$li("cv_finalround_list_forSynapse.rds"), tags$li("ah_finalround_list_forSynapse.rds"), tags$li("IFTA_finalround_list_forSynapse.rds"), tags$li("Glo_finalround_list_forSynapse.rds")))
  } else {
    div(class = "ok-card", strong("Modelos cargados correctamente. "), span("La aplicación está lista para calcular predicciones."))
  },
  sidebarLayout(
    sidebarPanel(width = 4,
      div(class = "panel-card",
        h3("Datos del donante"),
        numericInput("age", "Edad del donante (años)", value = 57, min = 0, max = 100, step = 1),
        selectInput("sex", "Sexo", choices = c("Mujer", "Hombre"), selected = "Mujer"),
        selectInput("donor_type", "Tipo de donante", choices = c("Vivo", "Fallecido"), selected = "Fallecido"),
        conditionalPanel(condition = "input.donor_type == 'Fallecido'", selectInput("vascular_death", "Causa de muerte cerebrovascular", choices = c("No", "Sí"), selected = "Sí"), selectInput("dcd", "Causa de muerte circulatoria / DCD", choices = c("No", "Sí"), selected = "No")),
        selectInput("hypertension", "Hipertensión", choices = c("No", "Sí"), selected = "No"),
        selectInput("diabetes", "Diabetes mellitus", choices = c("No", "Sí"), selected = "No"),
        selectInput("hcv", "Estado VHC", choices = c("No", "Sí"), selected = "No"),
        numericInput("bmi", "Índice de masa corporal / BMI (kg/m²)", value = 20, min = 10, max = 70, step = 0.1),
        numericInput("creatinine", "Creatinina sérica más baja (mg/dL)", value = 0.6, min = 0.1, max = 15, step = 0.1),
        selectInput("proteinuria", "Proteinuria", choices = c("No", "Sí"), selected = "No"),
        p(class = "small-muted", "Proteinuria positiva: tira reactiva ≥1 o UPCR ≥0.5 g/g."),
        actionButton("calculate", "Calcular biopsia virtual", class = "btn-primary", width = "100%")
      )
    ),
    mainPanel(width = 8,
      tabsetPanel(
        tabPanel("Resultados", br(), div(class = "panel-card", h3("Resumen de predicción"), uiOutput("summary_cards")), div(class = "panel-card", h3("Probabilidades por lesión y grado Banff"), DTOutput("probability_table")), div(class = "panel-card", h3("Glomeruloesclerosis"), htmlOutput("glo_output"))),
        tabPanel("Gráfico radar", br(), div(class = "panel-card", h3("Radar de probabilidades"), plotlyOutput("radar_plot", height = "560px"))),
        tabPanel("Nota clínica", br(), div(class = "panel-card", h3("Interpretación clínica automática"), verbatimTextOutput("clinical_note"))),
        tabPanel("Ayuda", br(), div(class = "panel-card", h3("Qué calcula esta aplicación"), p("La aplicación estima, a partir de 11 variables del donante, las probabilidades de cada grado Banff para cv, ah e IFTA. También estima el porcentaje continuo de glomeruloesclerosis."), h3("Advertencia"), p("Esta herramienta reproduce un sistema predictivo de investigación. No sustituye la valoración clínica, histológica ni la decisión del equipo de trasplante.")))
      )
    )
  )
)

server <- function(input, output, session) {
  prediction_result <- eventReactive(input$calculate, {
    validate(need(models_available, "Faltan los archivos .rds en la carpeta models/."))
    patient_values <- make_patient_values(input)
    probs <- list(
      cv = predict_probability_ensemble(loaded_models$cv, patient_values),
      ah = predict_probability_ensemble(loaded_models$ah, patient_values),
      IFTA = predict_probability_ensemble(loaded_models$IFTA, patient_values)
    )
    glo <- predict_regression_ensemble(loaded_models$glo, patient_values)
    list(patient_values = patient_values, probs = probs, glo = glo)
  }, ignoreInit = FALSE)

  output$summary_cards <- renderUI({
    res <- prediction_result()
    tagList(lapply(names(res$probs), function(nm) {
      p <- res$probs[[nm]]
      cls <- predicted_class(p)
      psev <- prob_ge2(p)
      div(style = "border-bottom: 1px solid #e5e7eb; padding: 12px 0;", h4(lesion_name(nm)), tags$p(tags$strong("Clase predicha: "), grade_label(cls)), tags$p(tags$strong("Probabilidad de grado ≥2: "), paste0(round(100 * psev, 1), "%")))
    }))
  })

  output$probability_table <- renderDT({
    res <- prediction_result()
    tab <- imap_dfr(res$probs, function(p, nm) {
      tibble(Lesión = lesion_name(nm), `Grado 0 (%)` = round(100 * p$X0, 1), `Grado 1 (%)` = round(100 * p$X1, 1), `Grado 2 (%)` = round(100 * p$X2, 1), `Grado 3 (%)` = round(100 * p$X3, 1), `Clase predicha` = grade_label(predicted_class(p)), `Probabilidad grado ≥2 (%)` = round(100 * prob_ge2(p), 1))
    })
    datatable(tab, rownames = FALSE, options = list(dom = "t", pageLength = 3))
  })

  output$glo_output <- renderUI({
    res <- prediction_result()
    HTML(paste0("<div class='result-number'>", round(res$glo, 2), "%</div><p>Porcentaje estimado de glomeruloesclerosis.</p>"))
  })

  output$clinical_note <- renderText({
    res <- prediction_result()
    clinical_interpretation(res$probs, res$glo)
  })

  output$radar_plot <- renderPlotly({
    res <- prediction_result()
    radar_data <- imap_dfr(res$probs, function(p, nm) {
      tibble(lesion = lesion_name(nm), grado = factor(c("Grado 0", "Grado 1", "Grado 2", "Grado 3"), levels = c("Grado 0", "Grado 1", "Grado 2", "Grado 3")), prob = as.numeric(p[1, paste0("X", 0:3)]))
    })
    plot_ly(type = "scatterpolar", mode = "lines+markers") %>%
      add_trace(data = radar_data[radar_data$lesion == "Arteriosclerosis (cv)", ], r = ~prob, theta = ~grado, name = "cv", fill = "toself") %>%
      add_trace(data = radar_data[radar_data$lesion == "Hialinosis arteriolar (ah)", ], r = ~prob, theta = ~grado, name = "ah", fill = "toself") %>%
      add_trace(data = radar_data[radar_data$lesion == "Fibrosis intersticial y atrofia tubular (IFTA)", ], r = ~prob, theta = ~grado, name = "IFTA", fill = "toself") %>%
      layout(polar = list(radialaxis = list(visible = TRUE, range = c(0, 1), tickformat = ".0%")), showlegend = TRUE, title = "Probabilidades predichas por grado Banff")
  })
}

shinyApp(ui = ui, server = server)
