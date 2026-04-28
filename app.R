library(shiny)
library(plotly)
library(DT)
library(dplyr)
library(tibble)
library(purrr)
library(caret)
library(caretEnsemble)
library(randomForest)
library(gbm)
library(xgboost)
library(nnet)
library(MASS)

options(shiny.maxRequestSize = 500 * 1024^2)

model_paths <- list(
  cv   = file.path("models", "cv_finalround_list_forSynapse.rds"),
  ah   = file.path("models", "ah_finalround_list_forSynapse.rds"),
  IFTA = file.path("models", "IFTA_finalround_list_forSynapse.rds"),
  glo  = file.path("models", "Glo_finalround_list_forSynapse.rds")
)

models_available <- all(file.exists(unlist(model_paths)))
models <- NULL
if (models_available) {
  models <- list(
    cv   = readRDS(model_paths$cv),
    ah   = readRDS(model_paths$ah),
    IFTA = readRDS(model_paths$IFTA),
    glo  = readRDS(model_paths$glo)
  )
}

is_train_model <- function(x) {
  inherits(x, "train") || (!is.null(x$finalModel) && !is.null(x$modelInfo) && !is.null(x$method))
}

collect_train_models <- function(x) {
  out <- list()
  walk <- function(obj, path = "model") {
    if (is.null(obj)) return(invisible(NULL))
    if (is_train_model(obj)) {
      out[[path]] <<- obj
      return(invisible(NULL))
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
    invisible(NULL)
  }
  walk(x)
  out
}

patient_values <- function(input) {
  deceased <- identical(input$donor_type, "Fallecido")
  female <- identical(input$sex, "Mujer")
  vals <- list(
    Age = as.numeric(input$age),
    Gender = ifelse(female, 1, 0),
    Gender1 = ifelse(female, 1, 0),
    Sex = ifelse(female, 1, 0),
    Sex1 = ifelse(female, 1, 0),
    Donor_type = ifelse(deceased, 1, 0),
    Donor_type1 = ifelse(deceased, 1, 0),
    Hypertension = ifelse(identical(input$hypertension, "Sí"), 1, 0),
    Hypertension1 = ifelse(identical(input$hypertension, "Sí"), 1, 0),
    Diabetes = ifelse(identical(input$diabetes, "Sí"), 1, 0),
    Diabetes1 = ifelse(identical(input$diabetes, "Sí"), 1, 0),
    Creatinine = as.numeric(input$creatinine),
    Proteinuria = ifelse(identical(input$proteinuria, "Sí"), 1, 0),
    Proteinuria1 = ifelse(identical(input$proteinuria, "Sí"), 1, 0),
    HCV_status = ifelse(identical(input$hcv, "Sí"), 1, 0),
    HCV_status1 = ifelse(identical(input$hcv, "Sí"), 1, 0),
    DCD = ifelse(deceased && identical(input$dcd, "Sí"), 1, 0),
    DCD1 = ifelse(deceased && identical(input$dcd, "Sí"), 1, 0),
    bmi = as.numeric(input$bmi),
    BMI = as.numeric(input$bmi),
    vascular_death = ifelse(deceased && identical(input$vascular_death, "Sí"), 1, 0),
    vascular_death1 = ifelse(deceased && identical(input$vascular_death, "Sí"), 1, 0)
  )
  vals
}

base_value <- function(nm, vals) {
  if (nm %in% names(vals)) return(vals[[nm]])
  if (grepl("1$", nm)) {
    nm0 <- sub("1$", "", nm)
    if (nm0 %in% names(vals)) return(vals[[nm0]])
  }
  nm1 <- paste0(nm, "1")
  if (nm1 %in% names(vals)) return(vals[[nm1]])
  NA_real_
}

extract_data_classes <- function(m) {
  possible_terms <- list(
    tryCatch(m$terms, error = function(e) NULL),
    tryCatch(m$finalModel$terms, error = function(e) NULL),
    tryCatch(m$finalModel$Terms, error = function(e) NULL)
  )
  dc <- NULL
  for (trm in possible_terms) {
    if (!is.null(trm)) {
      tmp <- attr(trm, "dataClasses")
      if (!is.null(tmp)) dc <- tmp
    }
  }
  if (is.null(dc)) return(character(0))
  dc <- dc[!names(dc) %in% c(".outcome", "y", "Y", "outcome")]
  dc[!is.na(names(dc)) & names(dc) != ""]
}

extract_xlevels <- function(m) {
  xl <- list()
  add <- function(z) {
    if (is.null(z) || !is.list(z)) return(NULL)
    for (nm in names(z)) {
      if (!is.null(z[[nm]]) && length(z[[nm]]) > 0) xl[[nm]] <<- as.character(z[[nm]])
    }
    NULL
  }
  add(tryCatch(m$finalModel$xlevels, error = function(e) NULL))
  add(tryCatch(m$finalModel$forest$xlevels, error = function(e) NULL))
  add(tryCatch(m$xlevels, error = function(e) NULL))
  xl
}

predictor_names_for_model <- function(m) {
  dc <- extract_data_classes(m)
  if (length(dc) > 0) return(names(dc))
  td <- tryCatch(m$trainingData, error = function(e) NULL)
  if (!is.null(td) && is.data.frame(td)) return(setdiff(names(td), ".outcome"))
  cn <- tryCatch(m$coefnames, error = function(e) NULL)
  if (!is.null(cn) && length(cn) > 0) return(as.character(cn))
  character(0)
}

choose_level <- function(nm, value, levels_vec) {
  lv <- as.character(levels_vec)
  if (length(lv) == 0) return(as.character(value))
  low <- tolower(lv)
  value <- as.integer(ifelse(is.na(value), 0, value))

  if (nm %in% c("Gender", "Gender1", "Sex", "Sex1")) {
    if (value == 1) {
      idx <- grep("female|woman|mujer|femenino|^f$|^1$|yes|true", low)
      if (length(idx) > 0) return(lv[idx[1]])
    } else {
      idx <- grep("male|man|hombre|masculino|^m$|^0$|no|false", low)
      if (length(idx) > 0) return(lv[idx[1]])
    }
  }

  if (nm %in% c("Donor_type", "Donor_type1")) {
    if (value == 1) {
      idx <- grep("deceased|dead|cadaver|fallecido|^1$|yes|true", low)
      if (length(idx) > 0) return(lv[idx[1]])
    } else {
      idx <- grep("living|live|vivo|^0$|no|false", low)
      if (length(idx) > 0) return(lv[idx[1]])
    }
  }

  if (value == 1) {
    idx <- grep("^1$|yes|si|sí|true|positive|pos", low)
    if (length(idx) > 0) return(lv[idx[1]])
  } else {
    idx <- grep("^0$|no|false|negative|neg", low)
    if (length(idx) > 0) return(lv[idx[1]])
  }

  if (length(lv) == 2) return(ifelse(value == 1, lv[2], lv[1]))
  lv[1]
}

build_one_newdata <- function(m, vals, mode = c("terms", "numeric", "factor")) {
  mode <- match.arg(mode)
  nms <- predictor_names_for_model(m)
  if (length(nms) == 0) {
    nms <- c("Age", "Gender", "Donor_type", "Hypertension", "Diabetes", "Creatinine",
             "Proteinuria", "HCV_status", "DCD", "bmi", "vascular_death")
  }

  dc <- extract_data_classes(m)
  xl <- extract_xlevels(m)
  td <- tryCatch(m$trainingData, error = function(e) NULL)

  out <- vector("list", length(nms))
  names(out) <- nms

  for (nm in nms) {
    v <- base_value(nm, vals)
    if (is.na(v) && !is.null(td) && is.data.frame(td) && nm %in% names(td)) {
      if (is.numeric(td[[nm]]) || is.integer(td[[nm]])) v <- median(td[[nm]], na.rm = TRUE)
      else v <- 0
    }
    if (is.na(v)) v <- 0

    cls <- if (nm %in% names(dc)) as.character(dc[[nm]]) else NA_character_
    lev <- xl[[nm]]
    if ((is.null(lev) || length(lev) == 0) && !is.null(td) && is.data.frame(td) && nm %in% names(td) && is.factor(td[[nm]])) {
      lev <- levels(td[[nm]])
    }

    want_factor <- FALSE
    want_ordered <- FALSE
    want_logical <- FALSE

    if (mode == "factor") want_factor <- !(nm %in% c("Age", "Creatinine", "bmi", "BMI"))
    if (mode == "terms") {
      want_factor <- identical(cls, "factor") || (!is.null(lev) && length(lev) > 0)
      want_ordered <- identical(cls, "ordered")
      want_logical <- identical(cls, "logical")
    }

    if (mode == "numeric") {
      out[[nm]] <- as.numeric(v)
    } else if (want_logical) {
      out[[nm]] <- as.logical(as.integer(v))
    } else if (want_factor || want_ordered) {
      if (is.null(lev) || length(lev) == 0) lev <- c("0", "1")
      sel <- choose_level(nm, v, lev)
      if (want_ordered) out[[nm]] <- ordered(sel, levels = lev) else out[[nm]] <- factor(sel, levels = lev)
    } else {
      if (identical(cls, "integer")) out[[nm]] <- as.integer(v)
      else out[[nm]] <- as.numeric(v)
    }
  }

  as.data.frame(out, check.names = FALSE)
}

predict_prob_one <- function(m, vals) {
  modes <- c("terms", "numeric", "factor")
  errs <- character(0)
  for (md in modes) {
    nd <- tryCatch(build_one_newdata(m, vals, md), error = function(e) NULL)
    if (is.null(nd)) next
    pred <- tryCatch(predict(m, newdata = nd, type = "prob"), error = function(e) {
      errs <<- c(errs, paste0(md, ": ", conditionMessage(e)))
      NULL
    })
    if (!is.null(pred)) return(pred)
  }
  attr(errs, "failed") <- TRUE
  errs
}

clean_prob <- function(p) {
  p <- as.data.frame(p, check.names = FALSE)
  nm <- names(p)
  nm <- gsub("^X", "", nm)
  nm <- gsub("[^0-9]", "", nm)
  names(p) <- nm
  for (k in as.character(0:3)) if (!k %in% names(p)) p[[k]] <- 0
  p <- p[, as.character(0:3), drop = FALSE]
  p <- as.data.frame(lapply(p, as.numeric), check.names = FALSE)
  p[is.na(p)] <- 0
  s <- rowSums(p)
  if (length(s) == 0 || s[1] <= 0) p[,] <- 0.25 else p <- p / s
  names(p) <- paste0("X", 0:3)
  p[1, , drop = FALSE]
}

predict_probability_ensemble <- function(container, vals) {
  ms <- collect_train_models(container)
  if (length(ms) == 0 && is_train_model(container)) ms <- list(model = container)
  if (length(ms) == 0) stop("No se encontraron modelos internos utilizables.")

  ok <- list()
  errs <- character(0)
  for (nm in names(ms)) {
    pr <- predict_prob_one(ms[[nm]], vals)
    if (is.character(pr) && isTRUE(attr(pr, "failed"))) {
      errs <- c(errs, paste0(nm, ": ", paste(head(pr, 1), collapse = "")))
    } else {
      ok[[length(ok) + 1]] <- clean_prob(pr)
    }
  }
  if (length(ok) == 0) stop(paste("No se pudieron calcular probabilidades reales.", paste(head(errs, 5), collapse = " | ")))
  arr <- simplify2array(lapply(ok, as.matrix))
  clean_prob(as.data.frame(apply(arr, c(1, 2), mean), check.names = FALSE))
}

predict_reg_one <- function(m, vals) {
  modes <- c("terms", "numeric", "factor")
  for (md in modes) {
    nd <- tryCatch(build_one_newdata(m, vals, md), error = function(e) NULL)
    if (is.null(nd)) next
    pred <- tryCatch(predict(m, newdata = nd), error = function(e) NULL)
    if (!is.null(pred)) {
      val <- suppressWarnings(as.numeric(pred[1]))
      if (!is.na(val)) return(val)
    }
  }
  NA_real_
}

predict_regression_ensemble <- function(container, vals) {
  ms <- collect_train_models(container)
  if (length(ms) == 0 && is_train_model(container)) ms <- list(model = container)
  if (length(ms) == 0) stop("No se encontraron modelos internos utilizables para glomeruloesclerosis.")
  y <- vapply(ms, predict_reg_one, vals = vals, FUN.VALUE = numeric(1))
  y <- y[!is.na(y)]
  if (length(y) == 0) stop("No se pudo calcular la glomeruloesclerosis real.")
  max(0, min(100, mean(y)))
}

predicted_class <- function(p) which.max(as.numeric(p[1, paste0("X", 0:3)])) - 1
prob_ge2 <- function(p) as.numeric(p$X2 + p$X3)

lesion_name <- function(code) {
  switch(code,
         cv = "Arteriosclerosis (cv)",
         ah = "Hialinosis arteriolar (ah)",
         IFTA = "Fibrosis intersticial y atrofia tubular (IFTA)")
}

grade_label <- function(x) {
  c("0" = "0 - ausente", "1" = "1 - leve", "2" = "2 - moderada", "3" = "3 - grave")[as.character(x)]
}

clinical_interpretation <- function(probs, glo) {
  txt <- character(0)
  for (nm in names(probs)) {
    p <- probs[[nm]]
    cls <- predicted_class(p)
    sev <- prob_ge2(p)
    if (sev >= 0.50 || cls >= 2) {
      txt <- c(txt, paste0(lesion_name(nm), ": alta probabilidad de lesión moderada o grave (grado ≥2: ", round(100 * sev, 1), "%). Considerar seguimiento estrecho y contextualización con biopsias posteriores."))
    } else if (sev >= 0.25) {
      txt <- c(txt, paste0(lesion_name(nm), ": probabilidad intermedia de lesión moderada o grave (grado ≥2: ", round(100 * sev, 1), "%). Interpretar junto con edad, comorbilidad y función renal del donante."))
    } else {
      txt <- c(txt, paste0(lesion_name(nm), ": predominan grados ausente o leve. Clase predicha: ", grade_label(cls), "."))
    }
  }
  if (glo >= 20) txt <- c(txt, paste0("Glomeruloesclerosis estimada elevada: ", round(glo, 1), "%."))
  else if (glo >= 10) txt <- c(txt, paste0("Glomeruloesclerosis estimada intermedia: ", round(glo, 1), "%."))
  else txt <- c(txt, paste0("Glomeruloesclerosis estimada baja: ", round(glo, 1), "%.") )
  paste(txt, collapse = "\n\n")
}

ui <- fluidPage(
  tags$head(tags$style(HTML("
    body { background-color:#f6f8fb; color:#1f2937; }
    .main-title { background:linear-gradient(135deg,#19324a,#295c7a); color:white; padding:22px; border-radius:16px; margin-bottom:20px; }
    .main-title h1 { margin-top:0; font-weight:700; }
    .panel-card { background:white; border-radius:16px; padding:18px; box-shadow:0 2px 10px rgba(0,0,0,.08); margin-bottom:16px; }
    .warning-card { background:#fff3cd; border-left:6px solid #f0ad4e; padding:14px; border-radius:10px; margin-bottom:16px; }
    .ok-card { background:#eaf7ee; border-left:6px solid #2e8b57; padding:14px; border-radius:10px; margin-bottom:16px; }
    .result-number { font-size:30px; font-weight:700; color:#19324a; }
    .small-muted { color:#6b7280; font-size:13px; }
    h3 { font-weight:700; color:#19324a; }
  "))),

  div(class = "main-title",
      h1("The Virtual Biopsy System"),
      h4("Biopsia virtual día cero para trasplante renal"),
      p("Predicción de lesiones Banff y porcentaje de glomeruloesclerosis a partir de parámetros clínicos del donante.")),

  if (!models_available) {
    div(class = "warning-card", h4("Modelos no encontrados"),
        p("Faltan archivos .rds en la carpeta models/."))
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
        conditionalPanel(condition = "input.donor_type == 'Fallecido'",
          selectInput("vascular_death", "Causa de muerte cerebrovascular", choices = c("No", "Sí"), selected = "Sí"),
          selectInput("dcd", "Causa de muerte circulatoria / DCD", choices = c("No", "Sí"), selected = "No")
        ),
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
        tabPanel("Resultados", br(),
          div(class = "panel-card", h3("Resumen de predicción"), uiOutput("summary_cards")),
          div(class = "panel-card", h3("Probabilidades por lesión y grado Banff"), DTOutput("probability_table")),
          div(class = "panel-card", h3("Glomeruloesclerosis"), htmlOutput("glo_output"))
        ),
        tabPanel("Gráfico radar", br(), div(class = "panel-card", h3("Radar de probabilidades"), plotlyOutput("radar_plot", height = "560px"))),
        tabPanel("Nota clínica", br(), div(class = "panel-card", h3("Interpretación clínica automática"), verbatimTextOutput("clinical_note"))),
        tabPanel("Ayuda", br(), div(class = "panel-card", h3("Qué calcula esta aplicación"),
          p("Estima probabilidades por grado Banff para cv, ah e IFTA, y porcentaje continuo de glomeruloesclerosis."),
          p("Herramienta predictiva de investigación: no sustituye la valoración clínica ni anatomopatológica.")))
      )
    )
  )
)

server <- function(input, output, session) {
  result <- eventReactive(input$calculate, {
    validate(need(models_available, "Faltan los archivos .rds en models/."))
    vals <- patient_values(input)
    probs <- list(
      cv = predict_probability_ensemble(models$cv, vals),
      ah = predict_probability_ensemble(models$ah, vals),
      IFTA = predict_probability_ensemble(models$IFTA, vals)
    )
    glo <- predict_regression_ensemble(models$glo, vals)
    list(probs = probs, glo = glo)
  }, ignoreInit = TRUE)

  output$summary_cards <- renderUI({
    res <- result()
    tagList(lapply(names(res$probs), function(nm) {
      p <- res$probs[[nm]]
      cls <- predicted_class(p)
      sev <- prob_ge2(p)
      div(style = "border-bottom:1px solid #e5e7eb; padding:12px 0;",
          h4(lesion_name(nm)),
          p(strong("Clase predicha: "), grade_label(cls)),
          p(strong("Probabilidad de grado ≥2: "), paste0(round(100 * sev, 1), "%")))
    }))
  })

  output$probability_table <- renderDT({
    res <- result()
    tab <- imap_dfr(res$probs, function(p, nm) tibble(
      Lesión = lesion_name(nm),
      `Grado 0 (%)` = round(100 * p$X0, 1),
      `Grado 1 (%)` = round(100 * p$X1, 1),
      `Grado 2 (%)` = round(100 * p$X2, 1),
      `Grado 3 (%)` = round(100 * p$X3, 1),
      `Clase predicha` = grade_label(predicted_class(p)),
      `Probabilidad grado ≥2 (%)` = round(100 * prob_ge2(p), 1)
    ))
    datatable(tab, rownames = FALSE, options = list(dom = "t", pageLength = 3))
  })

  output$glo_output <- renderUI({
    res <- result()
    HTML(paste0("<div class='result-number'>", round(res$glo, 2), "%</div><p>Porcentaje estimado de glomeruloesclerosis.</p>"))
  })

  output$clinical_note <- renderText({
    res <- result()
    clinical_interpretation(res$probs, res$glo)
  })

  output$radar_plot <- renderPlotly({
    res <- result()
    radar_data <- imap_dfr(res$probs, function(p, nm) tibble(
      lesion = lesion_name(nm),
      grado = factor(c("Grado 0", "Grado 1", "Grado 2", "Grado 3"), levels = c("Grado 0", "Grado 1", "Grado 2", "Grado 3")),
      prob = as.numeric(p[1, paste0("X", 0:3)])
    ))
    plot_ly(type = "scatterpolar", mode = "lines+markers") %>%
      add_trace(data = radar_data[radar_data$lesion == "Arteriosclerosis (cv)", ], r = ~prob, theta = ~grado, name = "cv", fill = "toself") %>%
      add_trace(data = radar_data[radar_data$lesion == "Hialinosis arteriolar (ah)", ], r = ~prob, theta = ~grado, name = "ah", fill = "toself") %>%
      add_trace(data = radar_data[radar_data$lesion == "Fibrosis intersticial y atrofia tubular (IFTA)", ], r = ~prob, theta = ~grado, name = "IFTA", fill = "toself") %>%
      layout(polar = list(radialaxis = list(visible = TRUE, range = c(0, 1), tickformat = ".0%")), showlegend = TRUE,
             title = "Probabilidades predichas por grado Banff")
  })
}

shinyApp(ui = ui, server = server)
