# ============================================================
# The Virtual Biopsy System - Shiny en castellano
# Versión de producción para Render usando los .rds originales.
# ============================================================

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

options(shiny.sanitize.errors = FALSE)

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

# -----------------------------
# Entrada clínica
# -----------------------------
make_patient <- function(input) {
  fallecido <- identical(input$donor_type, "Fallecido")
  mujer <- identical(input$sex, "Mujer")

  list(
    Age = as.numeric(input$age),
    Gender = ifelse(mujer, 1, 0),                       # 1=mujer, 0=hombre
    Donor_type = ifelse(fallecido, 1, 0),               # 1=fallecido, 0=vivo
    Hypertension = ifelse(input$hypertension == "Sí", 1, 0),
    Diabetes = ifelse(input$diabetes == "Sí", 1, 0),
    Creatinine = as.numeric(input$creatinine),
    Proteinuria = ifelse(input$proteinuria == "Sí", 1, 0),
    HCV_status = ifelse(input$hcv == "Sí", 1, 0),
    DCD = ifelse(fallecido && input$dcd == "Sí", 1, 0),
    bmi = as.numeric(input$bmi),
    vascular_death = ifelse(fallecido && input$vascular_death == "Sí", 1, 0)
  )
}

canonical_name <- function(x) {
  x <- gsub("`", "", x)
  x <- gsub("1$", "", x)
  x
}

patient_value <- function(patient, nm) {
  base <- canonical_name(nm)
  if (!is.null(patient[[base]])) return(patient[[base]])
  NA_real_
}

# Convierte 0/1 al nivel de factor usado en el modelo entrenado.
factor_value_for_model <- function(var, val, lev) {
  if (length(lev) == 0) return(as.character(val))
  lev_chr <- as.character(lev)
  lev_low <- tolower(lev_chr)
  base <- canonical_name(var)
  val <- suppressWarnings(as.numeric(val))
  if (is.na(val)) return(lev_chr[1])

  # Niveles explícitos 0/1
  if (as.character(val) %in% lev_chr) return(as.character(val))
  if (val == 0 && "0" %in% lev_chr) return("0")
  if (val == 1 && "1" %in% lev_chr) return("1")

  # Niveles yes/no, si/no
  yes_tokens <- c("yes", "y", "si", "sí", "true", "positive", "pos", "deceased", "female", "f")
  no_tokens  <- c("no", "n", "false", "negative", "neg", "living", "male", "m")

  if (base == "Gender") {
    if (val == 1) {
      hit <- which(lev_low %in% c("female", "f", "mujer", "woman"))
      if (length(hit)) return(lev_chr[hit[1]])
    } else {
      hit <- which(lev_low %in% c("male", "m", "hombre", "man"))
      if (length(hit)) return(lev_chr[hit[1]])
    }
  }

  if (base == "Donor_type") {
    if (val == 1) {
      hit <- which(lev_low %in% c("deceased", "deceased donor", "fallecido", "cadaveric"))
      if (length(hit)) return(lev_chr[hit[1]])
    } else {
      hit <- which(lev_low %in% c("living", "living donor", "vivo"))
      if (length(hit)) return(lev_chr[hit[1]])
    }
  }

  if (val == 1) {
    hit <- which(lev_low %in% yes_tokens)
    if (length(hit)) return(lev_chr[hit[1]])
  } else {
    hit <- which(lev_low %in% no_tokens)
    if (length(hit)) return(lev_chr[hit[1]])
  }

  # Último recurso: primer nivel para 0, segundo nivel para 1 si existe.
  if (val == 0) return(lev_chr[1])
  if (length(lev_chr) >= 2) return(lev_chr[2])
  lev_chr[1]
}

# Construye newdata usando EXACTAMENTE las columnas, clases y niveles que espera cada modelo.
newdata_for_model <- function(model, patient) {
  expected <- NULL
  template <- NULL

  if (!is.null(model$trainingData) && is.data.frame(model$trainingData)) {
    template <- model$trainingData
    expected <- setdiff(names(template), ".outcome")
  }

  if (is.null(expected) && !is.null(model$xNames)) {
    expected <- model$xNames
  }

  if (is.null(expected) && !is.null(model$terms)) {
    expected <- attr(model$terms, "term.labels")
  }

  if (is.null(expected) || length(expected) == 0) return(NULL)

  out <- as.data.frame(setNames(vector("list", length(expected)), expected), stringsAsFactors = FALSE)
  out[1, ] <- NA

  for (nm in expected) {
    val <- patient_value(patient, nm)

    if (!is.null(template) && nm %in% names(template)) {
      col <- template[[nm]]

      if (is.factor(col)) {
        lev <- levels(col)
        out[[nm]] <- factor(factor_value_for_model(nm, val, lev), levels = lev, ordered = is.ordered(col))
      } else if (is.logical(col)) {
        out[[nm]] <- as.logical(val)
      } else if (is.integer(col)) {
        out[[nm]] <- as.integer(val)
      } else if (is.numeric(col)) {
        out[[nm]] <- as.numeric(val)
      } else {
        out[[nm]] <- as.character(val)
      }
    } else {
      out[[nm]] <- as.numeric(val)
    }
  }

  out
}

# Extrae modelos predictivos desde contenedores/listas/caretList/caretEnsemble.
collect_models <- function(x, depth = 0) {
  if (depth > 8 || is.null(x)) return(list())

  cls <- class(x)
  if (inherits(x, "train") || inherits(x, "randomForest") || inherits(x, "gbm") ||
      inherits(x, "xgb.Booster") || inherits(x, "multinom") || inherits(x, "nnet") ||
      inherits(x, "lda")) {
    return(list(x))
  }

  if (is.list(x)) {
    res <- list()
    for (i in seq_along(x)) {
      res <- c(res, collect_models(x[[i]], depth + 1))
    }
    return(res)
  }

  list()
}

clean_prob <- function(p) {
  p <- as.data.frame(p)
  if (nrow(p) < 1) stop("predicción vacía")
  nms <- names(p)
  nms <- gsub("^X", "", nms)
  nms <- gsub("[^0-9]", "", nms)
  names(p) <- nms
  for (k in as.character(0:3)) {
    if (!k %in% names(p)) p[[k]] <- 0
  }
  p <- p[, as.character(0:3), drop = FALSE]
  p <- as.data.frame(lapply(p, as.numeric))
  p[is.na(p)] <- 0
  s <- rowSums(p)
  if (is.na(s[1]) || s[1] <= 0) {
    p[1, ] <- rep(0.25, 4)
  } else {
    p[1, ] <- p[1, ] / s[1]
  }
  names(p) <- paste0("X", 0:3)
  p
}

predict_one_prob <- function(model, patient) {
  nd <- newdata_for_model(model, patient)
  if (is.null(nd)) return(list(ok = FALSE, error = "sin estructura de variables"))

  out <- tryCatch({
    pred <- predict(model, newdata = nd, type = "prob")
    list(ok = TRUE, value = clean_prob(pred), error = NULL)
  }, error = function(e) {
    list(ok = FALSE, error = conditionMessage(e))
  })
  out
}

predict_ensemble_prob <- function(container, patient) {
  mods <- collect_models(container)
  if (length(mods) == 0) stop("El archivo .rds no contiene modelos predictivos reconocibles.")

  results <- lapply(mods, predict_one_prob, patient = patient)
  ok <- results[vapply(results, function(z) isTRUE(z$ok), logical(1))]

  if (length(ok) == 0) {
    errs <- unique(vapply(results, function(z) z$error %||% "error desconocido", character(1)))
    stop(paste("No se pudo calcular ninguna probabilidad. Primeros errores:", paste(head(errs, 4), collapse = " | ")))
  }

  arr <- simplify2array(lapply(ok, function(z) as.matrix(z$value)))
  avg <- as.data.frame(apply(arr, c(1, 2), mean))
  clean_prob(avg)
}

predict_one_reg <- function(model, patient) {
  nd <- newdata_for_model(model, patient)
  if (is.null(nd)) return(list(ok = FALSE, error = "sin estructura de variables"))

  tryCatch({
    pred <- predict(model, newdata = nd)
    val <- suppressWarnings(as.numeric(pred[1]))
    if (is.na(val)) stop("predicción no numérica")
    list(ok = TRUE, value = val, error = NULL)
  }, error = function(e) {
    list(ok = FALSE, error = conditionMessage(e))
  })
}

predict_ensemble_reg <- function(container, patient) {
  mods <- collect_models(container)
  if (length(mods) == 0) stop("El archivo .rds no contiene modelos predictivos reconocibles.")

  results <- lapply(mods, predict_one_reg, patient = patient)
  vals <- vapply(results, function(z) if (isTRUE(z$ok)) z$value else NA_real_, numeric(1))
  vals <- vals[!is.na(vals)]

  if (length(vals) == 0) {
    errs <- unique(vapply(results, function(z) z$error %||% "error desconocido", character(1)))
    stop(paste("No se pudo calcular glomeruloesclerosis. Primeros errores:", paste(head(errs, 4), collapse = " | ")))
  }

  max(0, min(100, mean(vals)))
}

`%||%` <- function(x, y) if (is.null(x)) y else x

lesion_name <- function(code) {
  switch(code,
         cv = "Arteriosclerosis (cv)",
         ah = "Hialinosis arteriolar (ah)",
         IFTA = "Fibrosis intersticial y atrofia tubular (IFTA)")
}

grade_label <- function(x) {
  c("0" = "0 - ausente", "1" = "1 - leve", "2" = "2 - moderada", "3" = "3 - grave")[as.character(x)]
}

predicted_class <- function(p) which.max(as.numeric(p[1, paste0("X", 0:3)])) - 1
prob_ge2 <- function(p) as.numeric(p$X2 + p$X3)

clinical_interpretation <- function(probs, glo) {
  txt <- c()
  for (nm in names(probs)) {
    p <- probs[[nm]]
    cls <- predicted_class(p)
    sev <- prob_ge2(p)
    lesion <- lesion_name(nm)
    if (sev >= 0.50 || cls >= 2) {
      txt <- c(txt, paste0(lesion, ": alta probabilidad de lesión moderada o grave (grado ≥2: ", round(100 * sev, 1), "%). Considerar seguimiento estrecho y contextualización con biopsias posteriores."))
    } else if (sev >= 0.25) {
      txt <- c(txt, paste0(lesion, ": probabilidad intermedia de lesión moderada o grave (grado ≥2: ", round(100 * sev, 1), "%). Interpretar junto al contexto clínico del donante."))
    } else {
      txt <- c(txt, paste0(lesion, ": predominan grados ausente o leve. Clase predicha: ", grade_label(cls), "."))
    }
  }
  if (glo >= 20) {
    txt <- c(txt, paste0("Glomeruloesclerosis estimada elevada: ", round(glo, 1), "%."))
  } else if (glo >= 10) {
    txt <- c(txt, paste0("Glomeruloesclerosis estimada intermedia: ", round(glo, 1), "%."))
  } else {
    txt <- c(txt, paste0("Glomeruloesclerosis estimada baja: ", round(glo, 1), "%."))
  }
  paste(txt, collapse = "\n\n")
}

# -----------------------------
# UI
# -----------------------------
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

  div(class = "main-title",
      h1("The Virtual Biopsy System"),
      h4("Biopsia virtual día cero para trasplante renal"),
      p("Predicción de lesiones Banff y porcentaje de glomeruloesclerosis a partir de parámetros clínicos del donante.")),

  if (!models_available) {
    div(class = "warning-card", h4("Modelos no encontrados"),
        p("Faltan archivos .rds dentro de la carpeta models/."))
  } else {
    div(class = "ok-card", strong("Modelos cargados correctamente. "),
        span("La aplicación está lista para calcular predicciones."))
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
        tabPanel("Ayuda", br(), div(class = "panel-card",
          h3("Uso"),
          p("Introduzca las 11 variables del donante y pulse Calcular biopsia virtual."),
          p("La app devuelve probabilidades por grado Banff para cv, ah e IFTA, una estimación continua de glomeruloesclerosis, un radar y una interpretación clínica automática."),
          p("Herramienta de apoyo basada en modelos publicados; no sustituye la evaluación clínica ni anatomopatológica.")
        ))
      )
    )
  )
)

# -----------------------------
# Server
# -----------------------------
server <- function(input, output, session) {
  prediction_result <- eventReactive(input$calculate, {
    validate(need(models_available, "Faltan los archivos .rds en la carpeta models/."))
    patient <- make_patient(input)

    probs <- list(
      cv = predict_ensemble_prob(loaded_models$cv, patient),
      ah = predict_ensemble_prob(loaded_models$ah, patient),
      IFTA = predict_ensemble_prob(loaded_models$IFTA, patient)
    )
    glo <- predict_ensemble_reg(loaded_models$glo, patient)
    list(patient = patient, probs = probs, glo = glo)
  }, ignoreInit = FALSE)

  output$summary_cards <- renderUI({
    res <- prediction_result()
    tagList(lapply(names(res$probs), function(nm) {
      p <- res$probs[[nm]]
      cls <- predicted_class(p)
      sev <- prob_ge2(p)
      div(style = "border-bottom: 1px solid #e5e7eb; padding: 12px 0;",
          h4(lesion_name(nm)),
          tags$p(tags$strong("Clase predicha: "), grade_label(cls)),
          tags$p(tags$strong("Probabilidad de grado ≥2: "), paste0(round(100 * sev, 1), "%")))
    }))
  })

  output$probability_table <- renderDT({
    res <- prediction_result()
    tab <- imap_dfr(res$probs, function(p, nm) {
      tibble(
        Lesión = lesion_name(nm),
        `Grado 0 (%)` = round(100 * p$X0, 1),
        `Grado 1 (%)` = round(100 * p$X1, 1),
        `Grado 2 (%)` = round(100 * p$X2, 1),
        `Grado 3 (%)` = round(100 * p$X3, 1),
        `Clase predicha` = grade_label(predicted_class(p)),
        `Probabilidad grado ≥2 (%)` = round(100 * prob_ge2(p), 1)
      )
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
      tibble(
        lesion = lesion_name(nm),
        grado = factor(c("Grado 0", "Grado 1", "Grado 2", "Grado 3"), levels = c("Grado 0", "Grado 1", "Grado 2", "Grado 3")),
        prob = as.numeric(p[1, paste0("X", 0:3)])
      )
    })

    plot_ly(type = "scatterpolar", mode = "lines+markers") %>%
      add_trace(data = radar_data[radar_data$lesion == "Arteriosclerosis (cv)", ], r = ~prob, theta = ~grado, name = "cv", fill = "toself") %>%
      add_trace(data = radar_data[radar_data$lesion == "Hialinosis arteriolar (ah)", ], r = ~prob, theta = ~grado, name = "ah", fill = "toself") %>%
      add_trace(data = radar_data[radar_data$lesion == "Fibrosis intersticial y atrofia tubular (IFTA)", ], r = ~prob, theta = ~grado, name = "IFTA", fill = "toself") %>%
      layout(polar = list(radialaxis = list(visible = TRUE, range = c(0, 1), tickformat = ".0%")), showlegend = TRUE,
             title = "Probabilidades predichas por grado Banff")
  })
}

shinyApp(ui = ui, server = server)
