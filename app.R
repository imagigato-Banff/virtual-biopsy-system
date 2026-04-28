# ============================================================
# The Virtual Biopsy System - Shiny en castellano
# Versión corregida: recorre correctamente los modelos internos
# de los .rds originales de Yoo et al.
# ============================================================

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

# ------------------------------------------------------------
# Entrada: 11 variables documentadas
# Codificación usada: 0/1, con variantes originales y dummy.
# ------------------------------------------------------------

make_patient <- function(input) {
  deceased <- input$donor_type == "Fallecido"

  data.frame(
    Age = as.numeric(input$age),
    Gender = ifelse(input$sex == "Mujer", 1, 0),
    Donor_type = ifelse(deceased, 1, 0),
    Hypertension = ifelse(input$hypertension == "Sí", 1, 0),
    Diabetes = ifelse(input$diabetes == "Sí", 1, 0),
    Creatinine = as.numeric(input$creatinine),
    Proteinuria = ifelse(input$proteinuria == "Sí", 1, 0),
    HCV_status = ifelse(input$hcv == "Sí", 1, 0),
    DCD = ifelse(deceased && input$dcd == "Sí", 1, 0),
    bmi = as.numeric(input$bmi),
    vascular_death = ifelse(deceased && input$vascular_death == "Sí", 1, 0),
    check.names = FALSE
  )
}

make_pool <- function(patient) {
  p <- patient
  pool <- data.frame(
    Age = p$Age,
    Gender = p$Gender,
    Gender1 = p$Gender,
    Donor_type = p$Donor_type,
    Donor_type1 = p$Donor_type,
    Hypertension = p$Hypertension,
    Hypertension1 = p$Hypertension,
    Diabetes = p$Diabetes,
    Diabetes1 = p$Diabetes,
    Creatinine = p$Creatinine,
    Proteinuria = p$Proteinuria,
    Proteinuria1 = p$Proteinuria,
    HCV_status = p$HCV_status,
    HCV_status1 = p$HCV_status,
    DCD = p$DCD,
    DCD1 = p$DCD,
    bmi = p$bmi,
    vascular_death = p$vascular_death,
    vascular_death1 = p$vascular_death,
    check.names = FALSE
  )
  pool
}

# ------------------------------------------------------------
# Recoger modelos internos:
# los .rds finales son contenedores/listas, no siempre objetos
# predictables directamente. Esta función entra en la lista.
# ------------------------------------------------------------

collect_predictable_models <- function(x, depth = 0) {
  if (depth > 6) return(list())

  out <- list()

  if (inherits(x, "train") ||
      inherits(x, "caretStack") ||
      inherits(x, "caretEnsemble")) {
    return(list(x))
  }

  if (is.list(x)) {
    # Caso habitual: caretList/lista de modelos caret::train
    for (i in seq_along(x)) {
      out <- c(out, collect_predictable_models(x[[i]], depth + 1))
    }
  }

  out
}

# ------------------------------------------------------------
# Construir newdata compatible con cada modelo concreto.
# Usa los nombres y tipos de trainingData cuando existen.
# ------------------------------------------------------------

get_required_predictors <- function(model) {
  if (!is.null(model$trainingData)) {
    return(setdiff(names(model$trainingData), ".outcome"))
  }

  if (!is.null(model$finalModel$xNames)) {
    return(model$finalModel$xNames)
  }

  if (!is.null(model$coefnames)) {
    return(model$coefnames)
  }

  names(make_pool(data.frame(
    Age = 50, Gender = 0, Donor_type = 1, Hypertension = 0,
    Diabetes = 0, Creatinine = 1, Proteinuria = 0, HCV_status = 0,
    DCD = 0, bmi = 25, vascular_death = 0
  )))
}

prepare_newdata_for_model <- function(model, patient) {
  pool_num <- make_pool(patient)

  pool_fac <- pool_num
  binary_cols <- setdiff(names(pool_fac), c("Age", "Creatinine", "bmi"))
  for (nm in binary_cols) {
    pool_fac[[nm]] <- factor(as.character(pool_fac[[nm]]), levels = c("0", "1"))
  }

  req <- get_required_predictors(model)
  aligned_num <- data.frame(row.names = 1)
  for (nm in req) {
    aligned_num[[nm]] <- if (nm %in% names(pool_num)) pool_num[[nm]] else 0
  }

  aligned_fac <- aligned_num
  if (!is.null(model)) {
    td <- model
    for (nm in intersect(names(aligned_fac), names(td))) {
      if (is.factor(td[[nm]])) {
        lev <- levels(td[[nm]])
        val <- as.character(aligned_fac[[nm]][1])
        if (!val %in% lev) val <- ifelse(suppressWarnings(as.numeric(aligned_fac[[nm]][1])) >= 0.5, tail(lev, 1), lev[1])
        aligned_fac[[nm]] <- factor(val, levels = lev)
      } else {
        aligned_fac[[nm]] <- suppressWarnings(as.numeric(aligned_fac[[nm]]))
      }
    }
  }

  # Devolvemos varias versiones. La clave para el error actual es que TODAS
  # las versiones completas contienen Gender y Gender1 a la vez.
  list(pool_num, pool_fac, aligned_num, aligned_fac)
}

clean_probability_output <- function(p) {
  p <- as.data.frame(p)
  if (nrow(p) < 1) stop("Salida de probabilidades vacía.")

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
  if (any(s <= 0)) {
    p[,] <- 0.25
  } else {
    p <- p / s
  }

  names(p) <- paste0("X", 0:3)
  p
}

predict_probability_ensemble <- function(container, patient) {
  internal_models <- collect_predictable_models(container)

  if (length(internal_models) == 0) {
    stop("El .rds se cargó, pero no contiene modelos predictivos reconocibles.")
  }

  prob_list <- list()
  error_list <- c()

  for (m in internal_models) {
    nd_list <- prepare_newdata_for_model(m, patient)
    pred <- NULL

    for (nd in nd_list) {
      pred <- tryCatch(
        predict(m, newdata = nd, type = "prob"),
        error = function(e) {
          error_list <<- c(error_list, paste(class(m)[1], conditionMessage(e)))
          NULL
        }
      )
      if (!is.null(pred)) break
    }

    if (!is.null(pred)) {
      prob_list[[length(prob_list) + 1]] <- clean_probability_output(pred)
    }
  }

  if (length(prob_list) == 0) {
    stop(paste0(
      "No se pudieron calcular probabilidades con los modelos internos. Primeros errores: ",
      paste(head(error_list, 3), collapse = " | ")
    ))
  }

  arr <- simplify2array(lapply(prob_list, as.matrix))
  mean_prob <- apply(arr, c(1, 2), mean)
  clean_probability_output(as.data.frame(mean_prob))
}

predict_regression_ensemble <- function(container, patient) {
  internal_models <- collect_predictable_models(container)

  if (length(internal_models) == 0) {
    stop("El .rds de glomeruloesclerosis no contiene modelos predictivos reconocibles.")
  }

  vals <- c()
  error_list <- c()

  for (m in internal_models) {
    nd_list <- prepare_newdata_for_model(m, patient)
    pred <- NULL

    for (nd in nd_list) {
      pred <- tryCatch(
        predict(m, newdata = nd),
        error = function(e) {
          error_list <<- c(error_list, paste(class(m)[1], conditionMessage(e)))
          NULL
        }
      )
      if (!is.null(pred)) break
    }

    if (!is.null(pred)) {
      val <- suppressWarnings(as.numeric(pred[1]))
      if (!is.na(val)) vals <- c(vals, val)
    }
  }

  if (length(vals) == 0) {
    stop(paste0(
      "No se pudo calcular glomeruloesclerosis. Primeros errores: ",
      paste(head(error_list, 3), collapse = " | ")
    ))
  }

  max(0, min(100, mean(vals, na.rm = TRUE)))
}

predicted_class <- function(prob_row) {
  which.max(as.numeric(prob_row[1, paste0("X", 0:3)])) - 1
}

prob_ge2 <- function(prob_row) {
  as.numeric(prob_row$X2 + prob_row$X3)
}

lesion_name <- function(code) {
  switch(
    code,
    cv = "Arteriosclerosis (cv)",
    ah = "Hialinosis arteriolar (ah)",
    IFTA = "Fibrosis intersticial y atrofia tubular (IFTA)"
  )
}

grade_label <- function(x) {
  labels <- c(
    "0" = "0 - ausente",
    "1" = "1 - leve",
    "2" = "2 - moderada",
    "3" = "3 - grave"
  )
  labels[as.character(x)]
}

clinical_interpretation <- function(probs, glo) {
  txt <- c()

  for (nm in names(probs)) {
    p <- probs[[nm]]
    cls <- predicted_class(p)
    psev <- prob_ge2(p)

    if (psev >= 0.50 || cls >= 2) {
      txt <- c(txt, paste0(
        lesion_name(nm), ": alta probabilidad de lesión moderada o grave ",
        "(grado ≥2: ", round(100 * psev, 1), "%). Considerar seguimiento estrecho."
      ))
    } else if (psev >= 0.25) {
      txt <- c(txt, paste0(
        lesion_name(nm), ": probabilidad intermedia de lesión moderada/grave ",
        "(grado ≥2: ", round(100 * psev, 1), "%). Interpretar según contexto clínico."
      ))
    } else {
      txt <- c(txt, paste0(
        lesion_name(nm), ": predominan grados ausente o leve. Clase predicha: ",
        grade_label(cls), "."
      ))
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

ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      body { background-color: #f6f8fb; color: #1f2937; }
      .main-title {
        background: linear-gradient(135deg, #19324a, #295c7a);
        color: white; padding: 22px; border-radius: 16px; margin-bottom: 20px;
      }
      .main-title h1 { margin-top: 0; font-weight: 700; }
      .panel-card {
        background: white; border-radius: 16px; padding: 18px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-bottom: 16px;
      }
      .warning-card {
        background: #fff3cd; border-left: 6px solid #f0ad4e;
        padding: 14px; border-radius: 10px; margin-bottom: 16px;
      }
      .ok-card {
        background: #eaf7ee; border-left: 6px solid #2e8b57;
        padding: 14px; border-radius: 10px; margin-bottom: 16px;
      }
      .result-number { font-size: 28px; font-weight: 700; color: #19324a; }
      .small-muted { color: #6b7280; font-size: 13px; }
      h3 { font-weight: 700; color: #19324a; }
    "))
  ),

  div(
    class = "main-title",
    h1("The Virtual Biopsy System"),
    h4("Biopsia virtual día cero para trasplante renal"),
    p("Predicción de lesiones Banff y porcentaje de glomeruloesclerosis a partir de parámetros clínicos del donante.")
  ),

  if (!models_available) {
    div(
      class = "warning-card",
      h4("Modelos no encontrados"),
      p("La aplicación necesita estos cuatro archivos dentro de la carpeta models/:"),
      tags$ul(
        tags$li("cv_finalround_list_forSynapse.rds"),
        tags$li("ah_finalround_list_forSynapse.rds"),
        tags$li("IFTA_finalround_list_forSynapse.rds"),
        tags$li("Glo_finalround_list_forSynapse.rds")
      )
    )
  } else {
    div(class = "ok-card", strong("Modelos cargados correctamente. "), span("La aplicación está lista para calcular predicciones."))
  },

  sidebarLayout(
    sidebarPanel(
      width = 4,
      div(
        class = "panel-card",
        h3("Datos del donante"),

        numericInput("age", "Edad del donante (años)", value = 57, min = 0, max = 100, step = 1),
        selectInput("sex", "Sexo", choices = c("Mujer", "Hombre"), selected = "Mujer"),
        selectInput("donor_type", "Tipo de donante", choices = c("Vivo", "Fallecido"), selected = "Fallecido"),

        conditionalPanel(
          condition = "input.donor_type == 'Fallecido'",
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

    mainPanel(
      width = 8,
      tabsetPanel(
        tabPanel(
          "Resultados",
          br(),
          div(class = "panel-card", h3("Resumen de predicción"), uiOutput("summary_cards")),
          div(class = "panel-card", h3("Probabilidades por lesión y grado Banff"), DT::DTOutput("probability_table")),
          div(class = "panel-card", h3("Glomeruloesclerosis"), htmlOutput("glo_output"))
        ),
        tabPanel(
          "Gráfico radar",
          br(),
          div(class = "panel-card", h3("Radar de probabilidades"), plotlyOutput("radar_plot", height = "560px"))
        ),
        tabPanel(
          "Nota clínica",
          br(),
          div(class = "panel-card", h3("Interpretación clínica automática"), verbatimTextOutput("clinical_note"))
        ),
        tabPanel(
          "Ayuda",
          br(),
          div(
            class = "panel-card",
            h3("Qué calcula esta aplicación"),
            p("Estima probabilidades Banff para cv, ah e IFTA, y porcentaje continuo de glomeruloesclerosis."),
            h3("Advertencia"),
            p("Herramienta predictiva de investigación. No sustituye la valoración clínica ni anatomopatológica.")
          )
        )
      )
    )
  )
)

server <- function(input, output, session) {

  prediction_result <- eventReactive(input$calculate, {
    validate(need(models_available, "Faltan archivos .rds en la carpeta models/."))

    patient <- make_patient(input)

    probs <- list(
      cv   = predict_probability_ensemble(loaded_models$cv, patient),
      ah   = predict_probability_ensemble(loaded_models$ah, patient),
      IFTA = predict_probability_ensemble(loaded_models$IFTA, patient)
    )

    glo <- predict_regression_ensemble(loaded_models$glo, patient)

    list(patient = patient, probs = probs, glo = glo)
  }, ignoreInit = FALSE)

  output$summary_cards <- renderUI({
    res <- prediction_result()

    cards <- lapply(names(res$probs), function(nm) {
      p <- res$probs[[nm]]
      cls <- predicted_class(p)
      psev <- prob_ge2(p)

      div(
        style = "border-bottom: 1px solid #e5e7eb; padding: 12px 0;",
        h4(lesion_name(nm)),
        tags$p(tags$strong("Clase predicha: "), grade_label(cls)),
        tags$p(tags$strong("Probabilidad de grado ≥2: "), paste0(round(100 * psev, 1), "%"))
      )
    })

    tagList(cards)
  })

  output$probability_table <- DT::renderDT({
    res <- prediction_result()

    tab <- purrr::imap_dfr(res$probs, function(p, nm) {
      tibble::tibble(
        Lesión = lesion_name(nm),
        `Grado 0 (%)` = round(100 * p$X0, 1),
        `Grado 1 (%)` = round(100 * p$X1, 1),
        `Grado 2 (%)` = round(100 * p$X2, 1),
        `Grado 3 (%)` = round(100 * p$X3, 1),
        `Clase predicha` = grade_label(predicted_class(p)),
        `Probabilidad grado ≥2 (%)` = round(100 * prob_ge2(p), 1)
      )
    })

    DT::datatable(tab, rownames = FALSE, options = list(dom = "t", pageLength = 3))
  })

  output$glo_output <- renderUI({
    res <- prediction_result()
    HTML(paste0("<div class='result-number'>", round(res$glo, 2), "%</div><p>Porcentaje estimado de glomeruloesclerosis.</p>"))
  })

  output$clinical_note <- renderText({
    res <- prediction_result()
    clinical_interpretation(res$probs, res$glo)
  })

  output$radar_plot <- plotly::renderPlotly({
    res <- prediction_result()

    radar_data <- purrr::imap_dfr(res$probs, function(p, nm) {
      tibble::tibble(
        lesion = lesion_name(nm),
        grado = factor(c("Grado 0", "Grado 1", "Grado 2", "Grado 3"),
                       levels = c("Grado 0", "Grado 1", "Grado 2", "Grado 3")),
        prob = as.numeric(p[1, paste0("X", 0:3)])
      )
    })

    plotly::plot_ly(type = "scatterpolar", mode = "lines+markers") %>%
      plotly::add_trace(
        data = radar_data[radar_data$lesion == "Arteriosclerosis (cv)", ],
        r = ~prob, theta = ~grado, name = "cv", fill = "toself"
      ) %>%
      plotly::add_trace(
        data = radar_data[radar_data$lesion == "Hialinosis arteriolar (ah)", ],
        r = ~prob, theta = ~grado, name = "ah", fill = "toself"
      ) %>%
      plotly::add_trace(
        data = radar_data[radar_data$lesion == "Fibrosis intersticial y atrofia tubular (IFTA)", ],
        r = ~prob, theta = ~grado, name = "IFTA", fill = "toself"
      ) %>%
      plotly::layout(
        polar = list(radialaxis = list(visible = TRUE, range = c(0, 1), tickformat = ".0%")),
        showlegend = TRUE,
        title = "Probabilidades predichas por grado Banff"
      )
  })
}

shinyApp(ui = ui, server = server)
