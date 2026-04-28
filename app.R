# ============================================================
# The Virtual Biopsy System - Shiny en castellano
# Réplica funcional usando los modelos .rds originales de Yoo et al.
# ============================================================

options(shiny.maxRequestSize = 500 * 1024^2)

required_packages <- c(
  "shiny",
  "plotly",
  "DT",
  "dplyr",
  "tidyr",
  "tibble",
  "purrr",
  "stringr",
  "caret",
  "caretEnsemble",
  "randomForest",
  "gbm",
  "xgboost",
  "nnet",
  "MASS"
)

install_if_missing <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))]
  if (length(missing) > 0) {
    install.packages(missing, repos = "https://cloud.r-project.org", dependencies = TRUE)
  }
  invisible(lapply(pkgs, library, character.only = TRUE))
}

install_if_missing(required_packages)

# ============================================================
# Rutas de modelos
# ============================================================

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

# ============================================================
# Variables usadas por los modelos originales
# Orden documentado en el código fuente:
# Age, Gender, Donor_type, Hypertension, Diabetes, Creatinine,
# Proteinuria, HCV_status, DCD, bmi, vascular_death
# ============================================================

base_predictors <- c(
  "Age",
  "Gender",
  "Donor_type",
  "Hypertension",
  "Diabetes",
  "Creatinine",
  "Proteinuria",
  "HCV_status",
  "DCD",
  "bmi",
  "vascular_death"
)

dummy_predictors <- c(
  "Age",
  "Gender1",
  "Donor_type1",
  "Hypertension1",
  "Diabetes1",
  "Creatinine",
  "Proteinuria1",
  "HCV_status1",
  "DCD1",
  "bmi",
  "vascular_death1"
)

# ============================================================
# Construcción del paciente individual
# ============================================================

make_patient <- function(input) {
  deceased <- input$donor_type == "Fallecido"

  gender <- ifelse(input$sex == "Mujer", 1, 0)
  donor_type <- ifelse(deceased, 1, 0)

  vascular_death <- ifelse(deceased && input$vascular_death == "Sí", 1, 0)
  dcd <- ifelse(deceased && input$dcd == "Sí", 1, 0)

  hypertension <- ifelse(input$hypertension == "Sí", 1, 0)
  diabetes <- ifelse(input$diabetes == "Sí", 1, 0)
  hcv <- ifelse(input$hcv == "Sí", 1, 0)
  proteinuria <- ifelse(input$proteinuria == "Sí", 1, 0)

  data.frame(
    Age = as.numeric(input$age),
    Gender = gender,
    Donor_type = donor_type,
    Hypertension = hypertension,
    Diabetes = diabetes,
    Creatinine = as.numeric(input$creatinine),
    Proteinuria = proteinuria,
    HCV_status = hcv,
    DCD = dcd,
    bmi = as.numeric(input$bmi),
    vascular_death = vascular_death,
    check.names = FALSE
  )
}

make_candidate_newdata <- function(patient) {
  numeric_original <- patient[, base_predictors, drop = FALSE]

  factor_original <- numeric_original
  binary_vars <- setdiff(base_predictors, c("Age", "Creatinine", "bmi"))
  for (v in binary_vars) {
    factor_original[[v]] <- factor(as.character(factor_original[[v]]), levels = c("0", "1"))
  }

  dummy_numeric <- data.frame(
    Age = patient$Age,
    Gender1 = patient$Gender,
    Donor_type1 = patient$Donor_type,
    Hypertension1 = patient$Hypertension,
    Diabetes1 = patient$Diabetes,
    Creatinine = patient$Creatinine,
    Proteinuria1 = patient$Proteinuria,
    HCV_status1 = patient$HCV_status,
    DCD1 = patient$DCD,
    bmi = patient$bmi,
    vascular_death1 = patient$vascular_death,
    check.names = FALSE
  )

  dummy_factor <- dummy_numeric
  binary_dummy <- setdiff(dummy_predictors, c("Age", "Creatinine", "bmi"))
  for (v in binary_dummy) {
    dummy_factor[[v]] <- factor(as.character(dummy_factor[[v]]), levels = c("0", "1"))
  }

  list(
    numeric_original = numeric_original,
    factor_original = factor_original,
    dummy_numeric = dummy_numeric,
    dummy_factor = dummy_factor
  )
}

# ============================================================
# Funciones robustas de predicción
# ============================================================

clean_probability_output <- function(p) {
  p <- as.data.frame(p)

  if (nrow(p) < 1) stop("La predicción de probabilidades está vacía.")

  names(p) <- gsub("^X", "", names(p))
  names(p) <- gsub("[^0-9]", "", names(p))

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

try_predict_probability <- function(model_object, patient) {
  candidates <- make_candidate_newdata(patient)

  possible_objects <- list()

  if (!is.null(model_object$ens_model)) {
    possible_objects <- c(possible_objects, list(model_object$ens_model))
  }

  if (!is.null(model_object$models)) {
    possible_objects <- c(possible_objects, model_object$models)
  }

  possible_objects <- c(possible_objects, list(model_object))

  for (obj in possible_objects) {
    for (nd in candidates) {
      pred <- tryCatch(
        predict(obj, newdata = nd, type = "prob"),
        error = function(e) NULL
      )

      if (!is.null(pred)) {
        return(clean_probability_output(pred))
      }
    }
  }

  # Si el ensemble no permite type = "prob", intentamos promediar modelos individuales
  if (!is.null(model_object$models)) {
    prob_list <- list()

    for (m in model_object$models) {
      for (nd in candidates) {
        pred <- tryCatch(
          predict(m, newdata = nd, type = "prob"),
          error = function(e) NULL
        )

        if (!is.null(pred)) {
          prob_list[[length(prob_list) + 1]] <- clean_probability_output(pred)
          break
        }
      }
    }

    if (length(prob_list) > 0) {
      arr <- simplify2array(lapply(prob_list, as.matrix))
      mean_prob <- apply(arr, c(1, 2), mean)
      mean_prob <- as.data.frame(mean_prob)
      return(clean_probability_output(mean_prob))
    }
  }

  stop("No se pudieron calcular probabilidades con este modelo.")
}

try_predict_regression <- function(model_object, patient) {
  candidates <- make_candidate_newdata(patient)

  possible_objects <- list()

  if (!is.null(model_object$ens_model)) {
    possible_objects <- c(possible_objects, list(model_object$ens_model))
  }

  if (!is.null(model_object$models)) {
    possible_objects <- c(possible_objects, model_object$models)
  }

  possible_objects <- c(possible_objects, list(model_object))

  values <- c()

  for (obj in possible_objects) {
    for (nd in candidates) {
      pred <- tryCatch(
        predict(obj, newdata = nd),
        error = function(e) NULL
      )

      if (!is.null(pred)) {
        val <- suppressWarnings(as.numeric(pred[1]))
        if (!is.na(val)) {
          values <- c(values, val)
          break
        }
      }
    }
  }

  if (length(values) == 0) {
    stop("No se pudo calcular la glomeruloesclerosis.")
  }

  val <- mean(values, na.rm = TRUE)
  val <- max(0, min(100, val))
  val
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
    p_sev <- prob_ge2(p)

    lesion <- lesion_name(nm)

    if (p_sev >= 0.50 || cls >= 2) {
      txt <- c(
        txt,
        paste0(
          lesion, ": alta probabilidad de lesión moderada o grave ",
          "(grado ≥2: ", round(100 * p_sev, 1), "%). ",
          "Interpretar con especial cautela en el contexto del donante y considerar seguimiento estrecho."
        )
      )
    } else if (p_sev >= 0.25) {
      txt <- c(
        txt,
        paste0(
          lesion, ": probabilidad intermedia de lesión moderada o grave ",
          "(grado ≥2: ", round(100 * p_sev, 1), "%). ",
          "Recomendable contextualizar con edad, comorbilidad y función renal del donante."
        )
      )
    } else {
      txt <- c(
        txt,
        paste0(
          lesion, ": predominan grados ausente o leve. ",
          "Clase predicha: ", grade_label(cls), "."
        )
      )
    }
  }

  if (glo >= 20) {
    txt <- c(
      txt,
      paste0(
        "Glomeruloesclerosis estimada elevada: ", round(glo, 1),
        "%. Este hallazgo virtual sugiere mayor carga crónica y debe interpretarse junto al resto de datos clínicos."
      )
    )
  } else if (glo >= 10) {
    txt <- c(
      txt,
      paste0(
        "Glomeruloesclerosis estimada intermedia: ", round(glo, 1),
        "%. Puede justificar vigilancia adicional según el contexto clínico."
      )
    )
  } else {
    txt <- c(
      txt,
      paste0(
        "Glomeruloesclerosis estimada baja: ", round(glo, 1),
        "%."
      )
    )
  }

  paste(txt, collapse = "\n\n")
}

# ============================================================
# Interfaz
# ============================================================

ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      body {
        background-color: #f6f8fb;
        color: #1f2937;
      }

      .main-title {
        background: linear-gradient(135deg, #19324a, #295c7a);
        color: white;
        padding: 22px;
        border-radius: 16px;
        margin-bottom: 20px;
      }

      .main-title h1 {
        margin-top: 0;
        font-weight: 700;
      }

      .panel-card {
        background: white;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 16px;
      }

      .warning-card {
        background: #fff3cd;
        border-left: 6px solid #f0ad4e;
        padding: 14px;
        border-radius: 10px;
        margin-bottom: 16px;
      }

      .ok-card {
        background: #eaf7ee;
        border-left: 6px solid #2e8b57;
        padding: 14px;
        border-radius: 10px;
        margin-bottom: 16px;
      }

      .result-number {
        font-size: 28px;
        font-weight: 700;
        color: #19324a;
      }

      .small-muted {
        color: #6b7280;
        font-size: 13px;
      }

      h3 {
        font-weight: 700;
        color: #19324a;
      }
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
    div(
      class = "ok-card",
      strong("Modelos cargados correctamente. "),
      span("La aplicación está lista para calcular predicciones.")
    )
  },

  sidebarLayout(
    sidebarPanel(
      width = 4,

      div(
        class = "panel-card",
        h3("Datos del donante"),

        numericInput(
          inputId = "age",
          label = "Edad del donante (años)",
          value = 57,
          min = 0,
          max = 100,
          step = 1
        ),

        selectInput(
          inputId = "sex",
          label = "Sexo",
          choices = c("Mujer", "Hombre"),
          selected = "Mujer"
        ),

        selectInput(
          inputId = "donor_type",
          label = "Tipo de donante",
          choices = c("Vivo", "Fallecido"),
          selected = "Fallecido"
        ),

        conditionalPanel(
          condition = "input.donor_type == 'Fallecido'",

          selectInput(
            inputId = "vascular_death",
            label = "Causa de muerte cerebrovascular",
            choices = c("No", "Sí"),
            selected = "Sí"
          ),

          selectInput(
            inputId = "dcd",
            label = "Causa de muerte circulatoria / DCD",
            choices = c("No", "Sí"),
            selected = "No"
          )
        ),

        selectInput(
          inputId = "hypertension",
          label = "Hipertensión",
          choices = c("No", "Sí"),
          selected = "No"
        ),

        selectInput(
          inputId = "diabetes",
          label = "Diabetes mellitus",
          choices = c("No", "Sí"),
          selected = "No"
        ),

        selectInput(
          inputId = "hcv",
          label = "Estado VHC",
          choices = c("No", "Sí"),
          selected = "No"
        ),

        numericInput(
          inputId = "bmi",
          label = "Índice de masa corporal / BMI (kg/m²)",
          value = 20,
          min = 10,
          max = 70,
          step = 0.1
        ),

        numericInput(
          inputId = "creatinine",
          label = "Creatinina sérica más baja (mg/dL)",
          value = 0.6,
          min = 0.1,
          max = 15,
          step = 0.1
        ),

        selectInput(
          inputId = "proteinuria",
          label = "Proteinuria",
          choices = c("No", "Sí"),
          selected = "No"
        ),

        p(
          class = "small-muted",
          "Proteinuria positiva: tira reactiva ≥1 o UPCR ≥0.5 g/g."
        ),

        actionButton(
          inputId = "calculate",
          label = "Calcular biopsia virtual",
          class = "btn-primary",
          width = "100%"
        )
      )
    ),

    mainPanel(
      width = 8,

      tabsetPanel(
        tabPanel(
          "Resultados",
          br(),

          div(
            class = "panel-card",
            h3("Resumen de predicción"),
            uiOutput("summary_cards")
          ),

          div(
            class = "panel-card",
            h3("Probabilidades por lesión y grado Banff"),
            DT::DTOutput("probability_table")
          ),

          div(
            class = "panel-card",
            h3("Glomeruloesclerosis"),
            htmlOutput("glo_output")
          )
        ),

        tabPanel(
          "Gráfico radar",
          br(),
          div(
            class = "panel-card",
            h3("Radar de probabilidades"),
            plotlyOutput("radar_plot", height = "560px")
          )
        ),

        tabPanel(
          "Nota clínica",
          br(),
          div(
            class = "panel-card",
            h3("Interpretación clínica automática"),
            verbatimTextOutput("clinical_note")
          )
        ),

        tabPanel(
          "Ayuda",
          br(),
          div(
            class = "panel-card",
            h3("Qué calcula esta aplicación"),
            p("La aplicación estima, a partir de 11 variables del donante, las probabilidades de cada grado Banff para:"),
            tags$ul(
              tags$li("cv: arteriosclerosis."),
              tags$li("ah: hialinosis arteriolar."),
              tags$li("IFTA: fibrosis intersticial y atrofia tubular.")
            ),
            p("También estima el porcentaje continuo de glomeruloesclerosis."),

            h3("Advertencia"),
            p("Esta herramienta reproduce un sistema predictivo de investigación. No sustituye la valoración clínica, histológica ni la decisión del equipo de trasplante.")
          )
        )
      )
    )
  )
)

# ============================================================
# Servidor
# ============================================================

server <- function(input, output, session) {

  prediction_result <- eventReactive(input$calculate, {
    validate(
      need(models_available, "Faltan los archivos .rds en la carpeta models/.")
    )

    patient <- make_patient(input)

    probs <- list(
      cv = try_predict_probability(loaded_models$cv, patient),
      ah = try_predict_probability(loaded_models$ah, patient),
      IFTA = try_predict_probability(loaded_models$IFTA, patient)
    )

    glo <- try_predict_regression(loaded_models$glo, patient)

    list(
      patient = patient,
      probs = probs,
      glo = glo
    )
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
        tags$p(
          tags$strong("Clase predicha: "),
          grade_label(cls)
        ),
        tags$p(
          tags$strong("Probabilidad de grado ≥2: "),
          paste0(round(100 * psev, 1), "%")
        )
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

    DT::datatable(
      tab,
      rownames = FALSE,
      options = list(
        dom = "t",
        pageLength = 3
      )
    )
  })

  output$glo_output <- renderUI({
    res <- prediction_result()

    HTML(
      paste0(
        "<div class='result-number'>",
        round(res$glo, 2),
        "%</div>",
        "<p>Porcentaje estimado de glomeruloesclerosis.</p>"
      )
    )
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
        grado = factor(
          c("Grado 0", "Grado 1", "Grado 2", "Grado 3"),
          levels = c("Grado 0", "Grado 1", "Grado 2", "Grado 3")
        ),
        prob = as.numeric(p[1, paste0("X", 0:3)])
      )
    })

    plotly::plot_ly(type = "scatterpolar", mode = "lines+markers") %>%
      plotly::add_trace(
        data = radar_data[radar_data$lesion == "Arteriosclerosis (cv)", ],
        r = ~prob,
        theta = ~grado,
        name = "cv",
        fill = "toself"
      ) %>%
      plotly::add_trace(
        data = radar_data[radar_data$lesion == "Hialinosis arteriolar (ah)", ],
        r = ~prob,
        theta = ~grado,
        name = "ah",
        fill = "toself"
      ) %>%
      plotly::add_trace(
        data = radar_data[radar_data$lesion == "Fibrosis intersticial y atrofia tubular (IFTA)", ],
        r = ~prob,
        theta = ~grado,
        name = "IFTA",
        fill = "toself"
      ) %>%
      plotly::layout(
        polar = list(
          radialaxis = list(
            visible = TRUE,
            range = c(0, 1),
            tickformat = ".0%"
          )
        ),
        showlegend = TRUE,
        title = "Probabilidades predichas por grado Banff"
      )
  })
}

shinyApp(ui = ui, server = server)