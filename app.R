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

alg_class <- c("rf", "gbm", "xgbTree", "lda", "avNNet", "mnom", "multinom")
alg_reg   <- c("rf", "gbm", "xgbTree", "avNNet", "ens_model")

as01 <- function(x) as.integer(isTRUE(x))

patient_values <- function(input) {
  deceased <- identical(input$donor_type, "Fallecido")
  female <- identical(input$sex, "Mujer")
  list(
    Age = as.numeric(input$age),
    Gender = as01(female),
    Gender1 = as01(female),
    Donor_type = as01(deceased),
    Donor_type1 = as01(deceased),
    Hypertension = as01(identical(input$hypertension, "Sí")),
    Hypertension1 = as01(identical(input$hypertension, "Sí")),
    Diabetes = as01(identical(input$diabetes, "Sí")),
    Diabetes1 = as01(identical(input$diabetes, "Sí")),
    Creatinine = as.numeric(input$creatinine),
    Proteinuria = as01(identical(input$proteinuria, "Sí")),
    Proteinuria1 = as01(identical(input$proteinuria, "Sí")),
    HCV_status = as01(identical(input$hcv, "Sí")),
    HCV_status1 = as01(identical(input$hcv, "Sí")),
    DCD = as01(deceased && identical(input$dcd, "Sí")),
    DCD1 = as01(deceased && identical(input$dcd, "Sí")),
    bmi = as.numeric(input$bmi),
    BMI = as.numeric(input$bmi),
    vascular_death = as01(deceased && identical(input$vascular_death, "Sí")),
    vascular_death1 = as01(deceased && identical(input$vascular_death, "Sí"))
  )
}

make_df <- function(vals, names_vec, binary_type = c("numeric", "factor01", "factorNoYes", "logical", "character01")) {
  binary_type <- match.arg(binary_type)
  out <- list()
  for (nm in names_vec) {
    v <- vals[[nm]]
    if (is.null(v)) {
      nm0 <- sub("1$", "", nm)
      v <- vals[[nm0]]
    }
    if (is.null(v)) v <- 0
    if (nm %in% c("Age", "Creatinine", "bmi", "BMI")) {
      out[[nm]] <- as.numeric(v)
    } else {
      if (binary_type == "numeric") out[[nm]] <- as.numeric(v)
      if (binary_type == "factor01") out[[nm]] <- factor(as.character(as.integer(v)), levels = c("0", "1"))
      if (binary_type == "factorNoYes") out[[nm]] <- factor(ifelse(as.integer(v) == 1, "Yes", "No"), levels = c("No", "Yes"))
      if (binary_type == "logical") out[[nm]] <- as.logical(as.integer(v))
      if (binary_type == "character01") out[[nm]] <- as.character(as.integer(v))
    }
  }
  as.data.frame(out, check.names = FALSE)
}

candidate_newdata <- function(vals) {
  orig <- c("Age", "Gender", "Donor_type", "Hypertension", "Diabetes", "Creatinine", "Proteinuria", "HCV_status", "DCD", "bmi", "vascular_death")
  dum  <- c("Age", "Gender1", "Donor_type1", "Hypertension1", "Diabetes1", "Creatinine", "Proteinuria1", "HCV_status1", "DCD1", "bmi", "vascular_death1")
  both <- unique(c(orig, dum, "BMI"))
  types <- c("factor01", "numeric", "factorNoYes", "logical", "character01")
  c(
    lapply(types, function(t) make_df(vals, orig, t)),
    lapply(types, function(t) make_df(vals, dum, t)),
    lapply(types, function(t) make_df(vals, both, t))
  )
}

model_candidates <- function(container, regression = FALSE) {
  if (is.null(container)) return(list())
  wanted <- if (regression) alg_reg else alg_class
  out <- list()
  if (inherits(container, "train") || inherits(container, "caretEnsemble") || inherits(container, "caretStack")) {
    out[["model"]] <- container
  }
  if (is.list(container)) {
    nms <- names(container)
    if (!is.null(nms)) {
      for (nm in intersect(wanted, nms)) {
        if (!is.null(container[[nm]])) out[[nm]] <- container[[nm]]
      }
      if (!regression && "mnLogit" %in% nms) out[["mnLogit"]] <- container[["mnLogit"]]
      if (regression && "models" %in% nms && is.list(container$models)) {
        for (nm in intersect(alg_reg, names(container$models))) out[[paste0("models_", nm)]] <- container$models[[nm]]
      }
    }
  }
  out
}

clean_prob <- function(x) {
  p <- as.data.frame(x, check.names = FALSE)
  if (nrow(p) < 1) stop("predicción vacía")
  n <- names(p)
  n2 <- gsub("^X", "", n)
  n2 <- gsub("[^0-9]", "", n2)
  names(p) <- n2
  for (k in as.character(0:3)) if (!k %in% names(p)) p[[k]] <- 0
  p <- p[, as.character(0:3), drop = FALSE]
  p <- as.data.frame(lapply(p, as.numeric), check.names = FALSE)
  p[is.na(p)] <- 0
  s <- rowSums(p)
  if (any(s <= 0)) p[,] <- 0.25 else p <- p / s
  names(p) <- paste0("X", 0:3)
  p[1, , drop = FALSE]
}

predict_one_prob <- function(m, nds) {
  for (nd in nds) {
    pr <- tryCatch(predict(m, newdata = nd, type = "prob"), error = function(e) NULL)
    if (!is.null(pr)) {
      cp <- tryCatch(clean_prob(pr), error = function(e) NULL)
      if (!is.null(cp)) return(cp)
    }
  }
  NULL
}

predict_probs <- function(container, vals) {
  mods <- model_candidates(container, regression = FALSE)
  if (length(mods) == 0) stop("Los RDS no exponen modelos predictivos de clasificación utilizables.")
  nds <- candidate_newdata(vals)
  probs <- list()
  for (nm in names(mods)) {
    pp <- predict_one_prob(mods[[nm]], nds)
    if (!is.null(pp)) probs[[nm]] <- pp
  }
  if (length(probs) == 0) stop("No se pudo obtener ninguna probabilidad válida desde los modelos.")
  arr <- simplify2array(lapply(probs, as.matrix))
  clean_prob(as.data.frame(apply(arr, c(1, 2), mean), check.names = FALSE))
}

predict_one_reg <- function(m, nds) {
  for (nd in nds) {
    pr <- tryCatch(predict(m, newdata = nd), error = function(e) NULL)
    if (!is.null(pr)) {
      val <- suppressWarnings(as.numeric(pr[1]))
      if (!is.na(val)) return(val)
    }
  }
  NULL
}

predict_glo <- function(container, vals) {
  mods <- model_candidates(container, regression = TRUE)
  if (length(mods) == 0) stop("Los RDS no exponen modelos predictivos de glomeruloesclerosis utilizables.")
  nds <- candidate_newdata(vals)
  values <- numeric(0)
  for (nm in names(mods)) {
    vv <- predict_one_reg(mods[[nm]], nds)
    if (!is.null(vv)) values <- c(values, vv)
  }
  if (length(values) == 0) stop("No se pudo obtener predicción válida de glomeruloesclerosis.")
  max(0, min(100, mean(values, na.rm = TRUE)))
}

predicted_class <- function(p) which.max(as.numeric(p[1, paste0("X", 0:3)])) - 1
prob_ge2 <- function(p) as.numeric(p$X2 + p$X3)
lesion_name <- function(x) switch(x, cv = "Arteriosclerosis (cv)", ah = "Hialinosis arteriolar (ah)", IFTA = "Fibrosis intersticial y atrofia tubular (IFTA)")
grade_label <- function(x) unname(c("0" = "0 - ausente", "1" = "1 - leve", "2" = "2 - moderada", "3" = "3 - grave")[as.character(x)])

clinical_note <- function(probs, glo) {
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
  if (glo >= 20) {
    txt <- c(txt, paste0("Glomeruloesclerosis estimada elevada: ", round(glo, 1), "%. Sugiere mayor carga crónica virtual y requiere valoración clínica cuidadosa."))
  } else if (glo >= 10) {
    txt <- c(txt, paste0("Glomeruloesclerosis estimada intermedia: ", round(glo, 1), "%. Puede justificar vigilancia adicional según el contexto clínico."))
  } else {
    txt <- c(txt, paste0("Glomeruloesclerosis estimada baja: ", round(glo, 1), "%."))
  }
  paste(txt, collapse = "\n\n")
}

ui <- fluidPage(
  tags$head(tags$style(HTML("
    body { background-color:#f6f8fb; color:#1f2937; }
    .main-title { background:linear-gradient(135deg,#19324a,#295c7a); color:white; padding:22px; border-radius:16px; margin-bottom:20px; }
    .main-title h1 { margin-top:0; font-weight:700; }
    .panel-card { background:white; border-radius:16px; padding:18px; box-shadow:0 2px 10px rgba(0,0,0,0.08); margin-bottom:16px; }
    .ok-card { background:#eaf7ee; border-left:6px solid #2e8b57; padding:14px; border-radius:10px; margin-bottom:16px; }
    .warning-card { background:#fff3cd; border-left:6px solid #f0ad4e; padding:14px; border-radius:10px; margin-bottom:16px; }
    .result-number { font-size:30px; font-weight:700; color:#19324a; }
    .small-muted { color:#6b7280; font-size:13px; }
    h3 { font-weight:700; color:#19324a; }
  "))),
  div(class = "main-title", h1("The Virtual Biopsy System"), h4("Biopsia virtual día cero para trasplante renal"), p("Predicción de lesiones Banff y porcentaje de glomeruloesclerosis a partir de parámetros clínicos del donante.")),
  if (!models_available) div(class = "warning-card", h4("Modelos no encontrados"), p("Faltan los cuatro .rds dentro de models/.")) else div(class = "ok-card", strong("Modelos cargados correctamente. "), span("La aplicación está lista para calcular predicciones.")),
  sidebarLayout(
    sidebarPanel(width = 4, div(class = "panel-card", h3("Datos del donante"),
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
    )),
    mainPanel(width = 8, tabsetPanel(
      tabPanel("Resultados", br(), div(class = "panel-card", h3("Resumen de predicción"), uiOutput("summary_cards")), div(class = "panel-card", h3("Probabilidades por lesión y grado Banff"), DTOutput("probability_table")), div(class = "panel-card", h3("Glomeruloesclerosis"), htmlOutput("glo_output"))),
      tabPanel("Gráfico radar", br(), div(class = "panel-card", h3("Radar de probabilidades"), plotlyOutput("radar_plot", height = "560px"))),
      tabPanel("Nota clínica", br(), div(class = "panel-card", h3("Interpretación clínica automática"), verbatimTextOutput("clinical_note"))),
      tabPanel("Ayuda", br(), div(class = "panel-card", h3("Qué calcula esta aplicación"), p("La aplicación estima probabilidades de cada grado Banff para cv, ah e IFTA, y el porcentaje continuo de glomeruloesclerosis."), h3("Advertencia"), p("Herramienta predictiva de investigación; no sustituye la valoración clínica o histológica.")))
    ))
  )
)

server <- function(input, output, session) {
  prediction_result <- eventReactive(input$calculate, {
    validate(need(models_available, "Faltan los archivos .rds en la carpeta models/."))
    vals <- patient_values(input)
    probs <- list(cv = predict_probs(models$cv, vals), ah = predict_probs(models$ah, vals), IFTA = predict_probs(models$IFTA, vals))
    glo <- predict_glo(models$glo, vals)
    list(probs = probs, glo = glo)
  }, ignoreInit = TRUE)

  output$summary_cards <- renderUI({
    res <- prediction_result()
    tagList(lapply(names(res$probs), function(nm) {
      p <- res$probs[[nm]]; cls <- predicted_class(p); sev <- prob_ge2(p)
      div(style = "border-bottom:1px solid #e5e7eb; padding:12px 0;", h4(lesion_name(nm)), tags$p(tags$strong("Clase predicha: "), grade_label(cls)), tags$p(tags$strong("Probabilidad de grado ≥2: "), paste0(round(100 * sev, 1), "%")))
    }))
  })

  output$probability_table <- renderDT({
    res <- prediction_result()
    tab <- imap_dfr(res$probs, function(p, nm) tibble(Lesión = lesion_name(nm), `Grado 0 (%)` = round(100 * p$X0, 1), `Grado 1 (%)` = round(100 * p$X1, 1), `Grado 2 (%)` = round(100 * p$X2, 1), `Grado 3 (%)` = round(100 * p$X3, 1), `Clase predicha` = grade_label(predicted_class(p)), `Probabilidad grado ≥2 (%)` = round(100 * prob_ge2(p), 1)))
    datatable(tab, rownames = FALSE, options = list(dom = "t", pageLength = 3))
  })

  output$glo_output <- renderUI({
    res <- prediction_result()
    HTML(paste0("<div class='result-number'>", round(res$glo, 2), "%</div><p>Porcentaje estimado de glomeruloesclerosis.</p>"))
  })

  output$clinical_note <- renderText({
    res <- prediction_result()
    clinical_note(res$probs, res$glo)
  })

  output$radar_plot <- renderPlotly({
    res <- prediction_result()
    radar_data <- imap_dfr(res$probs, function(p, nm) tibble(lesion = lesion_name(nm), grado = factor(c("Grado 0", "Grado 1", "Grado 2", "Grado 3"), levels = c("Grado 0", "Grado 1", "Grado 2", "Grado 3")), prob = as.numeric(p[1, paste0("X", 0:3)])))
    plot_ly(type = "scatterpolar", mode = "lines+markers") %>%
      add_trace(data = radar_data[radar_data$lesion == "Arteriosclerosis (cv)", ], r = ~prob, theta = ~grado, name = "cv", fill = "toself") %>%
      add_trace(data = radar_data[radar_data$lesion == "Hialinosis arteriolar (ah)", ], r = ~prob, theta = ~grado, name = "ah", fill = "toself") %>%
      add_trace(data = radar_data[radar_data$lesion == "Fibrosis intersticial y atrofia tubular (IFTA)", ], r = ~prob, theta = ~grado, name = "IFTA", fill = "toself") %>%
      layout(polar = list(radialaxis = list(visible = TRUE, range = c(0, 1), tickformat = ".0%")), showlegend = TRUE, title = "Probabilidades predichas por grado Banff")
  })
}

shinyApp(ui = ui, server = server)
